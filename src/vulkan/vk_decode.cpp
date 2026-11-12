// SPDX-License-Identifier: MIT
// Vulkan decode slot (see core/decode_backend.h). The host has located every code-block's
// bytes; here the frame's codestream and a table of block descriptors are copied to the
// device, one invocation decodes each block into its place in the wavelet planes, and
// the inverse wavelet and inverse color transform turn those into the picture, which is
// copied back to host memory. One submission per frame, on the slot's own queue.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/decode_backend.h"
#include "core/t1_dec_tables.h"
#include "dec_color_spv.h"
#include "dec_idwt_cols_spv.h"
#include "dec_idwt_rows_spv.h"
#include "dec_t1_spv.h"
#include "vulkan/vk_device.h"

namespace gail {

namespace {

enum DecodePipe { D_T1, D_ROWS, D_COLS, D_COLOR };

// Push-constant blocks (must match the GLSL declarations).
struct PcT1 {
    uint64_t bytes, descs, planes, tables;
    uint32_t nbytes, ndesc, planeWords, stride;
};
struct PcIdwt {
    uint64_t src, dst;
    uint32_t stride, planeSize, w, h;
};
struct PcColorOut {
    uint64_t planes, dst;
    uint32_t stride, planeSize, w, h, shift;
};

// One code-block as dec_t1.comp sees it (6 words).
struct BlockDesc {
    uint32_t off, len;
    uint32_t geom;  // (w - 1) | (h - 1) << 5 | orient << 10 | nbps << 12 | passes << 18
    uint32_t dst;   // index of its top-left sample in the planes; kNoBlock: an empty slot
    float halfStep;
    uint32_t pad;
};
constexpr uint32_t kNoBlock = 0xFFFFFFFFu;

enum { TS_START, TS_T1, TS_DWT, TS_END, TS_COUNT };

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class VulkanDecoderSlot : public DecoderSlot {
public:
    VulkanDecoderSlot(VulkanDevice& d, const FrameLayout& L, const DecodeSettings& s) : d_(d), L_(L), s_(s) {
        try {
            init();
        } catch (...) {
            release();
            throw;
        }
    }
    ~VulkanDecoderSlot() override { release(); }

    void decode(const FrameLayout& frame, const uint8_t* data, size_t size, const std::vector<CblkCode>& blocks, void* dst,
                ptrdiff_t strideBytes, DecodeTimings* tm) override {
        if (!sameGeometry(frame, L_) || (int)blocks.size() != L_.numCblks())
            throw std::logic_error("frame does not match the layout the slot was made for");
        if (size > 0xFFFFFFFFu) throw std::runtime_error("codestream too large");
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t W = (uint32_t)L_.width, H = (uint32_t)L_.height, npix = W * H;

        // The coded blocks, longest first, packed into subgroups of 32 by a byte budget: a
        // subgroup runs as long as its longest invocation, so a long block gets one to
        // itself and short ones share. Slots left over are marked empty.
        order_.clear();
        for (int i : wanted_)
            if (blocks[i].passes && (size_t)blocks[i].offset + blocks[i].len <= size) order_.push_back(i);
        std::sort(order_.begin(), order_.end(), [&](int a, int b) { return blocks[a].len > blocks[b].len; });
        ensureInput(size, order_.size() * 32 + 32);
        BlockDesc* descs = (BlockDesc*)hostDescs_.map;
        uint32_t n = 0, inGroup = 0, groupBytes = 0;
        const BlockDesc empty{0, 0, 0, kNoBlock, 0.f, 0};
        for (int i : order_) {
            const CblkCode& c = blocks[i];
            if (inGroup && (inGroup == 32 || groupBytes + c.len > budget_)) {
                for (; inGroup < 32; ++inGroup) descs[n++] = empty;
                inGroup = 0;
                groupBytes = 0;
            }
            const CodeBlock& cb = L_.cblks[i];
            const Subband& b = frame.bands[cb.band];
            BlockDesc d;
            d.off = c.offset;
            d.len = c.len;
            d.geom = (uint32_t)(cb.w - 1) | ((uint32_t)(cb.h - 1) << 5) | ((uint32_t)b.orient << 10) | ((uint32_t)c.nbps << 12) |
                     ((uint32_t)c.passes << 18);
            d.dst = (uint32_t)b.comp * npix + (uint32_t)(b.py0 + cb.y0) * W + (uint32_t)(b.px0 + cb.x0);
            d.halfStep = 0.5f * b.delta;
            d.pad = 0;
            descs[n++] = d;
            ++inGroup;
            groupBytes += c.len;
        }
        std::memcpy(hostBytes_.map, data, size);

        const bool tapWavelet = tm && tm->wavelet;
        begin();
        if (queries_) vkCmdResetQueryPool(cmd_, queries_, 0, TS_COUNT);
        stamp(TS_START);
        VkBufferCopy cBytes{0, 0, (VkDeviceSize)((size + 3) & ~(size_t)3)};
        vkCmdCopyBuffer(cmd_, hostBytes_.buf, bytes_.buf, 1, &cBytes);
        if (n) {
            VkBufferCopy cDescs{0, 0, (VkDeviceSize)n * sizeof(BlockDesc)};
            vkCmdCopyBuffer(cmd_, hostDescs_.buf, descs_.buf, 1, &cDescs);
        }
        vkCmdFillBuffer(cmd_, planesA_.buf, 0, (VkDeviceSize)npix * 12, 0);
        barrier();
        if (n) {
            const PcT1 pc{bytes_.addr, descs_.addr, planesA_.addr, tables_.addr, (uint32_t)size, n, npix * 3, W};
            dispatch(D_T1, pc, (n + 63) / 64);
            barrier();
        }
        stamp(TS_T1);
        if (tapWavelet) {  // tests: the dequantized coefficients, before the wavelet
            submitAndWait();
            for (int c = 0; c < 3; ++c) {
                tm->wavelet[c].resize(npix);
                readBack(planesA_, (VkDeviceSize)c * npix * 4, tm->wavelet[c].data(), (size_t)npix * 4);
            }
            begin();
        }
        for (int l = L_.levels; l > s_.reduce; --l) {
            const uint32_t w = (uint32_t)ws_[l - 1], h = (uint32_t)hs_[l - 1];
            const PcIdwt r{planesA_.addr, planesB_.addr, W, npix, w, h};
            dispatch(D_ROWS, r, h, 3);
            barrier();
            const PcIdwt k{planesB_.addr, planesA_.addr, W, npix, w, h};
            dispatch(D_COLS, k, (w + 31) / 32, (h + 63) / 64, 3);
            barrier();
        }
        if (!tapWavelet) stamp(TS_DWT);
        const PcColorOut pc{planesA_.addr, out_.addr, W, npix, (uint32_t)outW_, (uint32_t)outH_, s_.highBits ? 4u : 0u};
        dispatch(D_COLOR, pc, ((uint32_t)outW_ + 255) / 256, (uint32_t)outH_);
        barrier();
        const size_t outBytes = (size_t)outW_ * outH_ * 6;
        VkBufferCopy cOut{0, 0, (VkDeviceSize)outBytes};
        vkCmdCopyBuffer(cmd_, out_.buf, hostOut_.buf, 1, &cOut);
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        if (!tapWavelet) stamp(TS_END);
        submitAndWait();

        const size_t rowBytes = (size_t)outW_ * 6;
        if (strideBytes == (ptrdiff_t)rowBytes) {
            std::memcpy(dst, hostOut_.map, outBytes);
        } else {
            for (int y = 0; y < outH_; ++y)
                std::memcpy((uint8_t*)dst + (ptrdiff_t)y * strideBytes, (uint8_t*)hostOut_.map + (size_t)y * rowBytes, rowBytes);
        }
        if (tm) {
            uint64_t ts[TS_COUNT];
            if (queries_ && !tapWavelet &&
                vkGetQueryPoolResults(d_.dev_, queries_, 0, TS_COUNT, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                const double ms = d_.timestampPeriod_ * 1e-6;
                tm->t1 = (double)(ts[TS_T1] - ts[TS_START]) * ms;
                tm->dwt = (double)(ts[TS_DWT] - ts[TS_T1]) * ms;
                tm->color = (double)(ts[TS_END] - ts[TS_DWT]) * ms;
            }
            tm->total = msSince(t0);
        }
    }

private:
    void init() {
        if (s_.reduce < 0 || s_.reduce > L_.levels) throw std::runtime_error("more resolution levels left out than the stream has");
        for (int p = 0; p < VulkanDevice::kDecodePipes; ++p) d_.decodePipe(p);
        const uint32_t qi = d_.nextQueue_.fetch_add(1);
        queue_ = &d_.queues_[qi % d_.queues_.size()];
        // Bytes a subgroup of code-blocks may hold between them (see decode()).
        // GAIL_VK_DECODE_PACK overrides.
        if (const char* v = std::getenv("GAIL_VK_DECODE_PACK")) budget_ = (uint32_t)std::max(1, std::atoi(v));
        ws_.resize(L_.levels + 1);
        hs_.resize(L_.levels + 1);
        ws_[0] = L_.width;
        hs_[0] = L_.height;
        for (int l = 1; l <= L_.levels; ++l) {
            ws_[l] = (ws_[l - 1] + 1) / 2;
            hs_[l] = (hs_[l - 1] + 1) / 2;
        }
        decodedSize(L_, s_.reduce, outW_, outH_);
        for (int i = 0; i < L_.numCblks(); ++i)
            if (L_.bands[L_.cblks[i].band].res <= L_.levels - s_.reduce) wanted_.push_back(i);
        const size_t npix = (size_t)L_.width * L_.height;
        const size_t outBytes = (size_t)outW_ * outH_ * 6;
        planesA_ = d_.createBuffer(npix * 12, 0, kDevLocal);
        planesB_ = d_.createBuffer(npix * 12, 0, kDevLocal);
        out_ = d_.createBuffer(outBytes + 64, 0, kDevLocal);
        hostOut_ = d_.createBuffer(outBytes + 64, 0, kHostCached, kHostCoherent);
        tables_ = d_.createBuffer(kT1dZcSize + kT1dScSize, 0, kDevLocal);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = d_.family_;
        VKCHECK(vkCreateCommandPool(d_.dev_, &pci, nullptr, &pool_));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(d_.dev_, &cai, &cmd_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKCHECK(vkCreateFence(d_.dev_, &fci, nullptr, &fence_));
        if (d_.timestamps_) {
            VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = TS_COUNT;
            VKCHECK(vkCreateQueryPool(d_.dev_, &qpci, nullptr, &queries_));
        }
        ensureInput(1500000, (size_t)wanted_.size() + 32);  // a DCI frame is at most 250 Mbit/s over 24 frames

        // The context tables go to the device once, through the input staging buffer
        uint8_t* t = (uint8_t*)hostBytes_.map;
        buildT1DecodeTables(t, t + kT1dZcSize);
        begin();
        VkBufferCopy c{0, 0, kT1dZcSize + kT1dScSize};
        vkCmdCopyBuffer(cmd_, hostBytes_.buf, tables_.buf, 1, &c);
        submitAndWait();
    }

    // Room for a frame of `bytes` and `ndesc` descriptors, on the host and on the device.
    void ensureInput(size_t bytes, size_t ndesc) {
        if (bytes > capBytes_) {
            waitIdle();
            d_.destroy(hostBytes_);
            d_.destroy(bytes_);
            capBytes_ = 0;
            const size_t cap = (bytes + bytes / 2 + 3) & ~(size_t)3;
            hostBytes_ = d_.createBuffer(cap, 0, kHostCoherent);
            bytes_ = d_.createBuffer(cap, 0, kDevLocal);
            capBytes_ = cap;
        }
        if (ndesc > capDescs_) {
            waitIdle();
            d_.destroy(hostDescs_);
            d_.destroy(descs_);
            capDescs_ = 0;
            const size_t cap = ndesc + ndesc / 2;
            hostDescs_ = d_.createBuffer(cap * sizeof(BlockDesc), 0, kHostCoherent);
            descs_ = d_.createBuffer(cap * sizeof(BlockDesc), 0, kDevLocal);
            capDescs_ = cap;
        }
    }

    void begin() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd_, &bi));
    }
    void submitAndWait() {
        VKCHECK(vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VKCHECK(vkResetFences(d_.dev_, 1, &fence_));
        {
            std::lock_guard<std::mutex> l(queue_->m);
            VKCHECK(vkQueueSubmit(queue_->q, 1, &si, fence_));
        }
        VKCHECK(vkWaitForFences(d_.dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
    }
    void waitIdle() {
        if (!queue_) return;
        std::lock_guard<std::mutex> l(queue_->m);
        vkQueueWaitIdle(queue_->q);
    }
    // One stage's writes are visible to the next.
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
    }
    template <typename PC> void dispatch(DecodePipe p, const PC& pc, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        static_assert(sizeof(PC) <= kPushSize, "push constants too large");
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, d_.decodePipe(p));
        vkCmdPushConstants(cmd_, d_.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cmd_, x, y, z);
    }
    void stamp(uint32_t which) {
        if (queries_) vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries_, which);
    }
    // Copies part of a device buffer to host memory (tests only: one submission each).
    void readBack(Buffer& b, VkDeviceSize offset, void* dst, size_t bytes) {
        Buffer tmp = d_.createBuffer(bytes, 0, kHostCoherent);
        begin();
        VkBufferCopy c{offset, 0, (VkDeviceSize)bytes};
        vkCmdCopyBuffer(cmd_, b.buf, tmp.buf, 1, &c);
        submitAndWait();
        std::memcpy(dst, tmp.map, bytes);
        d_.destroy(tmp);
    }

    void release() {
        waitIdle();
        if (queries_) vkDestroyQueryPool(d_.dev_, queries_, nullptr);
        if (fence_) vkDestroyFence(d_.dev_, fence_, nullptr);
        if (pool_) vkDestroyCommandPool(d_.dev_, pool_, nullptr);
        queries_ = VK_NULL_HANDLE;
        fence_ = VK_NULL_HANDLE;
        pool_ = VK_NULL_HANDLE;
        for (Buffer* b : {&planesA_, &planesB_, &out_, &hostOut_, &tables_, &hostBytes_, &bytes_, &hostDescs_, &descs_}) d_.destroy(*b);
    }

    VulkanDevice& d_;
    const FrameLayout L_;
    const DecodeSettings s_;
    VulkanDevice::Queue* queue_ = nullptr;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    Buffer planesA_, planesB_, out_, hostOut_, tables_, hostBytes_, bytes_, hostDescs_, descs_;
    size_t capBytes_ = 0, capDescs_ = 0;
    uint32_t budget_ = 2000;
    std::vector<int> ws_, hs_, wanted_, order_;
    int outW_ = 0, outH_ = 0;
};

}  // namespace

VkPipeline VulkanDevice::decodePipe(int which) {
    std::lock_guard<std::mutex> l(decodeMutex_);
    // None of the decoder's shaders uses subgroup operations. Each is made on first use.
    if (!decodePipes_[which]) {
        switch (which) {
        case D_T1: decodePipes_[which] = makePipe(dec_t1_spv, sizeof dec_t1_spv, false); break;
        case D_ROWS: decodePipes_[which] = makePipe(dec_idwt_rows_spv, sizeof dec_idwt_rows_spv, false); break;
        case D_COLS: decodePipes_[which] = makePipe(dec_idwt_cols_spv, sizeof dec_idwt_cols_spv, false); break;
        default: decodePipes_[which] = makePipe(dec_color_spv, sizeof dec_color_spv, false); break;
        }
    }
    return decodePipes_[which];
}

std::unique_ptr<DecoderSlot> VulkanDevice::createDecoderSlot(const FrameLayout& L, const DecodeSettings& s) {
    return std::make_unique<VulkanDecoderSlot>(*this, L, s);
}

}  // namespace gail

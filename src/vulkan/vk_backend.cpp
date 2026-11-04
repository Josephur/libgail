// SPDX-License-Identifier: MIT
// Vulkan compute backend: colour conversion, 9/7 DWT, quantisation and EBCOT
// Tier-1 (context modelling + MQ coding) on any Vulkan 1.3 GPU.
//
// Each EncoderSlot owns a full set of device buffers and a command buffer that is
// recorded once and resubmitted for every frame, so several frames can be in
// flight per GPU (one per slot) while the CPU runs rate control / Tier-2.
#include "vulkan/vk_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "color_spv.h"
#include "color_yuv_spv.h"
#include "compact_spv.h"
#include "core/backend.h"
#include "core/t1_ref.h"
#include "kernels/color_params.h"
#include "dwt_cols_spv.h"
#include "dwt_rows_spv.h"
#include "quantize_spv.h"
#include "order_spv.h"
#include "scan_spv.h"
#include "t1_cm_spv.h"
#include "t1_mq_spv.h"

namespace gail {

namespace {

#define VKCHECK(x)                                                                                   \
    do {                                                                                             \
        VkResult r_ = (x);                                                                           \
        if (r_ != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan error ") + std::to_string(r_) + " at " #x); \
    } while (0)

constexpr uint32_t kPushSize = 128;

VkInstance g_instance = VK_NULL_HANDLE;
std::mutex g_instanceMutex;

// The shared instance, or VK_NULL_HANDLE when Vulkan is unusable here: no loader, or a
// loader without a driver (VK_ERROR_INCOMPATIBLE_DRIVER on a machine without a Vulkan
// GPU, e.g. a VM or server). Either way the Vulkan backend just has no devices.
VkInstance instance() {
    std::lock_guard<std::mutex> l(g_instanceMutex);
    static bool unusable = false;
    if (g_instance || unusable) return g_instance;
    unusable = true;  // until proven otherwise
    if (!vkLoadLoader()) return VK_NULL_HANDLE;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gail";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    if (const char* v = std::getenv("GAIL_VK_VALIDATION"); v && *v == '1') {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = layers;
    }
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (!vkLoadInstance(inst)) return VK_NULL_HANDLE;  // a loader older than Vulkan 1.3
    g_instance = inst;
    unusable = false;
    return g_instance;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress addr = 0;
    void* map = nullptr;
};

// Push-constant blocks (must match the GLSL declarations).
struct PcColor {
    uint64_t rgb, lut, params, planes, xyz, lut3d;
    uint32_t numPixels;
};
struct PcColorYuv {
    uint64_t src, lut, params, planes, xyz, lut3d;
    uint32_t numPixels, width, active, offset, fmt;
    float crR, cbG, crG, cbB;
};
struct PcDwt {
    uint64_t src, dst;
    uint32_t stride, planeSize, w, h;
};
struct PcQuant {
    uint64_t planes, cblkTab, bandTab, coeffs;
    uint32_t numCblks, stride, planeSize;
};
struct PcT1cm {
    uint64_t planes, cblkTab, info, passEnd, passDist, chunks, symbols, counters, bandTab, frameParams;
    uint32_t numCblks, symCapacity, maxPasses, stride, planeSize;
};
struct PcT1mq {
    uint64_t info, passEnd, passLen, chunks, symbols, outBuf, counters, perm;
    uint32_t numCblks, outCapacity;
};
struct PcOrder {
    uint64_t info, passEnd, perm;
    uint32_t numCblks;
};
struct PcScan {
    uint64_t info, offsets;
    uint32_t numCblks;
};
struct PcCompact {
    uint64_t info, offsets, src, passLen, passDist, hostInfo, hostPass, hostData;
    uint32_t numCblks, dataCapacity, passCapacity;
};

enum Pipe { P_COLOR, P_COLOR_YUV, P_ROWS, P_COLS, P_QUANT, P_T1CM, P_T1MQ, P_SCAN, P_COMPACT, P_ORDER, P_COUNT };

class VulkanDevice : public Device {
public:
    VulkanDevice(VkPhysicalDevice phys, const Logger& log) : phys_(phys), log_(log) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        name_ = props.deviceName;
        timestampPeriod_ = props.limits.timestampPeriod;
        vkGetPhysicalDeviceMemoryProperties(phys, &memProps_);

        VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceSubgroupSizeControlProperties sscp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        sgp.pNext = &sscp;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sgp;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        const VkSubgroupFeatureFlags need = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                            VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                            VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        if ((sgp.supportedOperations & need) != need || !(sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error(name_ + ": missing subgroup operations");
        if (sgp.subgroupSize != 32) {
            if (sscp.minSubgroupSize <= 32 && sscp.maxSubgroupSize >= 32 &&
                (sscp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
                forceSubgroup32_ = true;
            else
                throw std::runtime_error(name_ + ": Tier-1 kernel needs 32-wide subgroups");
        }

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
        family_ = UINT32_MAX;
        for (uint32_t i = 0; i < nq; ++i)  // prefer a dedicated compute queue
            if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                family_ = i;
                break;
            }
        if (family_ == UINT32_MAX)
            for (uint32_t i = 0; i < nq; ++i)
                if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    family_ = i;
                    break;
                }
        if (family_ == UINT32_MAX) throw std::runtime_error("no compute queue");
        if (!qf[family_].timestampValidBits) timestamps_ = false;
        // A transfer-only family maps to the copy (DMA) engines: uploads there overlap with
        // compute work of other frames in flight.
        xferFamily_ = family_;
        for (uint32_t i = 0; i < nq; ++i)
            if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(qf[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
                xferFamily_ = i;
                break;
            }

        // Several queues per family: each frame slot submits to its own queue, so the
        // stage barriers of one frame do not serialise against other frames and their
        // kernels overlap on the GPU (hiding the serial MQ tail and small dispatches).
        // Measured with 4 frames in flight (1 / 2 / 8 queues): RTX 5060 Ti 4K 75 / 97 / 95
        // fps, 2K 132 / 205 / 232 fps; RTX 3090 2K 134 / 184 / 217 fps, but 4K 101 / 87 / 82
        // fps (frames contend there; CUDA is the backend of choice on NVIDIA anyway).
        // GAIL_VK_QUEUES overrides.
        uint32_t wantQueues = 8;
        if (const char* q = std::getenv("GAIL_VK_QUEUES")) wantQueues = (uint32_t)std::max(1, std::atoi(q));
        numQueues_ = std::min<uint32_t>(qf[family_].queueCount, std::min<uint32_t>(wantQueues, 8));
        numXfer_ = xferFamily_ == family_ ? numQueues_ : std::min<uint32_t>(qf[xferFamily_].queueCount, numQueues_);
        std::vector<float> prio(8, 1.0f);
        VkDeviceQueueCreateInfo qcis[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO},
                                           {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
        qcis[0].queueFamilyIndex = family_;
        qcis[0].queueCount = numQueues_;
        qcis[0].pQueuePriorities = prio.data();
        qcis[1].queueFamilyIndex = xferFamily_;
        qcis[1].queueCount = numXfer_;
        qcis[1].pQueuePriorities = prio.data();

        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.synchronization2 = VK_TRUE;
        f13.subgroupSizeControl = forceSubgroup32_;
        f13.computeFullSubgroups = forceSubgroup32_;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.pNext = &f13;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.shaderInt8 = VK_TRUE;
        f12.timelineSemaphore = VK_TRUE;
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.pNext = &f12;
        f11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f11;
        f2.features.shaderInt64 = VK_TRUE;
        f2.features.shaderInt16 = VK_TRUE;

        // VK_EXT_memory_budget tells how much device memory this process may still take
        // (other processes included); the pipeline sizes the frames in flight from it.
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, exts.data());
        const char* devExts[1] = {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
        budgetExt_ = std::any_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
            return std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        });

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = xferFamily_ != family_ ? 2 : 1;
        dci.pQueueCreateInfos = qcis;
        dci.enabledExtensionCount = budgetExt_ ? 1 : 0;
        dci.ppEnabledExtensionNames = devExts;
        VKCHECK(vkCreateDevice(phys, &dci, nullptr, &dev_));
        for (uint32_t i = 0; i < numQueues_; ++i) {
            queues_.push_back({});
            vkGetDeviceQueue(dev_, family_, i, &queues_.back().q);
        }
        for (uint32_t i = 0; i < numXfer_; ++i) {
            xferQueues_.push_back({});
            vkGetDeviceQueue(dev_, xferFamily_, i, &xferQueues_.back().q);
        }

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushSize};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VKCHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &layout_));
        pipes_[P_COLOR] = makePipe(color_spv, sizeof color_spv, false);
        pipes_[P_COLOR_YUV] = makePipe(color_yuv_spv, sizeof color_yuv_spv, false);
        pipes_[P_ROWS] = makePipe(dwt_rows_spv, sizeof dwt_rows_spv, false);
        pipes_[P_COLS] = makePipe(dwt_cols_spv, sizeof dwt_cols_spv, false);
        pipes_[P_QUANT] = makePipe(quantize_spv, sizeof quantize_spv, false);
        pipes_[P_T1CM] = makePipe(t1_cm_spv, sizeof t1_cm_spv, true);
        pipes_[P_T1MQ] = makePipe(t1_mq_spv, sizeof t1_mq_spv, false);
        pipes_[P_SCAN] = makePipe(scan_spv, sizeof scan_spv, false);
        pipes_[P_COMPACT] = makePipe(compact_spv, sizeof compact_spv, false);
        pipes_[P_ORDER] = makePipe(order_spv, sizeof order_spv, false);
        if (log_) {
            char buf[512];
            std::snprintf(buf, sizeof buf, "vulkan: %s (%u compute queues, %u transfer queues, subgroup %u%s)",
                          name_.c_str(), numQueues_, numXfer_, sgp.subgroupSize, forceSubgroup32_ ? " -> 32" : "");
            log_(LogLevel::Info, buf);
        }
    }

    ~VulkanDevice() override {
        if (!dev_) return;
        vkDeviceWaitIdle(dev_);
        for (auto p : pipes_)
            if (p) vkDestroyPipeline(dev_, p, nullptr);
        vkDestroyPipelineLayout(dev_, layout_, nullptr);
        vkDestroyDevice(dev_, nullptr);
    }

    std::string name() const override { return name_ + " (Vulkan)"; }
    const Logger& log() const { return log_; }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override;
    uint64_t slotBytes(const FrameLayout& L, const EncodeSettings& s) const override;
    // The largest device-local heap: its size, and (with VK_EXT_memory_budget) how much of
    // the driver's budget for this process is still unused.
    DeviceMemory memory() const override {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        if (budgetExt_) mp.pNext = &bud;
        vkGetPhysicalDeviceMemoryProperties2(phys_, &mp);
        DeviceMemory m;
        uint32_t heap = UINT32_MAX;
        for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount; ++h) {
            const VkMemoryHeap& mh = mp.memoryProperties.memoryHeaps[h];
            if ((mh.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mh.size > m.total) {
                m.total = mh.size;
                heap = h;
            }
        }
        if (budgetExt_ && heap != UINT32_MAX) {
            m.budgeted = true;
            m.available = bud.heapBudget[heap] > bud.heapUsage[heap] ? bud.heapBudget[heap] - bud.heapUsage[heap] : 0;
        }
        return m;
    }

    VkDevice dev_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_;
    struct Queue {
        VkQueue q = VK_NULL_HANDLE;
        std::mutex m;
        Queue() = default;
        Queue(Queue&& o) noexcept : q(o.q) {}
    };
    std::deque<Queue> queues_, xferQueues_;
    uint32_t family_ = 0, xferFamily_ = 0, numQueues_ = 1, numXfer_ = 1;
    std::atomic<uint32_t> nextQueue_{0};
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipes_[P_COUNT] = {};
    float timestampPeriod_ = 1;
    bool timestamps_ = true;
    bool budgetExt_ = false;
    VkPhysicalDeviceMemoryProperties memProps_;

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want,
                        VkMemoryPropertyFlags fallback = 0) {
        Buffer b;
        b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        const uint32_t fams[2] = {family_, xferFamily_};
        if (xferFamily_ != family_) {
            bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bci.queueFamilyIndexCount = 2;
            bci.pQueueFamilyIndices = fams;
        }
        bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        // Running out of memory is a soft failure: the pipeline runs with fewer frames in
        // flight. Nothing of this buffer is left behind.
        auto outOfMemory = [&](VkResult r, const char* kind) {
            if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY && r != VK_ERROR_OUT_OF_HOST_MEMORY) return false;
            if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
            throw DeviceOutOfMemory(std::string("cannot allocate ") + std::to_string(size >> 20) + " MiB of " + kind);
        };
        VkResult r = vkCreateBuffer(dev_, &bci, nullptr, &b.buf);
        outOfMemory(r, "host memory");
        VKCHECK(r);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev_, b.buf, &req);
        uint32_t type = findType(req.memoryTypeBits, want);
        if (type == UINT32_MAX && fallback) type = findType(req.memoryTypeBits, fallback);
        if (type == UINT32_MAX) throw std::runtime_error("no suitable memory type");
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        r = vkAllocateMemory(dev_, &mai, nullptr, &b.mem);
        outOfMemory(r, (memProps_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                           ? "device memory"
                           : "host-visible memory");
        VKCHECK(r);
        VKCHECK(vkBindBufferMemory(dev_, b.buf, b.mem, 0));
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b.buf;
        b.addr = vkGetBufferDeviceAddress(dev_, &ai);
        if (memProps_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            VKCHECK(vkMapMemory(dev_, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        return b;
    }
    void destroy(Buffer& b) {
        if (b.map) vkUnmapMemory(dev_, b.mem);
        if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
        b = Buffer{};
    }

private:
    uint32_t findType(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memProps_.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    }
    VkPipeline makePipe(const uint32_t* code, size_t size, bool needSubgroup32) {
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = size;
        smci.pCode = code;
        VkShaderModule mod;
        VKCHECK(vkCreateShaderModule(dev_, &smci, nullptr, &mod));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        rss.requiredSubgroupSize = 32;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName = "main";
        if (needSubgroup32 && forceSubgroup32_) {
            cpci.stage.pNext = &rss;
            cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }
        cpci.layout = layout_;
        VkPipeline p;
        VKCHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
        vkDestroyShaderModule(dev_, mod, nullptr);
        return p;
    }

    std::string name_;
    bool forceSubgroup32_ = false;
    Logger log_;
};

constexpr VkMemoryPropertyFlags kDevLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kHostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kHostCached =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

enum Stamp { TS_START, TS_UPLOAD, TS_COLOR, TS_DWT, TS_QUANT, TS_T1CM, TS_T1MQ, TS_END, TS_COUNT };

// Buffer capacities of one slot, and the device-local memory they add up to. Kept in one
// place so the pipeline can size the number of frames in flight before anything is
// allocated.
struct SlotSizes {
    size_t npix = 0, n = 0, bands = 0, inBytes = 0, symCap = 0, outCap = 0, passCap = 0, dataCap = 0, lut3d = 0;
    // allocation granularity assumed for the estimate
    static size_t page(size_t b) { return (b + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1); }
    size_t deviceBytes() const {
        const size_t sizes[] = {(inBytes + 3) & ~(size_t)3, 65536 * 4, CP_COUNT * 4, lut3d,       npix * 12,
                                npix * 12,   n * kCblkArea * 4, n * 16,       bands * 16,  n * 16,      passCap * 4,
                                passCap * 4, passCap * 4,       n * 2 * 24 * 4, symCap,    outCap,      64,
                                (n + 1) * 8, n * 4,             npix * 6 + 64};
        size_t total = 0;
        for (size_t b : sizes) total += page(b);
        return total;
    }
};

SlotSizes slotSizes(const FrameLayout& L, const EncodeSettings& s) {
    SlotSizes z;
    z.npix = (size_t)L.width * L.height;
    z.n = (size_t)L.numCblks();
    z.bands = L.bands.size();
    z.inBytes = s.input.bytes();
    // Generous capacities; overflow is detected and falls back to the CPU coder.
    z.symCap = std::max<size_t>(64ull << 20, z.n * 6144);
    z.outCap = z.symCap / 2 + z.n * 80;
    z.passCap = z.n * kMaxPasses;
    z.dataCap = std::max<size_t>(32ull << 20, z.n * 2048);
    ColorConverter probe(s.color);  // sizes the optional 3D LUT buffer
    z.lut3d = std::max<size_t>(16, probe.lut3d().size() * 4);
    return z;
}

class VulkanSlot : public EncoderSlot {
public:
    VulkanSlot(VulkanDevice& d, const FrameLayout& L, const EncodeSettings& s) : d_(d), L_(L), s_(s) {
        try {
            init(d, L, s);
        } catch (...) {
            release();
            throw;
        }
    }
    ~VulkanSlot() override { release(); }

private:
    void init(VulkanDevice& d, const FrameLayout& L, const EncodeSettings& s) {
        const uint32_t qi = d.nextQueue_.fetch_add(1);
        queue_ = &d.queues_[qi % d.queues_.size()];
        xferQueue_ = &d.xferQueues_[qi % d.xferQueues_.size()];
        const SlotSizes z = slotSizes(L, s);
        const size_t npix = z.npix;
        n_ = (uint32_t)z.n;
        symCap_ = z.symCap;
        outCap_ = z.outCap;
        passCap_ = z.passCap;
        dataCap_ = z.dataCap;
        inBytes_ = z.inBytes;
        staging_ = d.createBuffer(inBytes_, 0, kHostCoherent);
        rgb_ = d.createBuffer((inBytes_ + 3) & ~3ull, 0, kDevLocal);
        lut_ = d.createBuffer(65536 * 4, 0, kDevLocal);
        params_ = d.createBuffer(CP_COUNT * 4, 0, kDevLocal);
        lut3d_ = d.createBuffer(z.lut3d, 0, kDevLocal);
        frameParams_ = d.createBuffer(64, 0, kHostCoherent);
        planesA_ = d.createBuffer(npix * 3 * 4, 0, kDevLocal);
        planesB_ = d.createBuffer(npix * 3 * 4, 0, kDevLocal);
        coeffs_ = d.createBuffer((size_t)n_ * kCblkArea * 4, 0, kDevLocal);
        cblkTab_ = d.createBuffer((size_t)n_ * 16, 0, kDevLocal);
        bandTab_ = d.createBuffer(L.bands.size() * 16, 0, kDevLocal);
        info_ = d.createBuffer((size_t)n_ * 16, 0, kDevLocal);
        passEnd_ = d.createBuffer(passCap_ * 4, 0, kDevLocal);
        passDist_ = d.createBuffer(passCap_ * 4, 0, kDevLocal);
        passLen_ = d.createBuffer(passCap_ * 4, 0, kDevLocal);
        chunks_ = d.createBuffer((size_t)n_ * 2 * 24 * 4, 0, kDevLocal);
        symbols_ = d.createBuffer(symCap_, 0, kDevLocal);
        out_ = d.createBuffer(outCap_, 0, kDevLocal);
        counters_ = d.createBuffer(64, 0, kDevLocal);
        offsets_ = d.createBuffer(((size_t)n_ + 1) * 8, 0, kDevLocal);
        perm_ = d.createBuffer((size_t)n_ * 4, 0, kDevLocal);
        xyz_ = d.createBuffer(npix * 6 + 64, 0, kDevLocal);
        hostInfo_ = d.createBuffer((size_t)n_ * 20, 0, kHostCached, kHostCoherent);
        hostPass_ = d.createBuffer(passCap_ * 8, 0, kHostCached, kHostCoherent);
        hostData_ = d.createBuffer(dataCap_, 0, kHostCached, kHostCoherent);
        hostSmall_ = d.createBuffer(128, 0, kHostCached, kHostCoherent);
        setSlopeThreshold(0.0);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = d.family_;
        VKCHECK(vkCreateCommandPool(d.dev_, &pci, nullptr, &pool_));
        pci.queueFamilyIndex = d.xferFamily_;
        VKCHECK(vkCreateCommandPool(d.dev_, &pci, nullptr, &xferPool_));
        uploadStatic();

        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 2;
        VkCommandBuffer cbs[2];
        VKCHECK(vkAllocateCommandBuffers(d.dev_, &cai, cbs));
        cmd_ = cbs[0];
        cmdDebug_ = cbs[1];
        cai.commandPool = xferPool_;
        cai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(d.dev_, &cai, &cmdUpload_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKCHECK(vkCreateFence(d.dev_, &fci, nullptr, &fence_));
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VKCHECK(vkCreateSemaphore(d.dev_, &sci, nullptr, &uploaded_));
        if (d.timestamps_) {
            VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = TS_COUNT;
            VKCHECK(vkCreateQueryPool(d.dev_, &qpci, nullptr, &queries_));
        }
        recordUpload(cmdUpload_);
        record(cmd_, false);
        record(cmdDebug_, true);
    }

    // Frees whatever init() got to (also after a failed construction). Only this slot's
    // queues are drained, so other slots of the device keep running: a slot can be given
    // up mid-stream when device memory gets scarce.
    void release() {
        if (queue_) {
            std::lock_guard<std::mutex> l(queue_->m);
            vkQueueWaitIdle(queue_->q);
        }
        if (xferQueue_) {
            std::lock_guard<std::mutex> l(xferQueue_->m);
            vkQueueWaitIdle(xferQueue_->q);
        }
        if (queries_) vkDestroyQueryPool(d_.dev_, queries_, nullptr);
        for (VkEvent e : events_) vkDestroyEvent(d_.dev_, e, nullptr);
        events_.clear();
        vkDestroySemaphore(d_.dev_, uploaded_, nullptr);
        vkDestroyFence(d_.dev_, fence_, nullptr);
        vkDestroyCommandPool(d_.dev_, pool_, nullptr);
        vkDestroyCommandPool(d_.dev_, xferPool_, nullptr);
        queries_ = VK_NULL_HANDLE;
        uploaded_ = VK_NULL_HANDLE;
        fence_ = VK_NULL_HANDLE;
        pool_ = xferPool_ = VK_NULL_HANDLE;
        for (Buffer* b : {&staging_, &rgb_, &lut_, &params_, &lut3d_, &frameParams_, &planesA_, &planesB_, &coeffs_, &cblkTab_,
                          &bandTab_, &info_, &passEnd_, &passDist_, &passLen_, &chunks_, &symbols_, &out_, &counters_,
                          &offsets_, &perm_, &xyz_, &hostInfo_, &hostPass_, &hostData_, &hostSmall_})
            d_.destroy(*b);
    }

public:
    void* inputBuffer() override { return staging_.map; }

    void setSlopeThreshold(double t) override {
        float* fp = (float*)frameParams_.map;
        fp[0] = (float)t;
        fp[1] = (float)s_.bitsPerSymbol;
    }

    void encode(const void* frame, Tier1Frame& out, FrameTimings* tm, FrameDebug* dbg) override {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t npix = (size_t)L_.width * L_.height;
        if (frame != staging_.map) std::memcpy(staging_.map, frame, inBytes_);
        submitAndWait(dbg ? cmdDebug_ : cmd_);

        const uint32_t* small = (const uint32_t*)hostSmall_.map;
        const uint32_t flags = small[2];
        const uint32_t totalData = small[16];  // scan totals copied after the counters
        const uint32_t totalPasses = small[17];
        if (dbg) {
            dbg->xyz12.resize(npix * 3);
            readBack(xyz_, dbg->xyz12.data(), npix * 6);
            if (dbg->full) {
                dbg->coeffs.resize((size_t)n_ * kCblkArea);
                readBack(coeffs_, dbg->coeffs.data(), (size_t)n_ * kCblkArea * 4);
                dbg->wavelet.resize(npix * 3);
                readBack(planesA_, dbg->wavelet.data(), npix * 12);
            }
        }
        if (flags || totalData > dataCap_ || totalPasses > passCap_) {
            // Capacity overflow (pathological content): redo Tier-1 on the CPU from the
            // GPU-quantised coefficients so the output stays exact.
            if (d_.log())
                d_.log()(LogLevel::Warning, "vulkan: Tier-1 buffer overflow (flags " + std::to_string(flags) +
                                               "), using CPU fallback for this frame");
            cpuFallback(out);
        } else {
            out.info.resize(n_);
            std::memcpy(out.info.data(), hostInfo_.map, (size_t)n_ * sizeof(CblkInfo));
            const uint32_t* hp = (const uint32_t*)hostPass_.map;
            out.passLen.assign(hp, hp + totalPasses);
            out.passDist.assign((const int32_t*)hp + passCap_, (const int32_t*)hp + passCap_ + totalPasses);
            out.data.assign((const uint8_t*)hostData_.map, (const uint8_t*)hostData_.map + totalData);
        }
        if (tm) {
            tm->symbols = small[0];
            tm->codedBytes = totalData;
            fillTimings(*tm);
            tm->total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }

private:
    void submitAndWait(VkCommandBuffer cb) {
        VkSubmitInfo up{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        up.commandBufferCount = 1;
        up.pCommandBuffers = &cmdUpload_;
        up.signalSemaphoreCount = 1;
        up.pSignalSemaphores = &uploaded_;
        {
            std::lock_guard<std::mutex> l(xferQueue_->m);
            VKCHECK(vkQueueSubmit(xferQueue_->q, 1, &up, VK_NULL_HANDLE));
        }
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &uploaded_;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        {
            std::lock_guard<std::mutex> l(queue_->m);
            VKCHECK(vkQueueSubmit(queue_->q, 1, &si, fence_));
        }
        VKCHECK(vkWaitForFences(d_.dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        VKCHECK(vkResetFences(d_.dev_, 1, &fence_));
    }

    // One-off transfer through a temporary host-visible buffer.
    void transfer(Buffer& devBuf, const void* src, void* dst, size_t bytes) {
        Buffer tmp = d_.createBuffer(bytes, 0, kHostCached, kHostCoherent);
        if (src) std::memcpy(tmp.map, src, bytes);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VKCHECK(vkAllocateCommandBuffers(d_.dev_, &cai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy c{0, 0, bytes};
        if (src) vkCmdCopyBuffer(cb, tmp.buf, devBuf.buf, 1, &c);
        else vkCmdCopyBuffer(cb, devBuf.buf, tmp.buf, 1, &c);
        vkEndCommandBuffer(cb);
        VkFence f;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(d_.dev_, &fci, nullptr, &f);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        {
            std::lock_guard<std::mutex> l(queue_->m);
            VKCHECK(vkQueueSubmit(queue_->q, 1, &si, f));
        }
        vkWaitForFences(d_.dev_, 1, &f, VK_TRUE, UINT64_MAX);
        vkDestroyFence(d_.dev_, f, nullptr);
        vkFreeCommandBuffers(d_.dev_, cai.commandPool, 1, &cb);
        if (dst) std::memcpy(dst, tmp.map, bytes);
        d_.destroy(tmp);
    }
    void readBack(Buffer& b, void* dst, size_t bytes) { transfer(b, nullptr, dst, bytes); }

    void uploadStatic() {
        ColorConverter cc(s_.color);
        transfer(lut_, cc.lut().data(), nullptr, 65536 * 4);
        transfer(params_, cc.params().data(), nullptr, CP_COUNT * 4);
        if (!cc.lut3d().empty()) transfer(lut3d_, cc.lut3d().data(), nullptr, cc.lut3d().size() * 4);
        std::vector<uint32_t> tab((size_t)n_ * 4);
        for (uint32_t i = 0; i < n_; ++i) {
            const CodeBlock& cb = L_.cblks[i];
            const Subband& b = L_.bands[cb.band];
            tab[4 * i] = (uint32_t)cb.band;
            tab[4 * i + 1] = (uint32_t)(b.px0 + cb.x0);
            tab[4 * i + 2] = (uint32_t)(b.py0 + cb.y0);
            tab[4 * i + 3] = (uint32_t)cb.w | ((uint32_t)cb.h << 8) | ((uint32_t)b.orient << 16) | ((uint32_t)b.comp << 20);
        }
        transfer(cblkTab_, tab.data(), nullptr, tab.size() * 4);
        std::vector<uint32_t> bt(L_.bands.size() * 4, 0);
        for (size_t i = 0; i < L_.bands.size(); ++i) {
            const Subband& b = L_.bands[i];
            const float inv = 1.0f / b.delta;
            const float scale = (float)((double)b.delta * b.delta * b.weight);
            std::memcpy(&bt[4 * i], &inv, 4);
            bt[4 * i + 1] = ((1u << b.Mb) << kFracBits) - 1;
            std::memcpy(&bt[4 * i + 2], &scale, 4);
        }
        transfer(bandTab_, bt.data(), nullptr, bt.size() * 4);
    }

    // Stage-to-stage dependency inside one frame. With events (default), the wait only
    // covers this frame's previous stage, so stages of other frames submitted to the same
    // queue may overlap; a pipeline barrier would also wait for every earlier submission.
    void stageSync(VkCommandBuffer cb) {
        if (!useEvents_) {
            barrier(cb);
            return;
        }
        if (evIdx_ >= events_.size()) {
            VkEvent e;
            VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
            eci.flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT;
            VKCHECK(vkCreateEvent(d_.dev_, &eci, nullptr, &e));
            events_.push_back(e);
        }
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
                           VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        VkEvent e = events_[evIdx_++];
        vkCmdSetEvent2(cb, e, &di);
        vkCmdWaitEvents2(cb, 1, &e, &di);
    }

    void barrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                             0, nullptr);
    }
    template <typename PC> void dispatch(VkCommandBuffer cb, Pipe p, const PC& pc, uint32_t x, uint32_t y = 1,
                                         uint32_t z = 1) {
        static_assert(sizeof(PC) <= kPushSize, "push constants too large");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d_.pipes_[p]);
        vkCmdPushConstants(cb, d_.layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cb, x, y, z);
    }
    void stamp(VkCommandBuffer cb, Stamp s) {
        if (queries_) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queries_, s);
    }

    void recordUpload(VkCommandBuffer cb) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(cb, &bi));
        VkBufferCopy c{0, 0, (VkDeviceSize)inBytes_};
        vkCmdCopyBuffer(cb, staging_.buf, rgb_.buf, 1, &c);
        VKCHECK(vkEndCommandBuffer(cb));
    }

    void record(VkCommandBuffer cb, bool debug) {
        const uint32_t W = L_.width, H = L_.height, npix = W * H;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(cb, &bi));
        if (queries_) vkCmdResetQueryPool(cb, queries_, 0, TS_COUNT);
        // Events are reset at the end of the previous use of this command buffer (below).
        evIdx_ = evBase_;
        stamp(cb, TS_START);
        vkCmdFillBuffer(cb, counters_.buf, 0, 64, 0);
        stageSync(cb);
        stamp(cb, TS_UPLOAD);

        const FrameFormat& in = s_.input;
        if (in.kind == FrameFormat::Kind::Planar) {
            PcColorYuv py{rgb_.addr, lut_.addr, params_.addr, planesA_.addr, debug ? xyz_.addr : 0, lut3d_.addr, npix, W,
                          (uint32_t)in.activeW | ((uint32_t)in.activeH << 16), (uint32_t)in.ox | ((uint32_t)in.oy << 16),
                          (uint32_t)in.bits | (in.fullRange ? 256u : 0u) | ((uint32_t)in.chroma << 9) |
                              ((uint32_t)in.sx << 11) | ((uint32_t)in.sy << 12) | (in.rgb ? 1u << 13 : 0u)};
            in.matrix(py.crR, py.cbG, py.crG, py.cbB);
            dispatch(cb, P_COLOR_YUV, py, (npix + 255) / 256);
        } else {
            PcColor pc{rgb_.addr, lut_.addr, params_.addr, planesA_.addr, debug ? xyz_.addr : 0, lut3d_.addr, npix};
            dispatch(cb, P_COLOR, pc, (npix + 255) / 256);
        }
        stageSync(cb);
        stamp(cb, TS_COLOR);

        uint32_t w = W, h = H;
        for (int l = 0; l < L_.levels; ++l) {
            PcDwt r{planesA_.addr, planesB_.addr, W, npix, w, h};
            dispatch(cb, P_ROWS, r, h, 3);
            stageSync(cb);
            PcDwt k{planesB_.addr, planesA_.addr, W, npix, w, h};
            dispatch(cb, P_COLS, k, (w + 31) / 32, (h + 63) / 64, 3);
            stageSync(cb);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        stamp(cb, TS_DWT);

        // Tier-1 quantises each block itself; the coefficient buffer is only filled for the
        // debug taps (and by the CPU fallback).
        if (debug) {
            dispatch(cb, P_QUANT, quantPc(), n_ * kCblkArea / 256);
            stageSync(cb);
        }
        stamp(cb, TS_QUANT);

        PcT1cm cm{planesA_.addr, cblkTab_.addr,     info_.addr,           passEnd_.addr, passDist_.addr,
                  chunks_.addr,  symbols_.addr,     counters_.addr,       bandTab_.addr, frameParams_.addr,
                  n_,            (uint32_t)symCap_, (uint32_t)kMaxPasses, W,             npix};
        dispatch(cb, P_T1CM, cm, (n_ + 3) / 4);
        stageSync(cb);
        stamp(cb, TS_T1CM);

        PcOrder po{info_.addr, passEnd_.addr, perm_.addr, n_};
        dispatch(cb, P_ORDER, po, 1);
        stageSync(cb);
        PcT1mq mq{info_.addr, passEnd_.addr,  passLen_.addr, chunks_.addr, symbols_.addr,
                  out_.addr,  counters_.addr, perm_.addr,    n_,           (uint32_t)outCap_};
        dispatch(cb, P_T1MQ, mq, (n_ + 63) / 64);
        stageSync(cb);
        stamp(cb, TS_T1MQ);

        PcScan sc{info_.addr, offsets_.addr, n_};
        dispatch(cb, P_SCAN, sc, 1);
        stageSync(cb);
        PcCompact cp{info_.addr,     offsets_.addr,  out_.addr,       passLen_.addr, passDist_.addr,
                     hostInfo_.addr, hostPass_.addr, hostData_.addr,  n_,            (uint32_t)dataCap_,
                     (uint32_t)passCap_};
        dispatch(cb, P_COMPACT, cp, n_);
        stageSync(cb);
        VkBufferCopy c1{0, 0, 64};
        vkCmdCopyBuffer(cb, counters_.buf, hostSmall_.buf, 1, &c1);
        VkBufferCopy c2{(VkDeviceSize)n_ * 8, 64, 8};  // totals: data bytes, passes
        vkCmdCopyBuffer(cb, offsets_.buf, hostSmall_.buf, 1, &c2);
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        stamp(cb, TS_END);
        for (size_t i = evBase_; i < evIdx_; ++i) vkCmdResetEvent2(cb, events_[i], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        evBase_ = evIdx_;  // the debug command buffer gets its own events
        VKCHECK(vkEndCommandBuffer(cb));
    }

    void fillTimings(FrameTimings& tm) {
        if (!queries_) return;
        uint64_t ts[TS_COUNT];
        if (vkGetQueryPoolResults(d_.dev_, queries_, 0, TS_COUNT, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return;
        const double k = d_.timestampPeriod_ * 1e-6;
        tm.upload = 0;  // runs on the copy engine, overlapped with other frames
        tm.color = (ts[TS_COLOR] - ts[TS_UPLOAD]) * k;
        tm.dwt = (ts[TS_DWT] - ts[TS_COLOR]) * k;
        tm.quant = (ts[TS_QUANT] - ts[TS_DWT]) * k;
        tm.t1 = (ts[TS_T1MQ] - ts[TS_QUANT]) * k;
        tm.t1cm = (ts[TS_T1CM] - ts[TS_QUANT]) * k;
        tm.t1mq = (ts[TS_T1MQ] - ts[TS_T1CM]) * k;
        tm.download = (ts[TS_END] - ts[TS_T1MQ]) * k;
        tm.gpu = (ts[TS_END] - ts[TS_START]) * k;
    }

    PcQuant quantPc() const {
        return PcQuant{planesA_.addr, cblkTab_.addr, bandTab_.addr, coeffs_.addr, n_, (uint32_t)L_.width,
                       (uint32_t)(L_.width * L_.height)};
    }
    // Fills the coefficient buffer from the wavelet planes of the last frame (one-off).
    void runQuant() {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VKCHECK(vkAllocateCommandBuffers(d_.dev_, &cai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cb, &bi));
        dispatch(cb, P_QUANT, quantPc(), n_ * kCblkArea / 256);
        barrier(cb);
        VKCHECK(vkEndCommandBuffer(cb));
        VkFence f;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKCHECK(vkCreateFence(d_.dev_, &fci, nullptr, &f));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        {
            std::lock_guard<std::mutex> l(queue_->m);
            VKCHECK(vkQueueSubmit(queue_->q, 1, &si, f));
        }
        vkWaitForFences(d_.dev_, 1, &f, VK_TRUE, UINT64_MAX);
        vkDestroyFence(d_.dev_, f, nullptr);
        vkFreeCommandBuffers(d_.dev_, pool_, 1, &cb);
    }
    void cpuFallback(Tier1Frame& out) {
        std::vector<uint32_t> coeffs((size_t)n_ * kCblkArea);
        runQuant();
        readBack(coeffs_, coeffs.data(), coeffs.size() * 4);
        out.info.assign(n_, CblkInfo{});
        out.data.clear();
        out.passLen.clear();
        out.passDist.clear();
        T1Result r;
        for (uint32_t i = 0; i < n_; ++i) {
            const CodeBlock& cb = L_.cblks[i];
            t1EncodeRef(coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, L_.bands[cb.band].orient, r);
            CblkInfo& ci = out.info[i];
            ci.nbps = r.nbps;
            ci.numPasses = r.numPasses;
            ci.dataOffset = (uint32_t)out.data.size();
            ci.dataLen = (uint32_t)r.data.size();
            ci.passBase = (uint32_t)out.passLen.size();
            out.data.insert(out.data.end(), r.data.begin(), r.data.end());
            out.passLen.insert(out.passLen.end(), r.passLen, r.passLen + r.numPasses);
            out.passDist.insert(out.passDist.end(), r.passDist, r.passDist + r.numPasses);
        }
    }

    VulkanDevice& d_;
    const FrameLayout& L_;
    EncodeSettings s_;
    uint32_t n_ = 0;
    size_t symCap_ = 0, outCap_ = 0, passCap_ = 0, dataCap_ = 0, inBytes_ = 0;
    Buffer staging_, rgb_, lut_, params_, lut3d_, frameParams_, planesA_, planesB_, coeffs_, cblkTab_, bandTab_, info_,
        passEnd_, passDist_, passLen_, chunks_, symbols_, out_, counters_, offsets_, perm_, xyz_, hostInfo_, hostPass_,
        hostData_, hostSmall_;
    VkCommandPool pool_ = VK_NULL_HANDLE, xferPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE, cmdDebug_ = VK_NULL_HANDLE, cmdUpload_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkSemaphore uploaded_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    VulkanDevice::Queue* queue_ = nullptr;
    VulkanDevice::Queue* xferQueue_ = nullptr;
    std::vector<VkEvent> events_;
    size_t evIdx_ = 0, evBase_ = 0;
    // Measured on NVIDIA: no gain over barriers (frames still serialise), so opt-in only.
    bool useEvents_ = std::getenv("GAIL_VK_EVENTS") != nullptr;
};

std::unique_ptr<EncoderSlot> VulkanDevice::createSlot(const FrameLayout& L, const EncodeSettings& s) {
    return std::make_unique<VulkanSlot>(*this, L, s);
}

uint64_t VulkanDevice::slotBytes(const FrameLayout& L, const EncodeSettings& s) const {
    return slotSizes(L, s).deviceBytes();
}

}  // namespace

namespace {

// GPUs in the order both listVulkanDevices() and createVulkanDevices() number them.
std::vector<VkPhysicalDevice> vulkanGpus(VkInstance inst) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> phys(n);
    vkEnumeratePhysicalDevices(inst, &n, phys.data());
    std::vector<VkPhysicalDevice> gpus;
    for (auto p : phys) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(p, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU && !std::getenv("GAIL_VK_ALLOW_CPU")) continue;
        gpus.push_back(p);
    }
    return gpus;
}

}  // namespace

std::vector<DeviceInfo> listVulkanDevices() {
    VkInstance inst = instance();
    if (!inst) return {};
    std::vector<DeviceInfo> out;
    const auto gpus = vulkanGpus(inst);
    for (size_t i = 0; i < gpus.size(); ++i) {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(gpus[i], nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(gpus[i], nullptr, &ne, exts.data());
        const bool hasPci = std::any_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
            return std::strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0;
        });
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        if (hasPci) id.pNext = &pci;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &id;
        vkGetPhysicalDeviceProperties2(gpus[i], &p2);
        DeviceInfo d;
        d.index = (int)i;
        d.name = p2.properties.deviceName;
        d.uuid = formatGpuUuid(id.deviceUUID);
        if (hasPci) {
            char s[32];
            std::snprintf(s, sizeof s, "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
            d.pciBusId = s;
        }
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(gpus[i], &mem);
        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h)
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) d.memoryBytes += mem.memoryHeaps[h].size;
        out.push_back(d);
    }
    return out;
}

std::vector<std::unique_ptr<Device>> createVulkanDevices(const std::vector<int>& select, const Logger& log) {
    VkInstance inst = instance();
    if (!inst) {
        if (log) log(LogLevel::Info, "vulkan: no Vulkan loader found");
        return {};
    }
    const auto gpus = vulkanGpus(inst);
    std::vector<std::unique_ptr<Device>> out;
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (!select.empty() && std::find(select.begin(), select.end(), (int)i) == select.end()) continue;
        try {
            out.push_back(std::make_unique<VulkanDevice>(gpus[i], log));
        } catch (const std::exception& e) {
            if (log) log(LogLevel::Warning, "vulkan: skipping device " + std::to_string(i) + ": " + e.what());
        }
    }
    return out;
}

}  // namespace gail

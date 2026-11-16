// SPDX-License-Identifier: MIT
// One Vulkan GPU as the encoder's and the decoder's slots see it: the logical device, its
// queues, buffers with device addresses, and compute pipelines from embedded SPIR-V.
#pragma once
#include "vulkan/vk_loader.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/backend.h"

namespace gail {

#define VKCHECK(x)                                                                                   \
    do {                                                                                             \
        VkResult r_ = (x);                                                                           \
        if (r_ != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan error ") + std::to_string(r_) + " at " #x); \
    } while (0)

constexpr uint32_t kPushSize = 128;

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress addr = 0;
    void* map = nullptr;
};

class VulkanDevice : public Device {
public:
    static constexpr int kMaxPipes = 16;
    static constexpr int kDecodePipes = 4;
    VulkanDevice(VkPhysicalDevice phys, const Logger& log);

    ~VulkanDevice() override {
        if (!dev_) return;
        vkDeviceWaitIdle(dev_);
        for (auto p : pipes_)
            if (p) vkDestroyPipeline(dev_, p, nullptr);
        for (auto p : decodePipes_)
            if (p) vkDestroyPipeline(dev_, p, nullptr);
        vkDestroyPipelineLayout(dev_, layout_, nullptr);
        vkDestroyDevice(dev_, nullptr);
    }

    std::string name() const override { return name_ + " (Vulkan)"; }
    std::string codestreamTag() const override { return "vulk"; }
    const Logger& log() const { return log_; }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override;
    uint64_t slotBytes(const FrameLayout& L, const EncodeSettings& s) const override;
    // Decoding (vk_decode.cpp)
    bool canDecode() const override { return true; }
    std::unique_ptr<DecoderSlot> createDecoderSlot(const FrameLayout& L, const DecodeSettings& s) override;
    uint64_t decoderSlotBytes(const FrameLayout& L, const DecodeSettings& s) const override {
        return gpuDecoderSlotBytes(L, s);
    }
    // The decoder's pipelines, made when the first decode slot asks for them.
    VkPipeline decodePipe(int which);
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
    VkPipeline pipes_[kMaxPipes] = {};  // the encoder's
    VkPipeline decodePipes_[kDecodePipes] = {};
    std::mutex decodeMutex_;
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

private:
    std::string name_;
    bool forceSubgroup32_ = false;
    Logger log_;
};

constexpr VkMemoryPropertyFlags kDevLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kHostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kHostCached =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

}  // namespace gail

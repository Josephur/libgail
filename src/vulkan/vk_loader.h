// SPDX-License-Identifier: MIT
// Vulkan entry points loaded at run time (libvulkan.so.1 / vulkan-1.dll), so libgail has
// no link-time dependency on the Vulkan loader: without one, the Vulkan backend simply
// reports no devices. The pointers live in namespace gail and use the standard names, so
// backend code calls them as usual and they cannot clash with a host app's own Vulkan.
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
// only LoadLibrary / GetProcAddress are needed: keep windows.h from defining min, max,
// small, near, far ... over ordinary C++ code
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>

#define GAIL_VK_GLOBAL_FUNCS(X) X(vkCreateInstance)

#define GAIL_VK_INSTANCE_FUNCS(X)                                                                      \
    X(vkAllocateCommandBuffers) X(vkAllocateMemory) X(vkBeginCommandBuffer) X(vkBindBufferMemory)      \
    X(vkCmdBindPipeline) X(vkCmdCopyBuffer) X(vkCmdDispatch) X(vkCmdFillBuffer)                        \
    X(vkCmdPipelineBarrier) X(vkCmdPushConstants) X(vkCmdResetEvent2) X(vkCmdResetQueryPool)           \
    X(vkCmdSetEvent2) X(vkCmdWaitEvents2) X(vkCmdWriteTimestamp) X(vkCreateBuffer)                     \
    X(vkCreateCommandPool) X(vkCreateComputePipelines) X(vkCreateDevice) X(vkCreateEvent)              \
    X(vkCreateFence) X(vkCreatePipelineLayout) X(vkCreateQueryPool) X(vkCreateSemaphore)               \
    X(vkCreateShaderModule) X(vkDestroyBuffer) X(vkDestroyCommandPool) X(vkDestroyDevice)              \
    X(vkDestroyEvent) X(vkDestroyFence) X(vkDestroyPipeline) X(vkDestroyPipelineLayout)                \
    X(vkDestroyQueryPool) X(vkDestroySemaphore) X(vkDestroyShaderModule) X(vkDeviceWaitIdle)           \
    X(vkEndCommandBuffer) X(vkEnumerateDeviceExtensionProperties) X(vkEnumeratePhysicalDevices)        \
    X(vkFreeCommandBuffers) X(vkFreeMemory)                                                            \
    X(vkGetBufferDeviceAddress) X(vkGetBufferMemoryRequirements) X(vkGetDeviceQueue)                   \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceMemoryProperties2)                     \
    X(vkGetPhysicalDeviceProperties)                                                                   \
    X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceQueueFamilyProperties)                      \
    X(vkGetQueryPoolResults) X(vkMapMemory) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkResetFences)       \
    X(vkUnmapMemory) X(vkWaitForFences)

namespace gail {

#define GAIL_VK_DECLARE(name) inline PFN_##name name = nullptr;
GAIL_VK_DECLARE(vkGetInstanceProcAddr)
GAIL_VK_GLOBAL_FUNCS(GAIL_VK_DECLARE)
GAIL_VK_INSTANCE_FUNCS(GAIL_VK_DECLARE)
#undef GAIL_VK_DECLARE

// Opens the Vulkan loader and resolves the global entry points. False if there is none.
inline bool vkLoadLoader() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
#if defined(_WIN32)
        HMODULE lib = LoadLibraryA("vulkan-1.dll");
        if (!lib) return;
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(lib, "vkGetInstanceProcAddr");
#else
        void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);  // macOS: the Vulkan SDK loader over MoltenVK
        if (!lib) lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return;
        vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#endif
        if (!vkGetInstanceProcAddr) return;
#define GAIL_VK_LOAD(name) name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);
        GAIL_VK_GLOBAL_FUNCS(GAIL_VK_LOAD)
#undef GAIL_VK_LOAD
        ok = vkCreateInstance != nullptr;
    });
    return ok;
}

// Resolves every other entry point through the instance (device functions too: one level
// of loader dispatch, negligible next to the GPU work). False if one is missing.
inline bool vkLoadInstance(VkInstance inst) {
    bool ok = true;
#define GAIL_VK_LOAD(name)                                          \
    name = (PFN_##name)vkGetInstanceProcAddr(inst, #name);          \
    ok = ok && name != nullptr;
    GAIL_VK_INSTANCE_FUNCS(GAIL_VK_LOAD)
#undef GAIL_VK_LOAD
    return ok;
}

}  // namespace gail

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <chrono>
#include <random>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <fstream>

#include "sha1-fast.h"
#include "common.h"

#define SELECTED_DEVICE 0

#define BAIL_ON_BAD_RESULT(result) \
    if (VK_SUCCESS != (result)) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(-1); }

typedef struct {
    bool found;
    uint32_t nonceLen;
    uint64_t nonce;
}NonceResult;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> deviceExtensions = {
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
};

// #define NDEBUG

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

void dumpGPUmem(uint8_t* payload, uint32_t memorySize) {
    printf("\nRaw gpu memory: \n");
    uint32_t* p = (uint32_t*) payload;
    for (uint32_t k = 0; k < memorySize / 4; k++) {
      printf("%3d: %08x\n", k * 4, p[k]);
    }
    printf("\n\n");
}

uint64_t rand_uint64() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

    return dist(gen);
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

void save_nonce(const char* filename, uint64_t nonce, uint32_t nonce_len) {
	FILE* f = fopen(filename, "wb");
	fwrite(&nonce, 1, nonce_len, f);
	fclose(f);
}

uint64_t start_hashing_file(const char* filename, uint8_t buf[64], uint32_t state[5]) {
    uint64_t len = 0;
    FILE* f = fopen(filename, "rb");

    state[0] = 0x67452301;
    state[1] = 0xEFCDAB89;
    state[2] = 0x98BADCFE;
    state[3] = 0x10325476;
    state[4] = 0xC3D2E1F0;
    
    while (true) {
        uint64_t read_size = fread(buf, 1, 64, f); // not too efficient to read this little, but anyway
        len += read_size;

        if (read_size != 64) {
            //printf("read: %d\nlen: %d\n", read_size, len);
            memset(buf + read_size, 0x00, 64 - read_size);
            break; 
        }

        sha1_compress(state, (uint32_t*) buf);
    }

    return len;
}

VkResult vkGetBestComputeQueueNPH(VkPhysicalDevice physicalDevice, uint32_t* queueFamilyIndex) {
    uint32_t queueFamilyPropertiesCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

    VkQueueFamilyProperties* const queueFamilyProperties = (VkQueueFamilyProperties*)alloca(
            sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

    // first try and find a queue that has just the compute bit set
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
        const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
                                          queueFamilyProperties[i].queueFlags);

        if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    // lastly get any queue that'll work for us
    for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
        // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
        const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
                                          queueFamilyProperties[i].queueFlags);

        if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
            *queueFamilyIndex = i;
            return VK_SUCCESS;
        }
    }

    return VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << "\n" << std::endl;
    return VK_FALSE;
}

NonceResult startCompute(const char* filename, uint32_t zeroBytes) {
    const VkApplicationInfo applicationInfo = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO, // VkStructureType    sType;
        0,                                  // const void*        pNext;
        "VKComputeSample",                  // const char*        pApplicationName;
        0,                                  // uint32_t           applicationVersion;
        "",                                 // const char*        pEngineName;
        0,                                  // uint32_t           engineVersion;
        VK_MAKE_VERSION(1, 0, 9)            // uint32_t           apiVersion;
    };

    VkInstanceCreateInfo instanceCreateInfo = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,   // VkStructureType             sType;
        0,                                        // const void*                 pNext;
        0,                                        // VkInstanceCreateFlags       flags;
        &applicationInfo,                         // const VkApplicationInfo*    pApplicationInfo;
        0,                                        // uint32_t                    enabledLayerCount;
        0,                                        // const char* const*          ppEnabledLayerNames;
        (uint32_t)(deviceExtensions.size()),      // uint32_t                    enabledExtensionCount;
        deviceExtensions.data()                   // const char* const*          ppEnabledExtensionNames;
    };

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();

        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;

        instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        instanceCreateInfo.enabledLayerCount = 0;
        instanceCreateInfo.pNext = nullptr;
    }

    VkInstance instance;
    BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));

    uint32_t physicalDeviceCount = 0;
    BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));

    VkPhysicalDevice* const physicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * physicalDeviceCount);

    BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

    VkPhysicalDevice selectedPhysicalDevice = physicalDevices[SELECTED_DEVICE];
    uint32_t queueFamilyIndex = 0;
    BAIL_ON_BAD_RESULT(vkGetBestComputeQueueNPH(selectedPhysicalDevice, &queueFamilyIndex));

    const float queuePrioritory = 1.0f;
    const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // VkStructureType             sType;
        0,                                          // const void*                 pNext;
        0,                                          // VkDeviceQueueCreateFlags    flags;
        queueFamilyIndex,                           // uint32_t                    queueFamilyIndex;
        1,                                          // uint32_t                    queueCount;
        &queuePrioritory                            // const float*                pQueuePriorities;
    };

    // enable 64 bit ints for shaders
    // const char* deviceExtensions[] = {
    //   VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME
    // };

    // VkPhysicalDeviceShaderAtomicInt64Features shaderAtomicInt64Features{};
    // shaderAtomicInt64Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
    // shaderAtomicInt64Features.shaderBufferInt64Atomics = VK_TRUE;
    // shaderAtomicInt64Features.shaderSharedInt64Atomics = VK_TRUE;

    // VkPhysicalDeviceFeatures2 deviceFeatures2{};
    // deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    // deviceFeatures2.pNext = &shaderAtomicInt64Features;

    const VkDeviceCreateInfo deviceCreateInfo = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, // VkStructureType                    sType;
        0,                                    // const void*                        pNext;
        0,                                    // VkDeviceCreateFlags                flags;
        1,                                    // uint32_t                           queueCreateInfoCount;
        &deviceQueueCreateInfo,               // const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
        0,                                    // uint32_t                           enabledLayerCount;
        0,                                    // const char* const*                 ppEnabledLayerNames;
        0,                                    // uint32_t                           enabledExtensionCount;
        0,                                    // const char* const*                 ppEnabledExtensionNames;
        0                                     // const VkPhysicalDeviceFeatures*    pEnabledFeatures;
    };

    VkDevice device;
    BAIL_ON_BAD_RESULT(vkCreateDevice(selectedPhysicalDevice, &deviceCreateInfo, 0, &device));

    VkPhysicalDeviceMemoryProperties properties;

    vkGetPhysicalDeviceMemoryProperties(selectedPhysicalDevice, &properties);

    // start hashing the file 
    uint8_t buf[64];
    uint8_t state[20];
    uint64_t len = start_hashing_file(filename, buf, (uint32_t*)state);

    // printf("Text to be hashed:\n%s\nWith length: %d\n", text.c_str(), textSize);

    // allocate 4 bytes for the length of the message, 20 bytes for the partial digest, and 64 bytes for the message buffer
    uint32_t inputBufferSize = 4 + (5 * 4) + (16 * 4) + (4 * 4);

    // make inputBufferSize a multiple of 64 (bytes) to make sure we follow the allignment requirements
    inputBufferSize += (64 - inputBufferSize % 64) % 64; 

    uint32_t outputBufferSize = (2 * 4) + (5 * 4);

    // make same for outputBufferSize
    outputBufferSize += (64 - outputBufferSize % 64) % 64; 

    // we are going to need two buffers from this one memory
    const VkDeviceSize memorySize = inputBufferSize + outputBufferSize;

    // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
        if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags)
            && (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags)
            && (VK_MEMORY_PROPERTY_HOST_CACHED_BIT & properties.memoryTypes[k].propertyFlags)
            && (memorySize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) {
            memoryTypeIndex = k;
            break;
        }
    }

    BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

    const VkMemoryAllocateInfo memoryAllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, // VkStructureType    sType;
        0,                                      // const void*        pNext;
        memorySize,                             // VkDeviceSize       allocationSize;
        memoryTypeIndex                         // uint32_t           memoryTypeIndex;
    };

    VkDeviceMemory memory;
    BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, 0, &memory));

    uint8_t *payload;
    BAIL_ON_BAD_RESULT(vkMapMemory(device, memory, 0, memorySize, 0, (void **)&payload));

    memcpy(payload +  0, (uint8_t*)&len,  4);   // uint len
    memcpy(payload +  4, state         , 20);   // uint inputHash[5]
    memcpy(payload + 24, buf           , 64);   // uint message[16]

    const uint32_t inputBufferMessageLen = 24 + 64;

    vkUnmapMemory(device, memory);

    VkBufferCreateInfo inBufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // VkStructureType        sType;
        0,                                    // const void*            pNext;
        0,                                    // VkBufferCreateFlags    flags;
        inputBufferSize,                      // VkDeviceSize           size;
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,   // VkBufferUsageFlags     usage;
        VK_SHARING_MODE_EXCLUSIVE,            // VkSharingMode          sharingMode;
        1,                                    // uint32_t               queueFamilyIndexCount;
        &queueFamilyIndex                     // const uint32_t*        pQueueFamilyIndices;
    };

    VkBufferCreateInfo outBufferCreateInfo = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // VkStructureType        sType;
        0,                                    // const void*            pNext;
        0,                                    // VkBufferCreateFlags    flags;
        outputBufferSize,                     // VkDeviceSize           size;
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,   // VkBufferUsageFlags     usage;
        VK_SHARING_MODE_EXCLUSIVE,            // VkSharingMode          sharingMode;
        1,                                    // uint32_t               queueFamilyIndexCount;
        &queueFamilyIndex                     // const uint32_t*        pQueueFamilyIndices;
    };

    VkBuffer in_buffer;
    BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &inBufferCreateInfo, 0, &in_buffer));

    BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, in_buffer, memory, 0 /* offset */));

    VkBuffer out_buffer;
    BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &outBufferCreateInfo, 0, &out_buffer));

    BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, out_buffer, memory, inputBufferSize /* skip over the input buffer */));

    // allocate push constants
    // const uint32_t pushConstantsSize = 4 * 4;
    // VkPushConstantRange pushConstants = {
    //     VK_SHADER_STAGE_VERTEX_BIT, // VkShaderStageFlags    stageFlags;
    //     0,                          // uint32_t              offset;
    //     pushConstantsSize           // uint32_t              size;
    // };

    auto shader = readFile("compute-hash.spv");

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,      // VkStructureType              sType;
        0,                                                // const void*                  pNext;
        0,                                                // VkShaderModuleCreateFlags    flags;
        shader.size(),                                    // size_t                       codeSize;
        reinterpret_cast<const uint32_t*>(shader.data())  // const uint32_t*              pCode;
    };

    VkShaderModule shader_module;

    BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module));

    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2] = {
        {
            0,                                  // uint32_t              binding;
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // VkDescriptorType      descriptorType;
            1,                                  // uint32_t              descriptorCount;
            VK_SHADER_STAGE_COMPUTE_BIT,        // VkShaderStageFlags    stageFlags;
            0                                   // const VkSampler*      pImmutableSamplers;
        },
        {
            1,                                  // uint32_t              binding;
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // VkDescriptorType      descriptorType;
            1,                                  // uint32_t              descriptorCount;
            VK_SHADER_STAGE_COMPUTE_BIT,        // VkShaderStageFlags    stageFlags;
            0                                   // const VkSampler*      pImmutableSamplers;
        }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  // VkStructureType                        sType;
        0,                                                    // const void*                            pNext;
        0,                                                    // VkDescriptorSetLayoutCreateFlags       flags;
        2,                                                    // uint32_t                               bindingCount;
        descriptorSetLayoutBindings                           // const VkDescriptorSetLayoutBinding*    pBindings;
    };

    VkDescriptorSetLayout descriptorSetLayout;
    BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, &descriptorSetLayout));

    // add pushConstantsRange here
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // VkStructureType                 sType;
        0,                                              // const void*                     pNext;
        0,                                              // VkPipelineLayoutCreateFlags     flags;
        1,                                              // uint32_t                        setLayoutCount;
        &descriptorSetLayout,                           // const VkDescriptorSetLayout*    pSetLayouts;
        0,                                              // uint32_t                        pushConstantRangeCount;
        0                                               // const VkPushConstantRange*      pPushConstantRanges;
    };

    VkPipelineLayout pipelineLayout;
    BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout));

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // VkStructureType                    sType;
        0,                                              // const void*                        pNext;
        0,                                              // VkPipelineCreateFlags              flags;
        {
            // VkPipelineShaderStageCreateInfo    stage;
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // VkStructureType                     sType;
            0,                                                    // const void*                         pNext;
            0,                                                    // VkPipelineShaderStageCreateFlags    flags;
            VK_SHADER_STAGE_COMPUTE_BIT,                          // VkShaderStageFlagBits               stage;
            shader_module,                                        // VkShaderModule                      module;
            "main",                                               // const char*                         pName;
            0                                                     // const VkSpecializationInfo*         pSpecializationInfo;
        },
        pipelineLayout,                                 // VkPipelineLayout                   layout;
        0,                                              // VkPipeline                         basePipelineHandle;
        0                                               // int32_t                            basePipelineIndex;
    };

    VkPipeline pipeline;
    BAIL_ON_BAD_RESULT(vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, 0, &pipeline));

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,      // VkStructureType             sType;
        0,                                               // const void*                 pNext;
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, // VkCommandPoolCreateFlags    flags;
        queueFamilyIndex                                 // uint32_t                    queueFamilyIndex;
    };

    VkDescriptorPoolSize descriptorPoolSize = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // VkDescriptorType    type;
        2                                   // uint32_t            descriptorCount;
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  // VkStructureType                sType;
        0,                                              // const void*                    pNext;
        0,                                              // VkDescriptorPoolCreateFlags    flags;
        1,                                              // uint32_t                       maxSets;
        1,                                              // uint32_t                       poolSizeCount;
        &descriptorPoolSize                             // const VkDescriptorPoolSize*    pPoolSizes;
    };

    VkDescriptorPool descriptorPool;
    BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, 0, &descriptorPool));

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // VkStructureType                 sType;
        0,                                              // const void*                     pNext;
        descriptorPool,                                 // VkDescriptorPool                descriptorPool;
        1,                                              // uint32_t                        descriptorSetCount;
        &descriptorSetLayout                            // const VkDescriptorSetLayout*    pSetLayouts;
    };

    VkDescriptorSet descriptorSet;
    BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    VkDescriptorBufferInfo in_descriptorBufferInfo = {
        in_buffer,    // VkBuffer        buffer;
        0,            // VkDeviceSize    offset;
        VK_WHOLE_SIZE // VkDeviceSize    range;
    };

    VkDescriptorBufferInfo out_descriptorBufferInfo = {
        out_buffer,    // VkBuffer        buffer;
        0,             // VkDeviceSize    offset;
        VK_WHOLE_SIZE  // VkDeviceSize    range;
    };

    VkWriteDescriptorSet writeDescriptorSet[2] = {
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // VkStructureType                  sType;
            0,                                      // const void*                      pNext;
            descriptorSet,                          // VkDescriptorSet                  dstSet;
            0,                                      // uint32_t                         dstBinding;
            0,                                      // uint32_t                         dstArrayElement;
            1,                                      // uint32_t                         descriptorCount;
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      // VkDescriptorType                 descriptorType;
            0,                                      // const VkDescriptorImageInfo*     pImageInfo;
            &in_descriptorBufferInfo,               // const VkDescriptorBufferInfo*    pBufferInfo;
            0                                       // const VkBufferView*              pTexelBufferView;
        },
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // VkStructureType                  sType;
            0,                                      // const void*                      pNext;
            descriptorSet,                          // VkDescriptorSet                  dstSet;
            1,                                      // uint32_t                         dstBinding;
            0,                                      // uint32_t                         dstArrayElement;
            1,                                      // uint32_t                         descriptorCount;
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      // VkDescriptorType                 descriptorType;
            0,                                      // const VkDescriptorImageInfo*     pImageInfo;
            &out_descriptorBufferInfo,              // const VkDescriptorBufferInfo*    pBufferInfo;
            0                                       // const VkBufferView*              pTexelBufferView;
        }
    };

    vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, 0);

    VkCommandPool commandPool;
    BAIL_ON_BAD_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // VkStructureType         sType;
        0,                                              // const void*             pNext;
        commandPool,                                    // VkCommandPool           commandPool;
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,                // VkCommandBufferLevel    level;
        1                                               // uint32_t                commandBufferCount;
    };

    VkCommandBuffer commandBuffer;
    BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,       // VkStructureType                          sType;
        0,                                                 // const void*                              pNext;
        0 /*VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT*/, // VkCommandBufferUsageFlags                flags;
        0                                                  // const VkCommandBufferInheritanceInfo*    pInheritanceInfo;
    };

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    // nonce len for loop
    NonceResult res{};
    uint32_t found, nonce;
    for (uint64_t nonce_len = 1; nonce_len <= MAX_NONCE_LEN; nonce_len++) {
        printf("=======================nonce_len: %ld============================\n", nonce_len);
        uint64_t max_val = 1ull << (nonce_len << 3); // 256 ^ i = 2 ^ 8 ^ i = 2 ^ (8 * i)
        uint64_t gpu_blocks = max_val / GPU_THREADS_PER_BLOCK;
        gpu_blocks = (gpu_blocks < MAX_GPU_BLOCKS)? gpu_blocks: MAX_GPU_BLOCKS;
        uint64_t total_threads = gpu_blocks * GPU_THREADS_PER_BLOCK;

        printf("max_val: %ld\ngpu_blocks: %ld\ntotal_threads: %ld\n", max_val, gpu_blocks, total_threads);

        BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, 0);

        // update push constants (this whole block should be in the inner for loop)
        // uint32_t constants[pushConstantsSize];
        // memcpy(constants     , (uint8_t*)&zeroBytes          , 4);   // uint zeroBytes
        // memcpy(constants +  4, (uint8_t*)&nonce_len          , 4);   // uint nonceLen
        // memcpy(constants +  8, (uint8_t*)(&start_point)      , 4);   // uint threadOffset
        // memcpy(constants + 12, ((uint8_t*)(&start_point)) + 4, 4);   // uint extraNonce

        // vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantsSize, constants);

        vkCmdDispatch(commandBuffer, gpu_blocks, 1, 1); // only one group (that will have only one thread)

        BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));

        VkSubmitInfo submitInfo = {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // VkStructureType                sType;
            0,                              // const void*                    pNext;
            0,                              // uint32_t                       waitSemaphoreCount;
            0,                              // const VkSemaphore*             pWaitSemaphores;
            0,                              // const VkPipelineStageFlags*    pWaitDstStageMask;
            1,                              // uint32_t                       commandBufferCount;
            &commandBuffer,                 // const VkCommandBuffer*         pCommandBuffers;
            0,                              // uint32_t                       signalSemaphoreCount;
            0                               // const VkSemaphore*             pSignalSemaphores;
        };

        // thread offset loop
        BAIL_ON_BAD_RESULT(vkMapMemory(device, memory, inputBufferMessageLen, memorySize - inputBufferMessageLen, 0, (void **)&payload));

        uint64_t maxExtraNonce = 1ull << ((MAX_NONCE_LEN - 4) << 3);
        maxExtraNonce = (nonce_len > 4)? maxExtraNonce: 1; // create extraNonce only when the nonce is greater than 4 bytes

        max_val = (nonce_len > 4)?((uint64_t)UINT32_MAX) + 1: max_val;

        for (uint64_t extraNonce = 0; extraNonce < maxExtraNonce; extraNonce++) {
            printf("%08lx\n", extraNonce); 
            for (uint64_t start_point = 0; start_point < max_val; start_point += total_threads) {
                // printf("Start: %08lx\n", start_point);
                // printf("Offset: %08x; ExtraNonce:%08x\n (start point: %016llx)", (uint32_t)start_point, *((uint32_t*)(&(start_point)) + 1), start_point);
                // update data
                memcpy(payload     , (uint8_t*)&zeroBytes     , 4);   // uint zeroBytes
                memcpy(payload +  4, (uint8_t*)&nonce_len     , 4);   // uint nonceLen
                memcpy(payload +  8, (uint8_t*)(&start_point) , 4);   // uint threadOffset
                memcpy(payload + 12, (uint8_t*)(&extraNonce)  , 4);   // uint extraNonce
                vkUnmapMemory(device, memory);

                BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));

                BAIL_ON_BAD_RESULT(vkQueueWaitIdle(queue));

                BAIL_ON_BAD_RESULT(vkMapMemory(device, memory, inputBufferMessageLen, memorySize - inputBufferMessageLen, 0, (void **)&payload));
                // read results
                memcpy((uint8_t*)&found, payload + inputBufferSize - inputBufferMessageLen    , 4);   // uint found
                memcpy((uint8_t*)&nonce, payload + inputBufferSize - inputBufferMessageLen + 4, 4);   // uint nonce

                if (found) {
                    res.found = found;
                    res.nonceLen = nonce_len;
                    res.nonce = (extraNonce << 32) | (nonce);
                    break;
                }
            }
            if (found) break;
        }
        vkUnmapMemory(device, memory);

        if (found) break;
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
    vkDestroyShaderModule(device, shader_module, NULL);
    vkDestroyBuffer(device, out_buffer, NULL);
    vkDestroyBuffer(device, in_buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    free(physicalDevices);

    return res;
}

int main() {
    NonceResult res;

    auto start = std::chrono::high_resolution_clock::now();
    res = startCompute("testfile.txt", 3);
    auto end = std::chrono::high_resolution_clock::now();

    if (res.found) {
        printf("\n!!!FOUND NONCE!!!\n\n");
        save_nonce("nonce.bin", res.nonce, res.nonceLen);
    }
    else {
        printf("\nDidn't find a valid nonce!\n\n");
    }

    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Elapsed: %lfs\n", elapsed);
}

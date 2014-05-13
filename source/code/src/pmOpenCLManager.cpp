
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#include <stdlib.h>

#include "pmOpenCLManager.h"
#include "pmLogger.h"
#include "pmUtility.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"

#include <string>

#ifdef MACOS
    #define OPENCL_LIBRARY (char*)"/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL"
#else
    #define OPENCL_LIBRARY (char*)"libOpenCL.so"
#endif

#ifdef SUPPORT_OPENCL
const int MIN_SUPPORTED_OPENCL_MAJOR_VERSION = 1;
const int MIN_SUPPORTED_OPENCL_MINOR_VERSION = 2;
#endif

namespace pm
{

/* class pmOpenCLManager */
pmOpenCLManager* pmOpenCLManager::GetOpenCLManager()
{
    static pmOpenCLManager sOpenCLManager;
    return sOpenCLManager.IsValid() ? &sOpenCLManager : NULL;
}

pmOpenCLManager::pmOpenCLManager()
    : mValid(false)
{
#ifdef SUPPORT_OPENCL
	try
	{
		mOpenCLDispatcher.reset(new pmOpenCLDispatcher());
        
        mValid = true;
	}
	catch(pmExceptionOpenCL& e)
	{
        switch(e.GetFailureId())
        {
            case pmExceptionOpenCL::LIBRARY_OPEN_FAILURE:
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more OpenCL libraries could not be loaded");
                break;

            case pmExceptionOpenCL::NO_OPENCL_DEVICES:
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "No OpenCL devices found");
                break;
                
            case pmExceptionOpenCL::DEVICE_FISSION_UNAVAILABLE:
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Device Fission Unavailable");
                break;
                
            case pmExceptionOpenCL::DEVICE_COUNT_MISMATCH:
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "OpenCL device count mismatch");
                break;
                
            default:
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "OpenCL library load error");
                break;
        }
	}
#endif
}

#ifdef SUPPORT_OPENCL
pmOpenCLDispatcher* pmOpenCLManager::GetOpenCLDispatcher() const
{
    return mOpenCLDispatcher.get_ptr();
}
#endif
    
bool pmOpenCLManager::IsValid() const
{
    return mValid;
}

#ifdef SUPPORT_OPENCL
std::string GetOpenCLErrorString(cl_int pError);
void* GetOpenCLSymbol(void* pLibPtr, const char* pSymbol);
    
std::string GetOpenCLErrorString(cl_int pError)
{
    // Code taken from http://www.khronos.org/message_boards/showthread.php/5912-error-to-string
    switch(pError)
    {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}
    

cl_int (*gFuncPtr_clGetPlatformIDs)(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
cl_int (*gFuncPtr_clGetPlatformInfo)(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_int (*gFuncPtr_clGetDeviceIDs)(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
cl_int (*gFuncPtr_clGetDeviceInfo)(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_int (*gFuncPtr_clCreateSubDevices)(cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_devices, cl_device_id* out_devices, cl_uint* num_devices_ret);
cl_int (*gFuncPtr_clRetainDevice)(cl_device_id device);
cl_int (*gFuncPtr_clReleaseDevice)(cl_device_id device);
cl_context (*gFuncPtr_clCreateContext)(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify) (const char* errinfo, const void* private_info, size_t cb, void* user_data), void* user_data, cl_int* errcode_ret);
cl_command_queue (*gFuncPtr_clCreateCommandQueue)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret);
cl_int (*gFuncPtr_clReleaseContext)(cl_context context);
cl_int (*gFuncPtr_clReleaseCommandQueue)(cl_command_queue command_queue);
cl_program (*gFuncPtr_clCreateProgramWithSource)(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret);
cl_int (*gFuncPtr_clBuildProgram)(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data), void* user_data);
cl_int (*gFuncPtr_clGetProgramBuildInfo)(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_kernel (*gFuncPtr_clCreateKernel)(cl_program program, const char* kernel_name, cl_int *errcode_ret);
cl_int (*gFuncPtr_clRetainKernel)(cl_kernel kernel);
cl_int (*gFuncPtr_clReleaseKernel)(cl_kernel kernel);
cl_int (*gFuncPtr_clReleaseProgram)(cl_program program);
cl_mem (*gFuncPtr_clCreateBuffer)(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);
cl_int (*gFuncPtr_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
cl_int (*gFuncPtr_clEnqueueNDRangeKernel)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
cl_int (*gFuncPtr_clWaitForEvents)(cl_uint num_events, const cl_event* event_list);


const cl_uint MAX_PLATFORMS = 16;
const size_t MAX_PROFILE_DATA_SIZE = 1024;
const cl_uint MAX_DEVICES_PER_PLATFORM = 256;
const size_t MAX_PARTITION_PROPERTIES = 16;
const size_t MAX_VENDOR_NAME_LENGTH = 128;
const size_t MAX_PROGRAM_ERROR_LOG_LENGTH = 8192;

    
void* GetOpenCLSymbol(void* pLibPtr, const char* pSymbol)
{
    void* lSymbolPtr = pmUtility::GetExportedSymbol(pLibPtr, pSymbol);
    if(!lSymbolPtr)
    {
        std::string lStr("Undefined OpenCL Symbol ");
        lStr += pSymbol;
        std::cout << lStr.c_str() << std::endl;
        PMTHROW(pmExceptionOpenCL(pmExceptionGPU::UNDEFINED_SYMBOL));
    }
    
    return lSymbolPtr;
}
    
#define THROW_OPENCL_ERROR(error) \
{ \
    std::cout << GetOpenCLErrorString(error) << std::endl; \
    PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::RUNTIME_ERROR, error)); \
}
    
#define CHECK_OPENCL_ERROR(error) \
{ \
    if(error != CL_SUCCESS) \
        THROW_OPENCL_ERROR(error); \
}

template<typename Prototype, typename... Args>
cl_int SafeExecuteOpenCL1_NoThrow(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    *(void**)(&pPrototype) = GetOpenCLSymbol(pLibPtr, pSymbol);
    return (*pPrototype)(std::forward<Args>(args)...);
}

template<typename Prototype, typename... Args>
void SafeExecuteOpenCL1(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    cl_int lError = SafeExecuteOpenCL1_NoThrow(pLibPtr, pSymbol, pPrototype, std::forward<Args>(args)...);
    CHECK_OPENCL_ERROR(lError);
}

template<typename RetType, typename Prototype, typename... Args>
RetType SafeExecuteOpenCL2(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    cl_int lErrorCode;

    *(void**)(&pPrototype) = GetOpenCLSymbol(pLibPtr, pSymbol);
    RetType lRetVal = (*pPrototype)(std::forward<Args>(args)..., &lErrorCode);
    CHECK_OPENCL_ERROR(lErrorCode);
    
    return lRetVal;
}

template<typename ArrayElemType, cl_uint maxArraySize, typename Prototype, typename... Args>
std::vector<ArrayElemType> SafeGetOpenCLArray(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    std::vector<ArrayElemType> lVector(maxArraySize);
    cl_uint lCount;
    
    SafeExecuteOpenCL1(pLibPtr, pSymbol, pPrototype, std::forward<Args>(args)..., maxArraySize, &lVector[0], &lCount);
    
    lVector.resize(lCount);
    return lVector;
}

template<typename ArrayElemType, typename SizeType, SizeType maxArraySize, typename Prototype, typename... Args>
std::vector<ArrayElemType> SafeGetOpenCLArray(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    std::vector<ArrayElemType> lVector(maxArraySize);
    SizeType lCount;
    
    SafeExecuteOpenCL1(pLibPtr, pSymbol, pPrototype, std::forward<Args>(args)..., maxArraySize * sizeof(ArrayElemType), &lVector[0], &lCount);
    
    lVector.resize(lCount / sizeof(ArrayElemType));
    return lVector;
}

template<size_t maxStrlen, typename Prototype, typename... Args>
std::string SafeGetOpenCLString(void* pLibPtr, const char* pSymbol, Prototype pPrototype, Args... args)
{
    std::vector<char> lVector = SafeGetOpenCLArray<char, size_t, maxStrlen>(pLibPtr, pSymbol, pPrototype, std::forward<Args>(args)...);
    
    return std::string(&lVector[0], lVector.size());
}

/* class pmOpenCLDispatcher */
pmOpenCLDispatcher::pmOpenCLDispatcher()
    : mRuntimeHandle(NULL)
{
	if((mRuntimeHandle = pmUtility::OpenLibrary(OPENCL_LIBRARY)) == NULL)
		PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::LIBRARY_OPEN_FAILURE));

    Initialize();
}

pmOpenCLDispatcher::~pmOpenCLDispatcher()
{
    for_each(mDeviceMap, [&] (typename decltype(mDeviceMap)::value_type& pPair)
    {
        SafeExecuteOpenCL1(mRuntimeHandle, "clReleaseDevice", gFuncPtr_clReleaseDevice, pPair.first);
        SafeExecuteOpenCL1(mRuntimeHandle, "clReleaseCommandQueue", gFuncPtr_clReleaseCommandQueue, pPair.second.queue);
    });
    
    for_each(mPlatformContextMap, [&] (typename decltype(mPlatformContextMap)::value_type& pPair)
    {
        SafeExecuteOpenCL1(mRuntimeHandle, "clReleaseContext", gFuncPtr_clReleaseContext, pPair.second);
    });

	try
	{
        pmUtility::CloseLibrary(mRuntimeHandle);
	}
	catch(pmIgnorableException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more OpenCL libraries could not be closed properly");
	}
}

void pmOpenCLDispatcher::Initialize()
{
    std::vector<cl_platform_id> lPlatformVector = SafeGetOpenCLArray<cl_platform_id, MAX_PLATFORMS>(mRuntimeHandle, "clGetPlatformIDs", gFuncPtr_clGetPlatformIDs);
    bool lCpuDevicesPartitioned = false;
    bool lNvidiaDevicesDone = false;
    size_t lNvidiaGpuDevicesCount = 0;
    
    for_each(lPlatformVector, [&] (cl_platform_id pPlatform)
    {
        std::string lProfile = SafeGetOpenCLString<MAX_PROFILE_DATA_SIZE>(mRuntimeHandle, "clGetPlatformInfo", gFuncPtr_clGetPlatformInfo, pPlatform, CL_PLATFORM_PROFILE);
        std::string lVersion = SafeGetOpenCLString<MAX_PROFILE_DATA_SIZE>(mRuntimeHandle, "clGetPlatformInfo", gFuncPtr_clGetPlatformInfo, pPlatform, CL_PLATFORM_VERSION);

    #if 0
        std::string lName = SafeGetOpenCLString<MAX_PROFILE_DATA_SIZE>(mRuntimeHandle, "clGetPlatformInfo", gFuncPtr_clGetPlatformInfo, pPlatform, CL_PLATFORM_NAME);
        std::string lVendor = SafeGetOpenCLString<MAX_PROFILE_DATA_SIZE>(mRuntimeHandle, "clGetPlatformInfo", gFuncPtr_clGetPlatformInfo, pPlatform, CL_PLATFORM_VENDOR);
        std::string lExtensions = SafeGetOpenCLString<MAX_PROFILE_DATA_SIZE>(mRuntimeHandle, "clGetPlatformInfo", gFuncPtr_clGetPlatformInfo, pPlatform, CL_PLATFORM_EXTENSIONS);
    #endif

        if(!strcmp(lProfile.c_str(), "FULL_PROFILE"))
        {
            size_t lDotIndex = lVersion.find(".");
            EXCEPTION_ASSERT(lDotIndex != std::string::npos);

            size_t lPostDotSpaceIndex = lVersion.find(" ", lDotIndex);
            EXCEPTION_ASSERT(lPostDotSpaceIndex != std::string::npos);
            
            size_t lMajorVersion = (size_t)atoi(lVersion.substr(7, lDotIndex - 7).c_str()); // Format OpenCL<Space><MajorVersion>.<MinorVersion><Space>
            size_t lMinorVersion = (size_t)atoi(lVersion.substr(lDotIndex + 1, lPostDotSpaceIndex - lDotIndex - 1).c_str());
            
            if((lMajorVersion > MIN_SUPPORTED_OPENCL_MAJOR_VERSION) || (lMajorVersion == MIN_SUPPORTED_OPENCL_MAJOR_VERSION && lMinorVersion >= MIN_SUPPORTED_OPENCL_MINOR_VERSION))
            {
                std::vector<cl_device_id> lDeviceVector = SafeGetOpenCLArray<cl_device_id, MAX_DEVICES_PER_PLATFORM>(mRuntimeHandle, "clGetDeviceIDs", gFuncPtr_clGetDeviceIDs, pPlatform, CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU);
                
                for_each(lDeviceVector, [&] (cl_device_id pDevice)
                {
                    cl_bool lIsDeviceAvailable = SafeGetOpenCLArray<cl_bool, size_t, 1>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_AVAILABLE)[0];
                    
                    if(lIsDeviceAvailable)
                    {
                        cl_device_type lDeviceType = SafeGetOpenCLArray<cl_device_type, size_t, 1>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_TYPE)[0];

                        if(lDeviceType == CL_DEVICE_TYPE_GPU && !lNvidiaDevicesDone)
                        {
                            std::string lVendor = SafeGetOpenCLString<MAX_VENDOR_NAME_LENGTH>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_VENDOR);

                            if(lVendor.find("NVIDIA") != std::string::npos)
                            {
                                mDeviceMap.emplace(std::piecewise_construct, std::forward_as_tuple(pDevice), std::forward_as_tuple(pPlatform));
                                mCudaDevices.emplace_back(pDevice);

                                lNvidiaDevicesDone = true;
                                ++lNvidiaGpuDevicesCount;
                            }
                        }
                        else if(lDeviceType == CL_DEVICE_TYPE_CPU && !lCpuDevicesPartitioned)
                        {
                            lCpuDevicesPartitioned = PartitionCpuDevice(pDevice, pPlatform);
                        }
                    }
                });
            }
        }
    });
    
    if(mDeviceMap.empty())
        PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::NO_OPENCL_DEVICES));
    
    if(!lCpuDevicesPartitioned)
        PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::DEVICE_FISSION_UNAVAILABLE));

    if(pmStubManager::GetStubManager()->GetProcessingElementsCPU() != mDeviceMap.size() - lNvidiaGpuDevicesCount)
        PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::DEVICE_COUNT_MISMATCH));
    
    if(pmStubManager::GetStubManager()->GetProcessingElementsGPU() != lNvidiaGpuDevicesCount)
        PMTHROW(pmExceptionOpenCL(pmExceptionOpenCL::DEVICE_COUNT_MISMATCH));

    for_each(mDeviceMap, [&] (typename decltype(mDeviceMap)::value_type& pPair)
    {
        mPlatformDevicesMap[pPair.second.platform].push_back(pPair.first);
    });
    
    typedef void (CL_CALLBACK* pfn_notify) (const char* errinfo, const void* private_info, size_t cb, void* user_data);
    
    for_each(mPlatformDevicesMap, [&] (typename decltype(mPlatformDevicesMap)::value_type& pPair)
    {
        cl_context_properties lProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)pPair.first, 0};
        cl_context lContext = SafeExecuteOpenCL2<cl_context>(mRuntimeHandle, "clCreateContext", gFuncPtr_clCreateContext, lProperties, (cl_uint)pPair.second.size(), (cl_device_id*)(&pPair.second[0]), (pfn_notify)NULL, (void*)NULL);

        EXCEPTION_ASSERT(lContext);
        mPlatformContextMap.emplace(pPair.first, lContext);
    });

    for_each(mDeviceMap, [&] (typename decltype(mDeviceMap)::value_type& pPair)
    {
        DEBUG_EXCEPTION_ASSERT(mPlatformContextMap.find(pPair.second.platform) != mPlatformContextMap.end());

        pPair.second.queue = SafeExecuteOpenCL2<cl_command_queue>(mRuntimeHandle, "clCreateCommandQueue", gFuncPtr_clCreateCommandQueue, mPlatformContextMap[pPair.second.platform], pPair.first, 0);
        
        EXCEPTION_ASSERT(pPair.second.queue);
    });
    
    SetupExecutionStubs();
}
    
bool pmOpenCLDispatcher::PartitionCpuDevice(cl_device_id pDevice, cl_platform_id pPlatform)
{
    cl_uint lMaxComputeUnits = SafeGetOpenCLArray<cl_uint, size_t, 1>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_MAX_COMPUTE_UNITS)[0];
    
    if(lMaxComputeUnits == 1)
    {
        mDeviceMap.emplace(std::piecewise_construct, std::forward_as_tuple(pDevice), std::forward_as_tuple(pPlatform));
        mCpuCoreDevices.emplace_back(pDevice);
        
        return true;
    }
    else
    {
        cl_uint lMaxPartitions = SafeGetOpenCLArray<cl_uint, size_t, 1>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_PARTITION_MAX_SUB_DEVICES)[0];
        
        if(lMaxPartitions == lMaxComputeUnits)
        {
            std::vector<cl_device_partition_property> lPartitionPropertyVector = SafeGetOpenCLArray<cl_device_partition_property, size_t, MAX_PARTITION_PROPERTIES>(mRuntimeHandle, "clGetDeviceInfo", gFuncPtr_clGetDeviceInfo, pDevice, CL_DEVICE_PARTITION_PROPERTIES);
            
            bool lPartitionsEqually = false;
            for_each(lPartitionPropertyVector, [&] (cl_device_partition_property pProperty)
            {
                if(pProperty == CL_DEVICE_PARTITION_EQUALLY)
                {
                    lPartitionsEqually = true;
                    return; // return from functor
                }
            });
            
            if(lPartitionsEqually)
            {
                cl_uint lNumDevices;
                std::vector<cl_device_id> lSubDevices(lMaxComputeUnits);
                const cl_device_partition_property lProperty[3] = {CL_DEVICE_PARTITION_EQUALLY, 1, 0};
                SafeExecuteOpenCL1(mRuntimeHandle, "clCreateSubDevices", gFuncPtr_clCreateSubDevices, pDevice, lProperty, lMaxComputeUnits, (cl_device_id*)&lSubDevices[0], &lNumDevices);
                
                for_each(lSubDevices, [&] (cl_device_id pSubDevice)
                {
                    mDeviceMap.emplace(std::piecewise_construct, std::forward_as_tuple(pSubDevice), std::forward_as_tuple(pPlatform));
                    mCpuCoreDevices.emplace_back(pSubDevice);
                    
                    SafeExecuteOpenCL1(mRuntimeHandle, "clRetainDevice", gFuncPtr_clRetainDevice, pSubDevice);
                });
                
                return true;
            }
        }
    }
    
    return false;
}
    
cl_device_id pmOpenCLDispatcher::GetCpuCoreDevice(size_t pIndex) const
{
    EXCEPTION_ASSERT(mCpuCoreDevices.size() > pIndex);
    
    return mCpuCoreDevices[pIndex];
}

cl_device_id pmOpenCLDispatcher::GetCudaDevice(size_t pIndex) const
{
    EXCEPTION_ASSERT(mCudaDevices.size() > pIndex);

    return mCudaDevices[pIndex];
}
    
void pmOpenCLDispatcher::SetupExecutionStubs()
{
    pmStubManager* lStubManager = pmStubManager::GetStubManager();

    size_t lCpuCores = lStubManager->GetProcessingElementsCPU();
    for(size_t i = 0; i < lCpuCores; ++i)
        lStubManager->GetCpuStub((uint)i)->SetOpenCLDevice(GetCpuCoreDevice(i));

    size_t lGpus = lStubManager->GetProcessingElementsGPU();
    for(size_t i = 0; i < lGpus; ++i)
        lStubManager->GetGpuStub((uint)i)->SetOpenCLDevice(GetCudaDevice(i));
}
    
std::shared_ptr<pmOpenCLKernel> pmOpenCLDispatcher::MakeKernel(std::string& pSource, std::string& pKernelName)
{
    return std::make_shared<pmOpenCLKernel>(pSource, pKernelName);
}

cl_mem pmOpenCLDispatcher::CreateBuffer(cl_device_id pDevice, size_t pSize, cl_mem_flags pFlags, void* pHostPtr)
{
    DEBUG_EXCEPTION_ASSERT(mDeviceMap.find(pDevice) != mDeviceMap.end());
    DEBUG_EXCEPTION_ASSERT(mPlatformDevicesMap.find(mDeviceMap[pDevice].platform) != mPlatformDevicesMap.end());
    
    cl_mem lBuffer = SafeExecuteOpenCL2<cl_mem>(mRuntimeHandle, "clCreateBuffer", gFuncPtr_clCreateBuffer, mPlatformContextMap[mDeviceMap.find(pDevice)->second.platform], pFlags,  pSize, pHostPtr);

    EXCEPTION_ASSERT(lBuffer);
    return lBuffer;
}
    
cl_event pmOpenCLDispatcher::ExecuteKernel(cl_device_id pDevice, pmOpenCLKernel& pKernel, const std::vector<std::pair<size_t, void*>>& pArgs, const std::vector<size_t>& pWorkGroupConf, const std::vector<size_t>& pWorkItemConf)
{
    EXCEPTION_ASSERT(pWorkGroupConf.size() <= 3);
    EXCEPTION_ASSERT(pWorkItemConf.size() <= 3);
    EXCEPTION_ASSERT(pWorkGroupConf.size() == pWorkItemConf.size());
    
    DEBUG_EXCEPTION_ASSERT(mDeviceMap.find(pDevice) != mDeviceMap.end());
    DEBUG_EXCEPTION_ASSERT(mPlatformDevicesMap.find(mDeviceMap[pDevice].platform) != mPlatformDevicesMap.end());

    cl_kernel lKernel = pKernel.GetKernel(mPlatformContextMap[mDeviceMap.find(pDevice)->second.platform]);

    for_each_with_index(pArgs, [&] (const std::pair<size_t, void*>& pPair, size_t pIndex)
    {
        SafeExecuteOpenCL1(mRuntimeHandle, "clSetKernelArg", gFuncPtr_clSetKernelArg, lKernel, (cl_uint)pIndex, pPair.first, pPair.second);
    });
    
    cl_event lEvent;
    SafeExecuteOpenCL1(mRuntimeHandle, "clEnqueueNDRangeKernel", gFuncPtr_clEnqueueNDRangeKernel, mDeviceMap.find(pDevice)->second.queue, lKernel, (cl_uint)pWorkGroupConf.size(), (const size_t*)NULL, &pWorkGroupConf[0], &pWorkItemConf[0], 0, (const cl_event*)NULL, &lEvent);

    return lEvent;
}
 
void pmOpenCLDispatcher::WaitForEvents(const std::vector<cl_event>& pEvents)
{
    SafeExecuteOpenCL1(mRuntimeHandle, "clWaitForEvents", gFuncPtr_clWaitForEvents, (cl_uint)pEvents.size(), &pEvents[0]);
}

void pmOpenCLDispatcher::ExecuteKernelBlocking(cl_device_id pDevice, pmOpenCLKernel& pKernel, const std::vector<std::pair<size_t, void*>>& pArgs, const std::vector<size_t>& pWorkGroupConf, const std::vector<size_t>& pWorkItemConf)
{
    cl_event lKernelEvent = ExecuteKernel(pDevice, pKernel, pArgs, pWorkGroupConf, pWorkItemConf);
    
    std::vector<cl_event> lVector(1, lKernelEvent);
    WaitForEvents(lVector);
}

/* class pmOpenCLKernel */
pmOpenCLKernel::pmOpenCLKernel(std::string& pSource, std::string& pKernelName)
{
    void* lRuntimeHandle = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mRuntimeHandle;
    auto& lPlatformContextMap = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mPlatformContextMap;

    for_each(lPlatformContextMap, [&] (const std::pair<cl_platform_id, cl_context>& pPair)
    {
        cl_program lProgram = CreateProgram(pPair.second, pSource);
        BuildProgram(pPair.first, lProgram);
        
        cl_kernel lKernel = CreateKernel(lProgram, pKernelName);

        SafeExecuteOpenCL1(lRuntimeHandle, "clRetainKernel", gFuncPtr_clRetainKernel, lKernel);
        mContextKernelMap.emplace(std::piecewise_construct, std::forward_as_tuple(pPair.second), std::forward_as_tuple(lProgram, lKernel));
    });
}
    
pmOpenCLKernel::~pmOpenCLKernel()
{
    void* lRuntimeHandle = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mRuntimeHandle;

    for_each(mContextKernelMap, [&] (typename decltype(mContextKernelMap)::value_type& pPair)
    {
        SafeExecuteOpenCL1(lRuntimeHandle, "clReleaseKernel", gFuncPtr_clReleaseKernel, pPair.second.second);
        SafeExecuteOpenCL1(lRuntimeHandle, "clReleaseProgram", gFuncPtr_clReleaseProgram, pPair.second.first);
    });
}
    
cl_kernel pmOpenCLKernel::GetKernel(cl_context pContext)
{
    DEBUG_EXCEPTION_ASSERT(mContextKernelMap.find(pContext) != mContextKernelMap.end());

    return mContextKernelMap[pContext].second;
}
    
cl_program pmOpenCLKernel::CreateProgram(cl_context pContext, std::string& pSource)
{
    void* lRuntimeHandle = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mRuntimeHandle;

    const char* lStr = pSource.c_str();
    
    cl_program lProgram = SafeExecuteOpenCL2<cl_program>(lRuntimeHandle, "clCreateProgramWithSource", gFuncPtr_clCreateProgramWithSource, pContext, 1, (const char**)&lStr, (const size_t*)NULL);

    EXCEPTION_ASSERT(lProgram);
    return lProgram;
}

void pmOpenCLKernel::BuildProgram(cl_platform_id pPlatform, cl_program pProgram)
{
    void* lRuntimeHandle = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mRuntimeHandle;
    auto& lPlatformDevicesMap = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mPlatformDevicesMap;

    typedef void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data);

    cl_int lErrorCode = SafeExecuteOpenCL1_NoThrow(lRuntimeHandle, "clBuildProgram", gFuncPtr_clBuildProgram, pProgram, (cl_uint)lPlatformDevicesMap[pPlatform].size(), (cl_device_id*)(&lPlatformDevicesMap[pPlatform][0]), (const char*)NULL, (pfn_notify)NULL, (void*)NULL);
    
    if(lErrorCode != CL_SUCCESS)
    {
        std::string lErrorLog = SafeGetOpenCLString<MAX_PROGRAM_ERROR_LOG_LENGTH>(lRuntimeHandle, "clGetProgramBuildInfo", gFuncPtr_clGetProgramBuildInfo, pProgram, lPlatformDevicesMap[pPlatform][0], CL_PROGRAM_BUILD_LOG);
        
        std::cout << "Failed to build program: " << lErrorLog << std::endl;

        THROW_OPENCL_ERROR(lErrorCode);
    }
}
    
cl_kernel pmOpenCLKernel::CreateKernel(cl_program pProgram, const std::string& pKernelName)
{
    void* lRuntimeHandle = pmOpenCLManager::GetOpenCLManager()->GetOpenCLDispatcher()->mRuntimeHandle;
    cl_kernel lKernel = SafeExecuteOpenCL2<cl_kernel>(lRuntimeHandle, "clCreateKernel", gFuncPtr_clCreateKernel, pProgram, pKernelName.c_str());
 
    EXCEPTION_ASSERT(lKernel);
    return lKernel;
}

#endif	// SUPPORT_OPENCL
    
}



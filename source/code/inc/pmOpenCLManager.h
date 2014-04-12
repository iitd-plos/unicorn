/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution,
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

#ifndef __PM_OPENCL_MANAGER__
#define __PM_OPENCL_MANAGER__

#include "pmBase.h"

#include <map>
#include <memory>

#ifdef SUPPORT_OPENCL

#ifdef MACOS
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#endif

namespace pm
{
    
#ifdef SUPPORT_OPENCL
class pmOpenCLDispatcher;
    
class pmOpenCLKernel
{
public:
    pmOpenCLKernel(std::string& pSource, std::string& pKernelName);
    ~pmOpenCLKernel();
    
    cl_kernel GetKernel(cl_context pContext);

private:
    cl_program CreateProgram(cl_context pContext, std::string& pSource);
    void BuildProgram(cl_platform_id pPlatform, cl_program pProgram);
    cl_kernel CreateKernel(cl_program pProgram, const std::string& pKernelName);

    std::map<cl_context, std::pair<cl_program, cl_kernel>> mContextKernelMap;
};
    
class pmOpenCLDispatcher : public pmBase
{
    friend class pmOpenCLKernel;

    struct deviceData
    {
        cl_platform_id platform;
        cl_command_queue queue;
        
        deviceData(cl_platform_id pPlatform)
        : platform(pPlatform)
        , queue(NULL)
        {}
    };

public:
    pmOpenCLDispatcher();
    ~pmOpenCLDispatcher();

    std::shared_ptr<pmOpenCLKernel> MakeKernel(std::string& pSource, std::string& pKernelName);
    cl_event ExecuteKernel(cl_device_id pDevice, pmOpenCLKernel& pKernel, const std::vector<std::pair<size_t, void*>>& pArgs, const std::vector<size_t>& pWorkGroupConf, const std::vector<size_t>& pWorkItemConf);
    void ExecuteKernelBlocking(cl_device_id pDevice, pmOpenCLKernel& pKernel, const std::vector<std::pair<size_t, void*>>& pArgs, const std::vector<size_t>& pWorkGroupConf, const std::vector<size_t>& pWorkItemConf);

    cl_mem CreateBuffer(cl_device_id pDevice, size_t pSize, cl_mem_flags pFlags, void* pHostPtr);
    
    void WaitForEvents(const std::vector<cl_event>& pEvents);
    
private:
    void Initialize();
    bool PartitionCpuDevice(cl_device_id pDevice, cl_platform_id pPlatform);

    cl_device_id GetCpuCoreDevice(size_t pIndex) const;
    cl_device_id GetCudaDevice(size_t pIndex) const;
    void SetupExecutionStubs();

    void* mRuntimeHandle;
    std::vector<cl_device_id> mCpuCoreDevices;
    std::vector<cl_device_id> mCudaDevices;
    std::map<cl_device_id, deviceData> mDeviceMap;
    std::map<cl_platform_id, cl_context> mPlatformContextMap;
    std::map<cl_platform_id, std::vector<cl_device_id>> mPlatformDevicesMap;
};
#endif

class pmOpenCLManager : public pmBase
{
public:
    static pmOpenCLManager* GetOpenCLManager();
#ifdef SUPPORT_OPENCL
    pmOpenCLDispatcher* GetOpenCLDispatcher() const;
#endif

private:
    pmOpenCLManager();
    
    bool IsValid() const;
    
    bool mValid;

#ifdef SUPPORT_OPENCL
    finalize_ptr<pmOpenCLDispatcher> mOpenCLDispatcher;
#endif
};
    
} // end namespace pm

#endif

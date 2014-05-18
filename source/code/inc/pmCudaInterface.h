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

#ifdef SUPPORT_CUDA

#ifndef __PM_CUDA_INTERFACE__
#define __PM_CUDA_INTERFACE__

#include "pmPublicDefinitions.h"
#include "pmInternalDefinitions.h"
#include "pmErrorDefinitions.h"

#include <vector>
#include <string>

namespace pm
{
    
struct pmCudaMemcpyCommand
{
    void* srcPtr;
    void* destPtr;
    size_t size;
    
    pmCudaMemcpyCommand(void* pSrcPtr, void* pDestPtr, size_t pSize)
    : srcPtr(pSrcPtr)
    , destPtr(pDestPtr)
    , size(pSize)
    {}
};

struct pmCudaSubtaskMemoryStruct
{
    void* cudaPtr;
    bool requiresLoad;
    bool isUncached;
    
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    void* pinnedPtr;
#endif
    
    pmCudaSubtaskMemoryStruct()
    : cudaPtr(NULL)
    , requiresLoad(false)
    , isUncached(false)
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    , pinnedPtr(NULL)
#endif
    {}
};

struct pmCudaSubtaskSecondaryBuffersStruct
{
    void* reservedMemCudaPtr;
    void* statusCudaPtr;
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    void* statusPinnedPtr;
#endif
    
    pmCudaSubtaskSecondaryBuffersStruct()
    : reservedMemCudaPtr(NULL)
    , statusCudaPtr(NULL)
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    , statusPinnedPtr(NULL)
#endif
    {}
};

class pmCudaStreamAutoPtr
{
public:
    pmCudaStreamAutoPtr();
    ~pmCudaStreamAutoPtr();
    
    void Initialize(void* pRuntimeHandle);

    void* GetStream();

    pmCudaStreamAutoPtr(pmCudaStreamAutoPtr& pPtr)
    : mRuntimeHandle(pPtr.mRuntimeHandle)
    , mStream(pPtr.mStream)
    {
        pPtr.mRuntimeHandle = NULL;
        pPtr.mStream = NULL;
    }
    
private:
    pmCudaStreamAutoPtr& operator=(const pmCudaStreamAutoPtr&);

    void* mRuntimeHandle;
    void* mStream;
};
    
class pmStubCUDA;
    
class pmCudaInterface
{
public:
    static void SetRuntimeHandle(void* pRuntimeHandle);

    static int GetCudaDriverVersion();

    static void* AllocateCudaMem(size_t pSize);
    static void DeallocateCudaMem(const void* pPtr);
    static void CopyDataToCudaDevice(void* pCudaPtr, const void* pHostPtr, size_t pSize);

    static void CountAndProbeProcessingElements();
    static void BindToDevice(size_t pDeviceIndex);
    static void UnbindFromDevice(size_t pDeviceIndex);
    
    static size_t GetCudaDeviceCount();
    static std::string GetDeviceName(size_t pDeviceIndex);
    static std::string GetDeviceDescription(size_t pDeviceIndex);
    
    static size_t GetCudaAlignment(size_t pDeviceIndex);
    static size_t GetAvailableCudaMem();
    
    static void WaitForStreamCompletion(pmCudaStreamAutoPtr& pStream);

    static void ForceResetAllCudaDevices();

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    static void* AllocatePinnedBuffer(size_t pSize);
    static void DeallocatePinnedBuffer(const void* pMem);
#endif

    static pmStatus InvokeKernel(pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, pmStatus* pStatusCudaPtr, pmCudaStreamAutoPtr& pStreamPtr);

private:
    static pmStatus ExecuteKernel(const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, pmDeviceInfo* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, void* pStream, pmStatus* pStatusCudaPtr);

    static size_t GetUnallocatableCudaMemSize();
    static void*& GetRuntimeHandle();
};
    
} // end namespace pm

#endif

#endif

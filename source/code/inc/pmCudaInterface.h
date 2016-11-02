
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

#ifdef SUPPORT_CUDA

#ifndef __PM_CUDA_INTERFACE__
#define __PM_CUDA_INTERFACE__

#include "pmPublicDefinitions.h"
#include "pmInternalDefinitions.h"
#include "pmErrorDefinitions.h"
#include "pmPublicUtilities.h"

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
    void* compressedMemCudaPtr;
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    void* statusPinnedPtr;
#endif
    
    pmCudaSubtaskSecondaryBuffersStruct()
    : reservedMemCudaPtr(NULL)
    , statusCudaPtr(NULL)
    , compressedMemCudaPtr(NULL)
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
class pmTaskProfiler;
    
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
    
    static pmStatus InvokeKernel(pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, pmStatus* pStatusCudaPtr, pmCudaStreamAutoPtr& pStreamPtr, pmReductionDataType pSentinelCompressionReductionDataType, void* pCompressedPtr
#ifdef ENABLE_TASK_PROFILING
         , pmTaskProfiler* pTaskProfiler
#endif
     );

private:
    static pmStatus ExecuteKernel(const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, pmDeviceInfo* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, void* pStream, pmStatus* pStatusCudaPtr);

    static bool CompressForSentinel(pmReductionDataType pReductionDataType, void* pCudaPtr, size_t pSize, void* pCompressedPtr, size_t& pCompressedSize, uint& pNonSentinelCount);

    static size_t GetUnallocatableCudaMemSize();
    static void*& GetRuntimeHandle();
};
    
} // end namespace pm

#endif

#endif

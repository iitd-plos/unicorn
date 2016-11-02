
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

#include "pmDispatcherGPU.h"
#include "pmExecutionStub.h"
#include "pmHardware.h"
#include "pmAddressSpace.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmSubscriptionManager.h"
#include "pmLogger.h"
#include "pmTask.h"
#include "pmUtility.h"

#ifdef MACOS
    #define CUDA_LIBRARY_DRIVER (char*)"libcuda.dylib"
    #define CUDA_LIBRARY_RUNTIME (char*)"libcudart.dylib"
#else
    #define CUDA_LIBRARY_DRIVER (char*)"libcuda.so"
    #define CUDA_LIBRARY_RUNTIME (char*)"libcudart.so"
#endif

#ifdef SUPPORT_CUDA
const int MIN_SUPPORTED_CUDA_DRIVER_VERSION = 4000;
#endif

namespace pm
{

/* class pmDispatcherGPU */
pmDispatcherGPU* pmDispatcherGPU::GetDispatcherGPU()
{
    static pmDispatcherGPU lDisptacherGPU;
    return &lDisptacherGPU;
}

pmDispatcherGPU::pmDispatcherGPU()
    : mCountGPU(0)
{
#ifdef SUPPORT_CUDA
	try
	{
		mDispatcherCUDA.reset(new pmDispatcherCUDA());
	}
	catch(pmExceptionGPU& e)
	{
		mDispatcherCUDA.reset(NULL);
        
        if(e.GetFailureId() == pmExceptionGPU::DRIVER_VERSION_UNSUPPORTED)
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Unsupported CUDA driver version");
        else
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be loaded");
	}
#endif
}

#ifdef SUPPORT_CUDA
pmDispatcherCUDA* pmDispatcherGPU::GetDispatcherCUDA()
{
	return mDispatcherCUDA.get_ptr();
}
#endif

size_t pmDispatcherGPU::GetCountGPU()
{
	return mCountGPU;
}

size_t pmDispatcherGPU::ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector)
{
#ifdef SUPPORT_CUDA
	size_t lCountCUDA = 0;
	if(mDispatcherCUDA.get_ptr())
	{
		lCountCUDA = pmCudaInterface::GetCudaDeviceCount();
		for(size_t i = 0; i < lCountCUDA; ++i)
			pStubVector.push_back(new pmStubCUDA(i, (uint)pStubVector.size()));
	}

	mCountGPU += lCountCUDA;
#endif

	return mCountGPU;
}

#ifdef SUPPORT_CUDA
/* class pmDispatcherCUDA */
pmDispatcherCUDA::pmDispatcherCUDA()
{
	if((mRuntimeHandle = pmUtility::OpenLibrary(CUDA_LIBRARY_RUNTIME)) == NULL)
		PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::LIBRARY_OPEN_FAILURE));

    pmCudaInterface::SetRuntimeHandle(mRuntimeHandle);

    if(pmCudaInterface::GetCudaDriverVersion() < MIN_SUPPORTED_CUDA_DRIVER_VERSION)
    {
        pmUtility::CloseLibrary(mRuntimeHandle);
        pmCudaInterface::SetRuntimeHandle(NULL);
        mRuntimeHandle = NULL;

        PMTHROW_NODUMP(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::DRIVER_VERSION_UNSUPPORTED));
    }
    
	pmCudaInterface::CountAndProbeProcessingElements();
}

pmDispatcherCUDA::~pmDispatcherCUDA()
{
	try
	{
        pmUtility::CloseLibrary(mRuntimeHandle);
	}
	catch(pmIgnorableException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be closed properly");
	}
}

pmStatus pmDispatcherCUDA::InvokeKernel(pmTask* pTask, pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, pmReductionDataType pSentinelCompressionReductionDataType, pmCudaStreamAutoPtr& pStreamPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands)
{
    void* lDeviceInfoCudaPtr = pStub->GetDeviceInfoCudaPtr();

    pmSubtaskInfo lSubtaskInfoCuda = pSubtaskInfo;
    lSubtaskInfoCuda.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfoCuda.subtaskId);

    const std::vector<pmCudaSubtaskMemoryStruct>& lSubtaskPointers = pStub->GetSubtaskPointersMap().find(pSubtaskInfo.subtaskId)->second;
    const pmCudaSubtaskSecondaryBuffersStruct& lSubtaskSecondaryBuffers = pStub->GetSubtaskSecondaryBuffersMap().find(pSubtaskInfo.subtaskId)->second;
    
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    for_each_with_index(pTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex)
    {
        void* lCpuPtr = pSubtaskInfo.memInfo[pAddressSpaceIndex].ptr;
        lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr = lSubtaskPointers[pAddressSpaceIndex].cudaPtr;

        if(lCpuPtr) // CPU Ptr is not available when subscription has compact view and there is no shadow memory
        {
            if(pSubtaskInfo.memInfo[pAddressSpaceIndex].readPtr)
            {
                size_t lOffset = reinterpret_cast<size_t>(pSubtaskInfo.memInfo[pAddressSpaceIndex].readPtr) - reinterpret_cast<size_t>(lCpuPtr);
                lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lOffset);
            }
            
            if(pSubtaskInfo.memInfo[pAddressSpaceIndex].writePtr)
            {
                size_t lOffset = reinterpret_cast<size_t>(pSubtaskInfo.memInfo[pAddressSpaceIndex].writePtr) - reinterpret_cast<size_t>(lCpuPtr);
                lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lOffset);
            }
        }
        else
        {
            EXCEPTION_ASSERT(pTask->GetAddressSpaceSubscriptionVisibility(pAddressSpace, pStub) == SUBSCRIPTION_COMPACT);

            const pmSplitInfo* lSplitInfo = ((pSubtaskInfo.splitInfo.splitCount == 0) ? NULL : &pSubtaskInfo.splitInfo);
            const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(pStub, pSubtaskInfo.subtaskId, lSplitInfo, pAddressSpaceIndex);
            
            if(!lCompactViewData.nonConsolidatedReadSubscriptionOffsets.empty())
                lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lCompactViewData.nonConsolidatedReadSubscriptionOffsets[0]);

            if(!lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.empty())
                lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lCompactViewData.nonConsolidatedWriteSubscriptionOffsets[0]);
        }
    });

    if(lSubtaskPointers.size() > pTask->GetAddressSpaceCount())
    {
        DEBUG_EXCEPTION_ASSERT(!lSubtaskInfoCuda.gpuContext.scratchBuffer);
        lSubtaskInfoCuda.gpuContext.scratchBuffer = lSubtaskPointers.back().cudaPtr;
    }
    
    if(lSubtaskSecondaryBuffers.reservedMemCudaPtr)
    {
        DEBUG_EXCEPTION_ASSERT(!lSubtaskInfoCuda.gpuContext.reservedGlobalMem);
        lSubtaskInfoCuda.gpuContext.reservedGlobalMem = lSubtaskSecondaryBuffers.reservedMemCudaPtr;
    }

    DEBUG_EXCEPTION_ASSERT(lSubtaskSecondaryBuffers.statusCudaPtr);
    pmStatus* lStatusCudaPtr = (pmStatus*)lSubtaskSecondaryBuffers.statusCudaPtr;

	return pmCudaInterface::InvokeKernel(pStub, pTaskInfo, pTaskInfoCuda, pStub->GetProcessingElement()->GetDeviceInfo(), lDeviceInfoCudaPtr, lSubtaskInfoCuda, pCudaLaunchConf, pKernelPtr, pCustomKernelPtr, pHostToDeviceCommands, pDeviceToHostCommands, lStatusCudaPtr, pStreamPtr, pSentinelCompressionReductionDataType, lSubtaskSecondaryBuffers.compressedMemCudaPtr
#ifdef ENABLE_TASK_PROFILING
         , pTask->GetTaskProfiler()
#endif
     );
}
    
void* pmDispatcherCUDA::GetRuntimeHandle()
{
    return mRuntimeHandle;
}

void* GetExportedSymbol(void* pLibHandle, char* pSymbol)
{
    return pmUtility::GetExportedSymbol(pLibHandle, pSymbol);
}

#endif

}

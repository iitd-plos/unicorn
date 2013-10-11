
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

const int MIN_SUPPORTED_CUDA_DRIVER_VERSION = 4000;

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

pmStatus pmDispatcherCUDA::InvokeKernel(pmTask* pTask, pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, pmCudaStreamAutoPtr& pStreamPtr)
{
    void* lDeviceInfoCudaPtr = pStub->GetDeviceInfoCudaPtr();

    pmSubtaskInfo lSubtaskInfoCuda = pSubtaskInfo;
    
    const std::vector<pmCudaSubtaskMemoryStruct>& lSubtaskPointers = pStub->GetSubtaskPointersMap().find(pSubtaskInfo.subtaskId)->second;
    const pmCudaSubtaskSecondaryBuffersStruct& lSubtaskSecondaryBuffers = pStub->GetSubtaskSecondaryBuffersMap().find(pSubtaskInfo.subtaskId)->second;
    
    for_each_with_index(pTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex)
    {
        void* lCpuPtr = lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr;

        lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr = lSubtaskPointers[pAddressSpaceIndex].cudaPtr;

        if(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].readPtr)
        {
            size_t lOffset = reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].readPtr) - reinterpret_cast<size_t>(lCpuPtr);
            lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lOffset);
        }
        
        if(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].writePtr)
        {
            size_t lOffset = reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].writePtr) - reinterpret_cast<size_t>(lCpuPtr);
            lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lSubtaskInfoCuda.memInfo[pAddressSpaceIndex].ptr) + lOffset);
        }
    });

    if(lSubtaskPointers.size() > pTask->GetAddressSpaceCount())
    {
        DEBUG_EXCEPTION_ASSERT(lSubtaskInfoCuda.gpuContext.scratchBuffer);
        lSubtaskInfoCuda.gpuContext.scratchBuffer = lSubtaskPointers.back().cudaPtr;
    }
    
    if(lSubtaskSecondaryBuffers.reservedMemCudaPtr)
    {
        DEBUG_EXCEPTION_ASSERT(lSubtaskInfoCuda.gpuContext.reservedGlobalMem);
        lSubtaskInfoCuda.gpuContext.reservedGlobalMem = lSubtaskSecondaryBuffers.reservedMemCudaPtr;
    }

    DEBUG_EXCEPTION_ASSERT(lSubtaskSecondaryBuffers.statusCudaPtr);
    pmStatus* lStatusCudaPtr = (pmStatus*)lSubtaskSecondaryBuffers.statusCudaPtr;

	return pmCudaInterface::InvokeKernel(pStub, pTaskInfo, pTaskInfoCuda, pStub->GetProcessingElement()->GetDeviceInfo(), lDeviceInfoCudaPtr, lSubtaskInfoCuda, pCudaLaunchConf, pKernelPtr, pCustomKernelPtr, ((pmStubCUDA*)pStub)->mHostToDeviceCommands, ((pmStubCUDA*)pStub)->mDeviceToHostCommands, lStatusCudaPtr, pStreamPtr);
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

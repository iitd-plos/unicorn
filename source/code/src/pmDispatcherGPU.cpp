
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
#include "pmMemSection.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmSubscriptionManager.h"
#include "pmLogger.h"
#include "pmTask.h"

#ifdef MACOS
    #define CUDA_LIBRARY_CUTIL (char*)"libcutil.dylib"
    #define CUDA_LIBRARY_CUDART (char*)"libcudart.dylib"
#else
    #define CUDA_LIBRARY_CUTIL (char*)"libcutil.so"
    #define CUDA_LIBRARY_CUDART (char*)"libcudart.so"
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
{
#ifdef SUPPORT_CUDA
	try
	{
		mDispatcherCUDA = new pmDispatcherCUDA();
	}
	catch(pmExceptionGPU& e)
	{
		mDispatcherCUDA = NULL;
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be loaded");
	}
#else
	mDispatcherCUDA = NULL;
#endif
}

pmDispatcherGPU::~pmDispatcherGPU()
{
	delete mDispatcherCUDA;
}

pmDispatcherCUDA* pmDispatcherGPU::GetDispatcherCUDA()
{
	return mDispatcherCUDA;
}

size_t pmDispatcherGPU::GetCountGPU()
{
	return mCountGPU;
}

size_t pmDispatcherGPU::ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector)
{
	size_t lCountCUDA = 0;
	if(mDispatcherCUDA)
	{
		lCountCUDA = mDispatcherCUDA->GetCountCUDA();
		for(size_t i=0; i<lCountCUDA; ++i)
			pStubVector.push_back(new pmStubCUDA(i, (uint)pStubVector.size()));
	}

	mCountGPU = lCountCUDA;
	return mCountGPU;
}

/* class pmDispatcherCUDA */
pmDispatcherCUDA::pmDispatcherCUDA()
{
#ifdef SUPPORT_CUDA
	//mCutilHandle = OpenLibrary(CUDA_LIBRARY_CUTIL);
	mRuntimeHandle = OpenLibrary(CUDA_LIBRARY_CUDART);

	//if(!mCutilHandle || !mRuntimeHandle)
	if(!mRuntimeHandle)
	{
		//CloseLibrary(mCutilHandle);
		CloseLibrary(mRuntimeHandle);
		
		PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::LIBRARY_OPEN_FAILURE));
	}

	CountAndProbeProcessingElements();
#endif
}

pmDispatcherCUDA::~pmDispatcherCUDA()
{
	try
	{
		//CloseLibrary(mCutilHandle);
		CloseLibrary(mRuntimeHandle);
	}
	catch(pmIgnorableException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be closed properly");
	}
}

size_t pmDispatcherCUDA::GetCountCUDA()
{
	return mCountCUDA;
}

pmStatus pmDispatcherCUDA::InvokeKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr)
{
#ifdef SUPPORT_CUDA
	pmTask* lTask = (pmTask*)(pTaskInfoCuda.taskHandle);
	uint lOriginatingMachineIndex = (uint)(*(lTask->GetOriginatingHost()));
	ulong lSequenceNumber = lTask->GetSequenceNumber();

    void* lDeviceInfoCudaPtr = dynamic_cast<pmStubCUDA*>(pStub)->GetDeviceInfoCudaPtr();
    pmLastCudaExecutionRecord& lLastExecutionRecord = dynamic_cast<pmStubCUDA*>(pStub)->GetLastExecutionRecord();

    pmMemSection* lMemSection = lTask->GetMemSectionRW();
	return InvokeKernel(pStub, lLastExecutionRecord, pTaskInfo, pTaskInfoCuda, pStub->GetProcessingElement()->GetDeviceInfo(), lDeviceInfoCudaPtr, pSubtaskInfo, pCudaLaunchConf, pOutputMemWriteOnly, pKernelPtr, pCustomKernelPtr, lOriginatingMachineIndex, lSequenceNumber, (lMemSection ? lMemSection->GetMem() : NULL));
#else	
        return pmSuccess;
#endif
}
    
void* pmDispatcherCUDA::CheckAndGetScratchBuffer(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId, size_t& pScratchBufferSize, pmScratchBufferInfo& pScratchBufferInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    return lTask->GetSubscriptionManager().CheckAndGetScratchBuffer(pStub, pSubtaskId, pScratchBufferSize, pScratchBufferInfo);
}
    
void pmDispatcherCUDA::GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetInputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pSubscriptionInfo);
}

void pmDispatcherCUDA::GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pReadSubscription, pSubscriptionInfo);
}

void pmDispatcherCUDA::GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetUnifiedOutputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pSubscriptionInfo);
}
    
void pmDispatcherCUDA::GetNonConsolidatedSubscriptionsForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubscriptionType pSubscriptionType, pmSubtaskInfo& pSubtaskInfo, std::vector<std::pair<size_t, size_t> >& pSubscriptionVector)
{
    subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;

    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    bool lRetVal = true;
    bool lIsInputMem = (pSubscriptionType == INPUT_MEM_READ_SUBSCRIPTION);
    if(lIsInputMem)
        lRetVal = lTask->GetSubscriptionManager().GetNonConsolidatedInputMemSubscriptionsForSubtask(pStub, pSubtaskInfo.subtaskId, lBegin, lEnd);
    else
        lRetVal = lTask->GetSubscriptionManager().GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskInfo.subtaskId,  (pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION), lBegin, lEnd);
    
    if(lRetVal)
    {
        for(lIter = lBegin; lIter != lEnd; ++lIter)
            pSubscriptionVector.push_back(std::make_pair(lIter->first, lIter->second.first));
    }
}

bool pmDispatcherCUDA::SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId1, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);
    
    return lTask->GetSubscriptionManager().SubtasksHaveMatchingSubscriptions(pStub, pSubtaskId1, pStub, pSubtaskId2, pSubscriptionType);
}
    
void pmDispatcherCUDA::MarkInsideUserCode(pmExecutionStub* pStub, ulong pSubtaskId)
{
    pStub->MarkInsideUserCode(pSubtaskId);
}

void pmDispatcherCUDA::MarkInsideLibraryCode(pmExecutionStub* pStub, ulong pSubtaskId)
{
    pStub->MarkInsideLibraryCode(pSubtaskId);
}
    
bool pmDispatcherCUDA::RequiresPrematureExit(pmExecutionStub* pStub, ulong pSubtaskId)
{
    return pStub->RequiresPrematureExit(pSubtaskId);
}

}


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

#define CUDA_LIBRARY_CUTIL (char*)"libcutil.so"
#define CUDA_LIBRARY_CUDART (char*)"libcudart.so"

namespace pm
{

pmDispatcherGPU* pmDispatcherGPU::mDispatcherGPU = NULL;

/* class pmDispatcherGPU */
pmDispatcherGPU* pmDispatcherGPU::GetDispatcherGPU()
{
	return mDispatcherGPU;
}

pmDispatcherGPU::pmDispatcherGPU()
{
    if(mDispatcherGPU)
        PMTHROW(pmFatalErrorException());
    
    mDispatcherGPU = this;

#ifdef SUPPORT_CUDA    
	try
	{
		mDispatcherCUDA = new pmDispatcherCUDA();
	}
	catch(pmExceptionGPU e)
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
	catch(pmIgnorableException e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be closed properly");
	}
}

size_t pmDispatcherCUDA::GetCountCUDA()
{
	return mCountCUDA;
}

pmStatus pmDispatcherCUDA::InvokeKernel(size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr)
{
#ifdef SUPPORT_CUDA
	pmTask* lTask = (pmTask*)(pTaskInfo.taskHandle);
	uint lOriginatingMachineIndex = (uint)(*(lTask->GetOriginatingHost()));
	ulong lSequenceNumber = lTask->GetSequenceNumber();

	return InvokeKernel(pBoundDeviceIndex, pTaskInfo, pSubtaskInfo, pCudaLaunchConf, pOutputMemWriteOnly, pKernelPtr, lOriginatingMachineIndex, lSequenceNumber);
#else	
        return pmSuccess;
#endif
}

void pmDispatcherCUDA::GetNonConsolidatedSubscriptionsForSubtask(uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, bool pIsInputMem, pmSubtaskInfo pSubtaskInfo, std::vector<std::pair<size_t, size_t> >& pSubscriptionVector)
{
    subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;

    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    if(pIsInputMem)
        lTask->GetSubscriptionManager().GetNonConsolidatedInputMemSubscriptionsForSubtask(pSubtaskInfo.subtaskId, lBegin, lEnd);
    else
        lTask->GetSubscriptionManager().GetNonConsolidatedOutputMemSubscriptionsForSubtask(pSubtaskInfo.subtaskId, lBegin, lEnd);
    
    if(lBegin == lEnd)
    {
        pSubscriptionVector.push_back(std::make_pair(0, pIsInputMem ? pSubtaskInfo.inputMemLength : pSubtaskInfo.outputMemLength));
    }
    else
    {
        for(lIter = lBegin; lIter != lEnd; ++lIter)
            pSubscriptionVector.push_back(std::make_pair(lIter->first, lIter->second.first));
    }
}

bool pmDispatcherCUDA::SubtasksHaveMatchingSubscriptions(uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId1, ulong pSubtaskId2, bool pIsInputMem)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);
    
    return lTask->GetSubscriptionManager().SubtasksHaveMatchingSubscriptions(pSubtaskId1, pSubtaskId2, pIsInputMem);
}

}

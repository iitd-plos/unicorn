
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

#include "pmBase.h"
#include "pmCallback.h"
#include "pmDispatcherGPU.h"
#include "pmTask.h"
#include "pmHardware.h"
#include "pmAddressSpace.h"
#include "pmExecutionStub.h"
#include "pmCudaInterface.h"

namespace pm
{

/* class pmDataDistributionCB */
pmDataDistributionCB::pmDataDistributionCB(pmDataDistributionCallback pCallback)
{
	mCallback = pCallback;
}

pmStatus pmDataDistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) const
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_PARTITIONING);
#endif
    
	if(!mCallback)
		return pmSuccess;

    pmSubtaskInfo lSubtaskInfo = pTask->GetPreSubscriptionSubtaskInfo(pSubtaskId, pSplitInfo);
    
    pmStatus lStatus = pmStatusUnavailable;
    pmJmpBufAutoPtr lJmpBufAutoPtr;

    sigjmp_buf lJmpBuf;
    int lJmpVal = sigsetjmp(lJmpBuf, 1);

    if(!lJmpVal)
    {
        lJmpBufAutoPtr.Reset(&lJmpBuf, pStub);
        lStatus = mCallback(pTask->GetTaskInfo(), pStub->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo);
    }
    else
    {
        lJmpBufAutoPtr.SetHasJumped();
         PMTHROW_NODUMP(pmPrematureExitException(true));
    }
    
    return lStatus;
}


/* class pmSubtaskCB */
pmSubtaskCB::pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom)
	: mCallback_CPU(pCallback_CPU)
	, mCallback_GPU_CUDA(pCallback_GPU_CUDA)
    , mCallback_GPU_Custom(pCallback_GPU_Custom)
{
}

bool pmSubtaskCB::IsCallbackDefinedForDevice(pmDeviceType pDeviceType) const
{
	switch(pDeviceType)
	{
		case CPU:
		{
			if(mCallback_CPU)
				return true;
			
			break;
		}

#ifdef SUPPORT_CUDA
		case GPU_CUDA:
		{
			if(mCallback_GPU_CUDA || mCallback_GPU_Custom)
				return true;

			break;
		}
#endif
	
		case MAX_DEVICE_TYPES:
			PMTHROW(pmFatalErrorException());
	}

	return false;
}
    
bool pmSubtaskCB::HasCustomGpuCallback() const
{
#ifdef SUPPORT_CUDA
    return (mCallback_GPU_Custom != NULL);
#else
    return false;
#endif
}
    
bool pmSubtaskCB::HasBothCpuAndGpuCallbacks() const
{
#ifdef SUPPORT_CUDA
    if(IsCallbackDefinedForDevice(CPU) && IsCallbackDefinedForDevice(GPU_CUDA))
        return true;
#endif
    
    return false;
}

pmStatus pmSubtaskCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo, void* pStreamPtr /* = NULL */) const
{    
    pmStatus lStatus = pmStatusUnavailable;

	switch(pStub->GetType())
	{
		case CPU:
		{
			if(!mCallback_CPU)
				return pmSuccess;

            const pmDeviceInfo& lDeviceInfo = pStub->GetProcessingElement()->GetDeviceInfo();

            pmJmpBufAutoPtr lJmpBufAutoPtr;
            
            sigjmp_buf lJmpBuf;
            int lJmpVal = sigsetjmp(lJmpBuf, 1);
            
            if(!lJmpVal)
            {
                lJmpBufAutoPtr.Reset(&lJmpBuf, pStub);
                lStatus = mCallback_CPU(pTaskInfo, lDeviceInfo, pSubtaskInfo);
            }
            else
            {
                lJmpBufAutoPtr.SetHasJumped();
                PMTHROW_NODUMP(pmPrematureExitException(true));
            }

			break;
		}

#ifdef SUPPORT_CUDA
		case GPU_CUDA:
		{
			if(!mCallback_GPU_CUDA && !mCallback_GPU_Custom)
				return pmSuccess;
            
			pmCudaLaunchConf& lCudaLaunchConf = pTask->GetSubscriptionManager().GetCudaLaunchConf(pStub, pSubtaskInfo.subtaskId, pSplitInfo);

            // pTaskInfo is task info with CUDA pointers; pTask->GetTaskInfo() is with CPU pointers
            lStatus = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->InvokeKernel(pTask, (pmStubCUDA*)pStub, pTask->GetTaskInfo(), pTaskInfo, pSubtaskInfo, lCudaLaunchConf, mCallback_GPU_CUDA, mCallback_GPU_Custom, *((pmCudaStreamAutoPtr*)pStreamPtr));
            
			break;
		}
#endif
		
		default:	
			PMTHROW(pmFatalErrorException());
	}
    
    pTask->GetSubscriptionManager().DropScratchBufferIfNotRequiredPostSubtaskExec(pStub, pSubtaskInfo.subtaskId, pSplitInfo);

	return lStatus;
}


/* class pmDataReductionCB */
pmDataReductionCB::pmDataReductionCB(pmDataReductionCallback pCallback)
{
	mCallback = pCallback;
}

pmStatus pmDataReductionCB::Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, bool pMultiAssign1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, bool pMultiAssign2) const
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_REDUCTION);
#endif

	if(!mCallback)
		return pmSuccess;
    
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    const pmSubtaskInfo& lSubtaskInfo1 = lSubscriptionManager.GetSubtaskInfo(pStub1, pSubtaskId1, pSplitInfo1);
    const pmSubtaskInfo& lSubtaskInfo2 = lSubscriptionManager.GetSubtaskInfo(pStub2, pSubtaskId2, pSplitInfo2);
    
	return mCallback(pTask->GetTaskInfo(), pStub1->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo1, pStub2->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo2);
}


/* class pmDataRedistributionCB */
pmDataRedistributionCB::pmDataRedistributionCB(pmDataRedistributionCallback pCallback)
{
	mCallback = pCallback;
}

pmStatus pmDataRedistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign) const
{
	if(!mCallback)
		return pmSuccess;

    const pmSubtaskInfo& lSubtaskInfo = pTask->GetSubscriptionManager().GetSubtaskInfo(pStub, pSubtaskId, pSplitInfo);
	return mCallback(pTask->GetTaskInfo(), pStub->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo);
}


/* class pmDeviceSelectionCB */
pmDeviceSelectionCB::pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback)
{
	mCallback = pCallback;
}

bool pmDeviceSelectionCB::Invoke(pmTask* pTask, const pmProcessingElement* pProcessingElement) const
{
	if(!mCallback)
		return pmSuccess;

	return mCallback(pTask->GetTaskInfo(), pProcessingElement->GetDeviceInfo());
}


/* class pmPreDataTransferCB */
pmPreDataTransferCB::pmPreDataTransferCB(pmPreDataTransferCallback pCallback)
{
	mCallback = pCallback;
}

pmStatus pmPreDataTransferCB::Invoke() const
{
	if(!mCallback)
		return pmSuccess;

	return pmSuccess;
	//return mCallback();
}


/* class pmPostDataTransferCB */
pmPostDataTransferCB::pmPostDataTransferCB(pmPostDataTransferCallback pCallback)
{
	mCallback = pCallback;
}

pmStatus pmPostDataTransferCB::Invoke() const
{
	if(!mCallback)
		return pmSuccess;

	return pmSuccess;
	//return mCallback();
}

};


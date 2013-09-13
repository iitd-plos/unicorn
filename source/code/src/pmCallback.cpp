
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
#include "pmMemSection.h"
#include "pmExecutionStub.h"

namespace pm
{

/* class pmCallback */
pmCallback::pmCallback()
{
}

pmCallback::~pmCallback()
{
}

/* class pmDataDistributionCB */
pmDataDistributionCB::pmDataDistributionCB(pmDataDistributionCallback pCallback)
{
	mCallback = pCallback;
}

pmDataDistributionCB::~pmDataDistributionCB()
{
}

pmStatus pmDataDistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_PARTITIONING);
#endif
    
	if(!mCallback)
		return pmSuccess;

    pmMemSection* lInputMemSection = pTask->GetMemSectionRO();
    pmMemSection* lOutputMemSection = pTask->GetMemSectionRW();
    
    void* lInputMem = (lInputMemSection && lInputMemSection->IsLazy()) ? (lInputMemSection->GetMem()) : NULL;
    void* lOutputMem = (lOutputMemSection && lOutputMemSection->IsLazy()) ? (lOutputMemSection->GetMem()) : NULL;
    
    size_t lInputMemSize = ((lInputMem && lInputMemSection) ? lInputMemSection->GetLength() : 0);
    size_t lOutputMemSize = ((lOutputMem && lOutputMemSection) ? lOutputMemSection->GetLength() : 0);
    
    pmLazyMemInfo lLazyMemInfo(lInputMem, lOutputMem, lInputMemSize, lOutputMemSize);
    
    pmStatus lStatus = pmStatusUnavailable;
    pmJmpBufAutoPtr lJmpBufAutoPtr;

    sigjmp_buf lJmpBuf;
    int lJmpVal = sigsetjmp(lJmpBuf, 1);

    if(!lJmpVal)
    {
        lJmpBufAutoPtr.Reset(&lJmpBuf, pStub);
        lStatus = mCallback(pTask->GetTaskInfo(), lLazyMemInfo, pStub->GetProcessingElement()->GetDeviceInfo(), pSubtaskId, pSplitInfo);
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

pmSubtaskCB::~pmSubtaskCB()
{
}

bool pmSubtaskCB::IsCallbackDefinedForDevice(pmDeviceType pDeviceType)
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
    
bool pmSubtaskCB::HasCustomGpuCallback()
{
#ifdef SUPPORT_CUDA
    return (mCallback_GPU_Custom != NULL);
#else
    return false;
#endif
}
    
bool pmSubtaskCB::HasBothCpuAndGpuCallbacks()
{
#ifdef SUPPORT_CUDA
    if(IsCallbackDefinedForDevice(CPU) && IsCallbackDefinedForDevice(GPU_CUDA))
        return true;
#endif
    
    return false;
}

pmStatus pmSubtaskCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, bool pOutputMemWriteOnly)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::SUBTASK_EXECUTION);
#endif
    
    pmStatus lStatus = pmStatusUnavailable;

	switch(pStub->GetType())
	{
		case CPU:
		{
			if(!mCallback_CPU)
				return pmSuccess;

            pmDeviceInfo& lDeviceInfo = pStub->GetProcessingElement()->GetDeviceInfo();

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
            
			pmCudaLaunchConf& lCudaLaunchConf = pTask->GetSubscriptionManager().GetCudaLaunchConf(pStub, pSubtaskId, pSplitInfo);

            // pTaskInfo is task info with CUDA pointers; pTask->GetTaskInfo() is with CPU pointers
            lStatus = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->InvokeKernel(pStub, pTask->GetTaskInfo(), pTaskInfo, pSubtaskInfo, lCudaLaunchConf, pOutputMemWriteOnly, mCallback_GPU_CUDA, mCallback_GPU_Custom);
            
			break;
		}
#endif
		
		default:	
			PMTHROW(pmFatalErrorException());
	}
    
    pTask->GetSubscriptionManager().DropScratchBufferIfNotRequiredPostSubtaskExec(pStub, pSubtaskId, pSplitInfo);

	return pmSuccess;
}


/* class pmDataReductionCB */
pmDataReductionCB::pmDataReductionCB(pmDataReductionCallback pCallback)
{
	mCallback = pCallback;
}

pmDataReductionCB::~pmDataReductionCB()
{
}

pmStatus pmDataReductionCB::Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, bool pMultiAssign1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, bool pMultiAssign2)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_REDUCTION);
#endif

	if(!mCallback)
		return pmSuccess;
    
    bool lOutputMemWriteOnly = false;

	pmSubtaskInfo lSubtaskInfo1, lSubtaskInfo2;
	pTask->GetSubtaskInfo(pStub1, pSubtaskId1, pSplitInfo1, pMultiAssign1, lSubtaskInfo1, lOutputMemWriteOnly);
	pTask->GetSubtaskInfo(pStub2, pSubtaskId2, pSplitInfo2, pMultiAssign2, lSubtaskInfo2, lOutputMemWriteOnly);
    
	return mCallback(pTask->GetTaskInfo(), pStub1->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo1, pStub2->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo2);
}


/* class pmDataRedistributionCB */
pmDataRedistributionCB::pmDataRedistributionCB(pmDataRedistributionCallback pCallback)
{
	mCallback = pCallback;
}

pmDataRedistributionCB::~pmDataRedistributionCB()
{
}

pmStatus pmDataRedistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign)
{
	if(!mCallback)
		return pmSuccess;

    bool lOutputMemWriteOnly = false;

	pmSubtaskInfo lSubtaskInfo;
	pTask->GetSubtaskInfo(pStub, pSubtaskId, pSplitInfo, pMultiAssign, lSubtaskInfo, lOutputMemWriteOnly);

	pmStatus lStatus = mCallback(pTask->GetTaskInfo(), pStub->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo);
    
    return lStatus;
}


/* class pmDeviceSelectionCB */
pmDeviceSelectionCB::pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback)
{
	mCallback = pCallback;
}

pmDeviceSelectionCB::~pmDeviceSelectionCB()
{
}

bool pmDeviceSelectionCB::Invoke(pmTask* pTask, pmProcessingElement* pProcessingElement)
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

pmPreDataTransferCB::~pmPreDataTransferCB()
{
}

pmStatus pmPreDataTransferCB::Invoke()
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

pmPostDataTransferCB::~pmPostDataTransferCB()
{
}

pmStatus pmPostDataTransferCB::Invoke()
{
	if(!mCallback)
		return pmSuccess;

	return pmSuccess;
	//return mCallback();
}

};


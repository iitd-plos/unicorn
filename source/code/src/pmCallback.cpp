
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
	: mCallback(pCallback)
{
}

pmStatus pmDataDistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) const
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_PARTITIONING);
#endif
    
	if(!mCallback)
		return pmSuccess;

    pmSubtaskInfo lSubtaskInfo = pTask->GetPreSubscriptionSubtaskInfo(pSubtaskId, pSplitInfo);
    lSubtaskInfo.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfo.subtaskId);
    
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

pmStatus pmDataDistributionCB::InvokeDirect(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) const
{
	if(!mCallback)
		return pmSuccess;

    pmSubtaskInfo lSubtaskInfo = pTask->GetPreSubscriptionSubtaskInfo(pSubtaskId, pSplitInfo);
    lSubtaskInfo.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfo.subtaskId);

    return mCallback(pTask->GetTaskInfo(), pStub->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo);
}


/* class pmSubtaskCB */
#ifdef SUPPORT_OPENCL
pmSubtaskCB::pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom, std::string pOpenCLImplementation)
#else
pmSubtaskCB::pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom)
#endif
	: mCallback_CPU(pCallback_CPU)
	, mCallback_GPU_CUDA(pCallback_GPU_CUDA)
    , mCallback_GPU_Custom(pCallback_GPU_Custom)
#ifdef SUPPORT_OPENCL
    , mOpenCLImplementation(pOpenCLImplementation)
#endif
{
}

bool pmSubtaskCB::IsCallbackDefinedForDevice(pmDeviceType pDeviceType) const
{
	switch(pDeviceType)
	{
		case CPU:
		{
			if(mCallback_CPU || !mOpenCLImplementation.empty())
				return true;
			
			break;
		}

#ifdef SUPPORT_CUDA
		case GPU_CUDA:
		{
			if(mCallback_GPU_CUDA || mCallback_GPU_Custom || !mOpenCLImplementation.empty())
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

bool pmSubtaskCB::HasOpenCLCallback() const
{
#ifdef SUPPORT_OPENCL
    return !mOpenCLImplementation.empty();
#else
    return false;
#endif
}
    
pmSubtaskCallback_CPU pmSubtaskCB::GetCpuCallback() const
{
    return mCallback_CPU;
}


#ifdef SUPPORT_CUDA
pmStatus pmSubtaskCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo, std::vector<pmCudaMemcpyCommand>* pHostToDeviceCommands /* = NULL */, std::vector<pmCudaMemcpyCommand>* pDeviceToHostCommands /* = NULL */, void* pStreamPtr /* = NULL */) const
#else
pmStatus pmSubtaskCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo) const
#endif
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

                pmSubtaskInfo lSubtaskInfo = pSubtaskInfo;
                lSubtaskInfo.subtaskId = pTask->GetPhysicalSubtaskId(pSubtaskInfo.subtaskId);

                lStatus = mCallback_CPU(pTaskInfo, lDeviceInfo, lSubtaskInfo);
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
            lStatus = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->InvokeKernel(pTask, (pmStubCUDA*)pStub, pTask->GetTaskInfo(), pTaskInfo, pSubtaskInfo, lCudaLaunchConf, mCallback_GPU_CUDA, mCallback_GPU_Custom, *((pmCudaStreamAutoPtr*)pStreamPtr), *pHostToDeviceCommands, *pDeviceToHostCommands);
            
			break;
		}
#endif
		
		default:	
			PMTHROW(pmFatalErrorException());
	}
    
    pTask->GetSubscriptionManager().DropScratchBuffersNotRequiredPostSubtaskExec(pStub, pSubtaskInfo.subtaskId, pSplitInfo);

	return lStatus;
}


/* class pmDataReductionCB */
pmDataReductionCB::pmDataReductionCB(pmDataReductionCallback pCallback)
	: mCallback(pCallback)
{
}

pmStatus pmDataReductionCB::Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, bool pMultiAssign1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, bool pMultiAssign2) const
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_REDUCTION);
#endif

	if(!mCallback)
		return pmSuccess;
    
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    pmSubtaskInfo lSubtaskInfo1 = lSubscriptionManager.GetSubtaskInfo(pStub1, pSubtaskId1, pSplitInfo1);
    pmSubtaskInfo lSubtaskInfo2 = lSubscriptionManager.GetSubtaskInfo(pStub2, pSubtaskId2, pSplitInfo2);
    
    lSubtaskInfo1.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfo1.subtaskId);
    lSubtaskInfo2.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfo2.subtaskId);
    
	return mCallback(pTask->GetTaskInfo(), pStub1->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo1, pStub2->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo2);
}


/* class pmDataRedistributionCB */
pmDataRedistributionCB::pmDataRedistributionCB(pmDataRedistributionCallback pCallback)
	: mCallback(pCallback)
{
}

pmStatus pmDataRedistributionCB::Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign) const
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

	if(!mCallback)
		return pmSuccess;

    pmSubtaskInfo lSubtaskInfo = pTask->GetSubscriptionManager().GetSubtaskInfo(pStub, pSubtaskId, pSplitInfo);
    lSubtaskInfo.subtaskId = pTask->GetPhysicalSubtaskId(lSubtaskInfo.subtaskId);

	return mCallback(pTask->GetTaskInfo(), pStub->GetProcessingElement()->GetDeviceInfo(), lSubtaskInfo);
}


/* class pmDeviceSelectionCB */
pmDeviceSelectionCB::pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback)
	: mCallback(pCallback)
{
}

bool pmDeviceSelectionCB::Invoke(pmTask* pTask, const pmProcessingElement* pProcessingElement) const
{
	if(!mCallback)
		return pmSuccess;

	return mCallback(pTask->GetTaskInfo(), pProcessingElement->GetDeviceInfo());
}


/* class pmPreDataTransferCB */
pmPreDataTransferCB::pmPreDataTransferCB(pmPreDataTransferCallback pCallback)
	: mCallback(pCallback)
{
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
	: mCallback(pCallback)
{
}

pmStatus pmPostDataTransferCB::Invoke() const
{
	if(!mCallback)
		return pmSuccess;

	return pmSuccess;
	//return mCallback();
}


/* class pmTaskCompletionCB */
pmTaskCompletionCB::pmTaskCompletionCB(pmTaskCompletionCallback pCallback)
	: mCallback(pCallback)
    , mUserData(NULL)
{
}

pmStatus pmTaskCompletionCB::Invoke(pmTask* pTask) const
{
	if(!mCallback)
		return pmSuccess;

	return mCallback(pTask->GetTaskInfo());
}
    
pmTaskCompletionCallback pmTaskCompletionCB::GetCallback() const
{
    return mCallback;
}
    
void* pmTaskCompletionCB::GetUserData() const
{
    return mUserData;
}

void pmTaskCompletionCB::SetUserData(void* pUserData)
{
    mUserData = pUserData;
}


};



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

pmStatus pmDataDistributionCB::Invoke(pmTask* pTask, ulong pSubtaskId, pmDeviceTypes pDeviceType)
{
	if(!mCallback)
		return pmSuccess;

	return mCallback(pTask->GetTaskInfo(), pSubtaskId, pDeviceType);
}


/* class pmSubtaskCB */
pmSubtaskCB::pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA)
{
	mCallback_CPU = pCallback_CPU;
	mCallback_GPU_CUDA = pCallback_GPU_CUDA;
}

pmSubtaskCB::~pmSubtaskCB()
{
}

bool pmSubtaskCB::IsCallbackDefinedForDevice(pmDeviceTypes pDeviceType)
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
			if(mCallback_GPU_CUDA)
				return true;

			break;
		}
#endif
	
		case MAX_DEVICE_TYPES:
			PMTHROW(pmFatalErrorException());
	}

	return false;
}

pmStatus pmSubtaskCB::Invoke(pmDeviceTypes pDeviceType, pmTask* pTask, ulong pSubtaskId, size_t pBoundHardwareDeviceIndex)
{
    bool lOutputMemWriteOnly = false;

	switch(pDeviceType)
	{
		case CPU:
		{
			if(!mCallback_CPU)
				return pmSuccess;

			pmSubtaskInfo lSubtaskInfo;
			pTask->GetSubtaskInfo(pSubtaskId, lSubtaskInfo, lOutputMemWriteOnly);
			return mCallback_CPU(pTask->GetTaskInfo(), lSubtaskInfo);

			break;
		}

#ifdef SUPPORT_CUDA
		case GPU_CUDA:
		{
			if(!mCallback_GPU_CUDA)
				return pmSuccess;
            
			pmSubtaskInfo lSubtaskInfo;
			pTask->GetSubtaskInfo(pSubtaskId, lSubtaskInfo, lOutputMemWriteOnly);
            
			pmCudaLaunchConf& lCudaLaunchConf = pTask->GetSubscriptionManager().GetCudaLaunchConf(pSubtaskId);
			return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->InvokeKernel(pBoundHardwareDeviceIndex, pTask->GetTaskInfo(), lSubtaskInfo, lCudaLaunchConf, lOutputMemWriteOnly, mCallback_GPU_CUDA);

			break;
		}
#endif
		
		default:	
			PMTHROW(pmFatalErrorException());
	}

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

pmStatus pmDataReductionCB::Invoke(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2)
{
	if(!mCallback)
		return pmSuccess;
    
    bool lOutputMemWriteOnly = false;

	pmSubtaskInfo lSubtaskInfo1, lSubtaskInfo2;
	pTask->GetSubtaskInfo(pSubtaskId1, lSubtaskInfo1, lOutputMemWriteOnly);
	pTask->GetSubtaskInfo(pSubtaskId2, lSubtaskInfo2, lOutputMemWriteOnly);

	return mCallback(pTask->GetTaskInfo(), lSubtaskInfo1, lSubtaskInfo2);
}


/* class pmDataRedistributionCB */
pmDataRedistributionCB::pmDataRedistributionCB(pmDataRedistributionCallback pCallback)
{
	mCallback = pCallback;
}

pmDataRedistributionCB::~pmDataRedistributionCB()
{
}

pmStatus pmDataRedistributionCB::Invoke(pmTask* pTask, ulong pSubtaskId)
{
	if(!mCallback)
		return pmSuccess;

    bool lOutputMemWriteOnly = false;

	pmSubtaskInfo lSubtaskInfo;
	pTask->GetSubtaskInfo(pSubtaskId, lSubtaskInfo, lOutputMemWriteOnly);

	return mCallback(pTask->GetTaskInfo(), lSubtaskInfo);
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


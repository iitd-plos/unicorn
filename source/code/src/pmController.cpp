
#include "pmController.h"
#include "pmCommunicator.h"
#include "pmDevicePool.h"
#include "pmNetwork.h"
#include "pmMemoryManager.h"
#include "pmTaskManager.h"
#include "pmScheduler.h"
#include "pmDispatcherGPU.h"
#include "pmStubManager.h"
#include "pmLogger.h"
#include "pmCallbackUnit.h"
#include "pmCallback.h"
#include "pmMemSection.h"
#include "pmTask.h"
#include "pmSignalWait.h"

namespace pm
{

#define SAFE_DESTROY(x, y) if(x) x->y();

pmController* pmController::mController = NULL;

pmController::pmController()
{
	mLastErrorCode = 0;
	mFinalizedHosts = 0;
	mSignalWait = NULL;
}

pmController::~pmController()
{
	delete mSignalWait;
}

pmController* pmController::GetController()
{
	if(!mController)
	{
		if(CreateAndInitializeController() == pmSuccess)
		{
			if(!pmLogger::GetLogger())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Logger Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmDispatcherGPU::GetDispatcherGPU())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Dispatcher Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmStubManager::GetStubManager())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Stub Manager Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Network Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmCommunicator::GetCommunicator())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Communicator Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmMachinePool::GetMachinePool())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Machine Pool Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Memory Manager Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmTaskManager::GetTaskManager())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Task Manager Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}

			if(!pmScheduler::GetScheduler())
			{
				pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Scheduler Initialization Failed");
				PMTHROW(pmFatalErrorException());
			}
		}
		else
		{
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Controller Initialization Failed");
			PMTHROW(pmFatalErrorException());
		}
	}

	return mController;
}

pmStatus pmController::DestroyController()
{
	SAFE_DESTROY(pmScheduler::GetScheduler(), DestroyScheduler);
	SAFE_DESTROY(pmTaskManager::GetTaskManager(), DestroyTaskManager);
	SAFE_DESTROY(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager(), DestroyMemoryManager);
	SAFE_DESTROY(pmMachinePool::GetMachinePool(), DestroyMachinePool);
	SAFE_DESTROY(pmCommunicator::GetCommunicator(), DestroyCommunicator);
	SAFE_DESTROY(NETWORK_IMPLEMENTATION_CLASS::GetNetwork(), DestroyNetwork);
	SAFE_DESTROY(pmStubManager::GetStubManager(), DestroyStubManager);
	SAFE_DESTROY(pmDispatcherGPU::GetDispatcherGPU(), DestroyDispatcherGPU);
	SAFE_DESTROY(pmLogger::GetLogger(), DestroyLogger);

	delete mController;
	mController = NULL;

	return pmSuccess;
}

pmStatus pmController::FinalizeController()
{
	if(pmScheduler::GetScheduler()->SendFinalizationSignal() != pmSuccess)
		PMTHROW(pmFatalErrorException());

	if(mSignalWait)
		PMTHROW(pmFatalErrorException());

	mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();
	mSignalWait->Wait();

	return DestroyController();
}

/* Only to be called on master controller (with mpi host id 0) */
pmStatus pmController::ProcessFinalization()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	++mFinalizedHosts;
	if(mFinalizedHosts == NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount())
		pmScheduler::GetScheduler()->BroadcastTerminationSignal();

	return pmSuccess;
}

pmStatus pmController::ProcessTermination()
{
	if(!mSignalWait)
		PMTHROW(pmFatalErrorException());
	
	mSignalWait->Signal();

	return pmSuccess;
}

pmStatus pmController::CreateAndInitializeController()
{
	mController = new pmController();

	if(mController)
		return pmSuccess;

	return pmFatalError;
}

/* Public API */
pmStatus pmController::RegisterCallbacks_Public(char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle)
{
	pmDataDistributionCB* lDataDistribution = NULL;
	pmSubtaskCB* lSubtask = NULL;
	pmDataReductionCB* lDataReduction = NULL;
	pmDeviceSelectionCB* lDeviceSelection = NULL;
	pmDataScatterCB* lDataScatter = NULL;
	pmPreDataTransferCB* lPreDataTransfer = NULL;
	pmPostDataTransferCB* lPostDataTransfer = NULL;
	pmCallbackUnit* lCallbackUnit = NULL;

	if(strlen(pKey) >= MAX_CB_KEY_LEN)
		PMTHROW(pmMaxKeyLengthExceeded);

	*pCallbackHandle = NULL;

	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		if(pCallbacks.dataDistribution)
			DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataDistribution, pmDataDistributionCB, new pmDataDistributionCB(pCallbacks.dataDistribution));
	if(pCallbacks.subtask_cpu || pCallbacks.subtask_gpu_cuda)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lSubtask, pmSubtaskCB, new pmSubtaskCB(pCallbacks.subtask_cpu, pCallbacks.subtask_gpu_cuda));
	if(pCallbacks.dataReduction)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataReduction, pmDataReductionCB, new pmDataReductionCB(pCallbacks.dataReduction));
	if(pCallbacks.dataScatter)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataScatter, pmDataScatterCB, new pmDataScatterCB(pCallbacks.dataScatter));
	if(pCallbacks.deviceSelection)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDeviceSelection, pmDeviceSelectionCB, new pmDeviceSelectionCB(pCallbacks.deviceSelection));
	if(pCallbacks.preDataTransfer)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPreDataTransfer, pmPreDataTransferCB, new pmPreDataTransferCB(pCallbacks.preDataTransfer));
	if(pCallbacks.postDataTransfer)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPostDataTransfer, pmPostDataTransferCB, new pmPostDataTransferCB(pCallbacks.postDataTransfer));
	DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lCallbackUnit, pmCallbackUnit, new pmCallbackUnit(pKey, lDataDistribution, lSubtask, lDataReduction, lDeviceSelection, lDataScatter, lPreDataTransfer, lPostDataTransfer));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

		*pCallbackHandle = lCallbackUnit;

	return pmSuccess;
}

pmStatus pmController::ReleaseCallbacks_Public(pmCallbackHandle pCallbackHandle)
{
	pmCallbackUnit* lCallbackUnit = static_cast<pmCallbackUnit*>(pCallbackHandle);

	delete lCallbackUnit->GetDataDistributionCB();
	delete lCallbackUnit->GetSubtaskCB();
	delete lCallbackUnit->GetDataReductionCB();
	delete lCallbackUnit->GetDeviceSelectionCB();
	delete lCallbackUnit->GetPreDataTransferCB();
	delete lCallbackUnit->GetPostDataTransferCB();
	delete lCallbackUnit;

	return pmSuccess;
}

pmStatus pmController::CreateMemory_Public(pmMemInfo pMemInfo, size_t pLength, pmMemHandle* pMem)
{
	*pMem = NULL;

	if(pMemInfo == INPUT_MEM_READ_ONLY)
	{
		pmMemSection* lInputMem = new pmInputMemSection(pLength);
		*pMem = lInputMem->GetMem();
	}
	else
	{
		pmMemSection* lOutputMem = new pmOutputMemSection(pLength, (pMemInfo == OUTPUT_MEM_WRITE_ONLY)?pmOutputMemSection::WRITE_ONLY:pmOutputMemSection::READ_WRITE);
		*pMem = lOutputMem->GetMem();
	}

	return pmSuccess;
}

pmStatus pmController::ReleaseMemory_Public(pmMemHandle pMem)
{
	pmMemSection* lMemSection = pmMemSection::FindMemSection(pMem);

	delete lMemSection;

	return pmSuccess;
}

pmStatus pmController::SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle)
{
	*pTaskHandle = NULL;

	pmMemSection* lInputMem = pmMemSection::FindMemSection(pTaskDetails.inputMem);
	pmMemSection* lOutputMem = pmMemSection::FindMemSection(pTaskDetails.outputMem);

	if(!dynamic_cast<pmInputMemSection*>(lInputMem) || !dynamic_cast<pmOutputMemSection*>(lOutputMem))
		PMTHROW(pmUnrecognizedMemoryException());

	pmCallbackUnit* lCallbackUnit = static_cast<pmCallbackUnit*>(pTaskDetails.callbackHandle);

	if(pTaskDetails.taskConfLength == 0)
		pTaskDetails.taskConf = NULL;

	*pTaskHandle = new pmLocalTask(pTaskDetails.taskConf, pTaskDetails.taskConfLength, pTaskDetails.taskId, lInputMem, lOutputMem, pTaskDetails.subtaskCount, lCallbackUnit, PM_LOCAL_MACHINE, PM_GLOBAL_CLUSTER, pTaskDetails.priority);

	pmTaskManager::GetTaskManager()->SubmitTask(static_cast<pmLocalTask*>(*pTaskHandle));

	return pmSuccess;
}

pmStatus pmController::WaitForTaskCompletion_Public(pmTaskHandle pTaskHandle)
{
	return (static_cast<pmLocalTask*>(pTaskHandle))->WaitForCompletion();
}

pmStatus pmController::ReleaseTask_Public(pmTaskHandle pTaskHandle)
{
	pmStatus lStatus = WaitForTaskCompletion_Public(pTaskHandle);

	delete static_cast<pmLocalTask*>(pTaskHandle);

	return lStatus;
}

pmStatus pmController::GetTaskExecutionTimeInSecs_Public(pmTaskHandle pTaskHandle, double* pTime)
{
	*pTime = (static_cast<pmLocalTask*>(pTaskHandle))->GetExecutionTimeInSecs();

	return pmSuccess;
}

pmStatus pmController::SubscribeToMemory_Public(pmTaskHandle pTaskHandle, ulong pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pScatterGatherInfo)
{
	return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().RegisterSubscription(pSubtaskId, pIsInputMemory, pScatterGatherInfo);
}

pmStatus pmController::SetCudaLaunchConf_Public(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf)
{
	return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().SetCudaLaunchConf(pSubtaskId, pCudaLaunchConf);
}

uint pmController::GetHostId_Public()
{
	if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork())
		PMTHROW(pmFatalErrorException());

	return NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId();
}

uint pmController::GetHostCount_Public()
{
	if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork())
		PMTHROW(pmFatalErrorException());

	return NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
}

} // end namespace pm

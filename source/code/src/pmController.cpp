
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

namespace pm
{

#define SAFE_DESTROY(x, y) if(x) x->y();

pmController* pmController::mController = NULL;

pmController* pmController::GetController()
{
	if(!mController)
	{
		if(CreateAndInitializeController() == pmSuccess)
		{
			if(!pmDispatcherGPU::GetDispatcherGPU()
			|| !pmStubManager::GetStubManager()
			|| !NETWORK_IMPLEMENTATION_CLASS::GetNetwork() 
			|| !pmCommunicator::GetCommunicator()
			|| !pmMachinePool::GetMachinePool()
			|| !MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()
			|| !pmTaskManager::GetTaskManager()
			|| !pmScheduler::GetScheduler()
			|| !pmLogger::GetLogger()
			)
				throw pmFatalErrorException();
		}
	}

	return mController;
}

pmStatus pmController::DestroyController()
{
	SAFE_DESTROY(pmLogger::GetLogger(), DestroyLogger);
	SAFE_DESTROY(pmScheduler::GetScheduler(), DestroyScheduler);
	SAFE_DESTROY(pmTaskManager::GetTaskManager(), DestroyTaskManager);
	SAFE_DESTROY(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager(), DestroyMemoryManager);
	SAFE_DESTROY(pmMachinePool::GetMachinePool(), DestroyMachinePool);
	SAFE_DESTROY(pmCommunicator::GetCommunicator(), DestroyCommunicator);
	SAFE_DESTROY(NETWORK_IMPLEMENTATION_CLASS::GetNetwork(), DestroyNetwork);
	SAFE_DESTROY(pmStubManager::GetStubManager(), DestroyStubManager);
	SAFE_DESTROY(pmDispatcherGPU::GetDispatcherGPU(), DestroyDispatcherGPU);
	
	delete mController;
	mController = NULL;

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
	pmDataDistributionCB* lDataDistribution;
	pmSubtaskCB* lSubtask;
	pmDataReductionCB* lDataReduction;
	pmDeviceSelectionCB* lDeviceSelection;
	pmPreDataTransferCB* lPreDataTransfer;
	pmPostDataTransferCB* lPostDataTransfer;
	pmCallbackUnit* lCallbackUnit;

	if(strlen(pKey) >= MAX_CB_KEY_LEN)
		throw pmMaxKeyLengthExceeded;

	*pCallbackHandle = NULL;

	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataDistribution, new pmDataDistributionCB(pCallbacks.dataDistribution));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lSubtask, new pmSubtaskCB(pCallbacks.subtask_cpu, pCallbacks.subtask_gpu_cuda));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataReduction, new pmDataReductionCB(pCallbacks.dataReduction));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDeviceSelection, new pmDeviceSelectionCB(pCallbacks.deviceSelection));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPreDataTransfer, new pmPreDataTransferCB(pCallbacks.preDataTransfer));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPostDataTransfer, new pmPostDataTransferCB(pCallbacks.postDataTransfer));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lCallbackUnit, new pmCallbackUnit(pKey, lDataDistribution, lSubtask, lDataReduction, lDeviceSelection, lPreDataTransfer, lPostDataTransfer));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

	*pCallbackHandle = lCallbackUnit;

	return pmSuccess;
}

pmStatus pmController::ReleaseCallbacks_Public(pmCallbackHandle* pCallbackHandle)
{
	pmCallbackUnit* lCallbackUnit = static_cast<pmCallbackUnit*>(*pCallbackHandle);

	delete lCallbackUnit->GetDataDistributionCB();
	delete lCallbackUnit->GetSubtaskCB();
	delete lCallbackUnit->GetDataReductionCB();
	delete lCallbackUnit->GetDeviceSelectionCB();
	delete lCallbackUnit->GetPreDataTransferCB();
	delete lCallbackUnit->GetPostDataTransferCB();
	delete lCallbackUnit;
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

pmStatus pmController::ReleaseMemory_Public(pmMemHandle* pMem)
{
	pmMemSection* lMemSection = pmMemSection::FindMemSection(*pMem);

	delete lMemSection;

	return pmSuccess;
}

pmStatus pmController::SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle)
{
	*pTaskHandle = NULL;

	pmMemSection* lInputMem = pmMemSection::FindMemSection(pTaskDetails.inputMem);
	pmMemSection* lOutputMem = pmMemSection::FindMemSection(pTaskDetails.outputMem);

	if(!dynamic_cast<pmInputMemSection*>(lInputMem) || !dynamic_cast<pmOutputMemSection*>(lOutputMem))
		throw pmUnrecognizedMemoryException();

	pmCallbackUnit* lCallbackUnit = static_cast<pmCallbackUnit*>(*(pTaskDetails.callbackHandle));

	if(pTaskDetails.taskConfLength == 0)
		pTaskDetails.taskConf = NULL;

	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, *pTaskHandle, new pmLocalTask(pTaskDetails.taskConf, pTaskDetails.taskConfLength, pTaskDetails.taskId, lInputMem, lOutputMem, pTaskDetails.subtaskCount, lCallbackUnit, PM_LOCAL_MACHINE, PM_GLOBAL_CLUSTER, pTaskDetails.priority));
		pmTaskManager::GetTaskManager()->SubmitTask(static_cast<pmLocalTask*>(*pTaskHandle));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

	return pmSuccess;
}

pmStatus pmController::WaitForTaskCompletion_Public(pmTaskHandle* pTaskHandle)
{
	return (static_cast<pmLocalTask*>(*pTaskHandle))->WaitForCompletion();
}

pmStatus pmController::ReleaseTask_Public(pmTaskHandle* pTaskHandle)
{
	pmStatus lStatus = WaitForTaskCompletion_Public(pTaskHandle);

	delete static_cast<pmLocalTask*>(*pTaskHandle);

	return lStatus;
}

pmStatus pmController::GetTaskExecutionTimeInSecs_Public(pmTaskHandle* pTaskHandle, double* pTime)
{
	*pTime = (static_cast<pmLocalTask*>(*pTaskHandle))->GetExecutionTimeInSecs();

	return pmSuccess;
}

pmStatus pmController::SubscribeToMemory_Public(pmTaskHandle pTaskHandle, ulong pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pScatterGatherInfo)
{
	return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().RegisterSubscription(pSubtaskId, pIsInputMemory, pScatterGatherInfo);
}

uint pmController::GetHostId_Public()
{
	if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork())
		throw pmFatalErrorException();

	return NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId();
}

uint pmController::GetHostCount_Public()
{
	if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork())
		throw pmFatalErrorException();

	return NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
}

} // end namespace pm

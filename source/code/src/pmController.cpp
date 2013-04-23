
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
#include "pmRedistributor.h"
#include "pmSubscriptionManager.h"
#include "pmUtility.h"
#include "pmReducer.h"

namespace pm
{

pmController::pmController()
    : mLastErrorCode(0)
	, mFinalizedHosts(0)
	, mSignalWait(NULL)
    , mResourceLock __LOCK_NAME__("pmController::mResourceLock")
{
#ifdef ENABLE_ACCUMULATED_TIMINGS
    pmAccumulatedTimesSorter::GetAccumulatedTimesSorter();
#endif

    pmLogger::GetLogger();
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    TLS_IMPLEMENTATION_CLASS::GetTls();
    pmDispatcherGPU::GetDispatcherGPU();
    pmStubManager::GetStubManager();
    pmCommunicator::GetCommunicator();
    pmMachinePool::GetMachinePool();
    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager();
    pmTaskManager::GetTaskManager();
    pmScheduler::GetScheduler();
    pmTimedEventManager::GetTimedEventManager();
    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool();
    
#ifdef DUMP_EVENT_TIMELINE
    pmStubManager::GetStubManager()->InitializeEventTimelines();
#endif
}

pmController::~pmController()
{
	delete mSignalWait;
}

pmController* pmController::GetController()
{
    static pmController lController;
    static bool lFirstCall = true;
    
    if(lFirstCall)
    {
        if(NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GlobalBarrier() != pmSuccess)
        {
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Global Initialization Barrier Failed");
            PMTHROW(pmFatalErrorException());
        }
     
        lFirstCall = false;
    }

    return &lController;
}

pmStatus pmController::DestroyController()
{
    pmTaskManager::GetTaskManager()->WaitForAllTasksToFinish();
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->FreezeReceptionAndFinishCommands(); // All commands are in scheduler queue after this stage and no more are generated by network
    pmScheduler::GetScheduler()->WaitForAllCommandsToFinish();
    
    pmMemSection::DeleteAllLocalMemSections();

	return pmSuccess;
}

pmStatus pmController::FinalizeController()
{
    pmMachine* lMasterHost = pmMachinePool::GetMachinePool()->GetMachine(0);
    
    if(lMasterHost != PM_LOCAL_MACHINE)
    {
        if(pmScheduler::GetScheduler()->SendFinalizationSignal() != pmSuccess)
            PMTHROW(pmFatalErrorException());

        pmCommunicatorCommand::hostFinalizationStruct lBroadcastData;
        pmCommunicatorCommandPtr lBroadcastCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::BROADCAST,pmCommunicatorCommand::HOST_FINALIZATION_TAG, lMasterHost, pmCommunicatorCommand::HOST_FINALIZATION_STRUCT, &lBroadcastData, 1, NULL, 0);
    
        pmCommunicator::GetCommunicator()->Broadcast(lBroadcastCommand);
    }
    else
    {
        if(mSignalWait)
            PMTHROW(pmFatalErrorException());

        mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();

        if(pmScheduler::GetScheduler()->SendFinalizationSignal() != pmSuccess)
            PMTHROW(pmFatalErrorException());
        
        mSignalWait->Wait();
    }

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

/* Only to be called on master controller (with mpi host id 0) */
pmStatus pmController::ProcessTermination()
{
	if(!mSignalWait)
		PMTHROW(pmFatalErrorException());
	
	mSignalWait->Signal();

	return pmSuccess;
}

/* Public API */
pmStatus pmController::RegisterCallbacks_Public(char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle)
{
	pmDataDistributionCB* lDataDistribution = NULL;
	pmSubtaskCB* lSubtask = NULL;
	pmDataReductionCB* lDataReduction = NULL;
	pmDeviceSelectionCB* lDeviceSelection = NULL;
	pmDataRedistributionCB* lDataRedistributionCB = NULL;
	pmPreDataTransferCB* lPreDataTransfer = NULL;
	pmPostDataTransferCB* lPostDataTransfer = NULL;
	pmCallbackUnit* lCallbackUnit = NULL;

	if(strlen(pKey) >= MAX_CB_KEY_LEN)
		PMTHROW(pmMaxKeyLengthExceeded);

	*pCallbackHandle = NULL;

	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
    if(pCallbacks.dataDistribution)
        DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataDistribution, pmDataDistributionCB, new pmDataDistributionCB(pCallbacks.dataDistribution));
	if(pCallbacks.subtask_cpu || pCallbacks.subtask_gpu_cuda || pCallbacks.subtask_gpu_custom)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lSubtask, pmSubtaskCB, new pmSubtaskCB(pCallbacks.subtask_cpu, pCallbacks.subtask_gpu_cuda, pCallbacks.subtask_gpu_custom));
	if(pCallbacks.dataReduction)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataReduction, pmDataReductionCB, new pmDataReductionCB(pCallbacks.dataReduction));
	if(pCallbacks.dataRedistribution)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDataRedistributionCB, pmDataRedistributionCB, new pmDataRedistributionCB(pCallbacks.dataRedistribution));
	if(pCallbacks.deviceSelection)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lDeviceSelection, pmDeviceSelectionCB, new pmDeviceSelectionCB(pCallbacks.deviceSelection));
	if(pCallbacks.preDataTransfer)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPreDataTransfer, pmPreDataTransferCB, new pmPreDataTransferCB(pCallbacks.preDataTransfer));
	if(pCallbacks.postDataTransfer)
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lPostDataTransfer, pmPostDataTransferCB, new pmPostDataTransferCB(pCallbacks.postDataTransfer));
	DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lCallbackUnit, pmCallbackUnit, new pmCallbackUnit(pKey, lDataDistribution, lSubtask, lDataReduction, lDeviceSelection, lDataRedistributionCB, lPreDataTransfer, lPostDataTransfer));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

    *pCallbackHandle = lCallbackUnit;
    
    if(NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GlobalBarrier() != pmSuccess)
    {
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Callback Registration Barrier Failed");
        PMTHROW(pmFatalErrorException());
    }

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

pmStatus pmController::CreateMemory_Public(size_t pLength, pmMemHandle* pMem)
{
	*pMem = NULL;

    pmMemSection* lMemSection = pmMemSection::CreateMemSection(pLength, PM_LOCAL_MACHINE);
    *pMem = new pmUserMemHandle(lMemSection);
    
	return pmSuccess;
}

pmStatus pmController::ReleaseMemory_Public(pmMemHandle pMem)
{
    if(!pMem)
        PMTHROW(pmFatalErrorException());

	pmMemSection* lMemSection = (reinterpret_cast<pmUserMemHandle*>(pMem))->GetMemSection();

    delete (reinterpret_cast<pmUserMemHandle*>(pMem));
	lMemSection->UserDelete();

	return pmSuccess;
}

pmStatus pmController::FetchMemory_Public(pmMemHandle pMem)
{
    if(!pMem)
        PMTHROW(pmFatalErrorException());

	pmMemSection* lMemSection = (reinterpret_cast<pmUserMemHandle*>(pMem))->GetMemSection();
    
    return lMemSection->Fetch(MAX_PRIORITY_LEVEL);
}

pmStatus pmController::FetchMemoryRange_Public(pmMemHandle pMem, size_t pOffset, size_t pLength)
{
    if(!pMem)
        PMTHROW(pmFatalErrorException());

	pmMemSection* lMemSection = (reinterpret_cast<pmUserMemHandle*>(pMem))->GetMemSection();
    
    return lMemSection->FetchRange(MAX_PRIORITY_LEVEL, pOffset, pLength);
}
    
pmStatus pmController::GetRawMemPtr_Public(pmMemHandle pMem, void** pPtr)
{
    if(!pMem)
        PMTHROW(pmFatalErrorException());

	pmMemSection* lMemSection = (reinterpret_cast<pmUserMemHandle*>(pMem))->GetMemSection();
    *pPtr = lMemSection->GetMem();

    return pmSuccess;
}

pmStatus pmController::SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle)
{
	*pTaskHandle = NULL;

	pmMemSection* lInputMem = pTaskDetails.inputMemHandle ? (reinterpret_cast<pmUserMemHandle*>(pTaskDetails.inputMemHandle))->GetMemSection() : NULL;
	pmMemSection* lOutputMem = pTaskDetails.outputMemHandle ? (reinterpret_cast<pmUserMemHandle*>(pTaskDetails.outputMemHandle))->GetMemSection() : NULL;
	pmCallbackUnit* lCallbackUnit = static_cast<pmCallbackUnit*>(pTaskDetails.callbackHandle);

	if(pTaskDetails.taskConfLength == 0)
		pTaskDetails.taskConf = NULL;

    scheduler::schedulingModel lModel = scheduler::PUSH;
    if(pTaskDetails.policy != SLOW_START)
    {
        if(pTaskDetails.policy == RANDOM_STEAL)
            lModel = scheduler::PULL;
        else if(pTaskDetails.policy == EQUAL_STATIC)
            lModel = scheduler::STATIC_EQUAL;
        else if(pTaskDetails.policy == PROPORTIONAL_STATIC)
            lModel = scheduler::STATIC_PROPORTIONAL;
        else
            PMTHROW(pmFatalErrorException());
    }
    
    if((lInputMem == lOutputMem) || (!lInputMem && !lOutputMem))
        PMTHROW(pmFatalErrorException());
        
	*pTaskHandle = new pmLocalTask(pTaskDetails.taskConf, pTaskDetails.taskConfLength, pTaskDetails.taskId, lInputMem, lOutputMem, pTaskDetails.inputMemInfo, pTaskDetails.outputMemInfo, pTaskDetails.subtaskCount, lCallbackUnit, pTaskDetails.timeOutInSecs, PM_LOCAL_MACHINE, PM_GLOBAL_CLUSTER, pTaskDetails.priority, lModel, pTaskDetails.multiAssignEnabled);

	pmTaskManager::GetTaskManager()->SubmitTask(static_cast<pmLocalTask*>(*pTaskHandle));

	return pmSuccess;
}

pmStatus pmController::WaitForTaskCompletion_Public(pmTaskHandle pTaskHandle)
{
    if(!pTaskHandle)
        PMTHROW(pmFatalErrorException());

	return (static_cast<pmLocalTask*>(pTaskHandle))->WaitForCompletion();
}

pmStatus pmController::ReleaseTask_Public(pmTaskHandle pTaskHandle)
{
    if(!pTaskHandle)
        PMTHROW(pmFatalErrorException());

	pmStatus lStatus = WaitForTaskCompletion_Public(pTaskHandle);
	static_cast<pmLocalTask*>(pTaskHandle)->UserDeleteTask();

	return lStatus;
}

pmStatus pmController::GetTaskExecutionTimeInSecs_Public(pmTaskHandle pTaskHandle, double* pTime)
{
    if(!pTaskHandle)
        PMTHROW(pmFatalErrorException());

	*pTime = (static_cast<pmLocalTask*>(pTaskHandle))->GetExecutionTimeInSecs();

	return pmSuccess;
}

pmStatus pmController::SubscribeToMemory_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSubscriptionType pSubscriptionType, pmSubscriptionInfo pSubscriptionInfo)
{
    pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId);
    
	return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().RegisterSubscription(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId, pSubscriptionType, pSubscriptionInfo);
}

pmStatus pmController::RedistributeData_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, size_t pOffset, size_t pLength, unsigned int pOrder)
{
    pmTask* lTask = static_cast<pmTask*>(pTaskHandle);
    lTask->GetRedistributor()->RedistributeData(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId, pOffset, pLength, pOrder);
    
    return pmSuccess;
}
    
pmStatus pmController::SetCudaLaunchConf_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf)
{
    pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId);

	return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().SetCudaLaunchConf(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId, pCudaLaunchConf);
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
    
void* pmController::GetScratchBuffer_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmScratchBufferInfo pScratchBufferInfo, size_t pBufferSize)
{
    pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId);
    
    return (static_cast<pmTask*>(pTaskHandle))->GetSubscriptionManager().GetScratchBuffer(static_cast<pmExecutionStub*>(pDeviceHandle), pSubtaskId, pScratchBufferInfo, pBufferSize);
    
    return NULL;
}
    
pmStatus pmController::pmReduceInts_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    if(!pTaskHandle || !pDevice1Handle || !pDevice2Handle || pReductionType >= MAX_REDUCTION_TYPES)
        PMTHROW(pmFatalErrorException());

    return (static_cast<pmTask*>(pTaskHandle))->GetReducer()->ReduceInts(static_cast<pmExecutionStub*>(pDevice1Handle), pSubtask1Id, static_cast<pmExecutionStub*>(pDevice2Handle), pSubtask2Id, pReductionType);
}

pmStatus pmController::pmReduceUInts_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    if(!pTaskHandle || !pDevice1Handle || !pDevice2Handle || pReductionType >= MAX_REDUCTION_TYPES)
        PMTHROW(pmFatalErrorException());

    return (static_cast<pmTask*>(pTaskHandle))->GetReducer()->ReduceUInts(static_cast<pmExecutionStub*>(pDevice1Handle), pSubtask1Id, static_cast<pmExecutionStub*>(pDevice2Handle), pSubtask2Id, pReductionType);
}

pmStatus pmController::pmReduceLongs_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    if(!pTaskHandle || !pDevice1Handle || !pDevice2Handle || pReductionType >= MAX_REDUCTION_TYPES)
        PMTHROW(pmFatalErrorException());

    return (static_cast<pmTask*>(pTaskHandle))->GetReducer()->ReduceLongs(static_cast<pmExecutionStub*>(pDevice1Handle), pSubtask1Id, static_cast<pmExecutionStub*>(pDevice2Handle), pSubtask2Id, pReductionType);
}

pmStatus pmController::pmReduceULongs_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    if(!pTaskHandle || !pDevice1Handle || !pDevice2Handle || pReductionType >= MAX_REDUCTION_TYPES)
        PMTHROW(pmFatalErrorException());

    return (static_cast<pmTask*>(pTaskHandle))->GetReducer()->ReduceULongs(static_cast<pmExecutionStub*>(pDevice1Handle), pSubtask1Id, static_cast<pmExecutionStub*>(pDevice2Handle), pSubtask2Id, pReductionType);
}

pmStatus pmController::pmReduceFloats_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    if(!pTaskHandle || !pDevice1Handle || !pDevice2Handle || pReductionType >= MAX_REDUCTION_TYPES)
        PMTHROW(pmFatalErrorException());

    return (static_cast<pmTask*>(pTaskHandle))->GetReducer()->ReduceFloats(static_cast<pmExecutionStub*>(pDevice1Handle), pSubtask1Id, static_cast<pmExecutionStub*>(pDevice2Handle), pSubtask2Id, pReductionType);
}

pmStatus pmController::MapFile_Public(const char* pPath)
{
    if(strlen(pPath) > MAX_FILE_SIZE_LEN - 1)
        PMTHROW(pmFatalErrorException());

    pmUtility::MapFileOnAllMachines(pPath);
    
    return pmSuccess;
}

void* pmController::GetMappedFile_Public(const char* pPath)
{
    return pmUtility::GetMappedFile(pPath);
}
    
pmStatus pmController::UnmapFile_Public(const char* pPath)
{
    if(strlen(pPath) > MAX_FILE_SIZE_LEN - 1)
        PMTHROW(pmFatalErrorException());

    pmUtility::UnmapFileOnAllMachines(pPath);
    
    return pmSuccess;
}

} // end namespace pm

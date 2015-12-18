
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

#include "pmScheduler.h"
#include "pmCommand.h"
#include "pmTask.h"
#include "pmTaskManager.h"
#include "pmSignalWait.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmSubtaskManager.h"
#include "pmMemoryManager.h"
#include "pmNetwork.h"
#include "pmDevicePool.h"
#include "pmAddressSpace.h"
#include "pmReducer.h"
#include "pmRedistributor.h"
#include "pmController.h"
#include "pmCallbackUnit.h"
#include "pmHeavyOperations.h"
#include "pmUtility.h"
#include "pmPreprocessorTask.h"

#ifdef USE_STEAL_AGENT_PER_NODE
    #include "pmStealAgent.h"
#endif

namespace pm
{

#ifdef DUMP_SCHEDULER_EVENT
const char* schedulerEventName[] =
{
	"NEW_SUBMISSION",
	"SUBTASK_EXECUTION",
	"STEAL_REQUEST_STEALER",
	"STEAL_PROCESS_TARGET",
	"STEAL_SUCCESS_TARGET",
	"STEAL_FAIL_TARGET",
	"STEAL_SUCCESS_STEALER",
	"STEAL_FAIL_STEALER",
	"SEND_ACKNOWLEDGEMENT",
	"RECEIVE_ACKNOWLEDGEMENT",
	"TASK_CANCEL",
	"TASK_FINISH",
    "TASK_COMPLETE",
	"SUBTASK_REDUCE",
	"COMMAND_COMPLETION",
    "HOST_FINALIZATION",
    "SUBTASK_RANGE_CANCEL",
    "REDISTRIBUTION_METADATA_EVENT",
    "RANGE_NEGOTIATION_EVENT",
    "RANGE_NEGOTIATION_SUCCESS_EVENT",
    "TERMINATE_TASK"
};
#endif
    
using namespace scheduler;
using namespace communicator;

#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_ack_transfer(const pmAddressSpace* addressSpace, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_ack_transfer(const pmAddressSpace* addressSpace, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
    
    sprintf(lStr, "Acknowledging address space %p (Remote mem (%d, %ld)) from offset %ld for length %ld to host %d", addressSpace, identifier.memOwnerHost, identifier.generationNumber, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_TRANSFER_ACK_DUMP(addressSpace, identifier, receiverOffset, offset, length, host) __dump_mem_ack_transfer(addressSpace, identifier, receiverOffset, offset, length, host);
#else
#define MEM_TRANSFER_ACK_DUMP(addressSpace, identifier, receiverOffset, offset, length, host)
#endif
    
#ifdef TRACK_SUBTASK_STEALS
void __dump_steal_request(uint sourceHost, uint targetHost, uint stealingDevice, uint targetDevice, double stealDeviceExecutionRate);
void __dump_steal_response(uint sourceHost, uint targetHost, uint stealingDevice, uint targetDevice, double targetDeviceExecutionRate, ulong transferredSubtasks);

void __dump_steal_request(uint sourceHost, uint targetHost, uint stealingDevice, uint targetDevice, double stealDeviceExecutionRate)
{
    char lStr[512];
    
    sprintf(lStr, "Steal request from host %d device %d at execution rate %f to host %d device %d", sourceHost, stealingDevice, stealDeviceExecutionRate, targetHost, targetDevice);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);    
}
    
void __dump_steal_response(uint sourceHost, uint targetHost, uint stealingDevice, uint targetDevice, double targetDeviceExecutionRate, ulong transferredSubtasks)
{
    char lStr[512];
    
    sprintf(lStr, "Steal response from host %d device %d at execution rate %f to host %d device %d assigning %ld subtasks", targetHost, targetDevice, targetDeviceExecutionRate, sourceHost, stealingDevice, transferredSubtasks);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);    
}
    
#define STEAL_REQUEST_DUMP(sourceHost, targetHost, stealingDevice, targetDevice, stealDeviceExecutionRate) __dump_steal_request(sourceHost, targetHost, stealingDevice, targetDevice, stealDeviceExecutionRate);
#define STEAL_RESPONSE_DUMP(sourceHost, targetHost, stealingDevice, targetDevice, targetDeviceExecutionRate, transferredSubtasks) __dump_steal_response(sourceHost, targetHost, stealingDevice, targetDevice, targetDeviceExecutionRate, transferredSubtasks);
#else
#define STEAL_REQUEST_DUMP(sourceHost, targetHost, stealingDevice, targetDevice, stealDeviceExecutionRate)
#define STEAL_RESPONSE_DUMP(sourceHost, targetHost, stealingDevice, targetDevice, targetDeviceExecutionRate, transferredSubtasks)
#endif

void SchedulerCommandCompletionCallback(const pmCommandPtr& pCommand)
{
	pmScheduler* lScheduler = pmScheduler::GetScheduler();
	lScheduler->CommandCompletionEvent(pCommand);
}

bool pmScheduler::SchedulingModelSupportsStealing(scheduler::schedulingModel pModel)
{
    return (pModel == PULL || pModel == PULL_WITH_AFFINITY);
}
    
pmScheduler::pmScheduler()
{
#ifdef TRACK_SUBTASK_EXECUTION
    mSubtasksAssigned = 0;
    mAcknowledgementsSent = 0;
#endif

	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(TASK_MEMORY_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SCATTERED_OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SCATTERED_OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SHADOW_MEM_TRANSFER_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SUBTASK_REDUCE_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SUBTASK_MEMORY_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(HOST_FINALIZATION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(REDISTRIBUTION_ORDER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(DATA_REDISTRIBUTION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(REDISTRIBUTION_OFFSETS_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SUBTASK_RANGE_CANCEL_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(NO_REDUCTION_REQD_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(MULTI_FILE_OPERATIONS_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_STRUCT);

	SetupPersistentCommunicationCommands();
}

pmScheduler::~pmScheduler()
{
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(TASK_MEMORY_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SCATTERED_OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SCATTERED_OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SHADOW_MEM_TRANSFER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SUBTASK_REDUCE_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SUBTASK_MEMORY_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(HOST_FINALIZATION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(REDISTRIBUTION_ORDER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(DATA_REDISTRIBUTION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(REDISTRIBUTION_OFFSETS_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SUBTASK_RANGE_CANCEL_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(NO_REDUCTION_REQD_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MULTI_FILE_OPERATIONS_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_STRUCT);

	DestroyPersistentCommunicationCommands();

#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down scheduler thread");
#endif
}

pmScheduler* pmScheduler::GetScheduler()
{
	static pmScheduler lScheduler;
    return &lScheduler;
}

pmCommandCompletionCallbackType pmScheduler::GetSchedulerCommandCompletionCallback()
{
	return SchedulerCommandCompletionCallback;
}

void pmScheduler::SetupPersistentCommunicationCommands()
{
    finalize_ptr<communicator::remoteSubtaskAssignStruct> lSubtaskAssignRecvData(new remoteSubtaskAssignStruct());
    finalize_ptr<communicator::taskEventStruct> lTaskEventRecvData(new taskEventStruct());
    finalize_ptr<communicator::stealRequestStruct> lStealRequestRecvData(new stealRequestStruct());
    finalize_ptr<communicator::stealResponseStruct> lStealResponseRecvData(new stealResponseStruct());
    finalize_ptr<communicator::subtaskRangeCancelStruct> lSubtaskRangeCancelData(new subtaskRangeCancelStruct());
    finalize_ptr<communicator::noReductionReqdStruct> lNoReductionReqdData(new noReductionReqdStruct());
    finalize_ptr<communicator::subtaskMemoryReduceStruct> lSubtaskMemoryReduceData(new subtaskMemoryReduceStruct());

#define PERSISTENT_RECV_COMMAND(tag, structEnumType, structType, recvDataPtr) pmCommunicatorCommand<structType>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, tag, NULL, structEnumType, recvDataPtr, 1, SchedulerCommandCompletionCallback)

	mRemoteSubtaskRecvCommand = PERSISTENT_RECV_COMMAND(REMOTE_SUBTASK_ASSIGNMENT_TAG, REMOTE_SUBTASK_ASSIGN_STRUCT, remoteSubtaskAssignStruct, lSubtaskAssignRecvData);
	mTaskEventRecvCommand = PERSISTENT_RECV_COMMAND(TASK_EVENT_TAG, TASK_EVENT_STRUCT, taskEventStruct, lTaskEventRecvData);
	mStealRequestRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_REQUEST_TAG, STEAL_REQUEST_STRUCT,	stealRequestStruct, lStealRequestRecvData);
    mStealResponseRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_RESPONSE_TAG, STEAL_RESPONSE_STRUCT, stealResponseStruct, lStealResponseRecvData);
    mSubtaskRangeCancelCommand = PERSISTENT_RECV_COMMAND(SUBTASK_RANGE_CANCEL_TAG, SUBTASK_RANGE_CANCEL_STRUCT, subtaskRangeCancelStruct, lSubtaskRangeCancelData);
    mNoReductionReqdCommand = PERSISTENT_RECV_COMMAND(NO_REDUCTION_REQD_TAG, NO_REDUCTION_REQD_STRUCT, noReductionReqdStruct, lNoReductionReqdData);
    mSubtaskMemoryReduceCommand = PERSISTENT_RECV_COMMAND(SUBTASK_MEMORY_REDUCE_TAG, SUBTASK_MEMORY_REDUCE_STRUCT, subtaskMemoryReduceStruct, lSubtaskMemoryReduceData);

    mRemoteSubtaskRecvCommand->SetPersistent();
    mTaskEventRecvCommand->SetPersistent();
    mStealRequestRecvCommand->SetPersistent();
    mStealResponseRecvCommand->SetPersistent();
    mSubtaskRangeCancelCommand->SetPersistent();
    mNoReductionReqdCommand->SetPersistent();
    mSubtaskMemoryReduceCommand->SetPersistent();
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mRemoteSubtaskRecvCommand);
    lNetwork->InitializePersistentCommand(mTaskEventRecvCommand);
    lNetwork->InitializePersistentCommand(mStealRequestRecvCommand);
    lNetwork->InitializePersistentCommand(mStealResponseRecvCommand);
    lNetwork->InitializePersistentCommand(mSubtaskRangeCancelCommand);
    lNetwork->InitializePersistentCommand(mNoReductionReqdCommand);
    lNetwork->InitializePersistentCommand(mSubtaskMemoryReduceCommand);

	pmCommunicator::GetCommunicator()->Receive(mRemoteSubtaskRecvCommand, false);
	pmCommunicator::GetCommunicator()->Receive(mTaskEventRecvCommand, false);
	pmCommunicator::GetCommunicator()->Receive(mStealRequestRecvCommand, false);
    pmCommunicator::GetCommunicator()->Receive(mStealResponseRecvCommand, false);
	pmCommunicator::GetCommunicator()->Receive(mSubtaskRangeCancelCommand, false);
    pmCommunicator::GetCommunicator()->Receive(mNoReductionReqdCommand, false);
    pmCommunicator::GetCommunicator()->Receive(mSubtaskMemoryReduceCommand, false);
	
	// Only MPI master host receives finalization signal
	if(pmMachinePool::GetMachinePool()->GetMachine(0) == PM_LOCAL_MACHINE)
	{
        finalize_ptr<communicator::hostFinalizationStruct> lHostFinalizationData(new hostFinalizationStruct());

		mHostFinalizationCommand = PERSISTENT_RECV_COMMAND(HOST_FINALIZATION_TAG, HOST_FINALIZATION_STRUCT, hostFinalizationStruct, lHostFinalizationData);
        
        mHostFinalizationCommand->SetPersistent();
        
        lNetwork->InitializePersistentCommand(mHostFinalizationCommand);
        
        pmCommunicator::GetCommunicator()->Receive(mHostFinalizationCommand, false);
	}
}

void pmScheduler::DestroyPersistentCommunicationCommands()
{
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();

    lNetwork->TerminatePersistentCommand(mRemoteSubtaskRecvCommand);
    lNetwork->TerminatePersistentCommand(mTaskEventRecvCommand);
    lNetwork->TerminatePersistentCommand(mStealRequestRecvCommand);
    lNetwork->TerminatePersistentCommand(mStealResponseRecvCommand);
    lNetwork->TerminatePersistentCommand(mSubtaskRangeCancelCommand);
    lNetwork->TerminatePersistentCommand(mNoReductionReqdCommand);
    lNetwork->TerminatePersistentCommand(mSubtaskMemoryReduceCommand);

	if(mHostFinalizationCommand.get())
        lNetwork->TerminatePersistentCommand(mHostFinalizationCommand);
}

void pmScheduler::SubmitTaskEvent(pmLocalTask* pLocalTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new taskSubmissionEvent(NEW_SUBMISSION, pLocalTask)), pLocalTask->GetPriority());
}

void pmScheduler::PushEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, bool pIsStealResponse)
{
#ifdef TRACK_SUBTASK_EXECUTION
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mSubtasksAssigned += pRange.endSubtask - pRange.startSubtask + 1;
    }
#endif

	SwitchThread(std::shared_ptr<schedulerEvent>(new subtaskExecEvent(SUBTASK_EXECUTION, pDevice, pRange, pIsStealResponse)), pRange.task->GetPriority());
}
    
#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::PushEvent(pmTask* pTask, const pmProcessingElement* pDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
#ifdef TRACK_SUBTASK_EXECUTION
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        
        auto lIter = pDiscontiguousStealData.begin(), lEndIter = pDiscontiguousStealData.end();
        for(; lIter != lEndIter; ++lIter)
        {
            ulong lFirstSubtask = *lIter;
            ++lIter;

            mSubtasksAssigned += ((*lIter - lFirstSubtask) + 1);
        }
    }
#endif

    SwitchThread(std::shared_ptr<schedulerEvent>(new subtaskExecDiscontiguousStealEvent(SUBTASK_EXECUTION_DISCONTIGUOUS_STEAL, pTask, pDevice, std::move(pDiscontiguousStealData))), pTask->GetPriority());
}
#endif

void pmScheduler::StealRequestEvent(const pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;

#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_WAIT, true);
#endif

	SwitchThread(std::shared_ptr<schedulerEvent>(new stealRequestEvent(STEAL_REQUEST_STEALER, pStealingDevice, pTask, pExecutionRate)), pTask->GetPriority());
}

void pmScheduler::StealProcessEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate, bool pShouldMultiAssign)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;
    
#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_SERVE, true);
#endif
    
	SwitchThread(std::shared_ptr<schedulerEvent>(new stealProcessEvent(STEAL_PROCESS_TARGET, pStealingDevice, pTargetDevice, pTask, pExecutionRate, pShouldMultiAssign)), pTask->GetPriority());
}

void pmScheduler::StealSuccessEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pRange.task))
        return;
    
	SwitchThread(std::shared_ptr<schedulerEvent>(new stealSuccessTargetEvent(STEAL_SUCCESS_TARGET, pStealingDevice, pTargetDevice, pRange)), pRange.task->GetPriority());
}

#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::StealSuccessEvent(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
    EXCEPTION_ASSERT(!pDiscontiguousStealData.empty() && (pDiscontiguousStealData.size() % 2 == 0));
    
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;
    
    SwitchThread(std::shared_ptr<schedulerEvent>(new stealSuccessDiscontiguousTargetEvent(STEAL_SUCCESS_DISCONTIGUOUS_TARGET, pTask, pStealingDevice, pTargetDevice, std::move(pDiscontiguousStealData))), pTask->GetPriority());
}
#endif

void pmScheduler::StealFailedEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;
    
	SwitchThread(std::shared_ptr<schedulerEvent>(new stealFailTargetEvent(STEAL_FAIL_TARGET, pStealingDevice, pTargetDevice, pTask)), pTask->GetPriority());
}

void pmScheduler::StealSuccessReturnEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pRange.task))
        return;
    
	SwitchThread(std::shared_ptr<schedulerEvent>(new stealSuccessStealerEvent(STEAL_SUCCESS_STEALER, pStealingDevice, pTargetDevice, pRange)), pRange.task->GetPriority());
}

#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::StealSuccessReturnEvent(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
    EXCEPTION_ASSERT(!pDiscontiguousStealData.empty() && (pDiscontiguousStealData.size() % 2 == 0));

    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;
    
    SwitchThread(std::shared_ptr<schedulerEvent>(new stealSuccessDiscontiguousStealerEvent(STEAL_SUCCESS_DISCONTIGUOUS_STEALER, pTask, pStealingDevice, pTargetDevice, std::move(pDiscontiguousStealData))), pTask->GetPriority());
}
#endif

void pmScheduler::StealFailedReturnEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return;
    
	SwitchThread(std::shared_ptr<schedulerEvent>(new stealFailStealerEvent(STEAL_FAIL_STEALER, pStealingDevice, pTargetDevice, pTask)), pTask->GetPriority());
}

void pmScheduler::AcknowledgementSendEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, size_t pTotalSplitCount)
{
#ifdef TRACK_SUBTASK_EXECUTION
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());

        mAcknowledgementsSent += (pRange.endSubtask - pRange.startSubtask + 1);
        std::cout << "[Host " << pmGetHostId() << "]: Device " << pDevice->GetGlobalDeviceIndex() << " sent " << (pRange.endSubtask - pRange.startSubtask + 1) << " acknowledgements for subtasks [" << pRange.startSubtask << " - " << pRange.endSubtask << "]" << std::endl;
    }
#endif

	SwitchThread(std::shared_ptr<schedulerEvent>(new sendAcknowledgementEvent(SEND_ACKNOWLEDGEMENT, pDevice, pRange, pExecStatus, std::move(pScatteredOwnershipVector), std::move(pAddressSpaceIndexVector), pTotalSplitCount)), pRange.task->GetPriority());
}

void pmScheduler::AcknowledgementSendEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, size_t pTotalSplitCount)
{
#ifdef TRACK_SUBTASK_EXECUTION
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());

        mAcknowledgementsSent += (pRange.endSubtask - pRange.startSubtask + 1);
        std::cout << "[Host " << pmGetHostId() << "]: Device " << pDevice->GetGlobalDeviceIndex() << " sent " << (pRange.endSubtask - pRange.startSubtask + 1) << " acknowledgements for subtasks [" << pRange.startSubtask << " - " << pRange.endSubtask << "]" << std::endl;
    }
#endif

	SwitchThread(std::shared_ptr<schedulerEvent>(new sendAcknowledgementEvent(SEND_ACKNOWLEDGEMENT, pDevice, pRange, pExecStatus, std::move(pOwnershipVector), std::move(pAddressSpaceIndexVector), pTotalSplitCount)), pRange.task->GetPriority());
}

void pmScheduler::AcknowledgementReceiveEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new receiveAcknowledgementEvent(RECEIVE_ACKNOWLEDGEMENT, pDevice, pRange, pExecStatus, std::move(pScatteredOwnershipVector), std::move(pAddressSpaceIndexVector))), pRange.task->GetPriority());
}

void pmScheduler::AcknowledgementReceiveEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new receiveAcknowledgementEvent(RECEIVE_ACKNOWLEDGEMENT, pDevice, pRange, pExecStatus, std::move(pOwnershipVector), std::move(pAddressSpaceIndexVector))), pRange.task->GetPriority());
}

void pmScheduler::TaskCancelEvent(pmTask* pTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new taskCancelEvent(TASK_CANCEL, pTask)), pTask->GetPriority());
}

void pmScheduler::TaskFinishEvent(pmTask* pTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new taskFinishEvent(TASK_FINISH, pTask)), pTask->GetPriority());
}

void pmScheduler::TaskCompleteEvent(pmLocalTask* pLocalTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new taskCompleteEvent(TASK_COMPLETE, pLocalTask)), pLocalTask->GetPriority());
}

void pmScheduler::ReduceRequestEvent(pmExecutionStub* pReducingStub, pmTask* pTask, const pmMachine* pDestMachine, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    pmSplitData lSplitData(pSplitInfo);

	SwitchThread(std::shared_ptr<schedulerEvent>(new subtaskReduceEvent(SUBTASK_REDUCE, pTask, pDestMachine, pReducingStub, pSubtaskId, lSplitData)), pTask->GetPriority());
}
    
void pmScheduler::NoReductionRequiredEvent(pmTask* pTask, const pmMachine* pDestMachine)
{
    SwitchThread(std::shared_ptr<schedulerEvent>(new noReductionRequiredEvent(NO_REDUCTION_REQD, pTask, pDestMachine)), pTask->GetPriority());
}

void pmScheduler::CommandCompletionEvent(const pmCommandPtr& pCommand)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new commandCompletionEvent(COMMAND_COMPLETION, pCommand)), pCommand->GetPriority());
}

void pmScheduler::RangeCancellationEvent(const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new subtaskRangeCancelEvent(SUBTASK_RANGE_CANCEL, pTargetDevice, pRange)), pRange.task->GetPriority());
}
    
void pmScheduler::RedistributionMetaDataEvent(pmTask* pTask, uint pAddressSpaceIndex, std::vector<redistributionOrderStruct>* pRedistributionData)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new redistributionMetaDataEvent(REDISTRIBUTION_METADATA_EVENT, pTask, pAddressSpaceIndex, pRedistributionData)), pTask->GetPriority());
}
    
void pmScheduler::RedistributionOffsetsEvent(pmTask* pTask, uint pAddressSpaceIndex, pmAddressSpace* pRedistributedAddressSpace, uint pDestHostId, std::vector<ulong>* pOffsetsData)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new redistributionOffsetsEvent(REDISTRIBUTION_OFFSETS_EVENT, pTask, pAddressSpaceIndex, pOffsetsData, pDestHostId, pRedistributedAddressSpace)), pTask->GetPriority());
}
    
void pmScheduler::RangeNegotiationEvent(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new rangeNegotiationEvent(RANGE_NEGOTIATION_EVENT, pRequestingDevice, pRange)), pRange.task->GetPriority());
}
    
void pmScheduler::RangeNegotiationSuccessEvent(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pNegotiatedRange)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new rangeNegotiationSuccessEvent(RANGE_NEGOTIATION_SUCCESS_EVENT, pRequestingDevice, pNegotiatedRange)), pNegotiatedRange.task->GetPriority());
}
    
void pmScheduler::TerminateTaskEvent(pmTask* pTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new taskTerminateEvent(TERMINATE_TASK, pTask)), pTask->GetPriority());
}
    
void pmScheduler::ReductionTerminationEvent(pmLocalTask* pLocalTask)
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new reductionTerminationEvent(REDUCTION_TERMINATION_EVENT, pLocalTask)), pLocalTask->GetPriority());
}

void pmScheduler::AffinityTransferEvent(pmLocalTask* pLocalTask, std::set<const pmMachine*>&& pMachines, const std::vector<ulong>* pLogicalToPhysicalSubtaskMapping)
{
    SwitchThread(std::shared_ptr<schedulerEvent>(new affinityTransferEvent(AFFINITY_TRANSFER_EVENT, pLocalTask, std::move(pMachines), pLogicalToPhysicalSubtaskMapping)), pLocalTask->GetPriority());
}
    
void pmScheduler::AllReductionsDoneEvent(pmLocalTask* pLocalTask, pmExecutionStub* pLastStub, ulong pLastSubtaskId, const pmSplitData& pLastSplitData)
{
    SwitchThread(std::shared_ptr<schedulerEvent>(new allReductionsDoneEvent(ALL_REDUCTIONS_DONE_EVENT, pLocalTask, pLastStub, pLastSubtaskId, pLastSplitData)), pLocalTask->GetPriority());
}
    
void pmScheduler::AddRegisterExternalReductionFinishEvent(pmTask* pTask)
{
    SwitchThread(std::shared_ptr<schedulerEvent>(new externalReductionFinishEvent(EXTERNAL_REDUCTION_FINISH_EVENT, pTask)), pTask->GetPriority());
}
    
void pmScheduler::SendFinalizationSignal()
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new hostFinalizationEvent(HOST_FINALIZATION, false)), MAX_CONTROL_PRIORITY);
}

void pmScheduler::BroadcastTerminationSignal()
{
	SwitchThread(std::shared_ptr<schedulerEvent>(new hostFinalizationEvent(HOST_FINALIZATION, true)), MAX_CONTROL_PRIORITY);
}

void pmScheduler::ThreadSwitchCallback(std::shared_ptr<schedulerEvent>& pEvent)
{
	try
	{
		ProcessEvent(*pEvent);
	}
	catch(pmException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from scheduler thread");
	}
}

void pmScheduler::ProcessEvent(schedulerEvent& pEvent)
{
#ifdef DUMP_SCHEDULER_EVENT
    char lStr[512];
    
    sprintf(lStr, "Host %d Scheduler Event: %s", pmGetHostId(), schedulerEventName[pEvent.eventId]);
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
#endif

	switch(pEvent.eventId)
	{
		case NEW_SUBMISSION:	/* Comes from application thread */
        {
            taskSubmissionEvent& lEvent = static_cast<taskSubmissionEvent&>(pEvent);
            StartLocalTaskExecution(lEvent.localTask);
            
            break;
        }

		case SUBTASK_EXECUTION:	/* Comes from network thread or from scheduler thread for local submissions */
        {
            subtaskExecEvent& lEvent = static_cast<subtaskExecEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEvent.range.task))
                PushToStub(lEvent.device, lEvent.range, lEvent.isStealResponse);
            
            break;
        }
            
    #ifdef USE_AFFINITY_IN_STEAL
        case SUBTASK_EXECUTION_DISCONTIGUOUS_STEAL:
        {
            subtaskExecDiscontiguousStealEvent& lEvent = static_cast<subtaskExecDiscontiguousStealEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEvent.task))
                PushToStub(lEvent.task, lEvent.device, std::move(lEvent.discontiguousStealData));
            
            break;
        }
    #endif

		case STEAL_REQUEST_STEALER:	/* Comes from stub thread */
        {
            stealRequestEvent& lEvent = static_cast<stealRequestEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEvent.task))
                StealSubtasks(lEvent.stealingDevice, lEvent.task, lEvent.stealingDeviceExecutionRate);
            
            break;
        }

		case STEAL_PROCESS_TARGET:	/* Comes from network thread */
        {
            stealProcessEvent& lEventDetails = static_cast<stealProcessEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                ServeStealRequest(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task, lEventDetails.stealingDeviceExecutionRate, lEventDetails.shouldMultiAssign);
            
            break;
        }

		case STEAL_SUCCESS_TARGET:	/* Comes from stub thread */
        {
            stealSuccessTargetEvent& lEventDetails = static_cast<stealSuccessTargetEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
                SendStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
            
        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.range.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_SERVE, false);
        #endif
            
            break;
        }

    #ifdef USE_AFFINITY_IN_STEAL
		case STEAL_SUCCESS_DISCONTIGUOUS_TARGET:	/* Comes from stub thread */
        {
            stealSuccessDiscontiguousTargetEvent& lEventDetails = static_cast<stealSuccessDiscontiguousTargetEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                SendStealResponse(lEventDetails.task, lEventDetails.stealingDevice, lEventDetails.targetDevice, std::move(lEventDetails.discontiguousStealData));
            
        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_SERVE, false);
        #endif
            
            break;
        }
    #endif

        case STEAL_FAIL_TARGET: /* Comes from stub thread */
        {
            stealFailTargetEvent& lEventDetails = static_cast<stealFailTargetEvent&>(pEvent);

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                SendFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
            
        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_SERVE, false);
        #endif
            
            break;
        }

		case STEAL_SUCCESS_STEALER: /* Comes from network thread */
        {
            stealSuccessStealerEvent& lEventDetails = static_cast<stealSuccessStealerEvent&>(pEvent);

        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.range.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_WAIT, false);
        #endif

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
                ReceiveStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
            
            break;
        }

    #ifdef USE_AFFINITY_IN_STEAL
		case STEAL_SUCCESS_DISCONTIGUOUS_STEALER: /* Comes from network thread */
        {
            stealSuccessDiscontiguousStealerEvent& lEventDetails = static_cast<stealSuccessDiscontiguousStealerEvent&>(pEvent);

        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_WAIT, false);
        #endif

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                ReceiveStealResponse(lEventDetails.task, lEventDetails.stealingDevice, lEventDetails.targetDevice, std::move(lEventDetails.discontiguousStealData));
            
            break;
        }
    #endif

        case STEAL_FAIL_STEALER: /* Comes from network thread */
        {
            stealFailStealerEvent& lEventDetails = static_cast<stealFailStealerEvent&>(pEvent);

        #ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(taskProfiler::SUBTASK_STEAL_WAIT, false);
        #endif

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                ReceiveFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
            
            break;
        }

		case SEND_ACKNOWLEDGEMENT:	/* Comes from stub thread */
        {
            sendAcknowledgementEvent& lEventDetails = static_cast<sendAcknowledgementEvent&>(pEvent);

            pmTask* lTask = lEventDetails.range.task;
        
            if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lTask))
                break;
        
            lTask->IncrementSubtasksExecuted(lEventDetails.range.endSubtask - lEventDetails.range.startSubtask + 1, lEventDetails.totalSplitCount);
            
            EXCEPTION_ASSERT(lEventDetails.ownershipVector.empty() || lEventDetails.scatteredOwnershipVector.empty());

            const pmMachine* lOriginatingHost = lTask->GetOriginatingHost();
            if(lOriginatingHost == PM_LOCAL_MACHINE)
            {
                if(lEventDetails.scatteredOwnershipVector.empty())
                    AcknowledgementReceiveEvent(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.ownershipVector), std::move(lEventDetails.addressSpaceIndexVector));
                else
                    AcknowledgementReceiveEvent(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.scatteredOwnershipVector), std::move(lEventDetails.addressSpaceIndexVector));

                return;
            }
            else
            {
                if(lEventDetails.scatteredOwnershipVector.empty())
                {
                    finalize_ptr<sendAcknowledgementPacked> lPackedData(new sendAcknowledgementPacked(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.ownershipVector), std::move(lEventDetails.addressSpaceIndexVector)));
                
                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<sendAcknowledgementPacked>::CreateSharedPtr(lTask->GetPriority(), SEND, SEND_ACKNOWLEDGEMENT_TAG, lOriginatingHost, SEND_ACKNOWLEDGEMENT_PACKED, lPackedData, 1);

                    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
                }
                else
                {
                    finalize_ptr<sendAcknowledgementScatteredPacked> lPackedData(new sendAcknowledgementScatteredPacked(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.scatteredOwnershipVector), std::move(lEventDetails.addressSpaceIndexVector)));
                
                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<sendAcknowledgementScatteredPacked>::CreateSharedPtr(lTask->GetPriority(), SEND, SEND_ACKNOWLEDGEMENT_SCATTERED_TAG, lOriginatingHost, SEND_ACKNOWLEDGEMENT_SCATTERED_PACKED, lPackedData, 1);

                    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
                }
            }

            break;
        }

		case RECEIVE_ACKNOWLEDGEMENT:
        {
            receiveAcknowledgementEvent& lEventDetails = static_cast<receiveAcknowledgementEvent&>(pEvent);

            EXCEPTION_ASSERT(lEventDetails.ownershipVector.empty() || lEventDetails.scatteredOwnershipVector.empty());

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
            {
                if(lEventDetails.scatteredOwnershipVector.empty())
                    ProcessAcknowledgement((pmLocalTask*)(lEventDetails.range.task), lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.ownershipVector), std::move(lEventDetails.addressSpaceIndexVector));
                else
                    ProcessAcknowledgement((pmLocalTask*)(lEventDetails.range.task), lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, std::move(lEventDetails.scatteredOwnershipVector), std::move(lEventDetails.addressSpaceIndexVector));
            }
            
            break;
        }

		case TASK_CANCEL:
        {
            taskCancelEvent& lEventDetails = static_cast<taskCancelEvent&>(pEvent);
            pmTask* lTask = lEventDetails.task;
            
            CancelAllSubtaskSplitDummyEventsOnLocalStubs(lTask);
            CancelAllSubtasksExecutingOnLocalStubs(lTask, false);
    
            break;
        }

		case TASK_FINISH:
        {
            taskFinishEvent& lEventDetails = static_cast<taskFinishEvent&>(pEvent);
            pmTask* lTask = lEventDetails.task;
        
            CommitShadowMemPendingOnAllStubs(lTask);
            lTask->MarkAllStubsScannedForShadowMemCommitMessages();

            lTask->MarkSubtaskExecutionFinished();
            pmTaskManager::GetTaskManager()->RegisterTaskFinish(*lTask->GetOriginatingHost(), lTask->GetSequenceNumber());

            ClearPendingTaskCommands(lTask);
            
            if(lTask->IsMultiAssignEnabled())
            {
                CancelAllSubtaskSplitDummyEventsOnLocalStubs(lTask);
                CancelAllSubtasksExecutingOnLocalStubs(lTask, true);
                lTask->MarkAllStubsScannedForCancellationMessages();
            }

            break;
        }

		case TASK_COMPLETE:
        {
            taskCompleteEvent& lEventDetails = static_cast<taskCompleteEvent&>(pEvent);
            pmLocalTask* lLocalTask = lEventDetails.localTask;

            lLocalTask->RegisterInternalTaskCompletionMessage();

            break;
        }

		case SUBTASK_REDUCE:
        {
            subtaskReduceEvent& lEventDetails = static_cast<subtaskReduceEvent&>(pEvent);
            
            std::unique_ptr<pmSplitInfo> lSplitInfoAutoPtr(lEventDetails.splitData.operator std::unique_ptr<pmSplitInfo>());
            
            pmSubscriptionManager& lSubscriptionManager = lEventDetails.task->GetSubscriptionManager();
            bool lHasScratchBuffers = lSubscriptionManager.HasScratchBuffers(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get());
            
            const pmAddressSpace* lAddressSpace = 0;
            size_t lReducibleAddressSpaces = 0;
            size_t lAddressSpaceIndex = 0;
            for_each_with_index(lEventDetails.task->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pIndex)
            {
                if(lEventDetails.task->IsWritable(pAddressSpace) && lEventDetails.task->IsReducible(pAddressSpace))
                {
                    lAddressSpace = pAddressSpace;
                    lAddressSpaceIndex = pIndex;
                    ++lReducibleAddressSpaces;
                }
            });

            bool lOptimalSendDone = false;

        #ifdef SUPPORT_LAZY_MEMORY
        #else
            // Check if reduction data can be transferred without mpi packing
            if(!lHasScratchBuffers && lReducibleAddressSpaces == 1 && NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->IsImplicitlyReducible(lEventDetails.task))
            {
                void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);
                ulong lOffset = 0, lLength = 0;
                
                if(lEventDetails.task->GetAddressSpaceSubscriptionVisibility(lAddressSpace, lEventDetails.reducingStub) == SUBSCRIPTION_NATURAL)
                {
                    pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);

                    subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                    lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
                    
                    EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

                    lOffset =  lBeginIter->first - lUnifiedSubscriptionInfo.offset;
                    lLength = (uint)lBeginIter->second.first;
                }
                else    // SUBSCRIPTION_COMPACT
                {
                    const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);

                    subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                    lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
                    
                    auto lCompactWriteIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
                    
                    EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

                    lOffset = *lCompactWriteIter;
                    lLength = (uint)lBeginIter->second.first;
                }
                
                pmReductionOpType lOpType;
                pmReductionDataType lDataType;
                
                findReductionOpAndDataType(lEventDetails.task->GetCallbackUnit()->GetDataReductionCB()->GetCallback(), lOpType, lDataType);

                char* lTargetMem = (static_cast<char*>(lShadowMem) + lOffset);

                ulong lCompressedLength = 0;
                std::shared_ptr<void> lCompressedMem;

                switch(lDataType)
                {
                    case REDUCE_INTS:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<int>((int*)lTargetMem, 0, lLength / sizeof(int), lCompressedLength);
                        break;
                    }
                        
                    case REDUCE_UNSIGNED_INTS:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<uint>((uint*)lTargetMem, 0, lLength / sizeof(uint), lCompressedLength);
                        break;
                    }
                        
                    case REDUCE_LONGS:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<long>((long*)lTargetMem, 0, lLength / sizeof(long), lCompressedLength);
                        break;
                    }
                        
                    case REDUCE_UNSIGNED_LONGS:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<ulong>((ulong*)lTargetMem, 0, lLength / sizeof(ulong), lCompressedLength);
                        break;
                    }
                        
                    case REDUCE_FLOATS:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<float>((float*)lTargetMem, 0, lLength / sizeof(float), lCompressedLength);
                        break;
                    }
                        
                    case REDUCE_DOUBLES:
                    {
                        lCompressedMem = pmUtility::CompressForSentinel<double>((double*)lTargetMem, 0, lLength / sizeof(double), lCompressedLength);
                        break;
                    }
                        
                    default:
                        PMTHROW(pmFatalErrorException());
                }

                finalize_ptr<subtaskMemoryReduceStruct> lData(new subtaskMemoryReduceStruct(*lEventDetails.task->GetOriginatingHost(), lEventDetails.task->GetSequenceNumber(), lEventDetails.subtaskId, lOffset, (lCompressedMem.get() ? lCompressedLength : lLength), std::numeric_limits<int>::max(), *PM_LOCAL_MACHINE, (lCompressedMem.get() != NULL)));
                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<subtaskMemoryReduceStruct>::CreateSharedPtr(lEventDetails.task->GetPriority(), SEND, SUBTASK_MEMORY_REDUCE_TAG, lEventDetails.machine, SUBTASK_MEMORY_REDUCE_STRUCT, lData, 1, NULL, (lCompressedMem.get() ? lCompressedMem.get() : lTargetMem));
                
                if(lCompressedMem.get())
                    lCommand->HoldExternalDataForLifetimeOfCommand(lCompressedMem);
                
                pmCommunicator::GetCommunicator()->SendReduce(lCommand, false);

                lOptimalSendDone = true;
            }
        #endif
            
            // Send reduction data as MPI_PACKED
            if(!lOptimalSendDone)
            {
                finalize_ptr<subtaskReducePacked> lPackedData(new subtaskReducePacked(lEventDetails.reducingStub, lEventDetails.task, lEventDetails.subtaskId, lSplitInfoAutoPtr.get()));
            
                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<subtaskReducePacked>::CreateSharedPtr(lEventDetails.task->GetPriority(), SEND, SUBTASK_REDUCE_TAG, lEventDetails.machine, SUBTASK_REDUCE_PACKED, lPackedData, 1, NULL, lEventDetails.task);

                pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
            }

            break;
        }
            
        case NO_REDUCTION_REQD:
        {
            noReductionRequiredEvent& lEventDetails = static_cast<noReductionRequiredEvent&>(pEvent);
            
            EXCEPTION_ASSERT(lEventDetails.machine != PM_LOCAL_MACHINE);
            
            finalize_ptr<noReductionReqdStruct> lData(new noReductionReqdStruct(lEventDetails.task));
            
            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<noReductionReqdStruct>::CreateSharedPtr(lEventDetails.task->GetPriority(), SEND, NO_REDUCTION_REQD_TAG, lEventDetails.machine, NO_REDUCTION_REQD_STRUCT, lData, 1);
            
            pmCommunicator::GetCommunicator()->Send(lCommand, false);
            
            break;
        }

		case COMMAND_COMPLETION:
        {
            commandCompletionEvent& lEventDetails = static_cast<commandCompletionEvent&>(pEvent);

            HandleCommandCompletion(lEventDetails.command);

            break;
        }
            
        case SUBTASK_RANGE_CANCEL:
        {
            subtaskRangeCancelEvent& lEventDetails = static_cast<subtaskRangeCancelEvent&>(pEvent);
            
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
            {
                pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetStub(lEventDetails.targetDevice);
                lStub->CancelSubtaskRange(lEventDetails.range);
            }
        
            break;
        }
        
        case REDISTRIBUTION_METADATA_EVENT:
        {
            redistributionMetaDataEvent& lEventDetails = static_cast<redistributionMetaDataEvent&>(pEvent);
            SendRedistributionData(lEventDetails.task, lEventDetails.addressSpaceIndex, lEventDetails.redistributionData);
            
            break;
        }
            
        case REDISTRIBUTION_OFFSETS_EVENT:
        {
            redistributionOffsetsEvent& lEventDetails = static_cast<redistributionOffsetsEvent&>(pEvent);
            SendRedistributionOffsets(lEventDetails.task, lEventDetails.addressSpaceIndex, lEventDetails.offsetsData, lEventDetails.redistributedAddressSpace, lEventDetails.destHostId);
            
            break;
        }

		case HOST_FINALIZATION:
        {
            hostFinalizationEvent& lEventDetails = static_cast<hostFinalizationEvent&>(pEvent);

            const pmMachine* lMasterHost = pmMachinePool::GetMachinePool()->GetMachine(0);

            if(lEventDetails.terminate)
            {
                // Only master host can broadcast the global termination signal
                if(lMasterHost != PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());

                finalize_ptr<hostFinalizationStruct> lBroadcastData(new hostFinalizationStruct(true));

                pmCommunicatorCommandPtr lBroadcastCommand = pmCommunicatorCommand<hostFinalizationStruct>::CreateSharedPtr(MAX_PRIORITY_LEVEL, BROADCAST, HOST_FINALIZATION_TAG, lMasterHost, HOST_FINALIZATION_STRUCT, lBroadcastData, 1, SchedulerCommandCompletionCallback);

                pmCommunicator::GetCommunicator()->Broadcast(lBroadcastCommand);
            }
            else
            {
                if(lMasterHost == PM_LOCAL_MACHINE)
                {
                    pmController::GetController()->ProcessFinalization();

                    return;
                }

                finalize_ptr<hostFinalizationStruct> lData(new hostFinalizationStruct(false));

                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<hostFinalizationStruct>::CreateSharedPtr(MAX_CONTROL_PRIORITY, SEND, HOST_FINALIZATION_TAG, lMasterHost, HOST_FINALIZATION_STRUCT, lData, 1);

                pmCommunicator::GetCommunicator()->Send(lCommand, false);
            }

            break;
        }
        
        case RANGE_NEGOTIATION_EVENT:
        {
            rangeNegotiationEvent& lEventDetails = static_cast<rangeNegotiationEvent&>(pEvent);
        
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
            {
                if(lEventDetails.range.originalAllottee == NULL || lEventDetails.range.originalAllottee->GetMachine() != PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());
            
                pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetStub(lEventDetails.range.originalAllottee);
                lStub->NegotiateRange(lEventDetails.requestingDevice, lEventDetails.range);
            }
        
            break;
        }

        case RANGE_NEGOTIATION_SUCCESS_EVENT:
        {
            rangeNegotiationSuccessEvent& lEventDetails = static_cast<rangeNegotiationSuccessEvent&>(pEvent);
        
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.negotiatedRange.task))
            {
                if(lEventDetails.negotiatedRange.originalAllottee == NULL || lEventDetails.requestingDevice->GetMachine() != PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());
            
                pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetStub(lEventDetails.requestingDevice);
                lStub->ProcessNegotiatedRange(lEventDetails.negotiatedRange);
            }
        
            break;
        }
        
        case TERMINATE_TASK:
        {
            taskTerminateEvent& lEventDetails = static_cast<taskTerminateEvent&>(pEvent);
            
            FreeTaskResourcesOnLocalStubs(lEventDetails.task);
        
            if(dynamic_cast<pmLocalTask*>(lEventDetails.task) != NULL)
            {
                delete lEventDetails.task;
                pmTaskManager::GetTaskManager()->DeleteTask(static_cast<pmLocalTask*>(lEventDetails.task));
            }
            else
            {
                delete lEventDetails.task;
                pmTaskManager::GetTaskManager()->DeleteTask(static_cast<pmRemoteTask*>(lEventDetails.task));
            }

            break;
        }
            
        case REDUCTION_TERMINATION_EVENT:
        {
            reductionTerminationEvent& lEventDetails = static_cast<reductionTerminationEvent&>(pEvent);
            SendReductionTerminationToMachines(lEventDetails.localTask);

            break;
        }
            
        case AFFINITY_TRANSFER_EVENT:
        {
            affinityTransferEvent& lEventDetails = static_cast<affinityTransferEvent&>(pEvent);
            SendAffinityDataToMachines(lEventDetails.localTask, lEventDetails.machines, *lEventDetails.logicalToPhysicalSubtaskMapping);
            
            break;
        }
            
        case ALL_REDUCTIONS_DONE_EVENT:
        {
            allReductionsDoneEvent& lEventDetails = static_cast<allReductionsDoneEvent&>(pEvent);

            std::unique_ptr<pmSplitInfo> lSplitInfoAutoPtr(lEventDetails.lastSplitData.operator std::unique_ptr<pmSplitInfo>());
            lEventDetails.localTask->AllReductionsDone(lEventDetails.lastStub, lEventDetails.lastSubtaskId, lSplitInfoAutoPtr.get());
            
            break;
        }
            
        case EXTERNAL_REDUCTION_FINISH_EVENT:
        {
            externalReductionFinishEvent& lEventDetails = static_cast<externalReductionFinishEvent&>(pEvent);
            
            lEventDetails.task->GetReducer()->RegisterExternalReductionFinish();
            
            break;
        }

        default:
            PMTHROW(pmFatalErrorException());
    }
}

void pmScheduler::FreeTaskResourcesOnLocalStubs(pmTask* pTask)
{
    pmStubManager* lManager = pmStubManager::GetStubManager();
    uint lStubCount = (uint)(lManager->GetStubCount());
    for(uint i = 0; i < lStubCount; ++i)
        lManager->GetStub(i)->FreeTaskResources(pTask);
}

void pmScheduler::CancelAllSubtasksExecutingOnLocalStubs(pmTask* pTask, bool pTaskListeningOnCancellation)
{
    pmStubManager* lManager = pmStubManager::GetStubManager();
    uint lStubCount = (uint)(lManager->GetStubCount());
    for(uint i = 0; i < lStubCount; ++i)
        lManager->GetStub(i)->CancelAllSubtasks(pTask, pTaskListeningOnCancellation);
}
    
void pmScheduler::CancelAllSubtaskSplitDummyEventsOnLocalStubs(pmTask* pTask)
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    pTask->GetSubtaskSplitter().FreezeDummyEvents();
#endif
}

void pmScheduler::CommitShadowMemPendingOnAllStubs(pmTask* pTask)
{
    pmStubManager* lManager = pmStubManager::GetStubManager();
    uint lStubCount = (uint)(lManager->GetStubCount());
    for(uint i = 0; i < lStubCount; ++i)
        lManager->GetStub(i)->ProcessDeferredShadowMemCommits(pTask);
}
    
void pmScheduler::NegotiateSubtaskRangeWithOriginalAllottee(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange)
{
    const pmProcessingElement* lOriginalAllottee = pRange.originalAllottee;
    
    DEBUG_EXCEPTION_ASSERT(lOriginalAllottee && lOriginalAllottee != pRequestingDevice);
    
	const pmMachine* lMachine = lOriginalAllottee->GetMachine();

	if(lMachine == PM_LOCAL_MACHINE)
	{
		RangeNegotiationEvent(pRequestingDevice, pRange);
	}
	else
	{
		finalize_ptr<remoteSubtaskAssignStruct> lSubtaskData(new remoteSubtaskAssignStruct(pRange.task->GetSequenceNumber(), pRange.startSubtask, pRange.endSubtask, *(pRange.task->GetOriginatingHost()), pRequestingDevice->GetGlobalDeviceIndex(), pRange.originalAllottee->GetGlobalDeviceIndex(), RANGE_NEGOTIATION));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<remoteSubtaskAssignStruct>::CreateSharedPtr(pRange.task->GetPriority(), SEND, REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, REMOTE_SUBTASK_ASSIGN_STRUCT, lSubtaskData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}
    
void pmScheduler::SendRangeNegotiationSuccess(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pNegotiatedRange)
{
    DEBUG_EXCEPTION_ASSERT(pNegotiatedRange.originalAllottee && pNegotiatedRange.originalAllottee != pRequestingDevice);
    
	const pmMachine* lMachine = pRequestingDevice->GetMachine();

	if(lMachine == PM_LOCAL_MACHINE)
	{
		RangeNegotiationSuccessEvent(pRequestingDevice, pNegotiatedRange);
	}
	else
	{
		finalize_ptr<remoteSubtaskAssignStruct> lSubtaskData(new remoteSubtaskAssignStruct(pNegotiatedRange.task->GetSequenceNumber(), pNegotiatedRange.startSubtask, pNegotiatedRange.endSubtask, *pNegotiatedRange.task->GetOriginatingHost(), pRequestingDevice->GetGlobalDeviceIndex(), pNegotiatedRange.originalAllottee->GetGlobalDeviceIndex(), SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<remoteSubtaskAssignStruct>::CreateSharedPtr(pNegotiatedRange.task->GetPriority(), SEND, REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, REMOTE_SUBTASK_ASSIGN_STRUCT, lSubtaskData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}
    
void pmScheduler::SendPostTaskOwnershipTransfer(pmAddressSpace* pAddressSpace, const pmMachine* pReceiverHost, std::shared_ptr<std::vector<ownershipChangeStruct> >& pChangeData)
{
    EXCEPTION_ASSERT(pReceiverHost != PM_LOCAL_MACHINE);
    
    finalize_ptr<ownershipTransferPacked> lPackedData(new ownershipTransferPacked(pAddressSpace, pChangeData));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<ownershipTransferPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, SEND, OWNERSHIP_TRANSFER_TAG, pReceiverHost, OWNERSHIP_TRANSFER_PACKED, lPackedData, 1);

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
}

void pmScheduler::SendPostTaskOwnershipTransfer(pmAddressSpace* pAddressSpace, const pmMachine* pReceiverHost, std::shared_ptr<std::vector<scatteredOwnershipChangeStruct> >& pChangeData)
{
    EXCEPTION_ASSERT(pReceiverHost != PM_LOCAL_MACHINE);
    
    finalize_ptr<scatteredOwnershipTransferPacked> lPackedData(new scatteredOwnershipTransferPacked(pAddressSpace, pChangeData));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<scatteredOwnershipTransferPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, SEND, SCATTERED_OWNERSHIP_TRANSFER_TAG, pReceiverHost, SCATTERED_OWNERSHIP_TRANSFER_PACKED, lPackedData, 1);

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
}

void pmScheduler::SendSubtaskRangeCancellationMessage(const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
 	const pmMachine* lMachine = pTargetDevice->GetMachine();
    if(lMachine == PM_LOCAL_MACHINE)
    {
        RangeCancellationEvent(pTargetDevice, pRange);
    }
    else
    {
		finalize_ptr<subtaskRangeCancelStruct> lRangeCancellationData(new subtaskRangeCancelStruct(pTargetDevice->GetGlobalDeviceIndex(), *pRange.task->GetOriginatingHost(), pRange.task->GetSequenceNumber(), pRange.startSubtask, pRange.endSubtask, (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pTargetDevice->GetGlobalDeviceIndex())));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<subtaskRangeCancelStruct>::CreateSharedPtr(pRange.task->GetPriority(), SEND, SUBTASK_RANGE_CANCEL_TAG, lMachine, SUBTASK_RANGE_CANCEL_STRUCT, lRangeCancellationData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
    }
}
    
void pmScheduler::SendRedistributionData(pmTask* pTask, uint pAddressSpaceIndex, std::vector<redistributionOrderStruct>* pRedistributionData)
{
    const pmMachine* lMachine = pTask->GetOriginatingHost();
    if(lMachine == PM_LOCAL_MACHINE)
    {
        pTask->GetRedistributor(pTask->GetAddressSpace(pAddressSpaceIndex))->PerformRedistribution(lMachine, pTask->GetSubtasksExecuted(), *pRedistributionData);
    }
    else
    {
        if((*pRedistributionData).empty())
        {
            (static_cast<pmRemoteTask*>(pTask))->MarkRedistributionFinished(pAddressSpaceIndex);
        }
        else
        {
            finalize_ptr<std::vector<redistributionOrderStruct>> lAutoPtr(pRedistributionData, false);

            finalize_ptr<dataRedistributionPacked> lPackedData(new dataRedistributionPacked(pTask, pAddressSpaceIndex, lAutoPtr));
            
            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<dataRedistributionPacked>::CreateSharedPtr(pTask->GetPriority(), SEND, DATA_REDISTRIBUTION_TAG, lMachine, DATA_REDISTRIBUTION_PACKED, lPackedData, 1);

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
        }
    }
}

void pmScheduler::SendRedistributionOffsets(pmTask* pTask, uint pAddressSpaceIndex, std::vector<ulong>* pOffsetsData, pmAddressSpace* pRedistributedAddressSpace, uint pDestHostId)
{
    DEBUG_EXCEPTION_ASSERT(pOffsetsData && !(*pOffsetsData).empty());

    const pmMachine* lMachine = pmMachinePool::GetMachinePool()->GetMachine(pDestHostId);

    if(lMachine == PM_LOCAL_MACHINE)
    {
        pTask->GetRedistributor(pTask->GetAddressSpace(pAddressSpaceIndex))->ReceiveGlobalOffsets(*pOffsetsData, pRedistributedAddressSpace->GetGenerationNumber());
    }
    else
    {
        finalize_ptr<std::vector<ulong>> lAutoPtr(pOffsetsData, false);

        finalize_ptr<redistributionOffsetsPacked> lPackedData(new redistributionOffsetsPacked(pTask, pAddressSpaceIndex, lAutoPtr, pRedistributedAddressSpace));
        
        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<redistributionOffsetsPacked>::CreateSharedPtr(pTask->GetPriority(), SEND, REDISTRIBUTION_OFFSETS_TAG, lMachine, REDISTRIBUTION_OFFSETS_PACKED, lPackedData, 1);

        pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
    }
}

void pmScheduler::AssignSubtasksToDevice(const pmProcessingElement* pDevice, pmLocalTask* pLocalTask)
{
	const pmMachine* lMachine = pDevice->GetMachine();

	ulong lStartingSubtask, lSubtaskCount;
    const pmProcessingElement* lOriginalAllottee = NULL;
	pLocalTask->GetSubtaskManager()->AssignSubtasksToDevice(pDevice, lSubtaskCount, lStartingSubtask, lOriginalAllottee);
    
	if(lSubtaskCount == 0)
		return;

#ifdef TRACK_SUBTASK_EXECUTION
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        std::cout << "Device " << pDevice->GetGlobalDeviceIndex() << " got " << lSubtaskCount << " subtasks [" << lStartingSubtask << " - " << (lStartingSubtask + lSubtaskCount - 1) << "]" << std::endl;
    }
#endif

	if(lMachine == PM_LOCAL_MACHINE)
	{
		pmSubtaskRange lRange(pLocalTask, lOriginalAllottee, lStartingSubtask, lStartingSubtask + lSubtaskCount - 1);
		PushEvent(pDevice, lRange, false);
	}
	else
	{
		finalize_ptr<remoteSubtaskAssignStruct> lSubtaskData(new remoteSubtaskAssignStruct(pLocalTask->GetSequenceNumber(), lStartingSubtask, lStartingSubtask + lSubtaskCount - 1, *(pLocalTask->GetOriginatingHost()), pDevice->GetGlobalDeviceIndex(), (lOriginalAllottee ? lOriginalAllottee->GetGlobalDeviceIndex() : pDevice->GetGlobalDeviceIndex()), SUBTASK_ASSIGNMENT_REGULAR));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<remoteSubtaskAssignStruct>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, REMOTE_SUBTASK_ASSIGN_STRUCT, lSubtaskData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}

void pmScheduler::AssignSubtasksToDevices(pmLocalTask* pLocalTask)
{
	const std::vector<const pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();

    for_each(lDevices, [&] (const pmProcessingElement* pDevice)
             {
                 AssignSubtasksToDevice(pDevice, pLocalTask);
             });
}

void pmScheduler::SendAffinityDataToMachines(pmLocalTask* pLocalTask, const std::set<const pmMachine*>& pMachines, const std::vector<ulong>& pLogicalToPhysicalSubtaskMappings)
{
    ulong* lData = const_cast<ulong*>(&pLogicalToPhysicalSubtaskMappings[0]);

    for_each(pMachines, [&] (const pmMachine* pMachine)
    {
		if(pMachine != PM_LOCAL_MACHINE)
		{
            pmAddressSpace* lAffinityAddressSpace = pLocalTask->GetAffinityAddressSpace();

            finalize_ptr<ulong, deleteArrayDeallocator<ulong>> lDataPtr(lData, false);
            finalize_ptr<affinityDataTransferPacked> lPackedData(new affinityDataTransferPacked((uint)(*(pLocalTask->GetOriginatingHost())), pLocalTask->GetSequenceNumber(), *lAffinityAddressSpace->GetMemOwnerHost(), lAffinityAddressSpace->GetGenerationNumber(), lAffinityAddressSpace->GetLength(), (uint)pLogicalToPhysicalSubtaskMappings.size(), std::move(lDataPtr)));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<affinityDataTransferPacked>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, AFFINITY_DATA_TRANSFER_TAG, pMachine, AFFINITY_DATA_TRANSFER_PACKED, lPackedData, 1);

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
		}
    });
}

void pmScheduler::AssignTaskToMachines(pmLocalTask* pLocalTask, std::set<const pmMachine*>& pMachines)
{
	std::set<const pmMachine*>::const_iterator lIter;
	for(lIter = pMachines.begin(); lIter != pMachines.end(); ++lIter)
	{
		const pmMachine* lMachine = *lIter;

		if(lMachine != PM_LOCAL_MACHINE)
		{
			finalize_ptr<remoteTaskAssignPacked> lPackedData(new remoteTaskAssignPacked(pLocalTask));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<remoteTaskAssignPacked>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, REMOTE_TASK_ASSIGNMENT_TAG, lMachine, REMOTE_TASK_ASSIGN_PACKED, lPackedData, 1);

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
		}
	}
}

void pmScheduler::SendTaskFinishToMachines(pmLocalTask* pLocalTask)
{
    std::set<const pmMachine*> lMachines = pLocalTask->GetAssignedMachines();

	std::set<const pmMachine*>::iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		const pmMachine* lMachine = *lIter;

		if(lMachine != PM_LOCAL_MACHINE)
		{
			finalize_ptr<taskEventStruct> lTaskEventData(new taskEventStruct(TASK_FINISH_EVENT, *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber()));

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<taskEventStruct>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, TASK_EVENT_TAG, lMachine, TASK_EVENT_STRUCT, lTaskEventData, 1);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}

    // Task master host must always be sent task finish even if none of it's devices were used in execution
    TaskFinishEvent(pLocalTask);
}

void pmScheduler::SendReductionTerminationToMachines(pmLocalTask* pLocalTask)
{
    std::set<const pmMachine*> lMachines = pLocalTask->GetAssignedMachines();

    lMachines.erase(PM_LOCAL_MACHINE);  // Master host is not to be sent the reduction termination event
    
	std::set<const pmMachine*>::iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		const pmMachine* lMachine = *lIter;

        finalize_ptr<taskEventStruct> lTaskEventData(new taskEventStruct(REDUCTION_TERMINATE_EVENT, *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber()));

        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<taskEventStruct>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, TASK_EVENT_TAG, lMachine, TASK_EVENT_STRUCT, lTaskEventData, 1);

        pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}
    
void pmScheduler::SendTaskCompleteToTaskOwner(pmTask* pTask)
{
    const pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

    if(lOriginatingHost == PM_LOCAL_MACHINE)
    {
        TaskCompleteEvent((pmLocalTask*)pTask);
    }
    else
    {
        finalize_ptr<taskEventStruct> lTaskEventData(new taskEventStruct(TASK_COMPLETE_EVENT, *pTask->GetOriginatingHost(), pTask->GetSequenceNumber()));

        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<taskEventStruct>::CreateSharedPtr(pTask->GetPriority(), SEND, TASK_EVENT_TAG, lOriginatingHost, TASK_EVENT_STRUCT, lTaskEventData, 1);

        pmCommunicator::GetCommunicator()->Send(lCommand, false);
    }
}

void pmScheduler::CancelTask(pmLocalTask* pLocalTask)
{
    const std::set<const pmMachine*>& lMachines = pLocalTask->GetAssignedMachines();

	std::set<const pmMachine*>::const_iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		const pmMachine* lMachine = *lIter;

		if(lMachine == PM_LOCAL_MACHINE)
		{
			TaskCancelEvent(pLocalTask);
		}
		else
		{
			// Send task cancel message to remote machines
			finalize_ptr<taskEventStruct> lTaskEventData(new taskEventStruct(TASK_CANCEL_EVENT, *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber()));

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<taskEventStruct>::CreateSharedPtr(pLocalTask->GetPriority(), SEND, TASK_EVENT_TAG, lMachine, TASK_EVENT_STRUCT, lTaskEventData, 1);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}
}

pmStatus pmScheduler::StartLocalTaskExecution(pmLocalTask* pLocalTask)
{
    ulong lTriggerTime = pLocalTask->GetTaskTimeOutTriggerTime();
    ulong lCurrentTime = GetIntegralCurrentTimeInSecs();

    if(lCurrentTime >= lTriggerTime)
        PMTHROW(pmFatalErrorException());   // Throw task TIMEDOUT from here
    
    pmTimedEventManager::GetTimedEventManager()->AddTaskTimeOutEvent(pLocalTask, lTriggerTime);
    
	const std::vector<const pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();
    
    if(lDevices.empty())
    {
        pLocalTask->MarkUserSideTaskCompletion();
        return pmNoCompatibleDevice;
    }

	std::set<const pmMachine*> lMachines;
    pmProcessingElement::GetMachines(lDevices, lMachines);

    /* PENDING --- All machines where the memories associated with the task are currently residing
     must also be sent in task definition */
	AssignTaskToMachines(pLocalTask, lMachines);

    if(pLocalTask->GetSchedulingModel() == scheduler::PULL_WITH_AFFINITY && pLocalTask->GetCallbackUnit()->GetDataDistributionCB() && !pLocalTask->HasReadOnlyLazyAddressSpace())
        pmPreprocessorTask::GetPreprocessorTask()->DeduceAffinity(pLocalTask, pLocalTask->GetAffinityCriterion());
    else
        pLocalTask->StartScheduling();

	return pmSuccess;
}

void pmScheduler::PushToStub(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, bool pIsStealResponse)
{
    pDevice->GetLocalExecutionStub()->Push(pRange, pIsStealResponse);
}

#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::PushToStub(pmTask* pTask, const pmProcessingElement* pDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
    pDevice->GetLocalExecutionStub()->Push(pTask, std::move(pDiscontiguousStealData));
}
#endif
    
#ifdef ENABLE_TWO_LEVEL_STEALING
const pmMachine* pmScheduler::RandomlySelectStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool& pShouldMultiAssign)
#else
const pmProcessingElement* pmScheduler::RandomlySelectStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool& pShouldMultiAssign)
#endif
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

	EXCEPTION_ASSERT(lStub);

	pmTaskExecStats& lTaskExecStats = pTask->GetTaskExecStats();

	uint lAttempts = lTaskExecStats.GetStealAttempts(lStub);
    
    const auto& lStealList = pTask->GetStealListForDevice(pStealingDevice);
    size_t lTargets = lStealList.size();
    
    uint lConsecutiveFailures = lTaskExecStats.GetFailedStealAttemptsSinceLastSuccessfulAttempt(lStub);
	if((lAttempts >= lTargets * MAX_STEAL_CYCLES_PER_DEVICE) || (lConsecutiveFailures >= lTargets))
		return NULL;

    // This device will only process a multi-assigned subtask in case it has not got anything in
    // consecutive lTargets/2 requests or if this device is too agressive like GPUs
    pShouldMultiAssign = ((lConsecutiveFailures >= lTargets/2) || (pStealingDevice->GetType() != CPU));

	lTaskExecStats.RecordStealAttempt(lStub);

#ifdef ENABLE_TWO_LEVEL_STEALING
#ifdef USE_STEAL_AGENT_PER_NODE
    if(pTask->GetStealAgent()->HasAnotherStubToStealFrom(pStealingDevice->GetLocalExecutionStub(), pShouldMultiAssign))
        return PM_LOCAL_MACHINE;
#endif
#endif

    return lStealList[lAttempts % lTargets];
}

void pmScheduler::StealSubtasks(const pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
    bool lShouldMultiAssign = true;

#ifdef ENABLE_TWO_LEVEL_STEALING
	const pmMachine* lTargetMachine = RandomlySelectStealTarget(pStealingDevice, pTask, lShouldMultiAssign);

    if(lTargetMachine)
    {
        STEAL_REQUEST_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*lTargetMachine), pStealingDevice->GetGlobalDeviceIndex(), std::numeric_limits<uint>::max(), pExecutionRate);

		if(lTargetMachine == PM_LOCAL_MACHINE)
		{
            StealProcessEvent(pStealingDevice, NULL, pTask, pExecutionRate, lShouldMultiAssign);
		}
		else
		{
			const pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

			finalize_ptr<stealRequestStruct> lStealRequestData(new stealRequestStruct(pStealingDevice->GetGlobalDeviceIndex(), *lOriginatingHost, pTask->GetSequenceNumber(), pExecutionRate, lShouldMultiAssign));

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<stealRequestStruct>::CreateSharedPtr(pTask->GetPriority(), SEND, STEAL_REQUEST_TAG, lTargetMachine, STEAL_REQUEST_STRUCT, lStealRequestData, 1);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
    }
#else
	const pmProcessingElement* lTargetDevice = RandomlySelectStealTarget(pStealingDevice, pTask, lShouldMultiAssign);
    if(lTargetDevice == pStealingDevice)
        lTargetDevice = RandomlySelectStealTarget(pStealingDevice, pTask, lShouldMultiAssign);

	if(lTargetDevice)
	{
        STEAL_REQUEST_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(lTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), lTargetDevice->GetGlobalDeviceIndex(), pExecutionRate);
    
		const pmMachine* lTargetMachine = lTargetDevice->GetMachine();

		if(lTargetMachine == PM_LOCAL_MACHINE)
		{
            if(lTargetDevice == pStealingDevice)
                StealFailedReturnEvent(pStealingDevice, lTargetDevice, pTask);
            else
                StealProcessEvent(pStealingDevice, lTargetDevice, pTask, pExecutionRate, lShouldMultiAssign);
		}
		else
		{
			const pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

			finalize_ptr<stealRequestStruct> lStealRequestData(new stealRequestStruct(pStealingDevice->GetGlobalDeviceIndex(), lTargetDevice->GetGlobalDeviceIndex(), *lOriginatingHost, pTask->GetSequenceNumber(), pExecutionRate, lShouldMultiAssign));

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<stealRequestStruct>::CreateSharedPtr(pTask->GetPriority(), SEND, STEAL_REQUEST_TAG, lTargetMachine, STEAL_REQUEST_STRUCT, lStealRequestData, 1);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}
#endif
}

void pmScheduler::ServeStealRequest(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate, bool pShouldMultiAssign)
{
#ifdef ENABLE_TWO_LEVEL_STEALING
    EXCEPTION_ASSERT(!pTargetDevice);

    pTargetDevice = RandomlySelectSecondLevelStealTarget(pStealingDevice, pTask, pShouldMultiAssign);
#endif

    if(pTargetDevice)
        pTargetDevice->GetLocalExecutionStub()->StealSubtasks(pTask, pStealingDevice, pExecutionRate, pShouldMultiAssign);
}

void pmScheduler::SendStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
    STEAL_RESPONSE_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(pTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), pRange.task->GetTaskExecStats().GetStubExecutionRate(pmStubManager::GetStubManager()->GetStub(pTargetDevice)), pRange.endSubtask - pRange.startSubtask + 1);

	const pmMachine* lMachine = pStealingDevice->GetMachine();
	if(lMachine == PM_LOCAL_MACHINE)
	{
		StealSuccessReturnEvent(pStealingDevice, pTargetDevice, pRange);
	}
	else
	{
		pmTask* lTask = pRange.task;
		const pmMachine* lOriginatingHost = lTask->GetOriginatingHost();

		finalize_ptr<stealResponseStruct> lStealResponseData(new stealResponseStruct(pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), *lOriginatingHost, lTask->GetSequenceNumber(), STEAL_SUCCESS_RESPONSE, pRange.startSubtask, pRange.endSubtask, (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pStealingDevice->GetGlobalDeviceIndex())));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<stealResponseStruct>::CreateSharedPtr(lTask->GetPriority(), SEND, STEAL_RESPONSE_TAG, lMachine, STEAL_RESPONSE_STRUCT, lStealResponseData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}

#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::SendStealResponse(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
#ifdef TRACK_SUBTASK_STEALS
//    for_each(pRangeVector, [&] (const pmSubtaskRange& range)
//    {
//        STEAL_RESPONSE_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(pTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), pRange.task->GetTaskExecStats().GetStubExecutionRate(pmStubManager::GetStubManager()->GetStub(pTargetDevice)), pRange.endSubtask - pRange.startSubtask + 1);
//    });
#endif

	const pmMachine* lMachine = pStealingDevice->GetMachine();
	if(lMachine == PM_LOCAL_MACHINE)
	{
        StealSuccessReturnEvent(pTask, pStealingDevice, pTargetDevice, std::move(pDiscontiguousStealData));
	}
	else
	{
		const pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

        finalize_ptr<stealSuccessDiscontiguousPacked> lPackedData(new stealSuccessDiscontiguousPacked(pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), *lOriginatingHost, pTask->GetSequenceNumber(), (uint)pDiscontiguousStealData.size(), std::move(pDiscontiguousStealData)));

        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<stealSuccessDiscontiguousPacked>::CreateSharedPtr(pTask->GetPriority(), SEND, STEAL_SUCCESS_DISCONTIGUOUS_TAG, lMachine, STEAL_SUCCESS_DISCONTIGUOUS_PACKED, lPackedData, 1);

        pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
	}
}
#endif
    
void pmScheduler::ReceiveStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
{
	pmTaskExecStats& lTaskExecStats = pRange.task->GetTaskExecStats();
	lTaskExecStats.RecordSuccessfulStealAttempt(pmStubManager::GetStubManager()->GetStub(pStealingDevice));

	PushEvent(pStealingDevice, pRange, true);
}

#ifdef USE_AFFINITY_IN_STEAL
void pmScheduler::ReceiveStealResponse(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
{
	pmTaskExecStats& lTaskExecStats = pTask->GetTaskExecStats();
	lTaskExecStats.RecordSuccessfulStealAttempt(pmStubManager::GetStubManager()->GetStub(pStealingDevice));
    
    PushEvent(pTask, pStealingDevice, std::move(pDiscontiguousStealData));
}
#endif

void pmScheduler::SendFailedStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    STEAL_RESPONSE_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(pTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), pTask->GetTaskExecStats().GetStubExecutionRate(pmStubManager::GetStubManager()->GetStub(pTargetDevice)), 0);

	const pmMachine* lMachine = pStealingDevice->GetMachine();
	if(lMachine == PM_LOCAL_MACHINE)
	{
		StealFailedReturnEvent(pStealingDevice, pTargetDevice, pTask);
	}
	else
	{
		const pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

		finalize_ptr<stealResponseStruct> lStealResponseData(new stealResponseStruct(pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), *lOriginatingHost, pTask->GetSequenceNumber(), STEAL_FAILURE_RESPONSE, 0, 0, 0));

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<stealResponseStruct>::CreateSharedPtr(pTask->GetPriority(), SEND, STEAL_RESPONSE_TAG, lMachine, STEAL_RESPONSE_STRUCT, lStealResponseData, 1);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
}

void pmScheduler::ReceiveFailedStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

    DEBUG_EXCEPTION_ASSERT(lStub);

	pmTaskExecStats& lTaskExecStats = pTask->GetTaskExecStats();
	lTaskExecStats.RecordFailedStealAttempt(pmStubManager::GetStubManager()->GetStub(pStealingDevice));

    StealRequestEvent(pStealingDevice, pTask, lTaskExecStats.GetStubExecutionRate(lStub));
}

void pmScheduler::RegisterPostTaskCompletionOwnershipTransfers(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, const std::vector<ownershipDataStruct>& pOwnershipVector, const std::vector<uint>& pAddressSpaceIndexVector)
{
    if(pOwnershipVector.empty())
        return;

    filtered_for_each_with_index(pRange.task->GetAddressSpaces(), [&pRange] (const pmAddressSpace* pAddressSpace) {return pRange.task->IsWritable(pAddressSpace);},
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
    {
        memoryIdentifierStruct lMemoryStruct(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber());
        
        std::vector<ownershipDataStruct>::const_iterator lDataIter = pOwnershipVector.begin() + pAddressSpaceIndexVector[pOutputAddressSpaceIndex];
        std::vector<ownershipDataStruct>::const_iterator lDataEndIter = pOwnershipVector.end();
        
        if(pOutputAddressSpaceIndex != pAddressSpaceIndexVector.size() - 1)
        {
            lDataEndIter = pOwnershipVector.begin() + pAddressSpaceIndexVector[pOutputAddressSpaceIndex + 1];
            --lDataEndIter;
        }

        const pmMachine* lMachine = pDevice->GetMachine();

        std::for_each(lDataIter, lDataEndIter, [&] (const ownershipDataStruct& pStruct)
        {
            pAddressSpace->TransferOwnershipPostTaskCompletion(vmRangeOwner(lMachine, pStruct.offset, lMemoryStruct), pStruct.offset, pStruct.length);
        });
    });
}

void pmScheduler::RegisterPostTaskCompletionOwnershipTransfers(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, const std::vector<scatteredOwnershipDataStruct>& pScatteredOwnershipVector, const std::vector<uint>& pAddressSpaceIndexVector)
{
    if(pScatteredOwnershipVector.empty())
        return;

    filtered_for_each_with_index(pRange.task->GetAddressSpaces(), [&pRange] (const pmAddressSpace* pAddressSpace) {return pRange.task->IsWritable(pAddressSpace);},
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
    {
        memoryIdentifierStruct lMemoryStruct(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber());
        
        std::vector<scatteredOwnershipDataStruct>::const_iterator lDataIter = pScatteredOwnershipVector.begin() + pAddressSpaceIndexVector[pOutputAddressSpaceIndex];
        std::vector<scatteredOwnershipDataStruct>::const_iterator lDataEndIter = pScatteredOwnershipVector.end();
        
        if(pOutputAddressSpaceIndex != pAddressSpaceIndexVector.size() - 1)
        {
            lDataEndIter = pScatteredOwnershipVector.begin() + pAddressSpaceIndexVector[pOutputAddressSpaceIndex + 1];
            --lDataEndIter;
        }

        const pmMachine* lMachine = pDevice->GetMachine();

        std::for_each(lDataIter, lDataEndIter, [&] (const scatteredOwnershipDataStruct& pStruct)
        {
            pAddressSpace->TransferOwnershipPostTaskCompletion(vmRangeOwner(lMachine, pStruct.offset, lMemoryStruct), pStruct.offset, pStruct.size, pStruct.step, pStruct.count);
        });
    });
}

// This method is executed at master host for the task
void pmScheduler::ProcessAcknowledgement(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector)
{
    RegisterPostTaskCompletionOwnershipTransfers(pDevice, pRange, pOwnershipVector, pAddressSpaceIndexVector);

    ProcessAcknowledgementCommon(pLocalTask, pDevice, pRange, pExecStatus);
}

void pmScheduler::ProcessAcknowledgement(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector)
{
    RegisterPostTaskCompletionOwnershipTransfers(pDevice, pRange, pScatteredOwnershipVector, pAddressSpaceIndexVector);
    
    ProcessAcknowledgementCommon(pLocalTask, pDevice, pRange, pExecStatus);
}
    
void pmScheduler::ProcessAcknowledgementCommon(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus)
{
    pmSubtaskManager* lSubtaskManager = pLocalTask->GetSubtaskManager();
    lSubtaskManager->RegisterSubtaskCompletion(pDevice, pRange.endSubtask - pRange.startSubtask + 1, pRange.startSubtask, pExecStatus);
    
    if(lSubtaskManager->HasTaskFinished())
    {
        SendTaskFinishToMachines(pLocalTask);
    }
    else
    {
        if(pLocalTask->GetSchedulingModel() == PUSH)
            AssignSubtasksToDevice(pDevice, pLocalTask);
    }
}

    
void pmScheduler::SendAcknowledgement(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount)
{
    // If task owner is not same as address space owner, then there is a problem.
    // We need to send ownership updates to address space owner and not task owner.
    // This works currently because an address space handle is only available on the
    // host where it has been created (i.e. address space owner) and task can only be
    // created with address space handle. This forces task owner to be same as address
    // space owner.

    if(pRange.task->GetOriginatingHost() != PM_LOCAL_MACHINE)
        RegisterPostTaskCompletionOwnershipTransfers(pDevice, pRange, pScatteredOwnershipVector, pAddressSpaceIndexVector);

	AcknowledgementSendEvent(pDevice, pRange, pExecStatus, std::move(pScatteredOwnershipVector), std::move(pAddressSpaceIndexVector), pTotalSplitCount);
}

void pmScheduler::SendAcknowledgement(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount)
{
    // If task owner is not same as address space owner, then there is a problem.
    // We need to send ownership updates to address space owner and not task owner.
    // This works currently because an address space handle is only available on the
    // host where it has been created (i.e. address space owner) and task can only be
    // created with address space handle. This forces task owner to be same as address
    // space owner.

    if(pRange.task->GetOriginatingHost() != PM_LOCAL_MACHINE)
        RegisterPostTaskCompletionOwnershipTransfers(pDevice, pRange, pOwnershipVector, pAddressSpaceIndexVector);

	AcknowledgementSendEvent(pDevice, pRange, pExecStatus, std::move(pOwnershipVector), std::move(pAddressSpaceIndexVector), pTotalSplitCount);
}

#ifdef ENABLE_TWO_LEVEL_STEALING
const pmProcessingElement* pmScheduler::RandomlySelectSecondLevelStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool pShouldMultiAssign)
{
#ifdef USE_STEAL_AGENT_PER_NODE
    pmExecutionStub* lRequestingStub = ((pStealingDevice->GetMachine() == PM_LOCAL_MACHINE) ? pStealingDevice->GetLocalExecutionStub() : NULL);
    pmExecutionStub* lStub = pTask->GetStealAgent()->GetStubWithMaxStealLikelihood(pShouldMultiAssign, lRequestingStub);

    if(!lStub)
    {
        pmScheduler::GetScheduler()->StealFailedEvent(pStealingDevice, pStealingDevice, pTask);
        return NULL;
    }
    
    return lStub->GetProcessingElement();
#else
    const std::vector<const pmProcessingElement*>& lDevices = (dynamic_cast<pmLocalTask*>(pTask) != NULL) ? (((pmLocalTask*)pTask)->GetAssignedDevices()) : (((pmRemoteTask*)pTask)->GetAssignedDevices());

    std::vector<const pmProcessingElement*> lLocalDevices;

#ifdef SUPPORT_SPLIT_SUBTASKS
    std::vector<std::vector<const pmProcessingElement*>> lDeviceGroups;
    std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>*> lQueryMap;
    ulong lUnsplittedDevices = 0;

    pTask->GetSubtaskSplitter().MakeDeviceGroups(lDevices, lDeviceGroups, lQueryMap, lUnsplittedDevices);

    if(!lDeviceGroups.empty())
    {
        filtered_for_each(lDeviceGroups, [&] (const std::vector<const pmProcessingElement*>& pVector) {return (pVector[0]->GetMachine() == PM_LOCAL_MACHINE && pStealingDevice != pVector[0]);},
        [&] (const std::vector<const pmProcessingElement*>& pVector)
        {
            lLocalDevices.emplace_back(pVector[0]);
        });
    }
    else
#endif
    {
        filtered_for_each(lDevices, [&] (const pmProcessingElement* pDevice) {return (pDevice->GetMachine() == PM_LOCAL_MACHINE && pStealingDevice != pDevice);}, [&] (const pmProcessingElement* pDevice)
        {
            lLocalDevices.emplace_back(pDevice);
        });
    }
        
    if(lLocalDevices.empty())
    {
        EXCEPTION_ASSERT(pStealingDevice->GetMachine() == PM_LOCAL_MACHINE);    // The stealer is the only device on the machine

        pmScheduler::GetScheduler()->StealFailedEvent(pStealingDevice, pStealingDevice, pTask);
        
        return NULL;
    }

    std::srand((unsigned int)time(NULL));

    return lLocalDevices[std::rand() % lLocalDevices.size()];
#endif
}
#endif
    
void pmScheduler::ClearPendingTaskCommands(pmTask* pTask)
{
    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->CancelTaskSpecificMemoryTransferEvents(pTask);
    DeleteMatchingCommands(pTask->GetPriority(), taskClearMatchFunc, pTask);
}
    
void pmScheduler::WaitForAllCommandsToFinish()
{
    WaitForQueuedCommands();
}

void pmScheduler::HandleCommandCompletion(const pmCommandPtr& pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);
    
	switch(lCommunicatorCommand->GetType())
	{
		case BROADCAST:
		{
			if(lCommunicatorCommand->GetTag() == HOST_FINALIZATION_TAG)
			{
                DEBUG_EXCEPTION_ASSERT(pmMachinePool::GetMachinePool()->GetMachine(0) == PM_LOCAL_MACHINE);
				pmController::GetController()->ProcessTermination();
			}

			break;
		}

		case SEND:
		{
			break;
		}

		case RECEIVE:
		{
			switch(lCommunicatorCommand->GetTag())
			{
				case MACHINE_POOL_TRANSFER_TAG:
				case DEVICE_POOL_TRANSFER_TAG:
				case UNKNOWN_LENGTH_TAG:
				case MAX_COMMUNICATOR_COMMAND_TAGS:
                {
					PMTHROW(pmFatalErrorException());
					break;
                }

				case REMOTE_TASK_ASSIGNMENT_TAG:
				{
					pmTaskManager::GetTaskManager()->CreateRemoteTask((remoteTaskAssignPacked*)(lCommunicatorCommand->GetData()));
					break;
				}

				case REMOTE_SUBTASK_ASSIGNMENT_TAG:
				{
					remoteSubtaskAssignStruct* lData = (remoteSubtaskAssignStruct*)(lCommunicatorCommand->GetData());

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					const pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

                    pmSubtaskRange lRange(NULL, NULL, lData->startSubtask, lData->endSubtask);
                
                    if(lData->assignmentType == SUBTASK_ASSIGNMENT_REGULAR)
                    {
                        // lRange.task will be set inside GetRemoteTaskOrEnqueueSubtasks
                        lRange.originalAllottee = ((lData->originalAllotteeGlobalIndex == lData->targetDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));

                        // Handling for out of order message receive (task received after subtask reception)
                        if(pmTaskManager::GetTaskManager()->GetRemoteTaskOrEnqueueSubtasks(lRange, lTargetDevice, lOriginatingHost, lData->sequenceNumber))
                            PushEvent(lTargetDevice, lRange, false);
                    }
                    else if(lData->assignmentType == RANGE_NEGOTIATION)
                    {
                        if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                        {
                            lRange.task = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                            lRange.originalAllottee = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex);
                        
                            RangeNegotiationEvent(lTargetDevice, lRange);
                        }
                    }
                    else if(lData->assignmentType == SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED)
                    {
                        lRange.task = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                        lRange.originalAllottee = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex);
                    
                        RangeNegotiationSuccessEvent(lTargetDevice, lRange);
                    }

					break;
				}

				case SEND_ACKNOWLEDGEMENT_TAG:
				{
					sendAcknowledgementPacked* lData = (sendAcknowledgementPacked*)(lCommunicatorCommand->GetData());
                    
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->ackStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->ackStruct.sequenceNumber);
                    
                    DEBUG_EXCEPTION_ASSERT(lOriginatingHost == PM_LOCAL_MACHINE);
                
                    const pmProcessingElement* lSourceDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.sourceDeviceGlobalIndex);
                    const pmProcessingElement* lOriginalAllottee = ((lData->ackStruct.originalAllotteeGlobalIndex == lData->ackStruct.sourceDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.originalAllotteeGlobalIndex));

                    pmSubtaskRange lRange(lTask, lOriginalAllottee, lData->ackStruct.startSubtask, lData->ackStruct.endSubtask);

                    AcknowledgementReceiveEvent(lSourceDevice, lRange, (pmStatus)(lData->ackStruct.execStatus), std::move(lData->ownershipVector), std::move(lData->addressSpaceIndexVector));
                
					break;
				}

				case SEND_ACKNOWLEDGEMENT_SCATTERED_TAG:
				{
					sendAcknowledgementScatteredPacked* lData = (sendAcknowledgementScatteredPacked*)(lCommunicatorCommand->GetData());
                    
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->ackStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->ackStruct.sequenceNumber);
                    
                    DEBUG_EXCEPTION_ASSERT(lOriginatingHost == PM_LOCAL_MACHINE);
                
                    const pmProcessingElement* lSourceDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.sourceDeviceGlobalIndex);
                    const pmProcessingElement* lOriginalAllottee = ((lData->ackStruct.originalAllotteeGlobalIndex == lData->ackStruct.sourceDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.originalAllotteeGlobalIndex));

                    pmSubtaskRange lRange(lTask, lOriginalAllottee, lData->ackStruct.startSubtask, lData->ackStruct.endSubtask);

                    AcknowledgementReceiveEvent(lSourceDevice, lRange, (pmStatus)(lData->ackStruct.execStatus), std::move(lData->scatteredOwnershipVector), std::move(lData->addressSpaceIndexVector));
                
					break;
				}

                case TASK_EVENT_TAG:
				{
					taskEventStruct* lData = (taskEventStruct*)(lCommunicatorCommand->GetData());

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lData->sequenceNumber);

                    if(lTask)
                    {
                        switch((taskEvents)(lData->taskEvent))
                        {
                            case TASK_FINISH_EVENT:
                            {
                                TaskFinishEvent(lTask);
                                break;
                            }

                            case TASK_COMPLETE_EVENT:
                            {
                                DEBUG_EXCEPTION_ASSERT(dynamic_cast<pmLocalTask*>(lTask));
                            
                                TaskCompleteEvent(static_cast<pmLocalTask*>(lTask));
                                break;
                            }
                                
                            case REDUCTION_TERMINATE_EVENT:
                            {
                                DEBUG_EXCEPTION_ASSERT(dynamic_cast<pmRemoteTask*>(lTask));
                                (static_cast<pmRemoteTask*>(lTask))->MarkReductionFinished();

                                break;
                            }

                            case TASK_CANCEL_EVENT:
                            {
                                TaskCancelEvent(lTask);
                                break;
                            }
                            
                            default:
                                PMTHROW(pmFatalErrorException());
                        }
                    }
                    else
                    {
                        finalize_ptr<taskEventStruct> lTaskEventData(new taskEventStruct((taskEvents)lData->taskEvent, lData->originatingHost, lData->sequenceNumber));
                        
                        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<taskEventStruct>::CreateSharedPtr(lCommunicatorCommand->GetPriority(), SEND, TASK_EVENT_TAG, PM_LOCAL_MACHINE, TASK_EVENT_STRUCT, lTaskEventData, 1);
                        
                        pmCommunicator::GetCommunicator()->Send(lCommand, false);
                    }

					break;
				}
                    
                case NO_REDUCTION_REQD_TAG:
                {
                    noReductionReqdStruct* lData = (noReductionReqdStruct*)(lCommunicatorCommand->GetData());
                    
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                    
                    lTask->GetReducer()->RegisterNoReductionReqdResponse();
                    
                    break;
                }

				case SUBTASK_REDUCE_TAG:
				{
					subtaskReducePacked* lData = (subtaskReducePacked*)(lCommunicatorCommand->GetData());

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->reduceStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->reduceStruct.sequenceNumber);
                
                    // Can not assign the received subtask to a stub which is already executing the same subtask (in case of multi-assign).
                    // This is because the incoming subtask will erase the original subtask from subscription manager making it crash.
                    // One solution is to use ProcessingElements in subscription manager rather than ExecutionStubs.
                    // For now, using a stub which has actually not registered that subtask
                    pmSubscriptionManager& lSubscriptionManager = lTask->GetSubscriptionManager();
                    pmStubManager* lStubManager = pmStubManager::GetStubManager();
                    
                    bool lStubFound = false;
                    
                    size_t lCpuStubCount = lStubManager->GetProcessingElementsCPU();
                    for(size_t i = 0; i < lCpuStubCount; ++i)
                    {
                        pmExecutionStub* lStub = lStubManager->GetCpuStub((uint)i);
                        if(!lSubscriptionManager.HasSubtask(lStub, lData->reduceStruct.subtaskId, NULL))
                        {
                            lStubFound = true;

                            lStub->RemoteSubtaskReduce(lTask, lCommunicatorCommand);
                            break;
                        }
                    }
                    
                    EXCEPTION_ASSERT(lStubFound);
                    
					break;
				}
                    
                case SUBTASK_MEMORY_REDUCE_TAG:
                {
                    pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);
                    subtaskMemoryReduceStruct* lReceiveStruct = (subtaskMemoryReduceStruct*)(lCommunicatorCommand->GetData());

                    const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->originatingHost);
                    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lReceiveStruct->sequenceNumber);

                    lTask->GetReducer()->PrepareForExternalReceive(*lReceiveStruct);
                    
                    break;
                }

				case DATA_REDISTRIBUTION_TAG:
				{
					dataRedistributionPacked* lData = (dataRedistributionPacked*)(lCommunicatorCommand->GetData());
                    
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->redistributionStruct.sequenceNumber);
                    
                    DEBUG_EXCEPTION_ASSERT(lOriginatingHost == PM_LOCAL_MACHINE);

                    lTask->GetRedistributor(lTask->GetAddressSpace(lData->redistributionStruct.addressSpaceIndex))->PerformRedistribution(pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.remoteHost), lData->redistributionStruct.subtasksAccounted, *lData->redistributionData.get_ptr());
                    
					break;
				}
                
				case REDISTRIBUTION_OFFSETS_TAG:
				{
					redistributionOffsetsPacked* lData = (redistributionOffsetsPacked*)(lCommunicatorCommand->GetData());
                    
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->redistributionStruct.sequenceNumber);
                    
                    DEBUG_EXCEPTION_ASSERT(lOriginatingHost != PM_LOCAL_MACHINE);

                    lTask->GetRedistributor(lTask->GetAddressSpace(lData->redistributionStruct.addressSpaceIndex))->ReceiveGlobalOffsets(*lData->offsetsData.get_ptr(), lData->redistributionStruct.redistributedMemGenerationNumber);
                    
					break;
				}

				case STEAL_REQUEST_TAG:
				{
					stealRequestStruct* lData = (stealRequestStruct*)(lCommunicatorCommand->GetData());

					const pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
                    
                #ifdef ENABLE_TWO_LEVEL_STEALING
					const pmProcessingElement* lTargetDevice = NULL;
                #else
					const pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);
                #endif

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
                    
                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                        StealProcessEvent(lStealingDevice, lTargetDevice, lTask, lData->stealingDeviceExecutionRate, (bool)lData->shouldMultiAssign);
                    }
                    
					break;
				}

				case STEAL_RESPONSE_TAG:
				{
					stealResponseStruct* lData = (stealResponseStruct*)(lCommunicatorCommand->GetData());

					const pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					const pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);

                        stealResponseType lResponseType = (stealResponseType)(lData->success);

                        switch(lResponseType)
                        {
                            case STEAL_SUCCESS_RESPONSE:
                            {
                                const pmProcessingElement* lOriginalAllottee = ((lData->originalAllotteeGlobalIndex == lData->stealingDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));
                                
                                pmSubtaskRange lRange(lTask, lOriginalAllottee, lData->startSubtask, lData->endSubtask);

                                StealSuccessReturnEvent(lStealingDevice, lTargetDevice, lRange);
                                
                                break;
                            }

                            case STEAL_FAILURE_RESPONSE:
                            {
                                StealFailedReturnEvent(lStealingDevice, lTargetDevice, lTask);

                                break;
                            }

                            default:
                                PMTHROW(pmFatalErrorException());
                        }
                    }

					break;
				}

				case OWNERSHIP_TRANSFER_TAG:
				{
					ownershipTransferPacked* lData = (ownershipTransferPacked*)(lCommunicatorCommand->GetData());
					pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lData->memIdentifier.memOwnerHost), lData->memIdentifier.generationNumber);
    
                    EXCEPTION_ASSERT(lAddressSpace);
                
                    lAddressSpace->ChangeOwnership(lData->transferData);

					break;
				}

				case SCATTERED_OWNERSHIP_TRANSFER_TAG:
				{
					scatteredOwnershipTransferPacked* lData = (scatteredOwnershipTransferPacked*)(lCommunicatorCommand->GetData());
					pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lData->memIdentifier.memOwnerHost), lData->memIdentifier.generationNumber);
    
                    EXCEPTION_ASSERT(lAddressSpace);
                
                    lAddressSpace->ChangeOwnership(lData->transferData);

					break;
				}

                case HOST_FINALIZATION_TAG:
				{
                    DEBUG_EXCEPTION_ASSERT(((hostFinalizationStruct*)(lCommunicatorCommand->GetData()))->terminate == false);
                    DEBUG_EXCEPTION_ASSERT(pmMachinePool::GetMachinePool()->GetMachine(0) == PM_LOCAL_MACHINE);

                    pmController::GetController()->ProcessFinalization();

					break;
				}
                
                case SUBTASK_RANGE_CANCEL_TAG:
                {
                    subtaskRangeCancelStruct* lData = (subtaskRangeCancelStruct*)(lCommunicatorCommand->GetData());
                
					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        const pmProcessingElement* lOriginalAllottee = ((lData->originalAllotteeGlobalIndex == lData->targetDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));

                        pmSubtaskRange lRange(pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber), lOriginalAllottee, lData->startSubtask, lData->endSubtask);
                        
                        const pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);
                        RangeCancellationEvent(lTargetDevice, lRange);
                    }
                    
                    break;
                }

                case MULTI_FILE_OPERATIONS_TAG:
                {
                    multiFileOperationsPacked* lData = (multiFileOperationsPacked*)(lCommunicatorCommand->GetData());
                    multiFileOperationsStruct& lStruct = lData->multiFileOpsStruct;
                    
                    switch((fileOperations)(lStruct.fileOp))
                    {
                        case MMAP_FILE:
                        {
                            uint lStartIndex = 0;
                            for(uint i = 0; i < lStruct.fileCount; ++i)
                            {
                                pmUtility::MapFile(std::string((char*)&(lData->fileNames.get_ptr()[lStartIndex]), (size_t)lData->fileNameLengthsArray.get_ptr()[i]).c_str());
                                lStartIndex += (size_t)lData->fileNameLengthsArray.get_ptr()[i];
                            }

                            pmUtility::SendMultiFileMappingAcknowledgement(lStruct.userId, pmMachinePool::GetMachinePool()->GetMachine(lStruct.sourceHost));

                            break;
                        }

                        case MUNMAP_FILE:
                        {
                            uint lStartIndex = 0;
                            for(uint i = 0; i < lStruct.fileCount; ++i)
                            {
                                pmUtility::UnmapFile(std::string((char*)&(lData->fileNames.get_ptr()[lStartIndex]), (size_t)lData->fileNameLengthsArray.get_ptr()[i]).c_str());
                                lStartIndex += (size_t)lData->fileNameLengthsArray.get_ptr()[i];
                            }

                            pmUtility::SendMultiFileUnmappingAcknowledgement(lStruct.userId, pmMachinePool::GetMachinePool()->GetMachine(lStruct.sourceHost));
                            
                            break;
                        }
                            
                        case MMAP_ACK:
                        {
                            pmUtility::RegisterMultiFileMappingResponse(lStruct.userId);
                            break;
                        }

                        case MUNMAP_ACK:
                        {
                            pmUtility::RegisterMultiFileUnmappingResponse(lStruct.userId);
                            break;
                        }

                        default:
                            PMTHROW(pmFatalErrorException());
                    }
                
                    break;
                }
                    
                case AFFINITY_DATA_TRANSFER_TAG:
                {
                    affinityDataTransferPacked* lData = (affinityDataTransferPacked*)(lCommunicatorCommand->GetData());
                    const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    pmRemoteTask* lRemoteTask = dynamic_cast<pmRemoteTask*>(pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber));
                    EXCEPTION_ASSERT(lRemoteTask);
                    
                #ifdef CENTRALIZED_AFFINITY_COMPUTATION
                    pmAddressSpace* lAffinityAddressSpace = pmAddressSpace::CheckAndCreateAddressSpace(lData->affinityAddressSpaceLength, pmMachinePool::GetMachinePool()->GetMachine(lData->affinityAddressSpace.memOwnerHost), lData->affinityAddressSpace.generationNumber);
                #else
                    pmAddressSpace* lAffinityAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lData->affinityAddressSpace.memOwnerHost), lData->affinityAddressSpace.generationNumber);
                #endif

                    EXCEPTION_ASSERT(lAffinityAddressSpace);

                    std::vector<ulong> lLogicalToPhysicalSubtaskMapping(lData->logicalToPhysicalSubtaskMapping.get_ptr(), lData->logicalToPhysicalSubtaskMapping.get_ptr() + lData->transferDataElements);
                    lRemoteTask->ReceiveAffinityData(std::move(lLogicalToPhysicalSubtaskMapping), lAffinityAddressSpace);
                    
                    break;
                }
                    
            #ifdef USE_AFFINITY_IN_STEAL
                case STEAL_SUCCESS_DISCONTIGUOUS_TAG:
                {
                    stealSuccessDiscontiguousPacked* lData = (stealSuccessDiscontiguousPacked*)(lCommunicatorCommand->GetData());
                    
					const pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					const pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);

                        StealSuccessReturnEvent(lTask, lStealingDevice, lTargetDevice, std::move(lData->discontiguousStealData));
                    }
                    
                    break;
                }
            #endif

				default:
                    PMTHROW(pmFatalErrorException());
			}

			break;
		}

		default:
			PMTHROW(pmFatalErrorException());
	}
}
    
bool taskClearMatchFunc(const schedulerEvent& pEvent, const void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case SUBTASK_EXECUTION:
        {
            const subtaskExecEvent& lEvent = static_cast<const subtaskExecEvent&>(pEvent);
            if(lEvent.range.task == static_cast<const pmTask*>(pCriterion))
                return true;
        
            break;
        }
        
        case STEAL_REQUEST_STEALER:
        {
            const stealRequestEvent& lEvent = static_cast<const stealRequestEvent&>(pEvent);
            if(lEvent.task == static_cast<const pmTask*>(pCriterion))
                return true;
            
            break;
        }

        case STEAL_PROCESS_TARGET:
        {
            const stealProcessEvent& lEvent = static_cast<const stealProcessEvent&>(pEvent);
            if(lEvent.task == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
            
        case STEAL_SUCCESS_TARGET:
        {
            const stealSuccessTargetEvent& lEvent = static_cast<const stealSuccessTargetEvent&>(pEvent);
            if(lEvent.range.task == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
            
        case STEAL_FAIL_TARGET:
        {
            const stealFailTargetEvent& lEvent = static_cast<const stealFailTargetEvent&>(pEvent);
            if(lEvent.task == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
            
        case STEAL_SUCCESS_STEALER:
        {
            const stealSuccessStealerEvent& lEvent = static_cast<const stealSuccessStealerEvent&>(pEvent);
            if(lEvent.range.task == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
            
        case STEAL_FAIL_STEALER:
        {
            const stealFailStealerEvent& lEvent = static_cast<const stealFailStealerEvent&>(pEvent);
            if(lEvent.task == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
        
        case SUBTASK_RANGE_CANCEL:
        {
            const subtaskRangeCancelEvent& lEvent = static_cast<const subtaskRangeCancelEvent&>(pEvent);
            if(lEvent.range.task == static_cast<const pmTask*>(pCriterion))
                return true;
            
            break;        
        }
        
        case RANGE_NEGOTIATION_EVENT:
        {
            const rangeNegotiationEvent& lEvent = static_cast<const rangeNegotiationEvent&>(pEvent);
            if(lEvent.range.task == static_cast<const pmTask*>(pCriterion))
                return true;
            
            break;        
        }
        
        case RANGE_NEGOTIATION_SUCCESS_EVENT:
        {
            const rangeNegotiationSuccessEvent& lEvent = static_cast<const rangeNegotiationSuccessEvent&>(pEvent);
            if(lEvent.negotiatedRange.task == static_cast<const pmTask*>(pCriterion))
                return true;
            
            break;        
        }
        
        default:
            return false;
   }
    
    return false;
}

} // end namespace pm




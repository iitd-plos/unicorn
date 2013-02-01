
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

#include "pmScheduler.h"
#include "pmCommand.h"
#include "pmTask.h"
#include "pmTaskManager.h"
#include "pmSignalWait.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmSubtaskManager.h"
#include "pmCommunicator.h"
#include "pmMemoryManager.h"
#include "pmNetwork.h"
#include "pmDevicePool.h"
#include "pmMemSection.h"
#include "pmReducer.h"
#include "pmRedistributor.h"
#include "pmController.h"
#include "pmCallbackUnit.h"

namespace pm
{
    
using namespace scheduler;

#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_forward(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, pmCommunicatorCommand::memoryIdentifierStruct& newIdentifier, ulong newOffset);
void __dump_mem_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
void __dump_mem_ack_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_forward(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, pmCommunicatorCommand::memoryIdentifierStruct&  newIdentifier, ulong newOffset)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Forwarding input mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    else
        sprintf(lStr, "Forwarding out mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Transferring input mem section %p (Remote mem (%d, %ld)) from offset %ld (Remote offset %ld) for length %ld to host %d", memSection,identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, host);
    else
        sprintf(lStr, "Transferring out mem section %p (Remote mem (%d, %ld)) from offset %ld (Remote Offset %ld) for length %ld to host %d", memSection, identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_ack_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Acknowledging input mem section %p (Remote mem (%d, %ld)) from offset %ld for length %ld to host %d", memSection, identifier.memOwnerHost, identifier.generationNumber, offset, length, host);
    else
        sprintf(lStr, "Acknowledging out mem section %p (Remote mem (%d, %ld)) from offset %ld for length %ld to host %d", memSection, identifier.memOwnerHost, identifier.generationNumber, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_TRANSFER_ACK_DUMP(memSection, identifier, receiverOffset, offset, length, host) __dump_mem_ack_transfer(memSection, identifier, receiverOffset, offset, length, host);
#define MEM_TRANSFER_DUMP(memSection, identifier, receiverOffset, offset, length, host) __dump_mem_transfer(memSection, identifier, receiverOffset, offset, length, host);
#define MEM_FORWARD_DUMP(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset) __dump_mem_forward(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset);
#else
#define MEM_TRANSFER_ACK_DUMP(memSection, identifier, receiverOffset, offset, length, host)
#define MEM_TRANSFER_DUMP(memSection, identifier, receiverOffset, offset, length, host)
#define MEM_FORWARD_DUMP(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset)
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
    
pmStatus SchedulerCommandCompletionCallback(pmCommandPtr pCommand)
{
	pmScheduler* lScheduler = pmScheduler::GetScheduler();
	return lScheduler->CommandCompletionEvent(pCommand);
}

static pmCommandCompletionCallback gCommandCompletionCallback = SchedulerCommandCompletionCallback;

pmScheduler* pmScheduler::mScheduler = NULL;

pmScheduler::pmScheduler()
{
    if(mScheduler)
        PMTHROW(pmFatalErrorException());
    
    mScheduler = this;

#ifdef TRACK_SUBTASK_EXECUTION
    mSubtasksAssigned = 0;
    mAcknowledgementsSent = 0;
#endif
    
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::HOST_FINALIZATION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::DATA_REDISTRIBUTION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_STRUCT);

	SetupPersistentCommunicationCommands();
}

pmScheduler::~pmScheduler()
{
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::HOST_FINALIZATION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_STRUCT);

	DestroyPersistentCommunicationCommands();

#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down scheduler thread");
#endif
}

pmScheduler* pmScheduler::GetScheduler()
{
	return mScheduler;
}

pmCommandCompletionCallback pmScheduler::GetUnknownLengthCommandCompletionCallback()
{
	return gCommandCompletionCallback;
}

pmStatus pmScheduler::SetupPersistentCommunicationCommands()
{
#define PERSISTENT_RECV_COMMAND(tag, structType, recvDataPtr) pmPersistentCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::RECEIVE, \
	pmCommunicatorCommand::tag, NULL, pmCommunicatorCommand::structType, recvDataPtr, 1, NULL, 0, gCommandCompletionCallback)

	mRemoteSubtaskRecvCommand = PERSISTENT_RECV_COMMAND(REMOTE_SUBTASK_ASSIGNMENT_TAG, REMOTE_SUBTASK_ASSIGN_STRUCT, &mSubtaskAssignRecvData);
	mTaskEventRecvCommand = PERSISTENT_RECV_COMMAND(TASK_EVENT_TAG, TASK_EVENT_STRUCT, &mTaskEventRecvData);
	mStealRequestRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_REQUEST_TAG, STEAL_REQUEST_STRUCT,	&mStealRequestRecvData);
	mStealResponseRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_RESPONSE_TAG, STEAL_RESPONSE_STRUCT, &mStealResponseRecvData);
	mMemTransferRequestCommand = PERSISTENT_RECV_COMMAND(MEMORY_TRANSFER_REQUEST_TAG, MEMORY_TRANSFER_REQUEST_STRUCT, &mMemTransferRequestData);
    mSubtaskRangeCancelCommand = PERSISTENT_RECV_COMMAND(SUBTASK_RANGE_CANCEL_TAG, SUBTASK_RANGE_CANCEL_STRUCT, &mSubtaskRangeCancelData);
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mRemoteSubtaskRecvCommand.get());
    lNetwork->InitializePersistentCommand(mTaskEventRecvCommand.get());
    lNetwork->InitializePersistentCommand(mStealRequestRecvCommand.get());
    lNetwork->InitializePersistentCommand(mStealResponseRecvCommand.get());
    lNetwork->InitializePersistentCommand(mMemTransferRequestCommand.get());
    lNetwork->InitializePersistentCommand(mSubtaskRangeCancelCommand.get());

	SetupNewRemoteSubtaskReception();
	SetupNewTaskEventReception();
	SetupNewStealRequestReception();
	SetupNewStealResponseReception();
	SetupNewMemTransferRequestReception();
    SetupNewSubtaskRangeCancelReception();
	
	// Only MPI master host receives finalization signal
	if(pmMachinePool::GetMachinePool()->GetMachine(0) == PM_LOCAL_MACHINE)
	{
		mHostFinalizationCommand = PERSISTENT_RECV_COMMAND(HOST_FINALIZATION_TAG, HOST_FINALIZATION_STRUCT, &mHostFinalizationData);
        
        lNetwork->InitializePersistentCommand(mHostFinalizationCommand.get());
        
		SetupNewHostFinalizationReception();
	}

	return pmSuccess;
}

pmStatus pmScheduler::DestroyPersistentCommunicationCommands()
{
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->TerminatePersistentCommand(mRemoteSubtaskRecvCommand.get());
    lNetwork->TerminatePersistentCommand(mTaskEventRecvCommand.get());
    lNetwork->TerminatePersistentCommand(mStealRequestRecvCommand.get());
    lNetwork->TerminatePersistentCommand(mStealResponseRecvCommand.get());
    lNetwork->TerminatePersistentCommand(mMemTransferRequestCommand.get());
    lNetwork->TerminatePersistentCommand(mSubtaskRangeCancelCommand.get());

	if(mHostFinalizationCommand.get())
        lNetwork->TerminatePersistentCommand(mHostFinalizationCommand.get());

	return pmSuccess;
}

pmStatus pmScheduler::SetupNewRemoteSubtaskReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mRemoteSubtaskRecvCommand, false);
}

pmStatus pmScheduler::SetupNewTaskEventReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mTaskEventRecvCommand, false);
}

pmStatus pmScheduler::SetupNewStealRequestReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mStealRequestRecvCommand, false);
}

pmStatus pmScheduler::SetupNewStealResponseReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mStealResponseRecvCommand, false);
}

pmStatus pmScheduler::SetupNewMemTransferRequestReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mMemTransferRequestCommand, false);
}

pmStatus pmScheduler::SetupNewHostFinalizationReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mHostFinalizationCommand, false);
}

pmStatus pmScheduler::SetupNewSubtaskRangeCancelReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mSubtaskRangeCancelCommand, false);
}

pmStatus pmScheduler::SubmitTaskEvent(pmLocalTask* pLocalTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = NEW_SUBMISSION;
	lEvent.submissionDetails.localTask = pLocalTask;

	return SwitchThread(lEvent, pLocalTask->GetPriority());
}

pmStatus pmScheduler::PushEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange)
{
#ifdef TRACK_SUBTASK_EXECUTION
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
    mSubtasksAssigned += pRange.endSubtask - pRange.startSubtask + 1;
#endif

	schedulerEvent lEvent;
	lEvent.eventId = SUBTASK_EXECUTION;
	lEvent.execDetails.device = pDevice;
	lEvent.execDetails.range = pRange;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmScheduler::StealRequestEvent(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return pmSuccess;

#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_WAIT, true);
#endif
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_REQUEST_STEALER;
	lEvent.stealRequestDetails.stealingDevice = pStealingDevice;
	lEvent.stealRequestDetails.task = pTask;
	lEvent.stealRequestDetails.stealingDeviceExecutionRate = pExecutionRate;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::StealProcessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return pmSuccess;
    
#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_SERVE, true);
#endif
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_PROCESS_TARGET;
	lEvent.stealProcessDetails.stealingDevice = pStealingDevice;
	lEvent.stealProcessDetails.targetDevice = pTargetDevice;
	lEvent.stealProcessDetails.task = pTask;
	lEvent.stealProcessDetails.stealingDeviceExecutionRate = pExecutionRate;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::StealSuccessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pRange.task))
        return pmSuccess;
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_SUCCESS_TARGET;
	lEvent.stealSuccessTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealSuccessTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealSuccessTargetDetails.range = pRange;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmScheduler::StealFailedEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return pmSuccess;
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_FAIL_TARGET;
	lEvent.stealFailTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealFailTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealFailTargetDetails.task = pTask;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::StealSuccessReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pRange.task))
        return pmSuccess;
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_SUCCESS_STEALER;
	lEvent.stealSuccessTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealSuccessTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealSuccessTargetDetails.range = pRange;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmScheduler::StealFailedReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
        return pmSuccess;
    
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_FAIL_STEALER;
	lEvent.stealFailTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealFailTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealFailTargetDetails.task = pTask;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::AcknowledgementSendEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, std::map<size_t, size_t>& pOwnershipMap)
{
#ifdef TRACK_SUBTASK_EXECUTION
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
    mAcknowledgementsSent += (pRange.endSubtask - pRange.startSubtask + 1);
    std::cout << "Device " << pDevice->GetGlobalDeviceIndex() << " sent " << (pRange.endSubtask - pRange.startSubtask + 1) << " acknowledgements for subtasks [" << pRange.startSubtask << " - " << pRange.endSubtask << "]" << std::endl;
#endif

	schedulerEvent lEvent;
	lEvent.eventId = SEND_ACKNOWLEDGEMENT;
	lEvent.ackSendDetails.device = pDevice;
	lEvent.ackSendDetails.range = pRange;
	lEvent.ackSendDetails.execStatus = pExecStatus;
    lEvent.ackSendDetails.ownershipData = new pmCommunicatorCommand::ownershipDataStruct[pOwnershipMap.size()];
    lEvent.ackSendDetails.dataElements = (uint)pOwnershipMap.size();
    
    std::map<size_t, size_t>::iterator lBeginIter = pOwnershipMap.begin(), lEndIter = pOwnershipMap.end();
    for(size_t index = 0; lBeginIter != lEndIter; ++index, ++lBeginIter)
    {
        lEvent.ackSendDetails.ownershipData[index].offset = lBeginIter->first;
        lEvent.ackSendDetails.ownershipData[index].length = lBeginIter->second;
    }

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmScheduler::AcknowledgementReceiveEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, pmCommunicatorCommand::ownershipDataStruct* pOwnershipData, uint pDataElements)
{
	schedulerEvent lEvent;
	lEvent.eventId = RECEIVE_ACKNOWLEDGEMENT;
	lEvent.ackReceiveDetails.device = pDevice;
	lEvent.ackReceiveDetails.range = pRange;
	lEvent.ackReceiveDetails.execStatus = pExecStatus;
	lEvent.ackReceiveDetails.ownershipData = pOwnershipData;
    lEvent.ackSendDetails.dataElements = pDataElements;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmScheduler::TaskCancelEvent(pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = TASK_CANCEL;
	lEvent.taskCancelDetails.task = pTask;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::TaskFinishEvent(pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = TASK_FINISH;
	lEvent.taskFinishDetails.task = pTask;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::TaskCompleteEvent(pmLocalTask* pLocalTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = TASK_COMPLETE;
	lEvent.taskCompleteDetails.localTask = pLocalTask;

	return SwitchThread(lEvent, pLocalTask->GetPriority());
}

pmStatus pmScheduler::ReduceRequestEvent(pmExecutionStub* pReducingStub, pmTask* pTask, pmMachine* pDestMachine, ulong pSubtaskId)
{
	schedulerEvent lEvent;
	lEvent.eventId = SUBTASK_REDUCE;
	lEvent.subtaskReduceDetails.task = pTask;
	lEvent.subtaskReduceDetails.machine = pDestMachine;
    lEvent.subtaskReduceDetails.reducingStub = pReducingStub;
	lEvent.subtaskReduceDetails.subtaskId = pSubtaskId;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmScheduler::MemTransferEvent(pmMemSection* pSrcMemSection, pmCommunicatorCommand::memoryIdentifierStruct& pDestMemIdentifier, ulong pOffset, ulong pLength, pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority)
{
	schedulerEvent lEvent;
	lEvent.eventId = MEMORY_TRANSFER;
	lEvent.memTransferDetails.srcMemSection = pSrcMemSection;
    lEvent.memTransferDetails.destMemIdentifier = pDestMemIdentifier;
	lEvent.memTransferDetails.offset = pOffset;
	lEvent.memTransferDetails.length = pLength;
	lEvent.memTransferDetails.machine = pDestMachine;
	lEvent.memTransferDetails.receiverOffset = pReceiverOffset;
	lEvent.memTransferDetails.priority = pPriority;
    lEvent.memTransferDetails.isForwarded = pIsForwarded;

	return SwitchThread(lEvent, pPriority);
}

pmStatus pmScheduler::CommandCompletionEvent(pmCommandPtr pCommand)
{
	schedulerEvent lEvent;
	lEvent.eventId = COMMAND_COMPLETION;
	lEvent.commandCompletionDetails.command = pCommand;

	return SwitchThread(lEvent, pCommand->GetPriority());
}

pmStatus pmScheduler::RangeCancellationEvent(pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
	schedulerEvent lEvent;
	lEvent.eventId = SUBTASK_RANGE_CANCEL;
    lEvent.subtaskRangeCancelDetails.targetDevice = pTargetDevice;
	lEvent.subtaskRangeCancelDetails.range = pRange;
    
	return SwitchThread(lEvent, pRange.task->GetPriority());
}
    
pmStatus pmScheduler::RedistributionMetaDataEvent(pmTask* pTask, std::vector<pmCommunicatorCommand::redistributionOrderStruct>* pRedistributionData, uint pCount)
{
	schedulerEvent lEvent;
	lEvent.eventId = REDISTRIBUTION_METADATA_EVENT;
	lEvent.redistributionMetaDataDetails.task = pTask;
    lEvent.redistributionMetaDataDetails.redistributionData = pRedistributionData;
    lEvent.redistributionMetaDataDetails.count = pCount;

	return SwitchThread(lEvent, pTask->GetPriority());
}
    
pmStatus pmScheduler::RangeNegotiationEvent(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange)
{
    schedulerEvent lEvent;
    lEvent.eventId = RANGE_NEGOTIATION_EVENT;
    lEvent.rangeNegotiationDetails.requestingDevice = pRequestingDevice;
    lEvent.rangeNegotiationDetails.range = pRange;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}
    
pmStatus pmScheduler::RangeNegotiationSuccessEvent(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pNegotiatedRange)
{
    schedulerEvent lEvent;
    lEvent.eventId = RANGE_NEGOTIATION_SUCCESS_EVENT;
    lEvent.rangeNegotiationSuccessDetails.requestingDevice = pRequestingDevice;
    lEvent.rangeNegotiationSuccessDetails.negotiatedRange = pNegotiatedRange;
    
	return SwitchThread(lEvent, pNegotiatedRange.task->GetPriority());
}
    
pmStatus pmScheduler::TerminateTaskEvent(pmTask* pTask)
{
    schedulerEvent lEvent;
    lEvent.eventId = TERMINATE_TASK;
    lEvent.taskTerminateDetails.task = pTask;

	return SwitchThread(lEvent, pTask->GetPriority());
}
    
pmStatus pmScheduler::SendFinalizationSignal()
{
	schedulerEvent lEvent;
	lEvent.eventId = HOST_FINALIZATION;
	lEvent.hostFinalizationDetails.terminate = false;

	return SwitchThread(lEvent, MAX_CONTROL_PRIORITY);    
}

pmStatus pmScheduler::BroadcastTerminationSignal()
{
	schedulerEvent lEvent;
	lEvent.eventId = HOST_FINALIZATION;
	lEvent.hostFinalizationDetails.terminate = true;

	return SwitchThread(lEvent, MAX_CONTROL_PRIORITY);    
}

pmStatus pmScheduler::ThreadSwitchCallback(schedulerEvent& pEvent)
{
	try
	{
		return ProcessEvent(pEvent);
	}
	catch(pmException e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from scheduler thread");
	}

	return pmSuccess;
}

pmStatus pmScheduler::ProcessEvent(schedulerEvent& pEvent)
{
#ifdef DUMP_SCHEDULER_EVENT
    char lStr[512];
    
    sprintf(lStr, "Host %d Scheduler Event: ", pmGetHostId(), schedulerEventName[pEvent.eventId]);
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
#endif
    
	switch(pEvent.eventId)
	{
		case NEW_SUBMISSION:	/* Comes from application thread */
        {
            pmLocalTask* lLocalTask = pEvent.submissionDetails.localTask;			
            return StartLocalTaskExecution(lLocalTask);
        }

		case SUBTASK_EXECUTION:	/* Comes from network thread or from scheduler thread for local submissions */
        {
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pEvent.execDetails.range.task))
                return PushToStub(pEvent.execDetails.device, pEvent.execDetails.range);
        }

		case STEAL_REQUEST_STEALER:	/* Comes from stub thread */
        {
            stealRequest& lRequest = pEvent.stealRequestDetails;

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lRequest.task))
                return StealSubtasks(lRequest.stealingDevice, lRequest.task, lRequest.stealingDeviceExecutionRate);
            
            break;
        }

		case STEAL_PROCESS_TARGET:	/* Comes from network thread */
        {
            stealProcess& lEventDetails = pEvent.stealProcessDetails;

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                return ServeStealRequest(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task, lEventDetails.stealingDeviceExecutionRate);
            
            break;
        }

		case STEAL_SUCCESS_TARGET:	/* Comes from stub thread */
        {
            pmStatus lStatus = pmSuccess;
            stealSuccessTarget& lEventDetails = pEvent.stealSuccessTargetDetails;

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
                lStatus = SendStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
            
#ifdef ENABLE_TASK_PROFILING
            lEventDetails.range.task->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_SERVE, false);
#endif
            return lStatus;            
        }

		case STEAL_FAIL_TARGET: /* Comes from stub thread */
        {
            pmStatus lStatus = pmSuccess;
            stealFailTarget& lEventDetails = pEvent.stealFailTargetDetails;

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                lStatus = SendFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
            
#ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_SERVE, false);
#endif
            return lStatus;
        }

		case STEAL_SUCCESS_STEALER: /* Comes from network thread */
        {
            stealSuccessStealer& lEventDetails = pEvent.stealSuccessStealerDetails;

#ifdef ENABLE_TASK_PROFILING
            lEventDetails.range.task->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_WAIT, false);
#endif            
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
                return ReceiveStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
            
            break;
        }

		case STEAL_FAIL_STEALER: /* Comes from network thread */
        {
            stealFailStealer& lEventDetails = pEvent.stealFailStealerDetails;

#ifdef ENABLE_TASK_PROFILING
            lEventDetails.task->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_STEAL_WAIT, false);
#endif
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.task))
                return ReceiveFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
            
            break;
        }

		case SEND_ACKNOWLEDGEMENT:	/* Comes from stub thread */
        {
            sendAcknowledgement& lEventDetails = pEvent.ackSendDetails;

            pmTask* lTask = lEventDetails.range.task;
        
            if(!pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lTask))
                break;
        
            lTask->IncrementSubtasksExecuted(lEventDetails.range.endSubtask - lEventDetails.range.startSubtask + 1);

            pmMachine* lOriginatingHost = lTask->GetOriginatingHost();
            if(lOriginatingHost == PM_LOCAL_MACHINE)
            {
                return AcknowledgementReceiveEvent(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, lEventDetails.ownershipData, lEventDetails.dataElements);
            }
            else
            {
                pmCommunicatorCommand::sendAcknowledgementPacked* lPackedData = new pmCommunicatorCommand::sendAcknowledgementPacked(lEventDetails.device, lEventDetails.range, lEventDetails.ownershipData, lEventDetails.dataElements, lEventDetails.execStatus);
                
                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG, lOriginatingHost, pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_PACKED, lPackedData, 1, NULL, 0, gCommandCompletionCallback);
                
                pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
            }

            break;
        }

		case RECEIVE_ACKNOWLEDGEMENT:
        {
            receiveAcknowledgement& lEventDetails = pEvent.ackReceiveDetails;

            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
                return ProcessAcknowledgement((pmLocalTask*)(lEventDetails.range.task), lEventDetails.device, lEventDetails.range, lEventDetails.execStatus, lEventDetails.ownershipData, lEventDetails.dataElements);
        }

		case TASK_CANCEL:
        {
            taskCancel& lEventDetails = pEvent.taskCancelDetails;
            pmTask* lTask = lEventDetails.task;

            CancelAllSubtasksExecutingOnLocalStubs(lTask, false);
            break;
        }

		case TASK_FINISH:
        {
            taskFinish& lEventDetails = pEvent.taskFinishDetails;
            pmTask* lTask = lEventDetails.task;
        
            if(lTask->GetMemSectionRW() && lTask->GetMemSectionRW()->IsReadWrite())
            {
                CommitShadowMemPendingOnAllStubs(lTask);
                lTask->MarkAllStubsScannedForShadowMemCommitMessages();
            }

            lTask->MarkSubtaskExecutionFinished();
            ClearPendingTaskCommands(lTask);
            
            if(lTask->IsMultiAssignEnabled())
            {
                CancelAllSubtasksExecutingOnLocalStubs(lTask, true);
                lTask->MarkAllStubsScannedForCancellationMessages();
            }

            break;
        }

		case TASK_COMPLETE:
        {
            taskComplete& lEventDetails = pEvent.taskCompleteDetails;
            pmLocalTask* lLocalTask = lEventDetails.localTask;

            lLocalTask->RegisterInternalTaskCompletionMessage();

            break;
        }

		case SUBTASK_REDUCE:
        {
            subtaskReduce& lEventDetails = pEvent.subtaskReduceDetails;

            pmCommunicatorCommand::subtaskReducePacked* lPackedData = new pmCommunicatorCommand::subtaskReducePacked(lEventDetails.reducingStub, lEventDetails.task, lEventDetails.subtaskId);

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.task->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::SUBTASK_REDUCE_TAG, lEventDetails.machine, pmCommunicatorCommand::SUBTASK_REDUCE_PACKED, lPackedData, 1, NULL, 0, gCommandCompletionCallback);

            pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);

            break;
        }

		case MEMORY_TRANSFER:
        {
            memTransfer& lEventDetails = pEvent.memTransferDetails;

            if(lEventDetails.machine == PM_LOCAL_MACHINE && !lEventDetails.isForwarded)
                PMTHROW(pmFatalErrorException());   // Cyclic reference
            
            pmCommunicatorCommand::memoryReceivePacked* lPackedData = NULL;
            
            // Check if the memory is residing locally or forward the request to the owner machine
            pmMemSection::pmMemOwnership lOwnerships;
            lEventDetails.srcMemSection->GetOwners(lEventDetails.offset, lEventDetails.length, lOwnerships);
            
            pmMemSection* lDestMemSection = NULL;
        
            pmMemSection::pmMemOwnership::iterator lStartIter = lOwnerships.begin(), lEndIter = lOwnerships.end(), lIter;
            for(lIter = lStartIter; lIter != lEndIter; ++lIter)
            {
                ulong lInternalOffset = lIter->first;
                ulong lInternalLength = lIter->second.first;
                pmMemSection::vmRangeOwner& lRangeOwner = lIter->second.second;
                
                if(lRangeOwner.host == PM_LOCAL_MACHINE)
                {
					pmMemSection* lOwnerMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lRangeOwner.memIdentifier.memOwnerHost), lRangeOwner.memIdentifier.generationNumber);
                
					if(!lOwnerMemSection)
                        PMTHROW(pmFatalErrorException());
                
                #ifdef ENABLE_MEM_PROFILING
                    lEventDetails.srcMemSection->RecordMemTransfer(lInternalLength);
                #endif
                
                    if(lEventDetails.machine == PM_LOCAL_MACHINE)
                    {
                        lDestMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.destMemIdentifier.memOwnerHost), lEventDetails.destMemIdentifier.generationNumber);
                    
                        if(!lDestMemSection)
                            PMTHROW(pmFatalErrorException());
                    
                        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lDestMemSection, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset));
                    }
                    else
                    {
                        lPackedData = new pmCommunicatorCommand::memoryReceivePacked(lEventDetails.destMemIdentifier.memOwnerHost, lEventDetails.destMemIdentifier.generationNumber, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset));
                    
                        MEM_TRANSFER_DUMP(lEventDetails.srcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)))

                        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.priority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_RECEIVE_TAG, lEventDetails.machine, pmCommunicatorCommand::MEMORY_RECEIVE_PACKED, lPackedData, 1, NULL, 0, gCommandCompletionCallback);

                        pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
                    }
                }
                else
                {
                    if(lEventDetails.isForwarded)
                        PMTHROW(pmFatalErrorException());
                    
                    pmCommunicatorCommand::memoryTransferRequest* lData = new pmCommunicatorCommand::memoryTransferRequest();
                    lData->sourceMemIdentifier.memOwnerHost = lRangeOwner.memIdentifier.memOwnerHost;
                    lData->sourceMemIdentifier.generationNumber = lRangeOwner.memIdentifier.generationNumber;
                    lData->destMemIdentifier.memOwnerHost = lEventDetails.destMemIdentifier.memOwnerHost;
                    lData->destMemIdentifier.generationNumber = lEventDetails.destMemIdentifier.generationNumber;
                    lData->receiverOffset = lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset;
                    lData->offset = lRangeOwner.hostOffset;
                    lData->length = lInternalLength;
                    lData->destHost = *(lEventDetails.machine);
                    lData->isForwarded = 1;
                    
                    MEM_FORWARD_DUMP(lEventDetails.srcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)), *lRangeOwner.host, lRangeOwner.memIdentifier, lRangeOwner.hostOffset)

                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG, lRangeOwner.host, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT, (void*)lData, 1, NULL, 0, gCommandCompletionCallback);
                    
                    pmCommunicator::GetCommunicator()->Send(lCommand);
                }
            }
            
            break;
        }

		case COMMAND_COMPLETION:
        {
            commandCompletion& lEventDetails = pEvent.commandCompletionDetails;

            HandleCommandCompletion(lEventDetails.command);

            break;
        }
            
        case SUBTASK_RANGE_CANCEL:
        {
            subtaskRangeCancel& lEventDetails = pEvent.subtaskRangeCancelDetails;
            
            if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lEventDetails.range.task))
            {
                pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetStub(lEventDetails.targetDevice);
                lStub->CancelSubtaskRange(lEventDetails.range);
            }
        
            break;
        }
        
        case REDISTRIBUTION_METADATA_EVENT:
        {
            redistributionMetaData& lEventDetails = pEvent.redistributionMetaDataDetails;
            SendRedistributionData(lEventDetails.task, lEventDetails.redistributionData, lEventDetails.count);
            
            break;
        }

		case HOST_FINALIZATION:
        {
            hostFinalization& lEventDetails = pEvent.hostFinalizationDetails;

            pmMachine* lMasterHost = pmMachinePool::GetMachinePool()->GetMachine(0);

            if(lEventDetails.terminate)
            {
                // Only master host can broadcast the global termination signal
                if(lMasterHost != PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());

                pmCommunicatorCommand::hostFinalizationStruct* lBroadcastData = new pmCommunicatorCommand::hostFinalizationStruct();
                lBroadcastData->terminate = true;

                pmCommunicatorCommandPtr lBroadcastCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_PRIORITY_LEVEL, pmCommunicatorCommand::BROADCAST,pmCommunicatorCommand::HOST_FINALIZATION_TAG, lMasterHost, pmCommunicatorCommand::HOST_FINALIZATION_STRUCT, lBroadcastData, 1, NULL, 0, gCommandCompletionCallback);

                pmCommunicator::GetCommunicator()->Broadcast(lBroadcastCommand);
            }
            else
            {
                if(lMasterHost == PM_LOCAL_MACHINE)
                    return pmController::GetController()->ProcessFinalization();

                pmCommunicatorCommand::hostFinalizationStruct* lData = new pmCommunicatorCommand::hostFinalizationStruct();
                lData->terminate = false;

                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::SEND, pmCommunicatorCommand::HOST_FINALIZATION_TAG, lMasterHost, pmCommunicatorCommand::HOST_FINALIZATION_STRUCT, lData, 1, NULL, 0, gCommandCompletionCallback);

                pmCommunicator::GetCommunicator()->Send(lCommand, false);
            }

            break;
        }
        
        case RANGE_NEGOTIATION_EVENT:
        {
            rangeNegotiation& lEventDetails = pEvent.rangeNegotiationDetails;
        
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
            rangeNegotiationSuccess& lEventDetails = pEvent.rangeNegotiationSuccessDetails;
        
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
            taskTerminate& lEventDetails = pEvent.taskTerminateDetails;
        
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
    }

	return pmSuccess;
}

void pmScheduler::CancelAllSubtasksExecutingOnLocalStubs(pmTask* pTask, bool pTaskListeningOnCancellation)
{
    pmStubManager* lManager = pmStubManager::GetStubManager();
    uint lStubCount = (uint)(lManager->GetStubCount());
    for(uint i = 0; i < lStubCount; ++i)
        lManager->GetStub(i)->CancelAllSubtasks(pTask, pTaskListeningOnCancellation);
}

void pmScheduler::CommitShadowMemPendingOnAllStubs(pmTask* pTask)
{
    pmStubManager* lManager = pmStubManager::GetStubManager();
    uint lStubCount = (uint)(lManager->GetStubCount());
    for(uint i = 0; i < lStubCount; ++i)
        lManager->GetStub(i)->ProcessDeferredShadowMemCommits(pTask);
}
    
pmStatus pmScheduler::NegotiateSubtaskRangeWithOriginalAllottee(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange)
{
    pmProcessingElement* lOriginalAllottee = pRange.originalAllottee;
    if(lOriginalAllottee == NULL || lOriginalAllottee == pRequestingDevice)
        PMTHROW(pmFatalErrorException());
    
	pmMachine* lMachine = lOriginalAllottee->GetMachine();

	if(lMachine == PM_LOCAL_MACHINE)
	{
		return RangeNegotiationEvent(pRequestingDevice, pRange);
	}
	else
	{
		pmCommunicatorCommand::remoteSubtaskAssignStruct* lSubtaskData = new pmCommunicatorCommand::remoteSubtaskAssignStruct();
		lSubtaskData->sequenceNumber = pRange.task->GetSequenceNumber();
		lSubtaskData->startSubtask = pRange.startSubtask;
		lSubtaskData->endSubtask = pRange.endSubtask;
        lSubtaskData->originalAllotteeGlobalIndex = pRange.originalAllottee->GetGlobalDeviceIndex();
		lSubtaskData->originatingHost = *(pRange.task->GetOriginatingHost());
		lSubtaskData->targetDeviceGlobalIndex = pRequestingDevice->GetGlobalDeviceIndex();
        lSubtaskData->assignmentType = pmCommunicatorCommand::RANGE_NEGOTIATION;

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pRange.task->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT, (void*)lSubtaskData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

    return pmSuccess;
}
    
pmStatus pmScheduler::SendRangeNegotiationSuccess(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pNegotiatedRange)
{
    pmProcessingElement* lOriginalAllottee = pNegotiatedRange.originalAllottee;
    if(lOriginalAllottee == NULL || lOriginalAllottee == pRequestingDevice)
        PMTHROW(pmFatalErrorException());
    
	pmMachine* lMachine = pRequestingDevice->GetMachine();

	if(lMachine == PM_LOCAL_MACHINE)
	{
		return RangeNegotiationSuccessEvent(pRequestingDevice, pNegotiatedRange);
	}
	else
	{
		pmCommunicatorCommand::remoteSubtaskAssignStruct* lSubtaskData = new pmCommunicatorCommand::remoteSubtaskAssignStruct();
		lSubtaskData->sequenceNumber = pNegotiatedRange.task->GetSequenceNumber();
		lSubtaskData->startSubtask = pNegotiatedRange.startSubtask;
		lSubtaskData->endSubtask = pNegotiatedRange.endSubtask;
        lSubtaskData->originalAllotteeGlobalIndex = pNegotiatedRange.originalAllottee->GetGlobalDeviceIndex();
		lSubtaskData->originatingHost = *(pNegotiatedRange.task->GetOriginatingHost());
		lSubtaskData->targetDeviceGlobalIndex = pRequestingDevice->GetGlobalDeviceIndex();
        lSubtaskData->assignmentType = pmCommunicatorCommand::SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED;

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pNegotiatedRange.task->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT, (void*)lSubtaskData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}
    
    return pmSuccess;
}
    
pmStatus pmScheduler::SendPostTaskOwnershipTransfer(pmMemSection* pMemSection, pmMachine* pReceiverHost, std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> >& pChangeData)
{
    if(pReceiverHost == PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());
    
    pmCommunicatorCommand::ownershipTransferPacked* lPackedData = new pmCommunicatorCommand::ownershipTransferPacked(pMemSection, pChangeData);
        
    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::SEND, pmCommunicatorCommand::OWNERSHIP_TRANSFER_TAG, pReceiverHost, pmCommunicatorCommand::OWNERSHIP_TRANSFER_PACKED, lPackedData, 1, NULL, 0, gCommandCompletionCallback);
    
    return pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
}
    
pmStatus pmScheduler::SendSubtaskRangeCancellationMessage(pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
 	pmMachine* lMachine = pTargetDevice->GetMachine();
    if(lMachine == PM_LOCAL_MACHINE)
    {
        return RangeCancellationEvent(pTargetDevice, pRange);
    }
    else
    {
		pmCommunicatorCommand::subtaskRangeCancelStruct* lRangeCancellationData = new pmCommunicatorCommand::subtaskRangeCancelStruct();
        lRangeCancellationData->targetDeviceGlobalIndex = pTargetDevice->GetGlobalDeviceIndex();
        lRangeCancellationData->originatingHost = *(pRange.task->GetOriginatingHost());
		lRangeCancellationData->sequenceNumber = pRange.task->GetSequenceNumber();
		lRangeCancellationData->startSubtask = pRange.startSubtask;
		lRangeCancellationData->endSubtask = pRange.endSubtask;
        lRangeCancellationData->originalAllotteeGlobalIndex = (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pTargetDevice->GetGlobalDeviceIndex());

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pRange.task->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_TAG, lMachine, pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_STRUCT, (void*)lRangeCancellationData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
    }
    
    return pmSuccess;
}
    
pmStatus pmScheduler::SendRedistributionData(pmTask* pTask, std::vector<pmCommunicatorCommand::redistributionOrderStruct>* pRedistributionData, uint pCount)
{
    pmMachine* lMachine = pTask->GetOriginatingHost();
    if(lMachine == PM_LOCAL_MACHINE)
    {
        return pTask->GetRedistributor()->PerformRedistribution(lMachine, pTask->GetSubtasksExecuted(), *pRedistributionData);
    }
    else
    {
        if((*pRedistributionData).empty())
        {
            (static_cast<pmRemoteTask*>(pTask))->MarkRedistributionFinished();
        }
        else
        {
            pmCommunicatorCommand::dataRedistributionPacked* lPackedData = new pmCommunicatorCommand::dataRedistributionPacked(pTask, &(*pRedistributionData)[0], pCount);
            
            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG, lMachine, pmCommunicatorCommand::DATA_REDISTRIBUTION_PACKED, lPackedData, 1, NULL, 0, gCommandCompletionCallback);
            
            return pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
        }
    }
    
    return pmSuccess;
}


pmStatus pmScheduler::AssignSubtasksToDevice(pmProcessingElement* pDevice, pmLocalTask* pLocalTask)
{
	pmMachine* lMachine = pDevice->GetMachine();

	ulong lStartingSubtask, lSubtaskCount;
    pmProcessingElement* lOriginalAllottee = NULL;
	pLocalTask->GetSubtaskManager()->AssignSubtasksToDevice(pDevice, lSubtaskCount, lStartingSubtask, lOriginalAllottee);
    
	if(lSubtaskCount == 0)
		return pmSuccess;

	if(lMachine == PM_LOCAL_MACHINE)
	{
		pmSubtaskRange lRange;
		lRange.task = pLocalTask;
		lRange.startSubtask = lStartingSubtask;
		lRange.endSubtask = lStartingSubtask + lSubtaskCount - 1;
        lRange.originalAllottee = lOriginalAllottee;

		return PushEvent(pDevice, lRange);
	}
	else
	{
		pmCommunicatorCommand::remoteSubtaskAssignStruct* lSubtaskData = new pmCommunicatorCommand::remoteSubtaskAssignStruct();
		lSubtaskData->sequenceNumber = pLocalTask->GetSequenceNumber();
		lSubtaskData->startSubtask = lStartingSubtask;
		lSubtaskData->endSubtask = lStartingSubtask + lSubtaskCount - 1;
        lSubtaskData->originalAllotteeGlobalIndex = (lOriginalAllottee ? lOriginalAllottee->GetGlobalDeviceIndex() : pDevice->GetGlobalDeviceIndex());
		lSubtaskData->originatingHost = *(pLocalTask->GetOriginatingHost());
		lSubtaskData->targetDeviceGlobalIndex = pDevice->GetGlobalDeviceIndex();
        lSubtaskData->assignmentType = pmCommunicatorCommand::SUBTASK_ASSIGNMENT_REGULAR;

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT_TAG, lMachine, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT, (void*)lSubtaskData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

	return pmSuccess;
}

pmStatus pmScheduler::AssignSubtasksToDevices(pmLocalTask* pLocalTask)
{
	std::vector<pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();

	size_t lSize = lDevices.size();
	for(size_t i=0; i<lSize; ++i)
		AssignSubtasksToDevice(lDevices[i], pLocalTask);

	return pmSuccess;
}

pmStatus pmScheduler::AssignTaskToMachines(pmLocalTask* pLocalTask, std::set<pmMachine*>& pMachines)
{
	std::set<pmMachine*>::iterator lIter;
	for(lIter = pMachines.begin(); lIter != pMachines.end(); ++lIter)
	{
		pmMachine* lMachine = *lIter;

		if(lMachine != PM_LOCAL_MACHINE)
		{
			pmCommunicatorCommand::remoteTaskAssignPacked* lPackedData = new pmCommunicatorCommand::remoteTaskAssignPacked(pLocalTask);

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT_TAG, lMachine, pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED,	lPackedData, 1, NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::SendTaskFinishToMachines(pmLocalTask* pLocalTask)
{
	std::vector<pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();
	std::set<pmMachine*> lMachines;

	pmProcessingElement::GetMachines(lDevices, lMachines);

    // Task master host must always be on the list even if none of it's devices were used in execution
    if(lMachines.find(PM_LOCAL_MACHINE) == lMachines.end())
        lMachines.insert(PM_LOCAL_MACHINE);

	std::set<pmMachine*>::iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		pmMachine* lMachine = *lIter;

		if(lMachine == PM_LOCAL_MACHINE)
        {
            TaskFinishEvent(pLocalTask);
        }
        else
		{
			pmCommunicatorCommand::taskEventStruct* lTaskEventData = new pmCommunicatorCommand::taskEventStruct();
			lTaskEventData->taskEvent = (uint)(pmCommunicatorCommand::TASK_FINISH_EVENT);
			lTaskEventData->originatingHost = *(pLocalTask->GetOriginatingHost());
			lTaskEventData->sequenceNumber = pLocalTask->GetSequenceNumber();

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::TASK_EVENT_TAG, lMachine, pmCommunicatorCommand::TASK_EVENT_STRUCT, lTaskEventData, 1, NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}

	return pmSuccess;
}
    
pmStatus pmScheduler::SendTaskCompleteToTaskOwner(pmTask* pTask)
{
    pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

    if(lOriginatingHost == PM_LOCAL_MACHINE)
    {
        TaskCompleteEvent((pmLocalTask*)pTask);
    }
    else
    {
        pmCommunicatorCommand::taskEventStruct* lTaskEventData = new pmCommunicatorCommand::taskEventStruct();
        lTaskEventData->taskEvent = (uint)(pmCommunicatorCommand::TASK_COMPLETE_EVENT);
        lTaskEventData->originatingHost = *(pTask->GetOriginatingHost());
        lTaskEventData->sequenceNumber = pTask->GetSequenceNumber();

        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::TASK_EVENT_TAG, lOriginatingHost, pmCommunicatorCommand::TASK_EVENT_STRUCT, lTaskEventData, 1, NULL, 0, gCommandCompletionCallback);

        pmCommunicator::GetCommunicator()->Send(lCommand, false);
    }

	return pmSuccess;
}

pmStatus pmScheduler::CancelTask(pmLocalTask* pLocalTask)
{
	std::vector<pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();
	std::set<pmMachine*> lMachines;

	pmProcessingElement::GetMachines(lDevices, lMachines);

	std::set<pmMachine*>::iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		pmMachine* lMachine = *lIter;

		if(lMachine == PM_LOCAL_MACHINE)
		{
			return TaskCancelEvent(pLocalTask);
		}
		else
		{
			// Send task cancel message to remote machines
			pmCommunicatorCommand::taskEventStruct* lTaskEventData = new pmCommunicatorCommand::taskEventStruct();
			lTaskEventData->taskEvent = (uint)(pmCommunicatorCommand::TASK_CANCEL_EVENT);
			lTaskEventData->originatingHost = *(pLocalTask->GetOriginatingHost());
			lTaskEventData->sequenceNumber = pLocalTask->GetSequenceNumber();

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::TASK_EVENT_TAG, lMachine, pmCommunicatorCommand::TASK_EVENT_STRUCT, lTaskEventData, 1, NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::StartLocalTaskExecution(pmLocalTask* pLocalTask)
{
	/* Steps -
	   1. Find candidate processing elements where this task can be executed (By comparing provided callbacks with devices available on each machine, also use cluster value) - Query Local Device Pool
	   2. Find current load on each of these machines from tasks of current or higher priority - Use latest value in the local device pool
	   3. Decide upon the initial set of processing elements where execution may start. Heavily loaded processing elements may be omitted initially and may participate later when load moderates
	   4. Depending upon the scheduling policy (push or pull), execute the task
	// For now, omit steps 2 and 3 - These are anyway not required for PULL scheduling policy
	 */

    ulong lTriggerTime = pLocalTask->GetTaskTimeOutTriggerTime();
    ulong lCurrentTime = GetIntegralCurrentTimeInSecs();

    if(lCurrentTime >= lTriggerTime)
        PMTHROW(pmFatalErrorException());   // Throw task TIMEDOUT from here
    
    pmTimedEventManager::GetTimedEventManager()->AddTaskTimeOutEvent(pLocalTask, lTriggerTime);
    
	std::set<pmProcessingElement*> lDevices;
	pLocalTask->FindCandidateProcessingElements(lDevices);
    
    if(lDevices.empty())
    {
        pLocalTask->MarkUserSideTaskCompletion();
        return pmNoCompatibleDevice;
    }

	pLocalTask->InitializeSubtaskManager(pLocalTask->GetSchedulingModel());

	std::set<pmMachine*> lMachines;
	pmProcessingElement::GetMachines(lDevices, lMachines);
	AssignTaskToMachines(pLocalTask, lMachines);

	AssignSubtasksToDevices(pLocalTask);

	return pmSuccess;
}

pmStatus pmScheduler::PushToStub(pmProcessingElement* pDevice, pmSubtaskRange& pRange)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pDevice);

	if(!lStub)
		PMTHROW(pmFatalErrorException());

	return lStub->Push(pRange);
}

pmProcessingElement* pmScheduler::RandomlySelectStealTarget(pmProcessingElement* pStealingDevice, pmTask* pTask)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

	if(!lStub)
		PMTHROW(pmFatalErrorException());

	pmTaskExecStats& lTaskExecStats = pTask->GetTaskExecStats();

	uint lAttempts = lTaskExecStats.GetStealAttempts(lStub);
    uint lDevices = pTask->GetAssignedDeviceCount();
	if((lAttempts >= lDevices * MAX_STEAL_CYCLES_PER_DEVICE) || (lTaskExecStats.GetFailedStealAttemptsSinceLastSuccessfulAttempt(lStub) >= lDevices))
		return NULL;

	lTaskExecStats.RecordStealAttempt(lStub);

    return (pTask->GetStealListForDevice(pStealingDevice))[lAttempts % lDevices];
}

pmStatus pmScheduler::StealSubtasks(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
	pmProcessingElement* lTargetDevice = RandomlySelectStealTarget(pStealingDevice, pTask);
    if(lTargetDevice == pStealingDevice)
        lTargetDevice = RandomlySelectStealTarget(pStealingDevice, pTask);
                          
	if(lTargetDevice)
	{
        STEAL_REQUEST_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(lTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), lTargetDevice->GetGlobalDeviceIndex(), pExecutionRate);
    
		pmMachine* lTargetMachine = lTargetDevice->GetMachine();

		if(lTargetMachine == PM_LOCAL_MACHINE)
		{
            if(lTargetDevice == pStealingDevice)
                return StealFailedReturnEvent(pStealingDevice, lTargetDevice, pTask);

			return StealProcessEvent(pStealingDevice, lTargetDevice, pTask, pExecutionRate);
		}
		else
		{
			pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

			pmCommunicatorCommand::stealRequestStruct* lStealRequestData = new pmCommunicatorCommand::stealRequestStruct();
			lStealRequestData->stealingDeviceGlobalIndex = pStealingDevice->GetGlobalDeviceIndex();
			lStealRequestData->targetDeviceGlobalIndex = lTargetDevice->GetGlobalDeviceIndex();
			lStealRequestData->originatingHost = *lOriginatingHost;
			lStealRequestData->sequenceNumber = pTask->GetSequenceNumber();
			lStealRequestData->stealingDeviceExecutionRate = pExecutionRate;

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_REQUEST_TAG, lTargetMachine, pmCommunicatorCommand::STEAL_REQUEST_STRUCT, lStealRequestData, 1, NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::ServeStealRequest(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate)
{
	return pTargetDevice->GetLocalExecutionStub()->StealSubtasks(pTask, pStealingDevice, pExecutionRate);
}

pmStatus pmScheduler::SendStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
    STEAL_RESPONSE_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(pTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), pRange.task->GetTaskExecStats().GetStubExecutionRate(pmStubManager::GetStubManager()->GetStub(pTargetDevice)), pRange.endSubtask - pRange.startSubtask + 1);

	pmMachine* lMachine = pStealingDevice->GetMachine();
	if(lMachine == PM_LOCAL_MACHINE)
	{
		return StealSuccessReturnEvent(pStealingDevice, pTargetDevice, pRange);
	}
	else
	{
		pmTask* lTask = pRange.task;
		pmMachine* lOriginatingHost = lTask->GetOriginatingHost();

		pmCommunicatorCommand::stealResponseStruct* lStealResponseData = new pmCommunicatorCommand::stealResponseStruct();
		lStealResponseData->stealingDeviceGlobalIndex = pStealingDevice->GetGlobalDeviceIndex();
		lStealResponseData->targetDeviceGlobalIndex = pTargetDevice->GetGlobalDeviceIndex();
		lStealResponseData->originatingHost = *lOriginatingHost;
        lStealResponseData->sequenceNumber = lTask->GetSequenceNumber();
		lStealResponseData->success = (ushort)(pmCommunicatorCommand::STEAL_SUCCESS_RESPONSE);
		lStealResponseData->startSubtask = pRange.startSubtask;
		lStealResponseData->endSubtask = pRange.endSubtask;
        lStealResponseData->originalAllotteeGlobalIndex = (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pStealingDevice->GetGlobalDeviceIndex());

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_RESPONSE_TAG, lMachine, pmCommunicatorCommand::STEAL_RESPONSE_STRUCT, lStealResponseData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

	return pmSuccess;
}

pmStatus pmScheduler::ReceiveStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange)
{
	pmTaskExecStats& lTaskExecStats = pRange.task->GetTaskExecStats();
	lTaskExecStats.RecordSuccessfulStealAttempt(pmStubManager::GetStubManager()->GetStub(pStealingDevice));

	return PushEvent(pStealingDevice, pRange);
}

pmStatus pmScheduler::SendFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
    STEAL_RESPONSE_DUMP((uint)(*(pStealingDevice->GetMachine())), (uint)(*(pTargetDevice->GetMachine())), pStealingDevice->GetGlobalDeviceIndex(), pTargetDevice->GetGlobalDeviceIndex(), pTask->GetTaskExecStats().GetStubExecutionRate(pmStubManager::GetStubManager()->GetStub(pTargetDevice)), 0);

	pmMachine* lMachine = pStealingDevice->GetMachine();
	if(lMachine == PM_LOCAL_MACHINE)
	{
		return StealFailedReturnEvent(pStealingDevice, pTargetDevice, pTask);
	}
	else
	{
		pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

		pmCommunicatorCommand::stealResponseStruct* lStealResponseData = new pmCommunicatorCommand::stealResponseStruct();
		lStealResponseData->stealingDeviceGlobalIndex = pStealingDevice->GetGlobalDeviceIndex();
		lStealResponseData->targetDeviceGlobalIndex = pTargetDevice->GetGlobalDeviceIndex();
		lStealResponseData->originatingHost = *lOriginatingHost;
        lStealResponseData->sequenceNumber = pTask->GetSequenceNumber();
		lStealResponseData->success = (ushort)(pmCommunicatorCommand::STEAL_FAILURE_RESPONSE);
		lStealResponseData->startSubtask = 0;	// dummy value
		lStealResponseData->endSubtask = 0;		// dummy value
        lStealResponseData->originalAllotteeGlobalIndex = 0;    // dummy value

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_RESPONSE_TAG, lMachine, pmCommunicatorCommand::STEAL_RESPONSE_STRUCT, lStealResponseData, 1, NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

	return pmSuccess;
}

pmStatus pmScheduler::ReceiveFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

	if(!lStub)
		PMTHROW(pmFatalErrorException());

	pmTaskExecStats& lTaskExecStats = pTask->GetTaskExecStats();
	lTaskExecStats.RecordFailedStealAttempt(pmStubManager::GetStubManager()->GetStub(pStealingDevice));

    return StealRequestEvent(pStealingDevice, pTask, lTaskExecStats.GetStubExecutionRate(lStub));
}


// This method is executed at master host for the task
pmStatus pmScheduler::ProcessAcknowledgement(pmLocalTask* pLocalTask, pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, pmCommunicatorCommand::ownershipDataStruct* pOwnershipData, uint pDataElements)
{
    pmMemSection* lMemSection = pRange.task->GetMemSectionRW();
    if(lMemSection)
    {
        pmMemSection::vmRangeOwner lRangeOwner;
        lRangeOwner.memIdentifier.memOwnerHost = *(lMemSection->GetMemOwnerHost());
        lRangeOwner.memIdentifier.generationNumber = lMemSection->GetGenerationNumber();
        
        for(size_t i=0; i<pDataElements; ++i)
        {
            lRangeOwner.host = pDevice->GetMachine();
            lRangeOwner.hostOffset = pOwnershipData[i].offset;
            lMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, pOwnershipData[i].offset, pOwnershipData[i].length);
        }
    }
    
    delete[] pOwnershipData;
    
	pmSubtaskManager* lSubtaskManager = pLocalTask->GetSubtaskManager();
	lSubtaskManager->RegisterSubtaskCompletion(pDevice, pRange.endSubtask - pRange.startSubtask + 1, pRange.startSubtask, pExecStatus);

	if(lSubtaskManager->HasTaskFinished())
	{
		SendTaskFinishToMachines(pLocalTask);
	}
	else
	{
		if(pLocalTask->GetSchedulingModel() == PUSH)
			return AssignSubtasksToDevice(pDevice, pLocalTask);
	}

	return pmSuccess;
}

pmStatus pmScheduler::SendAcknowledment(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, std::map<size_t, size_t>& pOwnershipMap)
{
    std::map<size_t, size_t> lEmptyMap;
	AcknowledgementSendEvent(pDevice, pRange, pExecStatus, (pRange.task->GetCallbackUnit()->GetDataRedistributionCB()) ? lEmptyMap : pOwnershipMap);
    
    if(pRange.task->GetOriginatingHost() != PM_LOCAL_MACHINE)
    {
        pmMemSection* lMemSection = pRange.task->GetMemSectionRW();
    
        if(lMemSection)
        {
            pmMemSection::vmRangeOwner lRangeOwner;
            lRangeOwner.host = PM_LOCAL_MACHINE;
            lRangeOwner.memIdentifier.memOwnerHost = *(lMemSection->GetMemOwnerHost());
            lRangeOwner.memIdentifier.generationNumber = lMemSection->GetGenerationNumber();
        
            std::map<size_t, size_t>::iterator lIter = pOwnershipMap.begin(), lEndIter = pOwnershipMap.end();
            for(; lIter != lEndIter; ++lIter)
            {
                lRangeOwner.hostOffset = lIter->first;
                lMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, lIter->first, lIter->second);
            }
        }
    }

	if(pRange.task->GetSchedulingModel() == PULL)
	{
		pmStubManager* lManager = pmStubManager::GetStubManager();
		pmExecutionStub* lStub = lManager->GetStub(pDevice);

		if(!lStub)
			PMTHROW(pmFatalErrorException());

		pmTaskExecStats& lTaskExecStats = pRange.task->GetTaskExecStats();
		return StealRequestEvent(pDevice, pRange.task, lTaskExecStats.GetStubExecutionRate(lStub));
	}

	return pmSuccess;
}
    
pmStatus pmScheduler::ClearPendingTaskCommands(pmTask* pTask)
{        
    return DeleteMatchingCommands(pTask->GetPriority(), taskClearMatchFunc, pTask);
}
    
pmStatus pmScheduler::WaitForAllCommandsToFinish()
{
    return WaitForQueuedCommands();
}

pmStatus pmScheduler::HandleCommandCompletion(pmCommandPtr pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::tr1::dynamic_pointer_cast<pmCommunicatorCommand>(pCommand);

	switch(lCommunicatorCommand->GetType())
	{
		case pmCommunicatorCommand::BROADCAST:
		{
			if(lCommunicatorCommand->GetTag() == pmCommunicatorCommand::HOST_FINALIZATION_TAG)
			{
                pmMachine* lMasterHost = pmMachinePool::GetMachinePool()->GetMachine(0);
                if(lMasterHost != PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());
                
				delete (pmCommunicatorCommand::hostFinalizationStruct*)(lCommunicatorCommand->GetData());

				pmController::GetController()->ProcessTermination();
			}

			break;
		}

		case pmCommunicatorCommand::SEND:
		{
			if(lCommunicatorCommand->GetTag() == pmCommunicatorCommand::SUBTASK_REDUCE_TAG)
			{
				pmCommunicatorCommand::subtaskReducePacked* lData = (pmCommunicatorCommand::subtaskReducePacked*)(lCommunicatorCommand->GetData());

				pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->reduceStruct.originatingHost);
				pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->reduceStruct.sequenceNumber);

                std::cout << "To Be Done: Pass Reducing Stub Here Or Comment these lines as shadow mem is an auto ptr" << std::endl;
                pmExecutionStub* lReducingStub = NULL;
				lTask->GetSubscriptionManager().DestroySubtaskShadowMem(lReducingStub, lData->reduceStruct.subtaskId);

                (static_cast<pmRemoteTask*>(lTask))->MarkReductionFinished();
			}
			else if(lCommunicatorCommand->GetTag() == pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG)
			{
				pmCommunicatorCommand::dataRedistributionPacked* lData = (pmCommunicatorCommand::dataRedistributionPacked*)(lCommunicatorCommand->GetData());
                
				pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.originatingHost);
				pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->redistributionStruct.sequenceNumber);
                
                if(lOriginatingHost == PM_LOCAL_MACHINE)
                    PMTHROW(pmFatalErrorException());

                (static_cast<pmRemoteTask*>(lTask))->MarkRedistributionFinished();
			}

			switch(lCommunicatorCommand->GetTag())
			{
				case pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT_TAG:
					delete (pmCommunicatorCommand::remoteTaskAssignPacked*)(lCommunicatorCommand->GetData());
					delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT_TAG:
					delete (pmCommunicatorCommand::remoteSubtaskAssignStruct*)(lCommunicatorCommand->GetData());
					break;
				case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG:
                    delete[] (pmCommunicatorCommand::ownershipDataStruct*)(((pmCommunicatorCommand::sendAcknowledgementPacked*)(lCommunicatorCommand->GetData()))->ownershipData);
					delete (pmCommunicatorCommand::sendAcknowledgementPacked*)(lCommunicatorCommand->GetData());
                    delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::TASK_EVENT_TAG:
					delete (pmCommunicatorCommand::taskEventStruct*)(lCommunicatorCommand->GetData());
					break;
				case pmCommunicatorCommand::STEAL_REQUEST_TAG:  // Steal Request Callback may be called after task deletion; do not de-reference task here
					delete (pmCommunicatorCommand::stealRequestStruct*)(lCommunicatorCommand->GetData());
					break;
				case pmCommunicatorCommand::STEAL_RESPONSE_TAG:  // Steal Request Callback may be called after task deletion; do not de-reference task here
					delete (pmCommunicatorCommand::stealResponseStruct*)(lCommunicatorCommand->GetData());
					break;
				case pmCommunicatorCommand::OWNERSHIP_TRANSFER_TAG:
					delete (pmCommunicatorCommand::ownershipTransferPacked*)(lCommunicatorCommand->GetData());
                    delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG:
					delete (pmCommunicatorCommand::memoryTransferRequest*)(lCommunicatorCommand->GetData());
					break;
				case pmCommunicatorCommand::MEMORY_RECEIVE_TAG:
					delete (pmCommunicatorCommand::memoryReceivePacked*)(lCommunicatorCommand->GetData());
					delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::SUBTASK_REDUCE_TAG:
					delete (pmCommunicatorCommand::subtaskReducePacked*)(lCommunicatorCommand->GetData());
					delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG:
					delete (pmCommunicatorCommand::dataRedistributionPacked*)(lCommunicatorCommand->GetData());
					delete[] (char*)(lCommunicatorCommand->GetSecondaryData());
					break;
				case pmCommunicatorCommand::HOST_FINALIZATION_TAG:
					delete (pmCommunicatorCommand::hostFinalizationStruct*)(lCommunicatorCommand->GetData());
					break;
                case pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_TAG:
                    delete (pmCommunicatorCommand::subtaskRangeCancelStruct*)(lCommunicatorCommand->GetData());
                    break;
				default:
					PMTHROW(pmFatalErrorException());
			}

			break;
		}

		case pmCommunicatorCommand::RECEIVE:
		{
			switch(lCommunicatorCommand->GetTag())
			{
				case pmCommunicatorCommand::MACHINE_POOL_TRANSFER_TAG:
				case pmCommunicatorCommand::DEVICE_POOL_TRANSFER_TAG:
				case pmCommunicatorCommand::UNKNOWN_LENGTH_TAG:
				case pmCommunicatorCommand::MAX_COMMUNICATOR_COMMAND_TAGS:
					PMTHROW(pmFatalErrorException());
					break;

				case pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT_TAG:
				{
					pmCommunicatorCommand::remoteTaskAssignPacked* lData = (pmCommunicatorCommand::remoteTaskAssignPacked*)(lCommunicatorCommand->GetData());

					pmTaskManager::GetTaskManager()->CreateRemoteTask(lData);

					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */
					delete[] (char*)(lData->taskConf.ptr);
					//delete[] (uint*)(lData->devices.ptr); // Freed inside destructor of remoteTaskAssignPacked class
					delete (pmCommunicatorCommand::remoteTaskAssignPacked*)(lData);

					break;
				}

				case pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT_TAG:
				{
					pmCommunicatorCommand::remoteSubtaskAssignStruct* lData = (pmCommunicatorCommand::remoteSubtaskAssignStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

                    pmSubtaskRange lRange;
                    lRange.startSubtask = lData->startSubtask;
                    lRange.endSubtask = lData->endSubtask;
                
                    if(lData->assignmentType == pmCommunicatorCommand::SUBTASK_ASSIGNMENT_REGULAR)
                    {
                        lRange.task = NULL;     // Will be set inside GetRemoteTaskOrEnqueueSubtasks
                        lRange.originalAllottee = ((lData->originalAllotteeGlobalIndex == lData->targetDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));

                        // Handling for out of order message receive (task received after subtask reception)
                        if(pmTaskManager::GetTaskManager()->GetRemoteTaskOrEnqueueSubtasks(lRange, lTargetDevice, lOriginatingHost, lData->sequenceNumber))
                            PushEvent(lTargetDevice, lRange);
                    }
                    else if(lData->assignmentType == pmCommunicatorCommand::RANGE_NEGOTIATION)
                    {
                        if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                        {
                            lRange.task = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                            lRange.originalAllottee = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex);
                        
                            RangeNegotiationEvent(lTargetDevice, lRange);
                        }
                    }
                    else if(lData->assignmentType == pmCommunicatorCommand::SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED)
                    {
                        lRange.task = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                        lRange.originalAllottee = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex);
                    
                        RangeNegotiationSuccessEvent(lTargetDevice, lRange);
                    }
                    
					SetupNewRemoteSubtaskReception();

					break;
				}

				case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG:
				{
					pmCommunicatorCommand::sendAcknowledgementPacked* lData = (pmCommunicatorCommand::sendAcknowledgementPacked*)(lCommunicatorCommand->GetData());
                    
					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->ackStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->ackStruct.sequenceNumber);
                    
                    if(lOriginatingHost != PM_LOCAL_MACHINE)
                        PMTHROW(pmFatalErrorException());
                
                    pmProcessingElement* lSourceDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.sourceDeviceGlobalIndex);
                    pmSubtaskRange lRange;
                    lRange.task = lTask;
                    lRange.startSubtask = lData->ackStruct.startSubtask;
                    lRange.endSubtask = lData->ackStruct.endSubtask;
                    lRange.originalAllottee = ((lData->ackStruct.originalAllotteeGlobalIndex == lData->ackStruct.sourceDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->ackStruct.originalAllotteeGlobalIndex));

                    AcknowledgementReceiveEvent(lSourceDevice, lRange, (pmStatus)(lData->ackStruct.execStatus), lData->ownershipData, lData->ackStruct.ownershipDataElements);
                
					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					//delete[] (pmCommunicatorCommand::ownershipDataStruct*)(lData->ownershipData);
					delete (pmCommunicatorCommand::sendAcknowledgementPacked*)(lData);
                    
					break;
				}

				case pmCommunicatorCommand::TASK_EVENT_TAG:
				{
					pmCommunicatorCommand::taskEventStruct* lData = (pmCommunicatorCommand::taskEventStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);

					switch((pmCommunicatorCommand::taskEvents)(lData->taskEvent))
					{
						case pmCommunicatorCommand::TASK_FINISH_EVENT:
						{
							TaskFinishEvent(lTask);
							break;
						}

						case pmCommunicatorCommand::TASK_COMPLETE_EVENT:
						{
                            if(dynamic_cast<pmLocalTask*>(lTask) == NULL)
                                PMTHROW(pmFatalErrorException());
                        
							TaskCompleteEvent(static_cast<pmLocalTask*>(lTask));
							break;
						}

						case pmCommunicatorCommand::TASK_CANCEL_EVENT:
						{
							TaskCancelEvent(lTask);
							break;
						}
                        
						default:
							PMTHROW(pmFatalErrorException());
					}

					SetupNewTaskEventReception();

					break;
				}

				case pmCommunicatorCommand::SUBTASK_REDUCE_TAG:
				{
					pmCommunicatorCommand::subtaskReducePacked* lData = (pmCommunicatorCommand::subtaskReducePacked*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->reduceStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->reduceStruct.sequenceNumber);

                    std::cout << "To Be Done: Handle non-contiguous subscriptions; find reducing stub (from reducer)" << std::endl;
					pmSubscriptionInfo lSubscriptionInfo;
					lSubscriptionInfo.offset = lData->reduceStruct.subscriptionOffset;
					lSubscriptionInfo.length = lData->reduceStruct.subtaskMemLength;
                
                    pmExecutionStub* lReducingStub = NULL;
					lTask->GetSubscriptionManager().RegisterSubscription(lReducingStub, lData->reduceStruct.subtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);

					lTask->GetSubscriptionManager().CreateSubtaskShadowMem(lReducingStub, lData->reduceStruct.subtaskId, (char*)(lData->subtaskMem.ptr), lData->subtaskMem.length);
					lTask->GetReducer()->AddSubtask(lReducingStub, lData->reduceStruct.subtaskId);

					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					delete[] (char*)(lData->subtaskMem.ptr);
					delete (pmCommunicatorCommand::subtaskReducePacked*)(lData);

					break;
				}

				case pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG:
				{
					pmCommunicatorCommand::dataRedistributionPacked* lData = (pmCommunicatorCommand::dataRedistributionPacked*)(lCommunicatorCommand->GetData());
                    
					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->redistributionStruct.sequenceNumber);
                    
                    if(lOriginatingHost != PM_LOCAL_MACHINE)
                        PMTHROW(pmFatalErrorException());

                    lTask->GetRedistributor()->PerformRedistribution(pmMachinePool::GetMachinePool()->GetMachine(lData->redistributionStruct.remoteHost), lData->redistributionStruct.subtasksAccounted, std::vector<pmCommunicatorCommand::redistributionOrderStruct>(lData->redistributionData, lData->redistributionData + lData->redistributionStruct.orderDataCount));
                    
					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					delete[] (pmCommunicatorCommand::redistributionOrderStruct*)(lData->redistributionData);
					delete (pmCommunicatorCommand::dataRedistributionPacked*)(lData);
                    
					break;
				}
                
				case pmCommunicatorCommand::MEMORY_RECEIVE_TAG:
				{
					pmCommunicatorCommand::memoryReceivePacked* lData = (pmCommunicatorCommand::memoryReceivePacked*)(lCommunicatorCommand->GetData());
					pmMemSection* lMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lData->receiveStruct.memOwnerHost), lData->receiveStruct.generationNumber);
                
					if(lMemSection)		// If memory still exists
						MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lMemSection, lData->receiveStruct.offset, lData->receiveStruct.length, lData->mem.ptr);

					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					delete[] (char*)(lData->mem.ptr);
					delete (pmCommunicatorCommand::memoryReceivePacked*)(lData);

					break;
				}

				case pmCommunicatorCommand::STEAL_REQUEST_TAG:
				{
					pmCommunicatorCommand::stealRequestStruct* lData = (pmCommunicatorCommand::stealRequestStruct*)(lCommunicatorCommand->GetData());

					pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
                    
                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                        StealProcessEvent(lStealingDevice, lTargetDevice, lTask, lData->stealingDeviceExecutionRate);
                    }
                    
					SetupNewStealRequestReception();

					break;
				}

				case pmCommunicatorCommand::STEAL_RESPONSE_TAG:
				{
					pmCommunicatorCommand::stealResponseStruct* lData = (pmCommunicatorCommand::stealResponseStruct*)(lCommunicatorCommand->GetData());

					pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);

                        pmCommunicatorCommand::stealResponseType lResponseType = (pmCommunicatorCommand::stealResponseType)(lData->success);

                        switch(lResponseType)
                        {
                            case pmCommunicatorCommand::STEAL_SUCCESS_RESPONSE:
                            {
                                pmSubtaskRange lRange;
                                lRange.task = lTask;
                                lRange.startSubtask = lData->startSubtask;
                                lRange.endSubtask = lData->endSubtask;
                                lRange.originalAllottee = ((lData->originalAllotteeGlobalIndex == lData->stealingDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));

                                StealSuccessReturnEvent(lStealingDevice, lTargetDevice, lRange);
                                break;
                            }

                            case pmCommunicatorCommand::STEAL_FAILURE_RESPONSE:
                            {
                                StealFailedReturnEvent(lStealingDevice, lTargetDevice, lTask);
                                break;
                            }

                            default:
                                PMTHROW(pmFatalErrorException());
                        }
                    }

					SetupNewStealResponseReception();

					break;
				}

				case pmCommunicatorCommand::OWNERSHIP_TRANSFER_TAG:
				{
					pmCommunicatorCommand::ownershipTransferPacked* lData = (pmCommunicatorCommand::ownershipTransferPacked*)(lCommunicatorCommand->GetData());
					pmMemSection* lMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lData->memIdentifier.memOwnerHost), lData->memIdentifier.generationNumber);
                
					if(!lMemSection)
                        PMTHROW(pmFatalErrorException());
                
                    std::vector<pmCommunicatorCommand::ownershipChangeStruct>* lChangeVector = lData->transferData.get();
                    std::vector<pmCommunicatorCommand::ownershipChangeStruct>::iterator lIter = lChangeVector->begin(), lEndIter = lChangeVector->end();
                    for(; lIter != lEndIter; ++lIter)
                        lMemSection->TransferOwnershipImmediate((*lIter).offset, (*lIter).length, pmMachinePool::GetMachinePool()->GetMachine((*lIter).newOwnerHost));

					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */
					delete (pmCommunicatorCommand::ownershipTransferPacked*)(lData);

					break;
				}
                
				case pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG:
				{
					pmCommunicatorCommand::memoryTransferRequest* lData = (pmCommunicatorCommand::memoryTransferRequest*)(lCommunicatorCommand->GetData());

					pmMemSection* lMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lData->sourceMemIdentifier.memOwnerHost), lData->sourceMemIdentifier.generationNumber);
					if(!lMemSection)
						PMTHROW(pmFatalErrorException());

					//MemTransferEvent(lMemSection, lData->destMemIdentifier, lData->offset, lData->length, pmMachinePool::GetMachinePool()->GetMachine(lData->destHost), lData->receiverOffset, lData->isForwarded, MAX_CONTROL_PRIORITY);

					SetupNewMemTransferRequestReception();

					break;
				}

				case pmCommunicatorCommand::HOST_FINALIZATION_TAG:
				{
					pmCommunicatorCommand::hostFinalizationStruct* lData = (pmCommunicatorCommand::hostFinalizationStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lMasterHost = pmMachinePool::GetMachinePool()->GetMachine(0);

					if(lData->terminate)
					{
						PMTHROW(pmFatalErrorException());
					}
					else
					{
						if(lMasterHost != PM_LOCAL_MACHINE)
							PMTHROW(pmFatalErrorException());

						pmController::GetController()->ProcessFinalization();
					}

					SetupNewHostFinalizationReception();

					break;
				}
                
                case pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_TAG:
                {
                    pmCommunicatorCommand::subtaskRangeCancelStruct* lData = (pmCommunicatorCommand::subtaskRangeCancelStruct*)(lCommunicatorCommand->GetData());
                
					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

                    if(pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(lOriginatingHost, lData->sequenceNumber))
                    {
                        pmSubtaskRange lRange;
                        lRange.task = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, lData->sequenceNumber);
                        lRange.startSubtask = lData->startSubtask;
                        lRange.endSubtask = lData->endSubtask;
                        lRange.originalAllottee = ((lData->originalAllotteeGlobalIndex == lData->targetDeviceGlobalIndex) ? NULL : pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->originalAllotteeGlobalIndex));
                        
                        pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);
                        RangeCancellationEvent(lTargetDevice, lRange);
                    }
                    
                    SetupNewSubtaskRangeCancelReception();
                
                    break;
                }

				default:
					PMTHROW(pmFatalErrorException());
			}

			break;
		}

		default:
			PMTHROW(pmFatalErrorException());
	}

	return pmSuccess;
}
    
bool taskClearMatchFunc(schedulerEvent& pEvent, void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case SUBTASK_EXECUTION:
        {
            if(pEvent.execDetails.range.task == (pmTask*)pCriterion)
                return true;
        
            break;
        }
        
        case STEAL_REQUEST_STEALER:
        {
            if(pEvent.stealRequestDetails.task == (pmTask*)pCriterion)
                return true;
            
            break;
        }

        case STEAL_PROCESS_TARGET:
        {
            if(pEvent.stealProcessDetails.task == (pmTask*)pCriterion)
                return true;

            break;
        }
            
        case STEAL_SUCCESS_TARGET:
        {
            if(pEvent.stealSuccessTargetDetails.range.task == (pmTask*)pCriterion)
                return true;

            break;
        }
            
        case STEAL_FAIL_TARGET:
        {
            if(pEvent.stealFailTargetDetails.task == (pmTask*)pCriterion)
                return true;

            break;
        }
            
        case STEAL_SUCCESS_STEALER:
        {
            if(pEvent.stealSuccessStealerDetails.range.task == (pmTask*)pCriterion)
                return true;

            break;
        }
            
        case STEAL_FAIL_STEALER:
        {
            if(pEvent.stealFailStealerDetails.task == (pmTask*)pCriterion)
                return true;

            break;
        }
        
        case SEND_ACKNOWLEDGEMENT:
        {
            if(pEvent.ackSendDetails.range.task == (pmTask*)pCriterion)
                break;
        }
        
        case RECEIVE_ACKNOWLEDGEMENT:
        {        
            if(pEvent.ackReceiveDetails.range.task == (pmTask*)pCriterion)
                break;
        }
        
        case SUBTASK_RANGE_CANCEL:
        {
            if(pEvent.subtaskRangeCancelDetails.range.task == (pmTask*)pCriterion)
                return true;
            
            break;        
        }
        
        case RANGE_NEGOTIATION_EVENT:
        {
            if(pEvent.rangeNegotiationDetails.range.task == (pmTask*)pCriterion)
                return true;
            
            break;        
        }
        
        case RANGE_NEGOTIATION_SUCCESS_EVENT:
        {
            if(pEvent.rangeNegotiationSuccessDetails.negotiatedRange.task == (pmTask*)pCriterion)
                return true;
            
            break;        
        }
        
        default:
            return false;
   }
    
    return false;
}

} // end namespace pm




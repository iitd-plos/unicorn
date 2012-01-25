
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

namespace pm
{

pmStatus SchedulerCommandCompletionCallback(pmCommandPtr pCommand)
{
	pmScheduler* lScheduler = pmScheduler::GetScheduler();
	return lScheduler->CommandCompletionEvent(pCommand);
}

static pmCommandCompletionCallback gCommandCompletionCallback = SchedulerCommandCompletionCallback;

pmScheduler* pmScheduler::mScheduler = NULL;

pmScheduler::pmScheduler()
{
	mRemoteSubtaskRecvCommand = mAcknowledgementRecvCommand = mTaskEventRecvCommand = mStealRequestRecvCommand = mStealResponseRecvCommand = mMemSubscriptionRequestCommand = NULL;

	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT);

	SetupPersistentCommunicationCommands();

	SwitchThread(NULL);	// Create an infinite loop in a new thread
}

pmScheduler::~pmScheduler()
{
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::TASK_EVENT_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::STEAL_REQUEST_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::STEAL_RESPONSE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT);

	DestroyPersistentCommunicationCommands();
}

pmScheduler* pmScheduler::GetScheduler()
{
	if(!mScheduler)
		mScheduler = new pmScheduler();

	return mScheduler;
}

pmStatus pmScheduler::DestroyScheduler()
{
	delete mScheduler;
	mScheduler = NULL;

	return pmSuccess;
}

pmCommandCompletionCallback pmScheduler::GetUnknownLengthCommandCompletionCallback()
{
	return gCommandCompletionCallback;
}

pmStatus pmScheduler::SetupPersistentCommunicationCommands()
{
#define PERSISTENT_RECV_COMMAND(tag, structType, recvDataPtr, recvStruct) pmPersistentCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::RECEIVE, \
	pmCommunicatorCommand::tag, NULL, pmCommunicatorCommand::structType, recvDataPtr, sizeof(pmCommunicatorCommand::recvStruct), NULL, 0, gCommandCompletionCallback)

	pmCommunicatorCommand::remoteSubtaskAssignStruct* lSubtaskAssignRecvData;
	pmCommunicatorCommand::sendAcknowledgementStruct* lSendAckRecvData;
	pmCommunicatorCommand::taskEventStruct* lTaskEventRecvData;
	pmCommunicatorCommand::stealRequestStruct* lStealRequestRecvData;
	pmCommunicatorCommand::stealResponseStruct* lStealResponseRecvData;
	pmCommunicatorCommand::memorySubscriptionRequest* lMemSubscriptionRequestData;

	START_DESTROY_ON_EXCEPTION(dBlock);

	DESTROY_PTR_ON_EXCEPTION(dBlock, lSubtaskAssignRecvData, new pmCommunicatorCommand::remoteSubtaskAssignStruct());
	DESTROY_PTR_ON_EXCEPTION(dBlock, lSendAckRecvData, new pmCommunicatorCommand::sendAcknowledgementStruct());
	DESTROY_PTR_ON_EXCEPTION(dBlock, lTaskEventRecvData, new pmCommunicatorCommand::taskEventStruct());
	DESTROY_PTR_ON_EXCEPTION(dBlock, lStealRequestRecvData, new pmCommunicatorCommand::stealRequestStruct());
	DESTROY_PTR_ON_EXCEPTION(dBlock, lStealResponseRecvData, new pmCommunicatorCommand::stealResponseStruct());

	mRemoteSubtaskRecvCommand = PERSISTENT_RECV_COMMAND(REMOTE_SUBTASK_ASSIGNMENT, REMOTE_SUBTASK_ASSIGN_STRUCT, lSubtaskAssignRecvData, remoteSubtaskAssignStruct);
	mAcknowledgementRecvCommand = PERSISTENT_RECV_COMMAND(SEND_ACKNOWLEDGEMENT_TAG, SEND_ACKNOWLEDGEMENT_STRUCT, lSendAckRecvData, sendAcknowledgementStruct);
	mTaskEventRecvCommand = PERSISTENT_RECV_COMMAND(TASK_EVENT_TAG, TASK_EVENT_STRUCT, lTaskEventRecvData, taskEventStruct);
	mStealRequestRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_REQUEST_TAG, STEAL_REQUEST_STRUCT,	lStealRequestRecvData, stealRequestStruct);
	mStealResponseRecvCommand = PERSISTENT_RECV_COMMAND(STEAL_RESPONSE_TAG, STEAL_RESPONSE_STRUCT, lStealResponseRecvData, stealResponseStruct);
	mMemSubscriptionRequestCommand = PERSISTENT_RECV_COMMAND(MEMORY_SUBSCRIPTION_TAG, MEMORY_SUBSCRIPTION_STRUCT, lMemSubscriptionRequestData, memorySubscriptionRequest);
	
	SetupNewRemoteSubtaskReception();
	SetupNewAcknowledgementReception();
	SetupNewTaskEventReception();
	SetupNewStealRequestReception();
	SetupNewStealResponseReception();
	SetupNewMemSubscriptionRequestReception();

	END_DESTROY_ON_EXCEPTION(dBlock);
}

pmStatus pmScheduler::DestroyPersistentCommunicationCommands()
{
	delete mRemoteSubtaskRecvCommand->GetData();
	delete mAcknowledgementRecvCommand->GetData();
	delete mTaskEventRecvCommand->GetData();
	delete mStealRequestRecvCommand->GetData();
	delete mStealResponseRecvCommand->GetData();
	delete mMemSubscriptionRequestCommand->GetData();
}

pmStatus pmScheduler::SetupNewRemoteSubtaskReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mRemoteSubtaskRecvCommand, false);
}

pmStatus pmScheduler::SetupNewAcknowledgementReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mAcknowledgementRecvCommand, false);
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

pmStatus pmScheduler::SetupNewMemSubscriptionRequestReception()
{
	return pmCommunicator::GetCommunicator()->Receive(mMemSubscriptionRequestCommand, false);
}

pmStatus pmScheduler::SubmitTaskEvent(pmLocalTask* pLocalTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = NEW_SUBMISSION;
	lEvent.submissionDetails.localTask = pLocalTask;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pLocalTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::PushEvent(pmProcessingElement* pDevice, pmScheduler::subtaskRange& pRange)
{
	schedulerEvent lEvent;
	lEvent.eventId = SUBTASK_EXECUTION;
	lEvent.execDetails.device = pDevice;
	lEvent.execDetails.range = pRange;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pRange.task->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealRequestEvent(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_REQUEST_STEALER;
	lEvent.stealRequestDetails.stealingDevice = pStealingDevice;
	lEvent.stealRequestDetails.task = pTask;
	lEvent.stealRequestDetails.stealingDeviceExecutionRate = pExecutionRate;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealProcessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_PROCESS_TARGET;
	lEvent.stealProcessDetails.stealingDevice = pStealingDevice;
	lEvent.stealProcessDetails.targetDevice = pTargetDevice;
	lEvent.stealProcessDetails.task = pTask;
	lEvent.stealProcessDetails.stealingDeviceExecutionRate = pExecutionRate;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealSuccessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, subtaskRange pRange)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_SUCCESS_TARGET;
	lEvent.stealSuccessTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealSuccessTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealSuccessTargetDetails.range = pRange;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pRange.task->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealFailedEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_FAIL_TARGET;
	lEvent.stealFailTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealFailTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealFailTargetDetails.task = pTask;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealSuccessReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, subtaskRange pRange)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_SUCCESS_STEALER;
	lEvent.stealSuccessTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealSuccessTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealSuccessTargetDetails.range = pRange;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pRange.task->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::StealFailedReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = STEAL_FAIL_STEALER;
	lEvent.stealFailTargetDetails.stealingDevice = pStealingDevice;
	lEvent.stealFailTargetDetails.targetDevice = pTargetDevice;
	lEvent.stealFailTargetDetails.task = pTask;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::AcknowledgementSendEvent(pmProcessingElement* pDevice, subtaskRange pRange, pmStatus pExecStatus)
{
	schedulerEvent lEvent;
	lEvent.eventId = SEND_ACKNOWLEDGEMENT;
	lEvent.ackSendDetails.device = pDevice;
	lEvent.ackSendDetails.range = pRange;
	lEvent.ackSendDetails.execStatus = pExecStatus;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pRange.task->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::AcknowledgementReceiveEvent(pmProcessingElement* pDevice, subtaskRange pRange, pmStatus pExecStatus)
{
	schedulerEvent lEvent;
	lEvent.eventId = RECEIVE_ACKNOWLEDGEMENT;
	lEvent.ackReceiveDetails.device = pDevice;
	lEvent.ackReceiveDetails.range = pRange;
	lEvent.ackReceiveDetails.execStatus = pExecStatus;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pRange.task->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::TaskCancelEvent(pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = TASK_CANCEL;
	lEvent.taskCancelDetails.task = pTask;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::TaskFinishEvent(pmTask* pTask)
{
	schedulerEvent lEvent;
	lEvent.eventId = TASK_FINISH;
	lEvent.taskFinishDetails.task = pTask;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::ReduceRequestEvent(pmTask* pTask, pmMachine* pDestMachine, ulong pSubtaskId)
{
	schedulerEvent lEvent;
	lEvent.eventId = SUBTASK_REDUCE;
	lEvent.subtaskReduceDetails.task = pTask;
	lEvent.subtaskReduceDetails.machine = pDestMachine;
	lEvent.subtaskReduceDetails.subtaskId = pSubtaskId;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pTask->GetPriority());

	mSignalWait.Signal();

	return lStatus;	
}

pmStatus pmScheduler::MemTransferEvent(pmMemSection* pSrcMemSection, ulong pOffset, ulong pLength, pmMachine* pDestMachine, ulong pDestMemBaseAddr, ushort pPriority)
{
	schedulerEvent lEvent;
	lEvent.eventId = MEMORY_TRANSFER;
	lEvent.memTransferDetails.memSection = pSrcMemSection;
	lEvent.memTransferDetails.offset = pOffset;
	lEvent.memTransferDetails.length = pLength;
	lEvent.memTransferDetails.machine = pDestMachine;
	lEvent.memTransferDetails.destMemBaseAddr = pDestMemBaseAddr;
	lEvent.memTransferDetails.priority = pPriority;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pPriority);

	mSignalWait.Signal();

	return lStatus;		
}

pmStatus pmScheduler::CommandCompletionEvent(pmCommandPtr pCommand)
{
	schedulerEvent lEvent;
	lEvent.eventId = COMMAND_COMPLETION;
	lEvent.commandCompletionDetails.command = pCommand;

	pmStatus lStatus = mPriorityQueue.InsertItem(lEvent, pCommand->GetPriority());

	mSignalWait.Signal();

	return lStatus;	
}

pmStatus pmScheduler::ThreadSwitchCallback(pmThreadCommandPtr pCommand)
{
	while(1)
	{
		mSignalWait.Wait();

		while(mPriorityQueue.GetSize() != 0)
		{
			schedulerEvent lEvent;
			mPriorityQueue.GetTopItem(lEvent);

			ProcessEvent(lEvent);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::ProcessEvent(schedulerEvent& pEvent)
{
	switch(pEvent.eventId)
	{
		case NEW_SUBMISSION:	/* Comes from application thread */
		{
			pmLocalTask* lLocalTask = pEvent.submissionDetails.localTask;			
			return StartLocalTaskExecution(lLocalTask);
		}

		case SUBTASK_EXECUTION:	/* Comes from network thread or from scheduler thread for local submissions */
		{
			return PushToStub(pEvent.execDetails.device, pEvent.execDetails.range);
		}

		case STEAL_REQUEST_STEALER:	/* Comes from stub thread */
		{
			stealRequest& lRequest = pEvent.stealRequestDetails;
			return StealSubtasks(lRequest.stealingDevice, lRequest.task, lRequest.stealingDeviceExecutionRate);
		}

		case STEAL_PROCESS_TARGET:	/* Comes from netwrok thread */
		{
			stealProcess& lEventDetails = pEvent.stealProcessDetails;
			return ServeStealRequest(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task, lEventDetails.stealingDeviceExecutionRate);
		}

		case STEAL_SUCCESS_TARGET:	/* Comes from stub thread */
		{
			stealSuccessTarget& lEventDetails = pEvent.stealSuccessTargetDetails;
			return SendStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
		}

		case STEAL_FAIL_TARGET: /* Comes from stub thread */
		{
			stealFailTarget& lEventDetails = pEvent.stealFailTargetDetails;
			return SendFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
		}

		case STEAL_SUCCESS_STEALER: /* Comes from network thread */
		{
			stealSuccessStealer& lEventDetails = pEvent.stealSuccessStealerDetails;
			return ReceiveStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.range);
		}

		case STEAL_FAIL_STEALER: /* Comes from network thread */
		{
			stealFailStealer& lEventDetails = pEvent.stealFailStealerDetails;
			return ReceiveFailedStealResponse(lEventDetails.stealingDevice, lEventDetails.targetDevice, lEventDetails.task);
		}

		case SEND_ACKNOWLEDGEMENT:	/* Comes from stub thread */
		{
			sendAcknowledgement& lEventDetails = pEvent.ackSendDetails;

			pmTask* lTask = lEventDetails.range.task;
			pmMachine* lOriginatingHost = lTask->GetOriginatingHost();

			lTask->IncrementSubtasksExecuted(lEventDetails.range.endSubtask - lEventDetails.range.startSubtask + 1);

			if(lOriginatingHost == PM_LOCAL_MACHINE)
			{
				return AcknowledgementReceiveEvent(lEventDetails.device, lEventDetails.range, lEventDetails.execStatus);
			}
			else
			{
				pmCommunicatorCommand::sendAcknowledgementStruct* lAckData = new pmCommunicatorCommand::sendAcknowledgementStruct();
				lAckData->sourceDeviceGlobalIndex = lEventDetails.device->GetGlobalDeviceIndex();
				lAckData->originatingHost = *(lOriginatingHost);
				lAckData->internalTaskId = (ulong)lTask;
				lAckData->startSubtask = lEventDetails.range.startSubtask;
				lAckData->endSubtask = lEventDetails.range.endSubtask;
				lAckData->execStatus = (uint)(lEventDetails.execStatus);

				pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG, 
					lOriginatingHost, pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT, (void*)lAckData, 
					sizeof(pmCommunicatorCommand::sendAcknowledgementStruct), NULL, 0, gCommandCompletionCallback);

				pmCommunicator::GetCommunicator()->Send(lCommand, false);
			}

			break;
		}

		case RECEIVE_ACKNOWLEDGEMENT:
		{
			receiveAcknowledgement& lEventDetails = pEvent.ackReceiveDetails;
			return ProcessAcknowledgement((pmLocalTask*)(lEventDetails.range.task), lEventDetails.device, lEventDetails.range, lEventDetails.execStatus);
		}

		case TASK_CANCEL:
		{
			taskCancel& lEventDetails = pEvent.taskCancelDetails;
			pmTask* lTask = lEventDetails.task;

			std::vector<pmProcessingElement*>& lDevices = (dynamic_cast<pmLocalTask*>(lTask) != NULL) ? (((pmLocalTask*)lTask)->GetAssignedDevices()) : (((pmRemoteTask*)lTask)->GetAssignedDevices());
			size_t lCount = lDevices.size();
			pmStubManager* lManager = pmStubManager::GetStubManager();
			for(size_t i=0; i<lCount; ++i)
			{
				if(lDevices[i]->GetMachine() == PM_LOCAL_MACHINE)
				{
					pmExecutionStub* lStub = lManager->GetStub(lDevices[i]);

					if(!lStub)
						throw pmFatalErrorException();

					lStub->CancelSubtasks(lTask);
				}
			}

			break;
		}

		case TASK_FINISH:
		{
			taskFinish& lEventDetails = pEvent.taskFinishDetails;
			pmTask* lTask = lEventDetails.task;

			if(lTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
				throw pmFatalErrorException();	// On local machine, task is cleared when user explicitly deletes that

			((pmRemoteTask*)lTask)->MarkSubtaskExecutionFinished();

			break;
		}

		case SUBTASK_REDUCE:
		{
			subtaskReduce& lEventDetails = pEvent.subtaskReduceDetails;
			
			pmCommunicatorCommand::subtaskReducePacked* lPackedData = new pmCommunicatorCommand::subtaskReducePacked(lEventDetails.task, lEventDetails.subtaskId);

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.task->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::SUBTASK_REDUCE_TAG, 
				lEventDetails.machine, pmCommunicatorCommand::SUBTASK_REDUCE_PACKED, lPackedData, sizeof(lPackedData), NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);

			break;
		}

		case MEMORY_TRANSFER:
		{
			memTransfer& lEventDetails = pEvent.memTransferDetails;
			
			pmCommunicatorCommand::memoryReceivePacked* lPackedData = new pmCommunicatorCommand::memoryReceivePacked(lEventDetails.destMemBaseAddr, lEventDetails.offset, lEventDetails.length, (void*)((char*)(lEventDetails.memSection->GetMem()) + lEventDetails.offset));

			if(dynamic_cast<pmOutputMemSection*>(lEventDetails.memSection))
				lEventDetails.memSection->SetRangeOwner(lEventDetails.machine, lEventDetails.destMemBaseAddr, lEventDetails.offset, lEventDetails.length);

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.priority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_RECEIVE_TAG, 
				lEventDetails.machine, pmCommunicatorCommand::MEMORY_RECEIVE_PACKED, lPackedData, sizeof(lPackedData), NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);

			break;
		}

		case COMMAND_COMPLETION:
		{
			commandCompletion& lEventDetails = pEvent.commandCompletionDetails;

			HandleCommandCompletion(lEventDetails.command);

			break;
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::AssignSubtasksToDevice(pmProcessingElement* pDevice, pmLocalTask* pLocalTask)
{
	pmMachine* lMachine = pDevice->GetMachine();

	ulong lStartingSubtask, lSubtaskCount;
	pLocalTask->GetSubtaskManager()->AssignSubtasksToDevice(pDevice, lSubtaskCount, lStartingSubtask);

	if(lSubtaskCount == 0)
		return pmSuccess;

	if(lMachine == PM_LOCAL_MACHINE)
	{
		pmScheduler::subtaskRange lRange;
		lRange.task = pLocalTask;
		lRange.startSubtask = lStartingSubtask;
		lRange.endSubtask = lStartingSubtask + lSubtaskCount - 1;

		return PushEvent(pDevice, lRange);
	}
	else
	{
		pmCommunicatorCommand::remoteSubtaskAssignStruct* lSubtaskData = new pmCommunicatorCommand::remoteSubtaskAssignStruct();
		lSubtaskData->internalTaskId = (ulong)pLocalTask;
		lSubtaskData->startSubtask = lStartingSubtask;
		lSubtaskData->endSubtask = lStartingSubtask + lSubtaskCount - 1;
		lSubtaskData->originatingHost = *(pLocalTask->GetOriginatingHost());
		lSubtaskData->targetDeviceGlobalIndex = pDevice->GetGlobalDeviceIndex();

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT, 
			lMachine, pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT, (void*)lSubtaskData, sizeof(pmCommunicatorCommand::remoteSubtaskAssignStruct), NULL, 0, gCommandCompletionCallback);

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
		pmMachine* lMachine = lIter._Mynode()->_Myval;

		if(lMachine != PM_LOCAL_MACHINE)
		{
			pmCommunicatorCommand::remoteTaskAssignPacked* lPackedData = new pmCommunicatorCommand::remoteTaskAssignPacked(pLocalTask);

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT, 
				lMachine, pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED,	lPackedData, sizeof(lPackedData), NULL, 0, gCommandCompletionCallback);

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
	std::set<pmMachine*>::iterator lIter;
	for(lIter = lMachines.begin(); lIter != lMachines.end(); ++lIter)
	{
		pmMachine* lMachine = lIter._Mynode()->_Myval;

		if(lMachine != PM_LOCAL_MACHINE)
		{
			// Send task finish message to machine lMachine
			// No action is required on local machine

			pmCommunicatorCommand::taskEventStruct* lTaskEventData = new pmCommunicatorCommand::taskEventStruct();
			lTaskEventData->taskEvent = (uint)(pmCommunicatorCommand::TASK_FINISH_EVENT);
			lTaskEventData->originatingHost = *(pLocalTask->GetOriginatingHost());
			lTaskEventData->internalTaskId = (ulong)pLocalTask;

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::TASK_EVENT_TAG, 
				lMachine, pmCommunicatorCommand::TASK_EVENT_STRUCT, lTaskEventData, sizeof(pmCommunicatorCommand::taskEventStruct), NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
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
		pmMachine* lMachine = lIter._Mynode()->_Myval;

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
			lTaskEventData->internalTaskId = (ulong)pLocalTask;

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pLocalTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::TASK_EVENT_TAG, 
				lMachine, pmCommunicatorCommand::TASK_EVENT_STRUCT, lTaskEventData, sizeof(pmCommunicatorCommand::taskEventStruct), NULL, 0, gCommandCompletionCallback);

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

	std::set<pmProcessingElement*> lDevices;
	pLocalTask->FindCandidateProcessingElements(lDevices);

	pLocalTask->InitializeSubtaskManager(pLocalTask->GetSchedulingModel());

	std::set<pmMachine*> lMachines;
	pmProcessingElement::GetMachines(lDevices, lMachines);
	AssignTaskToMachines(pLocalTask, lMachines);

	AssignSubtasksToDevices(pLocalTask);

	return pmSuccess;
}

pmStatus pmScheduler::PushToStub(pmProcessingElement* pDevice, subtaskRange pRange)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pDevice);

	if(!lStub)
		throw pmFatalErrorException();

	return lStub->Push(pRange);
}

pmProcessingElement* pmScheduler::RandomlySelectStealTarget(pmProcessingElement* pStealingDevice, pmTask* pTask)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

	if(!lStub)
		throw pmFatalErrorException();

	pmTaskExecStats lTaskExecStats = pTask->GetTaskExecStats();
	
	uint lAttempts = lTaskExecStats.GetStealAttempts(lStub);
	if(lAttempts >= pTask->GetAssignedDeviceCount())
		return NULL;

	lTaskExecStats.RecordStealAttempt(lStub);

	std::vector<pmProcessingElement*>& lRandomizedDevices = (dynamic_cast<pmLocalTask*>(pTask) != NULL) ? (((pmLocalTask*)pTask)->GetAssignedDevices()) : (((pmRemoteTask*)pTask)->GetAssignedDevices());

	return lRandomizedDevices[lAttempts];
}

pmStatus pmScheduler::StealSubtasks(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate)
{
	pmProcessingElement* lTargetDevice = RandomlySelectStealTarget(pStealingDevice, pTask);
	if(lTargetDevice)
	{
		pmMachine* lTargetMachine = lTargetDevice->GetMachine();

		if(lTargetMachine == PM_LOCAL_MACHINE)
		{
			return StealProcessEvent(pStealingDevice, lTargetDevice, pTask, pExecutionRate);
		}
		else
		{
			pmMachine* lOriginatingHost = pTask->GetOriginatingHost();

			pmCommunicatorCommand::stealRequestStruct* lStealRequestData = new pmCommunicatorCommand::stealRequestStruct();
			lStealRequestData->stealingDeviceGlobalIndex = pStealingDevice->GetGlobalDeviceIndex();
			lStealRequestData->targetDeviceGlobalIndex = lTargetDevice->GetGlobalDeviceIndex();
			lStealRequestData->originatingHost = *lOriginatingHost;

			if(lOriginatingHost == PM_LOCAL_MACHINE)
				lStealRequestData->internalTaskId = (ulong)pTask;
			else
				lStealRequestData->internalTaskId = ((pmRemoteTask*)pTask)->GetInternalTaskId();

			lStealRequestData->stealingDeviceExecutionRate = pExecutionRate;

			pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_REQUEST_TAG, 
				lTargetMachine, pmCommunicatorCommand::STEAL_REQUEST_STRUCT, lStealRequestData, sizeof(pmCommunicatorCommand::stealRequestStruct), NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lCommand, false);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::ServeStealRequest(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate)
{
	subtaskRange lStolenRange;
	lStolenRange.task = pTask;

	return pTargetDevice->GetLocalExecutionStub()->StealSubtasks(pTask, pStealingDevice, pExecutionRate);
}

pmStatus pmScheduler::SendStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, subtaskRange& pRange)
{
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

		if(lOriginatingHost == PM_LOCAL_MACHINE)
			lStealResponseData->internalTaskId = (ulong)lTask;
		else
			lStealResponseData->internalTaskId = ((pmRemoteTask*)lTask)->GetInternalTaskId();

		lStealResponseData->success = (ushort)(pmCommunicatorCommand::STEAL_SUCCESS_RESPONSE);
		lStealResponseData->startSubtask = pRange.startSubtask;
		lStealResponseData->endSubtask = pRange.endSubtask;

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_RESPONSE_TAG, 
			lMachine, pmCommunicatorCommand::STEAL_RESPONSE_STRUCT, lStealResponseData, sizeof(pmCommunicatorCommand::stealResponseStruct), NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

	return pmSuccess;
}

pmStatus pmScheduler::ReceiveStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, subtaskRange& pRange)
{
	return PushEvent(pStealingDevice, pRange);
}

pmStatus pmScheduler::SendFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
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

		if(lOriginatingHost == PM_LOCAL_MACHINE)
			lStealResponseData->internalTaskId = (ulong)pTask;
		else
			lStealResponseData->internalTaskId = ((pmRemoteTask*)pTask)->GetInternalTaskId();

		lStealResponseData->success = (ushort)(pmCommunicatorCommand::STEAL_FAILURE_RESPONSE);
		lStealResponseData->startSubtask = 0;	// dummy value
		lStealResponseData->endSubtask = 0;		// dummy value

		pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(pTask->GetPriority(), pmCommunicatorCommand::SEND, pmCommunicatorCommand::STEAL_RESPONSE_TAG, 
			lMachine, pmCommunicatorCommand::STEAL_RESPONSE_STRUCT, lStealResponseData, sizeof(pmCommunicatorCommand::stealResponseStruct), NULL, 0, gCommandCompletionCallback);

		pmCommunicator::GetCommunicator()->Send(lCommand, false);
	}

	return pmSuccess;
}

pmStatus pmScheduler::ReceiveFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	pmExecutionStub* lStub = lManager->GetStub(pStealingDevice);

	if(!lStub)
		throw pmFatalErrorException();

	pmTaskExecStats lTaskExecStats = pTask->GetTaskExecStats();
	
	return StealSubtasks(pStealingDevice, pTask, lTaskExecStats.GetStubExecutionRate(lStub));
}


// This method is executed at master host for the task
pmStatus pmScheduler::ProcessAcknowledgement(pmLocalTask* pLocalTask, pmProcessingElement* pDevice, subtaskRange pRange, pmStatus pExecStatus)
{
	pmSubtaskManager* lSubtaskManager = pLocalTask->GetSubtaskManager();
	lSubtaskManager->RegisterSubtaskCompletion(pDevice, pRange.endSubtask - pRange.startSubtask + 1, pRange.startSubtask, pExecStatus);

	if(lSubtaskManager->HasTaskFinished())
	{
		SendTaskFinishToMachines(pLocalTask);
		return pLocalTask->MarkSubtaskExecutionFinished();
		//return pLocalTask->MarkTaskEnd(lSubtaskManager->GetTaskExecutionStatus());
	}
	else
	{
		if(pLocalTask->GetSchedulingModel() == pmScheduler::PUSH)
			return AssignSubtasksToDevice(pDevice, pLocalTask);
	}

	return pmSuccess;
}

pmStatus pmScheduler::SendAcknowledment(pmProcessingElement* pDevice, subtaskRange pRange, pmStatus pExecStatus)
{
	AcknowledgementSendEvent(pDevice, pRange, pExecStatus);

	if(pRange.task->GetSchedulingModel() == pmScheduler::PULL)
	{
		pmStubManager* lManager = pmStubManager::GetStubManager();
		pmExecutionStub* lStub = lManager->GetStub(pDevice);

		if(!lStub)
			throw pmFatalErrorException();

		pmTaskExecStats lTaskExecStats = pRange.task->GetTaskExecStats();
		lTaskExecStats.ClearStealAttempts(lStub);
		
		return StealRequestEvent(pDevice, pRange.task, lTaskExecStats.GetStubExecutionRate(lStub));
	}

	return pmSuccess;
}

pmStatus pmScheduler::HandleCommandCompletion(pmCommandPtr pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::tr1::dynamic_pointer_cast<pmCommunicatorCommand>(pCommand);

	switch(lCommunicatorCommand->GetType())
	{
		case pmCommunicatorCommand::SEND:
		{
			switch(lCommunicatorCommand->GetTag())
			{
				case pmCommunicatorCommand::SUBTASK_REDUCE_TAG:
				{
					pmCommunicatorCommand::subtaskReducePacked* lData = (pmCommunicatorCommand::subtaskReducePacked*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->reduceStruct.originatingHost);
					pmTask* lTask = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->reduceStruct.internalTaskId);

					lTask->DestroySubtaskShadowMem(lData->reduceStruct.subtaskId);

					delete lTask;

					break;
				}
			}

			delete lCommunicatorCommand->GetData();
		}

		case pmCommunicatorCommand::RECEIVE:
		{
			switch(lCommunicatorCommand->GetTag())
			{
				case pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT:
				{
					pmCommunicatorCommand::remoteTaskAssignPacked* lData = (pmCommunicatorCommand::remoteTaskAssignPacked*)(lCommunicatorCommand->GetData());

					pmRemoteTask* lRemoteTask = pmTaskManager::GetTaskManager()->CreateRemoteTask(lData);

					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */
					delete[] lData->taskConf.ptr;
					delete[] lData->devices.ptr;
					delete[] lData;

					break;
				}

				case pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGNMENT:
				{
					pmCommunicatorCommand::remoteSubtaskAssignStruct* lData = (pmCommunicatorCommand::remoteSubtaskAssignStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					pmScheduler::subtaskRange lRange;
					lRange.task = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->internalTaskId);
					lRange.startSubtask = lData->startSubtask;
					lRange.endSubtask = lData->endSubtask;

					PushEvent(lTargetDevice, lRange);
					SetupNewRemoteSubtaskReception();
			
					break;
				}

				case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG:
				{
					pmCommunicatorCommand::sendAcknowledgementStruct* lData = (pmCommunicatorCommand::sendAcknowledgementStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmProcessingElement* lSourceDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->sourceDeviceGlobalIndex);

					pmScheduler::subtaskRange lRange;
					lRange.task = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->internalTaskId);
					lRange.startSubtask = lData->startSubtask;
					lRange.endSubtask = lData->endSubtask;

					AcknowledgementReceiveEvent(lSourceDevice, lRange, (pmStatus)(lData->execStatus));
					SetupNewAcknowledgementReception();

					break;
				}

				case pmCommunicatorCommand::TASK_EVENT_STRUCT:
				{
					pmCommunicatorCommand::taskEventStruct* lData = (pmCommunicatorCommand::taskEventStruct*)(lCommunicatorCommand->GetData());

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					pmRemoteTask* lRemoteTask = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->internalTaskId);

					switch((pmCommunicatorCommand::taskEvents)(lData->taskEvent))
					{
						case pmCommunicatorCommand::TASK_FINISH_EVENT:
						{
							TaskFinishEvent(lRemoteTask);
							break;
						}

						case pmCommunicatorCommand::TASK_CANCEL_EVENT:
						{
							TaskCancelEvent(lRemoteTask);
							break;
						}

						default:
							throw pmFatalErrorException();
					}
								 
					SetupNewTaskEventReception();

					break;
				}

				case pmCommunicatorCommand::SUBTASK_REDUCE_TAG:
				{
					pmCommunicatorCommand::subtaskReducePacked* lData = (pmCommunicatorCommand::subtaskReducePacked*)(lCommunicatorCommand->GetData());

					pmTask* lTask;
					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->reduceStruct.originatingHost);
					
					if(lOriginatingHost == PM_LOCAL_MACHINE)
						lTask = (pmLocalTask*)(lData->reduceStruct.internalTaskId);
					else
						lTask = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->reduceStruct.internalTaskId);

					pmSubscriptionInfo lSubscriptionInfo;
					lSubscriptionInfo.offset = lData->reduceStruct.subscriptionOffset;
					lSubscriptionInfo.length = lData->reduceStruct.subtaskMemLength;
					lTask->GetSubscriptionManager().RegisterSubscription(lData->reduceStruct.subtaskId, false, lSubscriptionInfo);

					lTask->CreateSubtaskShadowMem(lData->reduceStruct.subtaskId, (char*)(lData->subtaskMem.ptr), lData->subtaskMem.length);
					lTask->GetReducer()->AddSubtask(lData->reduceStruct.subtaskId);
					
					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					delete[] lData->subtaskMem.ptr;
					delete[] lData;

					break;
				}

				case pmCommunicatorCommand::MEMORY_RECEIVE_TAG:
				{
					pmCommunicatorCommand::memoryReceivePacked* lData = (pmCommunicatorCommand::memoryReceivePacked*)(lCommunicatorCommand->GetData());

					void* lMem = (void*)(lData->receiveStruct.receivingMemBaseAddr);
					pmMemSection* lMemSection = pmMemSection::FindMemSection(lMem);
					
					if(lMemSection)		// If memory still exists
						MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lMem, lMemSection, lData->receiveStruct.offset, lData->receiveStruct.length, lData->mem.ptr);
				
					/* The allocations are done in pmNetwork in UnknownLengthReceiveThread */					
					delete[] lData->mem.ptr;
					delete[] lData;

					break;
				}

				case pmCommunicatorCommand::STEAL_REQUEST_STRUCT:
				{
					pmTask* lTask;
					pmCommunicatorCommand::stealRequestStruct* lData = (pmCommunicatorCommand::stealRequestStruct*)(lCommunicatorCommand->GetData());

					pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);
					
					if(lOriginatingHost == PM_LOCAL_MACHINE)
						lTask = (pmLocalTask*)(lData->internalTaskId);
					else
						lTask = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->internalTaskId);

					StealProcessEvent(lStealingDevice, lTargetDevice, lTask, lData->stealingDeviceExecutionRate);
					SetupNewStealRequestReception();

					break;
				}

				case pmCommunicatorCommand::STEAL_RESPONSE_STRUCT:
				{
					pmTask* lTask;
					pmCommunicatorCommand::stealResponseStruct* lData = (pmCommunicatorCommand::stealResponseStruct*)(lCommunicatorCommand->GetData());

					pmProcessingElement* lStealingDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->stealingDeviceGlobalIndex);
					pmProcessingElement* lTargetDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lData->targetDeviceGlobalIndex);

					pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lData->originatingHost);

					if(lOriginatingHost == PM_LOCAL_MACHINE)
						lTask = (pmLocalTask*)(lData->internalTaskId);
					else
						lTask = pmTaskManager::GetTaskManager()->FindRemoteTask(lOriginatingHost, lData->internalTaskId);

					pmCommunicatorCommand::stealResponseType lResponseType = (pmCommunicatorCommand::stealResponseType)(lData->success);

					switch(lResponseType)
					{
						case pmCommunicatorCommand::STEAL_SUCCESS_RESPONSE:
						{
							pmScheduler::subtaskRange lRange;
							lRange.task = lTask;
							lRange.startSubtask = lData->startSubtask;
							lRange.endSubtask = lData->endSubtask;

							StealSuccessReturnEvent(lStealingDevice, lTargetDevice, lRange);
							break;
						}

						case pmCommunicatorCommand::STEAL_FAILURE_RESPONSE:
						{
							StealFailedReturnEvent(lStealingDevice, lTargetDevice, lTask);
							break;
						}

						default:
							throw pmFatalErrorException();
					}

					SetupNewStealResponseReception();

					break;
				}

				case pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT:
				{
					pmCommunicatorCommand::memorySubscriptionRequest* lData = (pmCommunicatorCommand::memorySubscriptionRequest*)(lCommunicatorCommand->GetData());

					pmMemSection* lMemSection = pmMemSection::FindMemSection((void*)(lData->ownerBaseAddr));
					if(!lMemSection)
						throw pmFatalErrorException();

					MemTransferEvent(lMemSection, lData->offset, lData->length, pmMachinePool::GetMachinePool()->GetMachine(lData->destHost), lData->receiverBaseAddr, MAX_CONTROL_PRIORITY);

					SetupNewMemSubscriptionRequestReception();

					break;
				}

				default:
					throw pmFatalErrorException();
			}

			break;
		}

		default:
			throw pmFatalErrorException();
	}

	return pmSuccess;
}

} // end namespace pm




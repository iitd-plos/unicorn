
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

#ifndef __PM_SCHEDULER__
#define __PM_SCHEDULER__

#include "pmBase.h"
#include "pmThread.h"

#include <set>

namespace pm
{

class pmCommunicatorCommand;
class pmHardware;
class pmTask;
class pmProcessingElement;
class pmMemSection;
class pmExecutionStub;

namespace scheduler
{

typedef enum schedulingModel
{
	PUSH,	/* subtasks are pushed from originating task manager to all schedulers to all stubs */
	PULL,	/* subtasks are pulled by stubs from their scheduler which pull from originating task manager */
    STATIC_EQUAL,  /* subtasks are equally and statically divided among all stubs */
    STATIC_PROPORTIONAL  /* subtasks are proportionally (as defined in configuration file STATIC_PROP_CONF_FILE) and statically divided among all stubs */
} schedulingModel;

typedef enum pushStrategy
{
	/** Name derived from TCP slow start. Initially, small number of subtasks are assigned to each processing element. An
	 *  acknowledgement is sent back upon completion of the allotted subtasks. The second time twice the number of subtasks
	 *  are allotted as were allotted during first allocation. This process continues and the allotment number grows in powers
	 *  of two as each processing element finishes it's allotted tasks. This continues until an equilibrium is found or all
	 *  subtasks are executed.
	 */
	SLOW_START
} pushStrategy;

typedef enum pullStrategy
{
	/** A random proessing element is selected and subtasks are stolen from it. Firstly, a steal attempt is made locally
	 *  on the machine. If no processing element on the machine can provide subtasks, then a global steal request is made
	 */
	RANDOM_STEALING_LOCAL_FIRST
} pullStrategy;

typedef enum eventIdentifier
{
	NEW_SUBMISSION,
	SUBTASK_EXECUTION,
	STEAL_REQUEST_STEALER,	/* Steal request genearted at source */
	STEAL_PROCESS_TARGET,	/* Steal request sent to target */
	STEAL_SUCCESS_TARGET,	/* Target sends success on steal */
	STEAL_FAIL_TARGET,		/* Target rejects steal request */
	STEAL_SUCCESS_STEALER,	/* Source processes receipt of STEAL_SUCCESS_RESPONSE */
	STEAL_FAIL_STEALER,		/* Source processes receipt of STEAL_FAIL_RESPONSE */
	SEND_ACKNOWLEDGEMENT,
	RECEIVE_ACKNOWLEDGEMENT,
	TASK_CANCEL,
	TASK_FINISH,
    TASK_COMPLETE,
	SUBTASK_REDUCE,
	COMMAND_COMPLETION,
    HOST_FINALIZATION,
    SUBTASK_RANGE_CANCEL,
    REDISTRIBUTION_METADATA_EVENT,
    RANGE_NEGOTIATION_EVENT,
    RANGE_NEGOTIATION_SUCCESS_EVENT,
    TERMINATE_TASK
} eventIdentifier;

typedef struct taskSubmission
{
	pmLocalTask* localTask;
} taskSubmission;

typedef struct subtaskExec
{
	pmProcessingElement* device;
	pmSubtaskRange range;
} subtaskExec;

typedef struct stealRequest
{
	pmProcessingElement* stealingDevice;
	pmTask* task;
	double stealingDeviceExecutionRate;
} stealRequest;

typedef struct stealProcess
{
	pmProcessingElement* stealingDevice;
	pmProcessingElement* targetDevice;
	pmTask* task;
	double stealingDeviceExecutionRate;
} stealProcess;

typedef struct stealSuccessTarget
{
	pmProcessingElement* stealingDevice;
	pmProcessingElement* targetDevice;
	pmSubtaskRange range;
} stealSuccessTarget;

typedef struct stealFailTarget
{
	pmProcessingElement* stealingDevice;
	pmProcessingElement* targetDevice;
	pmTask* task;
} stealFailTarget;

typedef struct stealSuccessStealer
{
	pmProcessingElement* stealingDevice;
	pmProcessingElement* targetDevice;
	pmSubtaskRange range;
} stealSuccessStealer;

typedef struct stealFailStealer
{
	pmProcessingElement* stealingDevice;
	pmProcessingElement* targetDevice;
	pmTask* task;
} stealFailStealer;

typedef struct sendAcknowledgement
{
	pmProcessingElement* device;
	pmSubtaskRange range;
	pmStatus execStatus;
    pmCommunicatorCommand::ownershipDataStruct* ownershipData;    // Alternating offsets and lengths
    uint dataElements;
} sendAcknowledgement;

typedef struct receiveAcknowledgement
{
	pmProcessingElement* device;
	pmSubtaskRange range;
	pmStatus execStatus;
    pmCommunicatorCommand::ownershipDataStruct* ownershipData;    // Alternating offsets and lengths
    uint dataElements;
} receiveAcknowledgement;

typedef struct taskCancel
{
	pmTask* task;
} taskCancel;

typedef struct taskFinish
{
	pmTask* task;
} taskFinish;

typedef struct taskComplete
{
	pmLocalTask* localTask;
} taskComplete;

typedef struct taskTerminate
{
	pmTask* task;
} taskTerminate;

typedef struct subtaskReduce
{
	pmTask* task;
	pmMachine* machine;
    pmExecutionStub* reducingStub;
	ulong subtaskId;
} subtaskReduce;

typedef struct commandCompletion
{
	pmCommandPtr command;
} commandCompletion;
    
typedef struct hostFinalization
{
    bool terminate; // true for final termination; false for task submission freeze
} hostFinalization;
    
typedef struct subtaskRangeCancel
{
	pmProcessingElement* targetDevice;
    pmSubtaskRange range;
} subtaskRangeCancel;
    
typedef struct redistributionMetaData
{
    pmTask* task;
    std::vector<pmCommunicatorCommand::redistributionOrderStruct>* redistributionData;
    uint count;
} redistributionMetaData;
    
typedef struct rangeNegotiation
{
    pmProcessingElement* requestingDevice;
    pmSubtaskRange range;
} rangeNegotiation;

typedef struct rangeNegotiationSuccess
{
    pmProcessingElement* requestingDevice;
    pmSubtaskRange negotiatedRange;
} rangeNegotiationSuccess;

typedef struct schedulerEvent : public pmBasicThreadEvent
{
	eventIdentifier eventId;
	union
	{
		taskSubmission submissionDetails;
		subtaskExec execDetails;
		stealRequest stealRequestDetails;
		stealProcess stealProcessDetails;
		stealSuccessTarget stealSuccessTargetDetails;
		stealFailTarget stealFailTargetDetails;
		stealSuccessStealer stealSuccessStealerDetails;
		stealFailStealer stealFailStealerDetails;
		sendAcknowledgement ackSendDetails;
		receiveAcknowledgement ackReceiveDetails;
		taskCancel taskCancelDetails;
		taskFinish taskFinishDetails;
        taskComplete taskCompleteDetails;
        taskTerminate taskTerminateDetails;
		subtaskReduce subtaskReduceDetails;
        hostFinalization hostFinalizationDetails;
        subtaskRangeCancel subtaskRangeCancelDetails;
        redistributionMetaData redistributionMetaDataDetails;
        rangeNegotiation rangeNegotiationDetails;
        rangeNegotiationSuccess rangeNegotiationSuccessDetails;
	};

	commandCompletion commandCompletionDetails;	
} schedulerEvent;

}

/**
 * \brief This class schedules, load balances and executes all tasks on this machine.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmScheduler : public THREADING_IMPLEMENTATION_CLASS<scheduler::schedulerEvent>
{
	friend pmStatus SchedulerCommandCompletionCallback(pmCommandPtr pCommand);

	public:

		virtual ~pmScheduler();

		static pmScheduler* GetScheduler();

		pmStatus SendAcknowledment(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, std::map<size_t, size_t>& pOwnershipMap);
		pmStatus ProcessAcknowledgement(pmLocalTask* pLocalTask, pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, pmCommunicatorCommand::ownershipDataStruct* pOwnershipData, uint pDataElements);

		virtual pmStatus ThreadSwitchCallback(scheduler::schedulerEvent& pEvent);

		pmStatus SendFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus SendStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);

		pmStatus SubmitTaskEvent(pmLocalTask* pLocalTask);
		pmStatus PushEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange);		// subtask range execution event
		pmStatus StealRequestEvent(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);
		pmStatus StealProcessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate);
		pmStatus StealSuccessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);
		pmStatus StealFailedEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus StealSuccessReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);
		pmStatus StealFailedReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus AcknowledgementSendEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, std::map<size_t, size_t>& pOwnershipMap);
		pmStatus AcknowledgementReceiveEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange, pmStatus pExecStatus, pmCommunicatorCommand::ownershipDataStruct* pOwnershipData, uint pDataElements);
		pmStatus TaskCancelEvent(pmTask* pTask);
        pmStatus TaskFinishEvent(pmTask* pTask);
        pmStatus TaskCompleteEvent(pmLocalTask* pLocalTask);
		pmStatus ReduceRequestEvent(pmExecutionStub* pReducingStub, pmTask* pTask, pmMachine* pDestMachine, ulong pSubtaskId);
		pmStatus MemTransferEvent(pmMemSection* pSrcMemSection, pmCommunicatorCommand::memoryIdentifierStruct& pDestMemIdentifier, ulong pOffset, ulong pLength, pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority);
		pmStatus CommandCompletionEvent(pmCommandPtr pCommand);
        pmStatus RangeCancellationEvent(pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);
        pmStatus RedistributionMetaDataEvent(pmTask* pTask, std::vector<pmCommunicatorCommand::redistributionOrderStruct>* pRedistributionData, uint pCount);
        pmStatus RangeNegotiationEvent(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange);
        pmStatus RangeNegotiationSuccessEvent(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pNegotiatedRange);
        pmStatus TerminateTaskEvent(pmTask* pTask);

        pmStatus SendPostTaskOwnershipTransfer(pmMemSection* pMemSection, pmMachine* pReceiverHost, std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> >& pChangeData);
        pmStatus SendSubtaskRangeCancellationMessage(pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);
    
		pmStatus HandleCommandCompletion(pmCommandPtr pCommand);

        void CancelAllSubtasksExecutingOnLocalStubs(pmTask* pTask, bool pTaskListeningOnCancellation);
        void CommitShadowMemPendingOnAllStubs(pmTask* pTask);
		pmStatus CancelTask(pmLocalTask* pLocalTask);

		pmCommandCompletionCallback GetUnknownLengthCommandCompletionCallback();
    
        pmStatus WaitForAllCommandsToFinish();

        pmStatus SendFinalizationSignal();
		pmStatus BroadcastTerminationSignal();
    
        pmStatus NegotiateSubtaskRangeWithOriginalAllottee(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange);
        pmStatus SendRangeNegotiationSuccess(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pNegotiatedRange);
        pmStatus SendRedistributionData(pmTask* pTask, std::vector<pmCommunicatorCommand::redistributionOrderStruct>* pRedistributionData, uint pCount);

        pmStatus SendTaskCompleteToTaskOwner(pmTask* pTask);

    private:
		pmScheduler();

		pmStatus SetupPersistentCommunicationCommands();
		pmStatus DestroyPersistentCommunicationCommands();

		pmStatus SetupNewRemoteSubtaskReception();
		pmStatus SetupNewTaskEventReception();
		pmStatus SetupNewStealRequestReception();
		pmStatus SetupNewStealResponseReception();
        pmStatus SetupNewMemTransferRequestReception();
        pmStatus SetupNewHostFinalizationReception();
        pmStatus SetupNewSubtaskRangeCancelReception();
    
		pmStatus ProcessEvent(scheduler::schedulerEvent& pEvent);

		pmStatus AssignTaskToMachines(pmLocalTask* pLocalTask, std::set<pmMachine*>& pMachines);

		pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, pmLocalTask* pLocalTask);
		pmStatus AssignSubtasksToDevices(pmLocalTask* pLocalTask);

		pmStatus StartLocalTaskExecution(pmLocalTask* pLocalTask);

		pmStatus PushToStub(pmProcessingElement* pDevice, pmSubtaskRange& pRange);

		pmProcessingElement* RandomlySelectStealTarget(pmProcessingElement* pStealingDevice, pmTask* pTask);
		pmStatus StealSubtasks(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);

		pmStatus ServeStealRequest(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate);
		pmStatus ReceiveFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus ReceiveStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);

        pmStatus ClearPendingTaskCommands(pmTask* pTask);
        pmStatus SendTaskFinishToMachines(pmLocalTask* pLocalTask);

		pmCommunicatorCommandPtr mRemoteTaskCommand;
		pmCommunicatorCommandPtr mReduceSubtaskCommand;

		pmPersistentCommunicatorCommandPtr mRemoteSubtaskRecvCommand;
		pmPersistentCommunicatorCommandPtr mTaskEventRecvCommand;
		pmPersistentCommunicatorCommandPtr mStealRequestRecvCommand;
		pmPersistentCommunicatorCommandPtr mStealResponseRecvCommand;
        pmPersistentCommunicatorCommandPtr mMemTransferRequestCommand;
        pmPersistentCommunicatorCommandPtr mHostFinalizationCommand;
        pmPersistentCommunicatorCommandPtr mSubtaskRangeCancelCommand;

        pmCommunicatorCommand::remoteSubtaskAssignStruct mSubtaskAssignRecvData;
        pmCommunicatorCommand::taskEventStruct mTaskEventRecvData;
        pmCommunicatorCommand::stealRequestStruct mStealRequestRecvData;
        pmCommunicatorCommand::stealResponseStruct mStealResponseRecvData;
        pmCommunicatorCommand::memoryTransferRequest mMemTransferRequestData;
        pmCommunicatorCommand::hostFinalizationStruct mHostFinalizationData;
        pmCommunicatorCommand::subtaskRangeCancelStruct mSubtaskRangeCancelData;
    
#ifdef TRACK_SUBTASK_EXECUTION
        ulong mSubtasksAssigned;
        ulong mAcknowledgementsSent;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTrackLock;
#endif
};

bool taskClearMatchFunc(scheduler::schedulerEvent& pEvent, void* pCriterion);

} // end namespace pm

#endif

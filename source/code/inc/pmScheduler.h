
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

namespace scheduler
{

typedef enum schedulingModel
{
	PUSH,	/* subtasks are pushed from originating task manager to all schedulers to all stubs */
	PULL	/* subtasks are pulled by stubs from their scheduler which pull from originating task manager */
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
	SUBTASK_REDUCE,
	MEMORY_TRANSFER,
	COMMAND_COMPLETION
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
} sendAcknowledgement;

typedef struct receiveAcknowledgement
{
	pmProcessingElement* device;
	pmSubtaskRange range;
	pmStatus execStatus;
} receiveAcknowledgement;

typedef struct taskCancel
{
	pmTask* task;
} taskCancel;

typedef struct taskFinish
{
	pmTask* task;
} taskFinish;

typedef struct subtaskReduce
{
	pmTask* task;
	pmMachine* machine;
	ulong subtaskId;
} subtaskReduce;

typedef struct memTransfer
{
	pmMemSection* memSection;
	ulong offset;
	ulong length;
	pmMachine* machine;
	ulong destMemBaseAddr;
	ushort priority;
} memTransfer;

typedef struct commandCompletion
{
	pmCommandPtr command;
} commandCompletion;

typedef struct schedulerEvent
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
		subtaskReduce subtaskReduceDetails;
		memTransfer memTransferDetails;
	};

	// Can't make this part of union as C++ standard does not allow anything with non-trivial constructor/copy-constructor/assignment to be part of an union
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
		pmStatus DestroyScheduler();

		pmStatus SendAcknowledment(pmProcessingElement* pDevice, pmSubtaskRange pRange, pmStatus pExecStatus);
		pmStatus ProcessAcknowledgement(pmLocalTask* pLocalTask, pmProcessingElement* pDevice, pmSubtaskRange pRange, pmStatus pExecStatus);

		virtual pmStatus ThreadSwitchCallback(scheduler::schedulerEvent& pEvent);

		pmStatus SendFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus SendStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);

		pmStatus SubmitTaskEvent(pmLocalTask* pLocalTask);
		pmStatus PushEvent(pmProcessingElement* pDevice, pmSubtaskRange& pRange);		// subtask range execution event
		pmStatus StealRequestEvent(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);
		pmStatus StealProcessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate);
		pmStatus StealSuccessEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange pRange);
		pmStatus StealFailedEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus StealSuccessReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange pRange);
		pmStatus StealFailedReturnEvent(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus AcknowledgementSendEvent(pmProcessingElement* pDevice, pmSubtaskRange pRange, pmStatus pExecStatus);
		pmStatus AcknowledgementReceiveEvent(pmProcessingElement* pDevice, pmSubtaskRange pRange, pmStatus pExecStatus);
		pmStatus TaskCancelEvent(pmTask* pTask);
		pmStatus TaskFinishEvent(pmTask* pTask);
		pmStatus ReduceRequestEvent(pmTask* pTask, pmMachine* pDestMachine, ulong pSubtaskId);
		pmStatus MemTransferEvent(pmMemSection* pSrcMemSection, ulong pOffset, ulong pLength, pmMachine* pDestMachine, ulong pDestMemBaseAddr, ushort pPriority);
		pmStatus CommandCompletionEvent(pmCommandPtr pCommand);

		pmStatus HandleCommandCompletion(pmCommandPtr pCommand);

		pmStatus CancelTask(pmLocalTask* pLocalTask);

		pmCommandCompletionCallback GetUnknownLengthCommandCompletionCallback();

	private:
		pmScheduler();

		pmStatus SetupPersistentCommunicationCommands();
		pmStatus DestroyPersistentCommunicationCommands();

		pmStatus SetupNewRemoteSubtaskReception();
		pmStatus SetupNewAcknowledgementReception();
		pmStatus SetupNewTaskEventReception();
		pmStatus SetupNewStealRequestReception();
		pmStatus SetupNewStealResponseReception();
		pmStatus SetupNewMemSubscriptionRequestReception();

		pmStatus ProcessEvent(scheduler::schedulerEvent& pEvent);

		pmStatus AssignTaskToMachines(pmLocalTask* pLocalTask, std::set<pmMachine*>& pMachines);

		pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, pmLocalTask* pLocalTask);
		pmStatus AssignSubtasksToDevices(pmLocalTask* pLocalTask);

		pmStatus StartLocalTaskExecution(pmLocalTask* pLocalTask);

		pmStatus PushToStub(pmProcessingElement* pDevice, pmSubtaskRange pRange);

		pmProcessingElement* RandomlySelectStealTarget(pmProcessingElement* pStealingDevice, pmTask* pTask);
		pmStatus StealSubtasks(pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);

		pmStatus ServeStealRequest(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate);
		pmStatus ReceiveFailedStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmTask* pTask);
		pmStatus ReceiveStealResponse(pmProcessingElement* pStealingDevice, pmProcessingElement* pTargetDevice, pmSubtaskRange& pRange);

		pmStatus SendTaskFinishToMachines(pmLocalTask* pLocalTask);

		pmCommunicatorCommandPtr mRemoteTaskCommand;
		pmCommunicatorCommandPtr mReduceSubtaskCommand;

		pmPersistentCommunicatorCommandPtr mRemoteSubtaskRecvCommand;
		pmPersistentCommunicatorCommandPtr mAcknowledgementRecvCommand;
		pmPersistentCommunicatorCommandPtr mTaskEventRecvCommand;
		pmPersistentCommunicatorCommandPtr mStealRequestRecvCommand;
		pmPersistentCommunicatorCommandPtr mStealResponseRecvCommand;
		pmPersistentCommunicatorCommandPtr mMemSubscriptionRequestCommand;

		static pmScheduler* mScheduler;
};

} // end namespace pm

#endif

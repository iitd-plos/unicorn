
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

#ifndef __PM_SCHEDULER__
#define __PM_SCHEDULER__

#include "pmBase.h"
#include "pmThread.h"
#include "pmCommunicator.h"

#include <set>

namespace pm
{

class pmHardware;
class pmTask;
class pmProcessingElement;
class pmAddressSpace;
class pmExecutionStub;

namespace scheduler
{

enum schedulingModel
{
	PUSH,	/* subtasks are pushed from originating task manager to all schedulers to all stubs */
	PULL,	/* subtasks are pulled by stubs from their scheduler which pull from originating task manager */
    PULL_WITH_AFFINITY, /* subtasks are pulled by stubs from their scheduler which pull from originating task manager (but the initial distribution is affinity based) */
    STATIC_EQUAL,  /* subtasks are equally and statically divided among all stubs */
    STATIC_PROPORTIONAL,  /* subtasks are proportionally (as defined in configuration file STATIC_PROP_CONF_FILE) and statically divided among all stubs */
    STATIC_EQUAL_NODE   /* subtasks are equally and statically divided among all nodes */
};

enum pushStrategy
{
	/** Name derived from TCP slow start. Initially, small number of subtasks are assigned to each processing element. An
	 *  acknowledgement is sent back upon completion of the allotted subtasks. The second time twice the number of subtasks
	 *  are allotted as were allotted during first allocation. This process continues and the allotment number grows in powers
	 *  of two as each processing element finishes it's allotted tasks. This continues until an equilibrium is found or all
	 *  subtasks are executed.
	 */
	SLOW_START
};

enum pullStrategy
{
	/** A random proessing element is selected and subtasks are stolen from it. Firstly, a steal attempt is made locally
	 *  on the machine. If no processing element on the machine can provide subtasks, then a global steal request is made
	 */
	RANDOM_STEALING_LOCAL_FIRST
};

enum eventIdentifier
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
    NO_REDUCTION_REQD,
	COMMAND_COMPLETION,
    HOST_FINALIZATION,
    SUBTASK_RANGE_CANCEL,
    REDISTRIBUTION_METADATA_EVENT,
    REDISTRIBUTION_OFFSETS_EVENT,
    RANGE_NEGOTIATION_EVENT,
    RANGE_NEGOTIATION_SUCCESS_EVENT,
    TERMINATE_TASK,
    REDUCTION_TERMINATION_EVENT,
    AFFINITY_TRANSFER_EVENT,
#ifdef USE_AFFINITY_IN_STEAL
    SUBTASK_EXECUTION_DISCONTIGUOUS_STEAL,
    STEAL_SUCCESS_DISCONTIGUOUS_TARGET,
    STEAL_SUCCESS_DISCONTIGUOUS_STEALER,
#endif
    ALL_REDUCTIONS_DONE_EVENT,
    EXTERNAL_REDUCTION_FINISH_EVENT,
    MAX_SCHEDULER_EVENTS
};

struct schedulerEvent : public pmBasicThreadEvent
{
	eventIdentifier eventId;
    
    schedulerEvent(eventIdentifier pEventId = MAX_SCHEDULER_EVENTS)
    : eventId(pEventId)
    {}
};

struct taskSubmissionEvent : public schedulerEvent
{
	pmLocalTask* localTask;
    
    taskSubmissionEvent(eventIdentifier pEventId, pmLocalTask* pLocalTask)
    : schedulerEvent(pEventId)
    , localTask(pLocalTask)
    {}
};

struct subtaskExecEvent : public schedulerEvent
{
	const pmProcessingElement* device;
	const pmSubtaskRange range;
    bool isStealResponse;
    
    subtaskExecEvent(eventIdentifier pEventId, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, bool pIsStealResponse)
    : schedulerEvent(pEventId)
    , device(pDevice)
    , range(pRange)
    , isStealResponse(pIsStealResponse)
    {}
};

#ifdef USE_AFFINITY_IN_STEAL
struct subtaskExecDiscontiguousStealEvent : public schedulerEvent
{
    pmTask* task;
	const pmProcessingElement* device;
    std::vector<ulong> discontiguousStealData;
    
    subtaskExecDiscontiguousStealEvent(eventIdentifier pEventId, pmTask* pTask, const pmProcessingElement* pDevice, std::vector<ulong>&& pDiscontiguousStealData)
    : schedulerEvent(pEventId)
    , task(pTask)
    , device(pDevice)
    , discontiguousStealData(std::move(pDiscontiguousStealData))
    {}
};
#endif

struct stealRequestEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	pmTask* task;
	double stealingDeviceExecutionRate;
    
    stealRequestEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, pmTask* pTask, double pStealingDeviceExecutionRate)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , task(pTask)
    , stealingDeviceExecutionRate(pStealingDeviceExecutionRate)
    {}
};

struct stealProcessEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
	pmTask* task;
	double stealingDeviceExecutionRate;
    bool shouldMultiAssign;
    
    stealProcessEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask, double pStealingDeviceExecutionRate, bool pShouldMultiAssign)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , task(pTask)
    , stealingDeviceExecutionRate(pStealingDeviceExecutionRate)
    , shouldMultiAssign(pShouldMultiAssign)
    {}
};

struct stealSuccessTargetEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
	const pmSubtaskRange range;
    
    stealSuccessTargetEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , range(pRange)
    {}
};

#ifdef USE_AFFINITY_IN_STEAL
struct stealSuccessDiscontiguousTargetEvent : public schedulerEvent
{
    pmTask* task;
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
    std::vector<ulong> discontiguousStealData;
    
    stealSuccessDiscontiguousTargetEvent(eventIdentifier pEventId, pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
    : schedulerEvent(pEventId)
    , task(pTask)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , discontiguousStealData(std::move(pDiscontiguousStealData))
    {}
};
#endif

struct stealFailTargetEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
	pmTask* task;
    
    stealFailTargetEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , task(pTask)
    {}
};

struct stealSuccessStealerEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
	const pmSubtaskRange range;
    
    stealSuccessStealerEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , range(pRange)
    {}
};

#ifdef USE_AFFINITY_IN_STEAL
struct stealSuccessDiscontiguousStealerEvent : public schedulerEvent
{
    pmTask* task;
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
    std::vector<ulong> discontiguousStealData;
    
    stealSuccessDiscontiguousStealerEvent(eventIdentifier pEventId, pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData)
    : schedulerEvent(pEventId)
    , task(pTask)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , discontiguousStealData(std::move(pDiscontiguousStealData))
    {}
};
#endif

struct stealFailStealerEvent : public schedulerEvent
{
	const pmProcessingElement* stealingDevice;
	const pmProcessingElement* targetDevice;
	pmTask* task;
    
    stealFailStealerEvent(eventIdentifier pEventId, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask)
    : schedulerEvent(pEventId)
    , stealingDevice(pStealingDevice)
    , targetDevice(pTargetDevice)
    , task(pTask)
    {}
};

struct sendAcknowledgementEvent : public schedulerEvent
{
	const pmProcessingElement* device;
	const pmSubtaskRange range;
	pmStatus execStatus;
    std::vector<communicator::ownershipDataStruct> ownershipVector;
    std::vector<communicator::scatteredOwnershipDataStruct> scatteredOwnershipVector;
    std::vector<uint> addressSpaceIndexVector;
    ulong totalSplitCount;
    
    sendAcknowledgementEvent(eventIdentifier pEventId, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipData, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount)
    : schedulerEvent(pEventId)
    , device(pDevice)
    , range(pRange)
    , execStatus(pExecStatus)
    , ownershipVector(pOwnershipData)
    , addressSpaceIndexVector(pAddressSpaceIndexVector)
    , totalSplitCount(pTotalSplitCount)
    {}

    sendAcknowledgementEvent(eventIdentifier pEventId, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipData, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount)
    : schedulerEvent(pEventId)
    , device(pDevice)
    , range(pRange)
    , execStatus(pExecStatus)
    , scatteredOwnershipVector(pScatteredOwnershipData)
    , addressSpaceIndexVector(pAddressSpaceIndexVector)
    , totalSplitCount(pTotalSplitCount)
    {}
};

struct receiveAcknowledgementEvent : public schedulerEvent
{
    const pmProcessingElement* device;
    const pmSubtaskRange range;
    pmStatus execStatus;
    std::vector<communicator::ownershipDataStruct> ownershipVector;
    std::vector<communicator::scatteredOwnershipDataStruct> scatteredOwnershipVector;
    std::vector<uint> addressSpaceIndexVector;
    
    receiveAcknowledgementEvent(eventIdentifier pEventId, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipData, std::vector<uint>&& pAddressSpaceIndexVector)
    : schedulerEvent(pEventId)
    , device(pDevice)
    , range(pRange)
    , execStatus(pExecStatus)
    , ownershipVector(pOwnershipData)
    , addressSpaceIndexVector(pAddressSpaceIndexVector)
    {}

    receiveAcknowledgementEvent(eventIdentifier pEventId, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipData, std::vector<uint>&& pAddressSpaceIndexVector)
    : schedulerEvent(pEventId)
    , device(pDevice)
    , range(pRange)
    , execStatus(pExecStatus)
    , scatteredOwnershipVector(pScatteredOwnershipData)
    , addressSpaceIndexVector(pAddressSpaceIndexVector)
    {}
};

struct taskCancelEvent : public schedulerEvent
{
	pmTask* task;
    
    taskCancelEvent(eventIdentifier pEventId, pmTask* pTask)
    : schedulerEvent(pEventId)
    , task(pTask)
    {}
};

struct taskFinishEvent : public schedulerEvent
{
	pmTask* task;
    
    taskFinishEvent(eventIdentifier pEventId, pmTask* pTask)
    : schedulerEvent(pEventId)
    , task(pTask)
    {}
};

struct taskCompleteEvent : public schedulerEvent
{
	pmLocalTask* localTask;
    
    taskCompleteEvent(eventIdentifier pEventId, pmLocalTask* pLocalTask)
    : schedulerEvent(pEventId)
    , localTask(pLocalTask)
    {}
};

struct taskTerminateEvent : public schedulerEvent
{
	pmTask* task;
    
    taskTerminateEvent(eventIdentifier pEventId, pmTask* pTask)
    : schedulerEvent(pEventId)
    , task(pTask)
    {}
};
    
struct noReductionRequiredEvent : public schedulerEvent
{
    pmTask* task;
    const pmMachine* machine;
    
    noReductionRequiredEvent(eventIdentifier pEventId, pmTask* pTask, const pmMachine* pMachine)
    : schedulerEvent(pEventId)
    , task(pTask)
    , machine(pMachine)
    {}
};

struct commandCompletionEvent : public schedulerEvent
{
	const pmCommandPtr command;
    
    commandCompletionEvent(eventIdentifier pEventId, const pmCommandPtr& pCommand)
    : schedulerEvent(pEventId)
    , command(pCommand)
    {}
};
    
struct hostFinalizationEvent : public schedulerEvent
{
    bool terminate; // true for final termination; false for task submission freeze
    
    hostFinalizationEvent(eventIdentifier pEventId, bool pTerminate)
    : schedulerEvent(pEventId)
    , terminate(pTerminate)
    {}
};
    
struct subtaskRangeCancelEvent : public schedulerEvent
{
	const pmProcessingElement* targetDevice;
    const pmSubtaskRange range;
    
    subtaskRangeCancelEvent(eventIdentifier pEventId, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange)
    : schedulerEvent(pEventId)
    , targetDevice(pTargetDevice)
    , range(pRange)
    {}
};
    
struct redistributionMetaDataEvent : public schedulerEvent
{
    typedef std::vector<communicator::redistributionOrderStruct> redistributionOrderVectorType;

    pmTask* task;
    uint addressSpaceIndex;
    redistributionOrderVectorType* redistributionData;
    
    redistributionMetaDataEvent(eventIdentifier pEventId, pmTask* pTask, uint pAddressSpaceIndex, redistributionOrderVectorType* pRedistributionData)
    : schedulerEvent(pEventId)
    , task(pTask)
    , addressSpaceIndex(pAddressSpaceIndex)
    , redistributionData(pRedistributionData)
    {}
};

struct redistributionOffsetsEvent : public schedulerEvent
{
    pmTask* task;
    uint addressSpaceIndex;
    std::vector<ulong>* offsetsData;
    uint destHostId;
    pmAddressSpace* redistributedAddressSpace;
    
    redistributionOffsetsEvent(eventIdentifier pEventId, pmTask* pTask, uint pAddressSpaceIndex, std::vector<ulong>* pOffsetsData, uint pDestHostId, pmAddressSpace* pRedistributedAddressSpace)
    : schedulerEvent(pEventId)
    , task(pTask)
    , addressSpaceIndex(pAddressSpaceIndex)
    , offsetsData(pOffsetsData)
    , destHostId(pDestHostId)
    , redistributedAddressSpace(pRedistributedAddressSpace)
    {}
};
    
struct rangeNegotiationEvent : public schedulerEvent
{
    const pmProcessingElement* requestingDevice;
    const pmSubtaskRange range;
    
    rangeNegotiationEvent(eventIdentifier pEventId, const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange)
    : schedulerEvent(pEventId)
    , requestingDevice(pRequestingDevice)
    , range(pRange)
    {}
};

struct rangeNegotiationSuccessEvent : public schedulerEvent
{
    const pmProcessingElement* requestingDevice;
    const pmSubtaskRange negotiatedRange;
    
    rangeNegotiationSuccessEvent(eventIdentifier pEventId, const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pNegotiatedRange)
    : schedulerEvent(pEventId)
    , requestingDevice(pRequestingDevice)
    , negotiatedRange(pNegotiatedRange)
    {}
};
    
struct reductionTerminationEvent : public schedulerEvent
{
	pmLocalTask* localTask;
    
    reductionTerminationEvent(eventIdentifier pEventId, pmLocalTask* pLocalTask)
    : schedulerEvent(pEventId)
    , localTask(pLocalTask)
    {}
};
    
struct affinityTransferEvent : public schedulerEvent
{
    pmLocalTask* localTask;
    std::set<const pmMachine*> machines;
    const std::vector<ulong>* logicalToPhysicalSubtaskMapping;
    
    affinityTransferEvent(eventIdentifier pEventId, pmLocalTask* pLocalTask, std::set<const pmMachine*>&& pMachines, const std::vector<ulong>* pLogicalToPhysicalSubtaskMapping)
    : schedulerEvent(pEventId)
    , localTask(pLocalTask)
    , machines(pMachines)
    , logicalToPhysicalSubtaskMapping(pLogicalToPhysicalSubtaskMapping)
    {}
};
    
struct allReductionsDoneEvent : public schedulerEvent
{
    pmLocalTask* localTask;
    pmExecutionStub* lastStub;
    ulong lastSubtaskId;
    pmSplitData lastSplitData;
    
    allReductionsDoneEvent(eventIdentifier pEventId, pmLocalTask* pLocalTask, pmExecutionStub* pLastStub, ulong pLastSubtaskId, const pmSplitData& pLastSplitData)
    : schedulerEvent(pEventId)
    , localTask(pLocalTask)
    , lastStub(pLastStub)
    , lastSubtaskId(pLastSubtaskId)
    , lastSplitData(pLastSplitData)
    {}
};
    
struct externalReductionFinishEvent : public schedulerEvent
{
    pmTask* task;
    
    externalReductionFinishEvent(eventIdentifier pEventId, pmTask* pTask)
    : schedulerEvent(pEventId)
    , task(pTask)
    {}
};

}

/**
 * \brief This class schedules, load balances and executes all tasks on this machine.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmScheduler : public THREADING_IMPLEMENTATION_CLASS<scheduler::schedulerEvent>
{
	friend void SchedulerCommandCompletionCallback(const pmCommandPtr& pCommand);
    
	public:
        virtual ~pmScheduler();
		static pmScheduler* GetScheduler();
    
        static bool SchedulingModelSupportsStealing(scheduler::schedulingModel pModel);

        void SendAcknowledgement(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount);
        void SendAcknowledgement(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount);
        void ProcessAcknowledgement(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);
		void ProcessAcknowledgement(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);

        virtual void ThreadSwitchCallback(std::shared_ptr<scheduler::schedulerEvent>& pEvent);

		void SendFailedStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask);
		void SendStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);

		void SubmitTaskEvent(pmLocalTask* pLocalTask);
		void PushEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, bool pIsStealResponse);		// subtask range execution event
		void StealRequestEvent(const pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);
		void StealProcessEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate, bool pMultiAssign);
		void StealSuccessEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);
        void StealFailedEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask);
		void StealSuccessReturnEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);
		void StealFailedReturnEvent(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask);

    #ifdef USE_AFFINITY_IN_STEAL
        void PushEvent(pmTask* pTask, const pmProcessingElement* pDevice, std::vector<ulong>&& pDiscontiguousStealData);
        void SendStealResponse(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData);
    
        void StealSuccessEvent(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData);
        void StealSuccessReturnEvent(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData);
    #endif

        void AcknowledgementSendEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount);
        void AcknowledgementSendEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector, ulong pTotalSplitCount);
		void AcknowledgementReceiveEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);
		void AcknowledgementReceiveEvent(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<communicator::ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);
		void TaskCancelEvent(pmTask* pTask);
        void TaskFinishEvent(pmTask* pTask);
        void TaskCompleteEvent(pmLocalTask* pLocalTask);
        void NoReductionRequiredEvent(pmTask* pTask, const pmMachine* pDestMachine);
    
        void CommandCompletionEvent(const pmCommandPtr& pCommand);
        void RangeCancellationEvent(const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);
        void RedistributionMetaDataEvent(pmTask* pTask, uint pAddressSpaceIndex, std::vector<communicator::redistributionOrderStruct>* pRedistributionData);
        void RedistributionOffsetsEvent(pmTask* pTask, uint pAddressSpaceIndex, pmAddressSpace* pRedistributedAddressSpace, uint pDestHostId, std::vector<ulong>* pOffsetsData);
        void RangeNegotiationEvent(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange);
        void RangeNegotiationSuccessEvent(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pNegotiatedRange);
        void TerminateTaskEvent(pmTask* pTask);
        void ReductionTerminationEvent(pmLocalTask* pLocalTask);
        void AffinityTransferEvent(pmLocalTask* pLocalTask, std::set<const pmMachine*>&& pMachines, const std::vector<ulong>* pLogicalToPhysicalSubtaskMapping);
    
        void AllReductionsDoneEvent(pmLocalTask* pLocalTask, pmExecutionStub* pLastStub, ulong pLastSubtaskId, const pmSplitData& pLastSplitData);
        void AddRegisterExternalReductionFinishEvent(pmTask* pTask);

        void SendPostTaskOwnershipTransfer(pmAddressSpace* pAddressSpace, const pmMachine* pReceiverHost, std::shared_ptr<std::vector<communicator::ownershipChangeStruct> >& pChangeData);
        void SendPostTaskOwnershipTransfer(pmAddressSpace* pAddressSpace, const pmMachine* pReceiverHost, std::shared_ptr<std::vector<communicator::scatteredOwnershipChangeStruct> >& pChangeData);
        void SendSubtaskRangeCancellationMessage(const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);

        void HandleCommandCompletion(const pmCommandPtr& pCommand);
        void FreeTaskResourcesOnLocalStubs(pmTask* pTask);
        void CancelAllSubtasksExecutingOnLocalStubs(pmTask* pTask, bool pTaskListeningOnCancellation);
        void CancelAllSubtaskSplitDummyEventsOnLocalStubs(pmTask* pTask);
        void CommitShadowMemPendingOnAllStubs(pmTask* pTask);
		void CancelTask(pmLocalTask* pLocalTask);

        pmCommandCompletionCallbackType GetSchedulerCommandCompletionCallback();
    
        void WaitForAllCommandsToFinish();

        void SendFinalizationSignal();
		void BroadcastTerminationSignal();
    
        void NegotiateSubtaskRangeWithOriginalAllottee(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange);
        void SendRangeNegotiationSuccess(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pNegotiatedRange);
        void SendRedistributionData(pmTask* pTask, uint pAddressSpaceIndex, std::vector<communicator::redistributionOrderStruct>* pRedistributionData);
        void SendRedistributionOffsets(pmTask* pTask, uint pAddressSpaceIndex, std::vector<ulong>* pOffsetsData, pmAddressSpace* pRedistributedAddressSpace, uint pDestHostId);

        void SendTaskCompleteToTaskOwner(pmTask* pTask);
    
        void AssignSubtasksToDevices(pmLocalTask* pLocalTask);

    private:
		pmScheduler();

		void SetupPersistentCommunicationCommands();
		void DestroyPersistentCommunicationCommands();

        void ProcessEvent(scheduler::schedulerEvent& pEvent);

		void AssignTaskToMachines(pmLocalTask* pLocalTask, std::set<const pmMachine*>& pMachines);

		void AssignSubtasksToDevice(const pmProcessingElement* pDevice, pmLocalTask* pLocalTask);

		pmStatus StartLocalTaskExecution(pmLocalTask* pLocalTask);

		void PushToStub(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, bool pIsStealResponse);
    
        void ProcessAcknowledgementCommon(pmLocalTask* pLocalTask, const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus);

    #ifdef ENABLE_TWO_LEVEL_STEALING
		const pmMachine* RandomlySelectStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool& pShouldMultiAssign);
        const pmProcessingElement* RandomlySelectSecondLevelStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool pShouldMultiAssign);
    #else
		const pmProcessingElement* RandomlySelectStealTarget(const pmProcessingElement* pStealingDevice, pmTask* pTask, bool& pShouldMultiAssign);
    #endif

		void StealSubtasks(const pmProcessingElement* pStealingDevice, pmTask* pTask, double pExecutionRate);

		void ServeStealRequest(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask, double pExecutionRate, bool pShouldMultiAssign);
		void ReceiveFailedStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, pmTask* pTask);
		void ReceiveStealResponse(const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, const pmSubtaskRange& pRange);
    
    #ifdef USE_AFFINITY_IN_STEAL
        void PushToStub(pmTask* pTask, const pmProcessingElement* pDevice, std::vector<ulong>&& pDiscontiguousStealData);
        void ReceiveStealResponse(pmTask* pTask, const pmProcessingElement* pStealingDevice, const pmProcessingElement* pTargetDevice, std::vector<ulong>&& pDiscontiguousStealData);
    #endif

        void RegisterPostTaskCompletionOwnershipTransfers(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, const std::vector<communicator::ownershipDataStruct>& pOwnershipVector, const std::vector<uint>& pAddressSpaceIndexVector);
        void RegisterPostTaskCompletionOwnershipTransfers(const pmProcessingElement* pDevice, const pmSubtaskRange& pRange, const std::vector<communicator::scatteredOwnershipDataStruct>& pScatteredOwnershipVector, const std::vector<uint>& pAddressSpaceIndexVector);
    
        void ClearPendingTaskCommands(pmTask* pTask);
        void SendTaskFinishToMachines(pmLocalTask* pLocalTask);
    
        void SendReductionTerminationToMachines(pmLocalTask* pLocalTask);
    
        void SendAffinityDataToMachines(pmLocalTask* pLocalTask, const std::set<const pmMachine*>& pMachines, const std::vector<ulong>& pLogicalToPhysicalSubtaskMappings);

        pmCommunicatorCommandPtr mRemoteSubtaskRecvCommand;
		pmCommunicatorCommandPtr mTaskEventRecvCommand;
		pmCommunicatorCommandPtr mStealRequestRecvCommand;
        pmCommunicatorCommandPtr mStealResponseRecvCommand;
        pmCommunicatorCommandPtr mHostFinalizationCommand;
        pmCommunicatorCommandPtr mSubtaskRangeCancelCommand;
        pmCommunicatorCommandPtr mNoReductionReqdCommand;
        pmCommunicatorCommandPtr mSubtaskMemoryReduceCommand;

    #ifdef TRACK_SUBTASK_EXECUTION
        ulong mSubtasksAssigned;
        ulong mAcknowledgementsSent;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTrackLock;
    #endif
};

bool taskClearMatchFunc(const scheduler::schedulerEvent& pEvent, const void* pCriterion);

} // end namespace pm

#endif

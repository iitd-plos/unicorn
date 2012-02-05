
#ifndef __PM_EXECUTION_STUB__
#define __PM_EXECUTION_STUB__

#include "pmBase.h"
#include "pmThread.h"

namespace pm
{

class pmTask;
class pmProcessingElement;
class pmSubscriptionManager;
class pmReducer;

/**
 * \brief The controlling thread of each processing element.
 */

namespace execStub
{

typedef enum eventIdentifier
{
        THREAD_BIND,
        SUBTASK_EXEC,
        SUBTASK_REDUCE,
        SUBTASK_CANCEL,
        SUBTASK_STEAL
} eventIdentifier;

typedef struct threadBind
{
} threadBind;

typedef struct subtaskExec
{
        pmSubtaskRange range;
        bool rangeExecutedOnce;
        ulong lastExecutedSubtaskId;
} subtaskExec;

typedef struct subtaskReduce
{
        pmTask* task;
        ulong subtaskId1;
        ulong subtaskId2;
} subtaskReduce;

typedef struct subtaskCancel
{
        pmTask* task;   /* not to be dereferenced */
        ushort priority;
} subtaskCancel;

typedef struct subtaskSteal
{
        pmTask* task;
        pmProcessingElement* requestingDevice;
        double requestingDeviceExecutionRate;
} subtaskSteal;

typedef struct stubEvent
{
        eventIdentifier eventId;
        union
        {
                threadBind bindDetails;
                subtaskExec execDetails;
                subtaskReduce reduceDetails;
                subtaskCancel cancelDetails;
                subtaskSteal stealDetails;
        };
} stubEvent;

}

class pmExecutionStub : public THREADING_IMPLEMENTATION_CLASS<execStub::stubEvent>
{
	public:
		pmExecutionStub(uint pDeviceIndexOnMachine);
		virtual ~pmExecutionStub();

		virtual pmStatus BindToProcessingElement() = 0;

		virtual pmStatus Push(pmSubtaskRange pRange);
		virtual pmStatus ThreadSwitchCallback(execStub::stubEvent& pEvent);

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceTypes GetType() = 0;

		pmProcessingElement* GetProcessingElement();

		pmStatus ReduceSubtasks(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2);
		pmStatus StealSubtasks(pmTask* pTask, pmProcessingElement* pRequestingDevice, double pExecutionRate);
		pmStatus CancelSubtasks(pmTask* pTask);

	protected:
		bool IsHighPriorityEventWaiting(ushort pPriority);
		pmStatus CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId);
		pmStatus CommonPostExecuteOnCPU(pmTask* pTask, ulong pSubtaskId);

		virtual pmStatus DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2);

	private:
		pmStatus ProcessEvent(execStub::stubEvent& pEvent);
		virtual pmStatus Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId) = 0;

		uint mDeviceIndexOnMachine;
		size_t mCoreId;
};

class pmStubGPU : public pmExecutionStub
{
	public:
		pmStubGPU(uint pDeviceIndexOnMachine);
		virtual ~pmStubGPU();

		virtual pmStatus BindToProcessingElement() = 0;

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceTypes GetType() = 0;

		virtual pmStatus Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId) = 0;
		virtual pmStatus Execute(pmTask* pTask, ulong pSubtaskId) = 0;

	private:
};

class pmStubCPU : public pmExecutionStub
{
	public:
		pmStubCPU(size_t pCoreId, uint pDeviceIndexOnMachine);
		virtual ~pmStubCPU();

		virtual pmStatus BindToProcessingElement();
		virtual size_t GetCoreId();

		virtual std::string GetDeviceName();
		virtual std::string GetDeviceDescription();

		virtual pmDeviceTypes GetType();

		virtual pmStatus Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId);
		virtual pmStatus Execute(pmTask* pTask, ulong pSubtaskId);

	private:
 		size_t mCoreId;
};

class pmStubCUDA : public pmStubGPU
{
	public:
		pmStubCUDA(size_t pDeviceIndex, uint pDeviceIndexOnMachine);
		virtual ~pmStubCUDA();

		virtual pmStatus BindToProcessingElement();

		virtual std::string GetDeviceName();
		virtual std::string GetDeviceDescription();

		virtual pmDeviceTypes GetType();

		virtual pmStatus Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId);
		virtual pmStatus Execute(pmTask* pTask, ulong pSubtaskId);

	private:
		size_t mDeviceIndex;
};

bool execEventMatchFunc(execStub::stubEvent& pEvent, void* pCriterion);

} // end namespace pm

#endif

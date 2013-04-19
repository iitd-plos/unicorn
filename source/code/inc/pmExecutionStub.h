
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

#ifndef __PM_EXECUTION_STUB__
#define __PM_EXECUTION_STUB__

#include "pmBase.h"
#include "pmThread.h"

#ifdef DUMP_EVENT_TIMELINE
    #include "pmEventTimeline.h"
#endif

#include <setjmp.h>

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
    THREAD_BIND
    , SUBTASK_EXEC
    , SUBTASK_REDUCE
    , NEGOTIATED_RANGE
	, FREE_GPU_RESOURCES
    , POST_HANDLE_EXEC_COMPLETION
    , DEFERRED_SHADOW_MEM_COMMITS
    , CHECK_REDUCTION_FINISH
#ifdef DUMP_EVENT_TIMELINE
    , INIT_EVENT_TIMELINE
#endif
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
    pmExecutionStub* stub2;
    ulong subtaskId2;
} subtaskReduce;

typedef struct negotiatedRange
{
    pmSubtaskRange range;
} negotiatedRange;
    
typedef struct execCompletion
{
    pmSubtaskRange range;
    pmStatus execStatus;
} execCompletion;
    
typedef struct deferredShadowMemCommits
{
    pmTask* task;
} deferredShadowMemCommits;
    
typedef struct checkReductionFinish
{
    pmTask* task;
} checkReductionFinish;
    
#ifdef DUMP_EVENT_TIMELINE
typedef struct initTimeline
{
} initTimeline;
#endif

typedef struct stubEvent : public pmBasicThreadEvent
{
    eventIdentifier eventId;
    union
    {
        threadBind bindDetails;
        subtaskExec execDetails;
        subtaskReduce reduceDetails;
        negotiatedRange negotiatedRangeDetails;
        execCompletion execCompletionDetails;
        deferredShadowMemCommits deferredShadowMemCommitsDetails;
        checkReductionFinish checkReductionFinishDetails;
    #ifdef DUMP_EVENT_TIMELINE
        initTimeline initTimelineDetails;
    #endif
    };

    virtual bool BlocksSecondaryCommands();
} stubEvent;

}

class pmExecutionStub : public THREADING_IMPLEMENTATION_CLASS<execStub::stubEvent>
{
	public:
		pmExecutionStub(uint pDeviceIndexOnMachine);
		virtual ~pmExecutionStub();

		virtual pmStatus BindToProcessingElement() = 0;

		virtual pmStatus Push(pmSubtaskRange& pRange);
		virtual pmStatus ThreadSwitchCallback(execStub::stubEvent& pEvent);

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceType GetType() = 0;

		pmProcessingElement* GetProcessingElement();

        pmStatus ThreadBindEvent();
    #ifdef DUMP_EVENT_TIMELINE
        pmStatus InitializeEventTimeline();
    #endif
		pmStatus ReduceSubtasks(pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2);
		pmStatus StealSubtasks(pmTask* pTask, pmProcessingElement* pRequestingDevice, double pRequestingDeviceExecutionRate);
		pmStatus CancelAllSubtasks(pmTask* pTask, bool pTaskListeningOnCancellation);
        pmStatus CancelSubtaskRange(pmSubtaskRange& pRange);
        pmStatus ProcessNegotiatedRange(pmSubtaskRange& pRange);
        void ProcessDeferredShadowMemCommits(pmTask* pTask);
        void CheckReductionFinishEvent(pmTask* pTask);

        pmStatus NegotiateRange(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange);

        bool RequiresPrematureExit(ulong pSubtaskId);

        void MarkInsideLibraryCode(ulong pSubtaskId);
        void MarkInsideUserCode(ulong pSubtaskId);
    
        void SetupJmpBuf(sigjmp_buf* pJmpBuf, ulong pSubtaskId);
        void UnsetupJmpBuf(ulong pSubtaskId);
    
        void WaitForNetworkFetch(std::vector<pmCommunicatorCommandPtr>& pNetworkCommands);
    
	protected:
		bool IsHighPriorityEventWaiting(ushort pPriority);
		pmStatus CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId);
		pmStatus CommonPostExecuteOnCPU(pmTask* pTask, ulong pSubtaskId);

		pmStatus FreeGpuResources();

		virtual pmStatus DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2);

	private:
        void MarkInsideLibraryCodeInternal(ulong pSubtaskId);
        void MarkInsideUserCodeInternal(ulong pSubtaskId);
    
        typedef struct currentSubtaskStats
        {
            pmTask* task;
            ulong subtaskId;
            ulong parentRangeStartSubtask;
            bool originalAllottee;
            double startTime;
            bool reassigned;    // the current subtask has been negotiated
            bool forceAckFlag;  // send acknowledgement for the entire parent range after current subtask is stolen/negotiated
            bool executingLibraryCode;
            bool prematureTermination;
            bool taskListeningOnCancellation;
            sigjmp_buf* jmpBuf;
            pmAccumulatorCommandPtr* accumulatorCommandPtr;
        
            currentSubtaskStats(pmTask* pTask, ulong pSubtaskId, bool pOriginalAllottee, ulong pParentRangeStartSubtask, sigjmp_buf* pJmpBuf, double pStartTime);
        } currentSubtaskStats;
    
        typedef class currentSubtaskTerminus
        {
            public:
                currentSubtaskTerminus(bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmExecutionStub* pStub);
                void Terminating(currentSubtaskStats* pStats);
        
            private:
                bool& mReassigned;
                bool& mForceAckFlag;
                bool& mPrematureTermination;
                pmExecutionStub* mStub;
        } currentSubtaskTerminus;
    
		pmStatus ProcessEvent(execStub::stubEvent& pEvent);
        virtual pmStatus Execute(pmTask* pTask, ulong pSubtaskId) = 0;
        pmStatus ExecuteWrapper(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong pParentRangeStartSubtask, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination);
        pmStatus ExecuteWrapperInternal(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong pParentRangeStartSubtask, currentSubtaskTerminus& pTerminus);
        void PostHandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus);
        void HandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus);
        pmStatus CommonPostNegotiationOnCPU(pmTask* pTask, ulong pSubtaskId);
        void CommitRange(pmSubtaskRange& pRange, pmStatus pExecStatus);
        void CommitSubtaskShadowMem(pmTask* pTask, ulong pSubtaskId);
        void DeferShadowMemCommit(pmTask* pTask, ulong pSubtaskId);
        void CancelCurrentlyExecutingSubtask(bool pTaskListeningOnCancellation);
        void TerminateCurrentSubtask();
        void RaiseCurrentSubtaskTerminationSignalInThread();
        
    #ifdef DUMP_EVENT_TIMELINE
        std::string GetEventTimelineName();
    #endif
    
		uint mDeviceIndexOnMachine;
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mCurrentSubtaskLock;
        currentSubtaskStats* mCurrentSubtaskStats;  // Subtask currently being executed
        std::map<std::pair<pmTask*, ulong>, std::vector<pmProcessingElement*> > mSecondaryAllotteeMap;  // PULL model: secondary allottees of a subtask
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mDeferredShadowMemCommitsLock;
        std::map<pmTask*, std::vector<ulong> > mDeferredShadowMemCommits;

    #ifdef DUMP_EVENT_TIMELINE
        std::auto_ptr<pmEventTimeline> mEventTimelineAutoPtr;
    #endif
};

class pmStubGPU : public pmExecutionStub
{
	public:
		pmStubGPU(uint pDeviceIndexOnMachine);
		virtual ~pmStubGPU();

		virtual pmStatus BindToProcessingElement() = 0;

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceType GetType() = 0;

		virtual pmStatus FreeResources() = 0;
		virtual pmStatus FreeExecutionResources() = 0;

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

		virtual pmDeviceType GetType();

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

		virtual pmDeviceType GetType();

		virtual pmStatus FreeResources();
		virtual pmStatus FreeExecutionResources();

		virtual pmStatus Execute(pmTask* pTask, ulong pSubtaskId);

    #ifdef SUPPORT_CUDA
        void* GetDeviceInfoCudaPtr();
    #endif

	private:
		size_t mDeviceIndex;

    #ifdef SUPPORT_CUDA
        void* mDeviceInfoCudaPtr;
    #endif
};

bool execEventMatchFunc(execStub::stubEvent& pEvent, void* pCriterion);

} // end namespace pm

#endif

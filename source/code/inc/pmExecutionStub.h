
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

#ifndef __PM_EXECUTION_STUB__
#define __PM_EXECUTION_STUB__

#include "pmBase.h"
#include "pmThread.h"
#include "pmSubtaskSplitter.h"
#include "pmCommunicator.h"
#include "pmCache.h"

#ifdef DUMP_EVENT_TIMELINE
    #include "pmEventTimeline.h"
#endif

#ifdef SUPPORT_CUDA
#include "pmMemChunk.h"
#include "pmAllocatorCollection.h"
#include "pmCudaInterface.h"
#endif

#include <setjmp.h>
#include <string.h>

#include <queue>

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

enum eventIdentifier
{
    THREAD_BIND
    , SUBTASK_EXEC
    , SUBTASK_REDUCE
    , EXTERNAL_MEM_REDUCE
    , NEGOTIATED_RANGE
	, FREE_GPU_RESOURCES
    , POST_HANDLE_EXEC_COMPLETION
    , DEFERRED_SHADOW_MEM_COMMITS
    , REDUCTION_FINISH
    , PROCESS_REDISTRIBUTION_BUCKET
    , FREE_TASK_RESOURCES
#ifdef DUMP_EVENT_TIMELINE
    , INIT_EVENT_TIMELINE
#endif
#ifdef SUPPORT_SPLIT_SUBTASKS
    , SPLIT_SUBTASK_CHECK
#endif
    , REMOTE_SUBTASK_REDUCE
    , CANCEL_SUBTASK_RANGE
    , CANCEL_ALL_SUBTASKS
    , DIRECT_EXTERNAL_REDUCTION
    , MAX_EXEC_STUB_EVENTS
};

struct stubEvent : public pmBasicBlockableThreadEvent
{
    eventIdentifier eventId;

    stubEvent(eventIdentifier pEventId = MAX_EXEC_STUB_EVENTS)
    :eventId(pEventId)
    {}
    
    virtual bool BlocksSecondaryOperations();
};
    
struct threadBindEvent : public stubEvent
{
    size_t physicalMemory;
    size_t totalStubCount;
    
    threadBindEvent(eventIdentifier pEventId, size_t pPhysicalMemory, size_t pTotalStubCount)
    : stubEvent(pEventId)
    , physicalMemory(pPhysicalMemory)
    , totalStubCount(pTotalStubCount)
    {}
};

struct subtaskExecEvent : public stubEvent
{
    pmSubtaskRange range;
    bool rangeExecutedOnce;
    ulong lastExecutedSubtaskId;
    
    subtaskExecEvent(eventIdentifier pEventId, const pmSubtaskRange& pRange, bool pRangeExecutedOnce, ulong pLastExecutedSubtaskId)
    : stubEvent(pEventId)
    , range(pRange)
    , rangeExecutedOnce(pRangeExecutedOnce)
    , lastExecutedSubtaskId(pLastExecutedSubtaskId)
    {}

#ifdef USE_STEAL_AGENT_PER_NODE
    virtual void EventNotification(void* pThreadQueue, bool pSubmitted);
#endif
};

struct subtaskReduceEvent : public stubEvent
{
    pmTask* task;
    ulong subtaskId1;
    pmExecutionStub* stub2;
    ulong subtaskId2;
    pmSplitData splitData1;
    pmSplitData splitData2;
    
    subtaskReduceEvent(eventIdentifier pEventId, pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitData& pSplitData1, pmSplitData& pSplitData2)
    : stubEvent(pEventId)
    , task(pTask)
    , subtaskId1(pSubtaskId1)
    , stub2(pStub2)
    , subtaskId2(pSubtaskId2)
    , splitData1(pSplitData1)
    , splitData2(pSplitData2)
    {}
};

struct externalMemReduceEvent : public stubEvent
{
    pmTask* task;
    const pmCommandPtr command;

    externalMemReduceEvent(eventIdentifier pEventId, pmTask* pTask, const pmCommandPtr& pCommand)
    : stubEvent(pEventId)
    , task(pTask)
    , command(pCommand)
    {}
};

struct negotiatedRangeEvent : public stubEvent
{
    pmSubtaskRange range;
    
    negotiatedRangeEvent(eventIdentifier pEventId, const pmSubtaskRange& pRange)
    : stubEvent(pEventId)
    , range(pRange)
    {}
};

#ifdef SUPPORT_CUDA
struct freeGpuResourcesEvent : public stubEvent
{
    freeGpuResourcesEvent(eventIdentifier pEventId)
    : stubEvent(pEventId)
    {}
};
#endif
    
struct execCompletionEvent : public stubEvent
{
    pmSubtaskRange range;
    pmStatus execStatus;
    
    execCompletionEvent(eventIdentifier pEventId, pmSubtaskRange& pRange, pmStatus pExecStatus)
    : stubEvent(pEventId)
    , range(pRange)
    , execStatus(pExecStatus)
    {}
};
    
struct deferredShadowMemCommitsEvent : public stubEvent
{
    pmTask* task;
    
    deferredShadowMemCommitsEvent(eventIdentifier pEventId, pmTask* pTask)
    : stubEvent(pEventId)
    , task(pTask)
    {}
};
    
struct reductionFinishEvent : public stubEvent
{
    pmTask* task;

    reductionFinishEvent(eventIdentifier pEventId, pmTask* pTask)
    : stubEvent(pEventId)
    , task(pTask)
    {}
};
    
struct processRedistributionBucketEvent : public stubEvent
{
    pmTask* task;
    uint addressSpaceIndex;
    size_t bucketIndex;
    
    processRedistributionBucketEvent(eventIdentifier pEventId, pmTask* pTask, uint pAddressSpaceIndex, size_t pBucketIndex)
    : stubEvent(pEventId)
    , task(pTask)
    , addressSpaceIndex(pAddressSpaceIndex)
    , bucketIndex(pBucketIndex)
    {}
};

struct freeTaskResourcesEvent : public stubEvent
{
    const pmMachine* taskOriginatingHost;
    ulong taskSequenceNumber;
    
    freeTaskResourcesEvent(eventIdentifier pEventId, const pmMachine* pTaskOriginatingHost, ulong pTaskSequenceNumber)
    : stubEvent(pEventId)
    , taskOriginatingHost(pTaskOriginatingHost)
    , taskSequenceNumber(pTaskSequenceNumber)
    {}
};
    
#ifdef DUMP_EVENT_TIMELINE
struct initTimelineEvent : public stubEvent
{
    initTimelineEvent(eventIdentifier pEventId)
    : stubEvent(pEventId)
    {}
};
#endif
    
#ifdef SUPPORT_SPLIT_SUBTASKS
struct splitSubtaskCheckEvent : public stubEvent
{
    pmTask* task;
    
    splitSubtaskCheckEvent(eventIdentifier pEventId, pmTask* pTask)
    : stubEvent(pEventId)
    , task(pTask)
    {}
};
#endif
    
struct remoteSubtaskReduceEvent : public stubEvent
{
    pmTask* task;
    const pmCommunicatorCommandPtr commandPtr;
    
    remoteSubtaskReduceEvent(eventIdentifier pEventId, pmTask* pTask, const pmCommunicatorCommandPtr& pCommandPtr)
    : stubEvent(pEventId)
    , task(pTask)
    , commandPtr(pCommandPtr)
    {}
};

struct cancelSubtaskRangeEvent : public stubEvent
{
    pmSubtaskRange range;
    
    cancelSubtaskRangeEvent(eventIdentifier pEventId, const pmSubtaskRange& pRange)
    : stubEvent(pEventId)
    , range(pRange)
    {}
};

struct cancelAllSubtasksEvent : public stubEvent
{
    pmTask* task;
    bool taskListeningOnCancellation;
    
    cancelAllSubtasksEvent(eventIdentifier pEventId, pmTask* pTask, bool pTaskListeningOnCancellation)
    : stubEvent(pEventId)
    , task(pTask)
    , taskListeningOnCancellation(pTaskListeningOnCancellation)
    {}
};
    
struct directExternalReductionPlaceHolderEvent : public stubEvent
{
    pmTask* task;

    directExternalReductionPlaceHolderEvent(eventIdentifier pEventId, pmTask* pTask)
    : stubEvent(pEventId)
    , task(pTask)
    {}
};

}

class pmExecutionStub : public THREADING_IMPLEMENTATION_CLASS<execStub::stubEvent>
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    friend class pmSplitGroup;
#endif
    
	public:
		pmExecutionStub(uint pDeviceIndexOnMachine);
		virtual ~pmExecutionStub();

		virtual void BindToProcessingElement() = 0;

		void Push(const pmSubtaskRange& pRange, bool pIsStealResponse);
    
    #ifdef USE_AFFINITY_IN_STEAL
        void Push(pmTask* pTask, std::vector<ulong>&& pDiscontiguousStealData);
    #endif
    
		virtual void ThreadSwitchCallback(std::shared_ptr<execStub::stubEvent>& pEvent);

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceType GetType() const = 0;

		const pmProcessingElement* GetProcessingElement() const;

        void ThreadBindEvent(size_t pPhysicalMemory, size_t pTotalStubCount);
    #ifdef DUMP_EVENT_TIMELINE
        void InitializeEventTimeline();
    #endif
		void ReduceSubtasks(pmTask* pTask, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2);
        void ReduceExternalMemory(pmTask* pTask, const pmCommandPtr& pCommand);
		void StealSubtasks(pmTask* pTask, const pmProcessingElement* pRequestingDevice, double pRequestingDeviceExecutionRate, bool pShouldMultiAssign);
		void CancelAllSubtasks(pmTask* pTask, bool pTaskListeningOnCancellation);
        void CancelSubtaskRange(const pmSubtaskRange& pRange);
        void ProcessNegotiatedRange(const pmSubtaskRange& pRange);
        void ProcessDeferredShadowMemCommits(pmTask* pTask);
        void ReductionFinishEvent(pmTask* pTask);
        void ProcessRedistributionBucket(pmTask* pTask, uint pAddressSpaceIndex, size_t pBucketIndex);
        void FreeTaskResources(pmTask* pTask);
        void RemoteSubtaskReduce(pmTask* pTask, const pmCommunicatorCommandPtr& pCommandPtr);

    #ifdef SUPPORT_SPLIT_SUBTASKS
        void SplitSubtaskCheckEvent(pmTask* pTask);
        void RemoveSplitSubtaskCheckEvent(pmTask* pTask);
    #endif
    
        void AddPlaceHolderEventForDirectExternalReduction(pmTask* pTask);

        void NegotiateRange(const pmProcessingElement* pRequestingDevice, const pmSubtaskRange& pRange);

        bool RequiresPrematureExit();

        void MarkInsideLibraryCode();
        void MarkInsideUserCode();
    
        void SetupJmpBuf(sigjmp_buf* pJmpBuf);
        void UnsetupJmpBuf(bool pHasJumped);
    
        void WaitForNetworkFetch(const std::vector<pmCommandPtr>& pNetworkCommands);

        void CommitRange(pmSubtaskRange& pRange, pmStatus pExecStatus);

    #ifdef SUPPORT_OPENCL
        void SetOpenCLDevice(void* pOpenCLDevice);
    #endif

    protected:
		bool IsHighPriorityEventWaiting(ushort pPriority);
		void CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, bool pPrefetch, pmSplitInfo* pSplitInfo);

    #ifdef SUPPORT_CUDA
		void FreeGpuResources();
    #endif

		void DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2);

        virtual ulong FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, pmSplitInfo* pSplitInfo, bool pMultiAssign) = 0;
        virtual void WaitForSubtaskExecutionToFinish(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) = 0;
        virtual void CleanupPostSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, ulong pCleanupEndSubtaskId, pmSplitInfo* pSplitInfo) = 0;
        virtual void TerminateUserModeExecution() = 0;

	private:
        void PushInternal(const pmSubtaskRange& pRange, bool pIsStealResponse);

        void CancelAllSubtasksInternal(pmTask* pTask, bool pTaskListeningOnCancellation);
        void CancelSubtaskRangeInternal(const pmSubtaskRange& pRange);

        void CheckTermination();
        void AddSubtaskRangeToExecutionQueue(const std::shared_ptr<execStub::stubEvent>& pSharedPtr);

        typedef struct currentSubtaskRangeStats
        {
            pmTask* task;
            ulong startSubtaskId;
            ulong endSubtaskId;
            ulong currentSubtaskId;
            ulong parentRangeStartSubtask;
            bool currentSubtaskIdValid; // there is a duration when this data is populated and when first subtask starts execution. In between, this is false.
            bool currentSubtaskInPostDataFetchStage;
            bool originalAllottee;
            double startTime;
            bool reassigned;    // the current subtask range has been negotiated
            bool forceAckFlag;  // send acknowledgement for the entire parent range after current subtask range is stolen/negotiated
            bool prematureTermination;
            bool taskListeningOnCancellation;
            sigjmp_buf* jmpBuf;
            pmCommandPtr* accumulatorCommandPtr;
        #ifdef SUPPORT_SPLIT_SUBTASKS
            pmExecutionStub* splitSubtaskSourceStub;
            pmSplitData splitData;
        #endif

            currentSubtaskRangeStats(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, bool pOriginalAllottee, ulong pParentRangeStartSubtask, sigjmp_buf* pJmpBuf, double pStartTime, pmSplitInfo* pSplitInfo, pmExecutionStub* pSplitSubtaskSourceStub);
            
            void ResetEndSubtaskId(ulong pEndSubtaskId);
        } currentSubtaskRangeStats;
    
        typedef class currentSubtaskRangeTerminus
        {
            public:
                currentSubtaskRangeTerminus(bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmExecutionStub* pStub);
                void Terminating(currentSubtaskRangeStats* pStats);
        
            private:
                bool& mReassigned;
                bool& mForceAckFlag;
                bool& mPrematureTermination;
                pmExecutionStub* mStub;
        } currentSubtaskRangeTerminus;

        ulong GetStealCount(pmTask* pTask, const pmProcessingElement* pRequestingDevice, ulong pAvailableSubtasks, double pLocalExecutionRate, double pRequestingDeviceExecutionRate);
    
        void ProcessEvent(execStub::stubEvent& pEvent);
        virtual void Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr, pmSplitInfo* pSplitInfo = NULL) = 0;

    #ifdef DUMP_EVENT_TIMELINE
        ulong ExecuteWrapper(const pmSubtaskRange& pCurrentRange, const execStub::subtaskExecEvent& pEvent, bool pIsMultiAssign, pmSubtaskRangeExecutionTimelineAutoPtr& pRangeExecTimelineAutoPtr, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmStatus& pStatus);
    #else
        ulong ExecuteWrapper(const pmSubtaskRange& pCurrentRange, const execStub::subtaskExecEvent& pEvent, bool pIsMultiAssign, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmStatus& pStatus);
    #endif
    
    #ifdef SUPPORT_SPLIT_SUBTASKS
        bool CheckSplittedExecution(execStub::subtaskExecEvent& pEvent);
        void ExecutePendingSplit(std::unique_ptr<pmSplitSubtask>&& pSplitSubtaskAutoPtr, bool pSecondaryOperationsBlocked);
        double ExecuteSplitSubtask(const std::unique_ptr<pmSplitSubtask>& pSplitSubtaskAutoPtr, bool pSecondaryOperationsBlocked, bool pMultiAssign, bool& pPrematureTermination, bool& pReassigned, bool& pForceAckFlag);
        void HandleSplitSubtaskExecutionCompletion(pmTask* pTask, const splitter::splitRecord& pSplitRecord, pmStatus pExecStatus);
        void CommitSplitSubtask(pmSubtaskRange& pRange, const splitter::splitRecord& pSplitRecord, pmStatus pExecStatus);
        bool UpdateSecondaryAllotteeMap(std::pair<pmTask*, ulong>& pPair, const pmProcessingElement* pRequestingDevice);
        bool UpdateSecondaryAllotteeMapInternal(std::pair<pmTask*, ulong>& pPair, const pmProcessingElement* pRequestingDevice);
        void CancelPostNegotiationSplittedExecutionInternal(std::vector<pmExecutionStub*>& pStubsToBeCancelled, const pmSubtaskRange& pRange);
    #endif
    
        void ExecuteSubtaskRange(execStub::subtaskExecEvent& pEvent);
    
        void PostHandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus);
        void HandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus);
        void CommonPostNegotiationOnCPU(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, pmSplitInfo* pSplitInfo);
        void DeferShadowMemCommit(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void CancelCurrentlyExecutingSubtaskRange(bool pTaskListeningOnCancellation);
        void TerminateCurrentSubtaskRange();
        void ClearSecondaryAllotteeMap(pmSubtaskRange& pRange);
        void SendSplitAcknowledgement(const pmSubtaskRange& pRange, const std::map<ulong, std::vector<pmExecutionStub*>>& pMap, pmStatus pExecStatus, ulong pTotalSplitCount);
    
        void IssueStealRequestIfRequired(pmTask* pTask);
    
    #ifdef DUMP_EVENT_TIMELINE
        std::string GetEventTimelineName();
    #endif
    
		uint mDeviceIndexOnMachine;

        volatile sig_atomic_t mExecutingLibraryCode;
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mCurrentSubtaskRangeLock;
        currentSubtaskRangeStats* mCurrentSubtaskRangeStats;  // Subtasks currently being executed
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mSecondaryAllotteeLock;
        std::map<std::pair<pmTask*, ulong>, std::vector<const pmProcessingElement*> > mSecondaryAllotteeMap;  // PULL model: secondary allottees of a subtask range
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mDeferredShadowMemCommitsLock;
        std::map<pmTask*, std::vector<std::pair<ulong, pmSplitData> > > mDeferredShadowMemCommits;
    
        std::queue<ulong> mCurrentSubtaskQueue; // list of subtasks currently executing on the stub
        std::map<ulong, TIMER_IMPLEMENTATION_CLASS> mCurrentSubtaskTimersMap; // timers for subtasks currently executing on the stub

    #ifdef SUPPORT_SPLIT_SUBTASKS
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mPushAckLock;
        std::map<pmTask*, std::list<std::pair<std::pair<ulong, ulong>, std::map<ulong, std::vector<pmExecutionStub*>>>>> mPushAckHolder;   // Key - Task, Value - List of (Start, End Subtask, Map of subtask id versus vector of stubs that executed splits)
    #endif
    
    #ifdef SUPPORT_OPENCL
        void* mOpenCLDevice;    // cl_device_id
    #endif
    
    #ifdef PROACTIVE_STEAL_REQUESTS
        std::map<pmTask*, bool> mStealRequestIssuedMap;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mStealRequestIssuedMapLock;
    #endif
    
    protected:
    #ifdef DUMP_EVENT_TIMELINE
        std::unique_ptr<pmEventTimeline> mEventTimelineAutoPtr;
    #endif
};

class pmStubGPU : public pmExecutionStub
{
	public:
		pmStubGPU(uint pDeviceIndexOnMachine);
		virtual ~pmStubGPU();

		virtual void BindToProcessingElement() = 0;

		virtual std::string GetDeviceName() = 0;
		virtual std::string GetDeviceDescription() = 0;

		virtual pmDeviceType GetType() const = 0;

		virtual void FreeResources() = 0;
		virtual void FreeExecutionResources() = 0;

		virtual void Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr, pmSplitInfo* pmSplitInfo = NULL) = 0;

        virtual void PurgeAddressSpaceEntriesFromGpuCache(const pmAddressSpace* pAddressSpace) = 0;

    protected:
        virtual ulong FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, pmSplitInfo* pSplitInfo, bool pMultiAssign) = 0;
        virtual void WaitForSubtaskExecutionToFinish(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) = 0;
        virtual void CleanupPostSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, ulong pCleanupEndSubtaskId, pmSplitInfo* pSplitInfo) = 0;
        virtual void TerminateUserModeExecution() = 0;

	private:
};

class pmStubCPU : public pmExecutionStub
{
	public:
		pmStubCPU(size_t pCoreId, uint pDeviceIndexOnMachine);
		virtual ~pmStubCPU();

		virtual void BindToProcessingElement();
		virtual size_t GetCoreId();

		virtual std::string GetDeviceName();
		virtual std::string GetDeviceDescription();

		virtual pmDeviceType GetType() const;
    
		virtual void Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr, pmSplitInfo* pmSplitInfo = NULL);

    protected:
        virtual ulong FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, pmSplitInfo* pSplitInfo, bool pMultiAssign);
        virtual void WaitForSubtaskExecutionToFinish(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        virtual void CleanupPostSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, ulong pCleanupEndSubtaskId, pmSplitInfo* pSplitInfo);
        virtual void TerminateUserModeExecution();
    
	private:
 		size_t mCoreId;
};

#ifdef SUPPORT_CUDA
class pmStubCUDA : public pmStubGPU
{
    friend class pmDispatcherCUDA;

    public:
        typedef pmCache<pmCudaCacheKey, pmCudaCacheValue, pmCudaCacheHasher, pmCudaCacheEvictor, CUDA_CACHE_EVICTION_POLICY> pmCudaCacheType;

        pmStubCUDA(size_t pDeviceIndex, uint pDeviceIndexOnMachine);

		virtual void BindToProcessingElement();

		virtual std::string GetDeviceName();
		virtual std::string GetDeviceDescription();

		virtual pmDeviceType GetType() const;

		virtual void FreeResources();
		virtual void FreeExecutionResources();

		virtual void Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr, pmSplitInfo* pmSplitInfo = NULL);
    
        void* GetDeviceInfoCudaPtr();
        void FreeTaskResources(const pmMachine* pOriginatingHost, ulong pSequenceNumber);
    
        void StreamFinishCallback(void* pCudaStream);

        void ReserveMemory(size_t pPhysicalMemory, size_t pTotalStubCount);
    
        pmAllocatorCollection<pmCudaMemChunkTraits>* GetCudaChunkCollection();
    
        const std::map<ulong, std::vector<pmCudaSubtaskMemoryStruct>>& GetSubtaskPointersMap() const;
        const std::map<ulong, pmCudaSubtaskSecondaryBuffersStruct>& GetSubtaskSecondaryBuffersMap() const;

        size_t GetDeviceIndex();
    
        virtual void PurgeAddressSpaceEntriesFromGpuCache(const pmAddressSpace* pAddressSpace);
    
    #ifdef DUMP_CUDA_CACHE_STATISTICS
        void RecordCudaCacheAllocation(size_t pLength);
        void RecordCudaCacheEviction(size_t pLength);
    #endif

    protected:
        virtual ulong FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, pmSplitInfo* pSplitInfo, bool pMultiAssign);
        virtual void WaitForSubtaskExecutionToFinish(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        virtual void CleanupPostSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, ulong pCleanupEndSubtaskId, pmSplitInfo* pSplitInfo);
        virtual void TerminateUserModeExecution();

        void PopulateMemcpyCommands(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, const pmSubtaskInfo& pSubtaskInfo, std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands);
    
    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        void CopyDataToPinnedBuffers(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, const pmSubtaskInfo& pSubtaskInfo);
        pmStatus CopyDataFromPinnedBuffers(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, const pmSubtaskInfo& pSubtaskInfo);
    #endif

	private:
        void* CreateTaskConf(const pmTaskInfo& pTaskInfo);
        void DestroyTaskConf(void* pTaskConfCudaPtr);
    
        void* CreateDeviceInfoCudaPtr(const pmDeviceInfo& pDeviceInfo);
        void DestroyDeviceInfoCudaPtr(void* pDeviceInfoCudaPtr);
    
        bool CheckSubtaskMemoryRequirements(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, std::map<ulong, std::vector<std::shared_ptr<pmCudaCacheValue>>>& pPreventCachePurgeMap, size_t pCudaAlignment);

        void* AllocateMemoryOnDevice(size_t pLength, size_t pCudaAlignment, pmAllocatorCollection<pmCudaMemChunkTraits>& pChunkCollection);
        bool AllocateMemoryForDeviceCopy(size_t pLength, size_t pCudaAlignment, pmCudaSubtaskMemoryStruct& pMemoryStruct, pmAllocatorCollection<pmCudaMemChunkTraits>& pChunkCollection);
    
        std::unique_ptr<pmCudaCacheKey> MakeCudaCacheKey(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pAddressSpaceIndex, const pmAddressSpace* pAddressSpace, pmSubscriptionVisibilityType pVisibilityType);
    
        ulong PrepareSubtasksForExecution(const pmSubtaskRange& pSubtaskRange, pmSplitInfo* pSplitInfo);
        bool InitializeCudaStream(std::shared_ptr<pmCudaStreamAutoPtr>& pSharedPtr);
    
        void CleanupPostSubtaskExecution(pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        size_t mDeviceIndex;

        std::map<std::pair<const pmMachine*, ulong>, pmTaskInfo> mTaskInfoCudaMap; // pair of task originating host and sequence number
        void* mDeviceInfoCudaPtr;
    
        pmCudaCacheType mCudaCache;
        pmAllocatorCollection<pmCudaMemChunkTraits> mCudaChunkCollection;    // Chunks for cachable memory i.e. address spaces
        pmAllocatorCollection<pmCudaMemChunkTraits> mScratchChunkCollection; // Chunks for non-cacheable memory i.e. scratch buffer, reserved mem, etc.

        std::map<ulong, std::vector<pmCudaSubtaskMemoryStruct>> mSubtaskPointersMap;  // subtask id versus CUDA and pinned pointers
        std::map<ulong, pmCudaSubtaskSecondaryBuffersStruct> mSubtaskSecondaryBuffersMap;
        std::map<ulong, std::vector<pmCudaCacheKey>> mCacheKeys;

        std::map<ulong, std::shared_ptr<pmCudaStreamAutoPtr>> mCudaStreams; // subtask id versus stream
        std::map<ulong, std::vector<std::shared_ptr<pmCudaCacheValue>>> mPreventCachePurgeMap;

    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        pmAllocatorCollection<pmPinnedMemChunkTraits> mPinnedChunkCollection;
    #else
        const pmStatus mStatusCopySrc;
        pmStatus mStatusCopyDest;
    #endif

    #ifdef DUMP_CUDA_CACHE_STATISTICS
        ulong mCudaCacheAllocationLength, mCudaCacheEvictionLength;
        uint mCudaCacheAllocationCount, mCudaCacheEvictionCount;
    #endif
};
#endif

bool execEventMatchFunc(const execStub::stubEvent& pEvent, const void* pCriterion);
bool execEventRangeMatchFunc(const execStub::stubEvent& pEvent, const void* pCriterion);
    
#ifdef SUPPORT_SPLIT_SUBTASKS
bool splitSubtaskCheckEventMatchFunc(const execStub::stubEvent& pEvent, const void* pCriterion);
#endif
    
} // end namespace pm

#endif

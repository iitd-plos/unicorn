
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

#ifndef __PM_TASK__
#define __PM_TASK__

#include "pmBase.h"
#include "pmScheduler.h"
#include "pmTaskExecStats.h"
#include "pmSubscriptionManager.h"
#include "pmSubtaskSplitter.h"
#include "pmPoolAllocator.h"

#ifdef ENABLE_TASK_PROFILING
    #include "pmTaskProfiler.h"
#endif

#include <map>
#include <set>
#include <vector>

namespace pm
{

class pmCallbackUnit;
class pmSubtaskManager;
class pmAddressSpace;
class pmMachine;
class pmCluster;
class pmReducer;
class pmRedistributor;
class pmAffinityTable;

#ifdef USE_STEAL_AGENT_PER_NODE
    class pmStealAgent;
#endif

extern pmMachine* PM_LOCAL_MACHINE;
extern pmCluster* PM_GLOBAL_CLUSTER;

/**
 * \brief The representation of a parallel task.
 */

class pmTask : public pmBase
{
    friend void AddressSpacesLockCallback(const pmCommandPtr& pCountDownCommand);

	protected:
        pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, const pmCluster* pCluster, ushort pPriority, scheduler::schedulingModel pSchedulingModel, ushort pTaskFlags, pmAffinityCriterion pAffinityCriterion);

	public:
		virtual ~pmTask();

        const std::vector<pmTaskMemory>& GetTaskMemVector() const;
        pmAddressSpace* GetAddressSpace(size_t pIndex) const;
        size_t GetAddressSpaceCount() const;
        std::vector<pmAddressSpace*>& GetAddressSpaces();
        const std::vector<pmAddressSpace*>& GetAddressSpaces() const;
        uint GetAddressSpaceIndex(const pmAddressSpace* pAddressSpace) const;

		ulong GetTaskId() const;
		const pmCallbackUnit* GetCallbackUnit() const;
		ulong GetSubtaskCount() const;
		const pmMachine* GetOriginatingHost() const;
		const pmCluster* GetCluster() const;
		ushort GetPriority() const;
		uint GetAssignedDeviceCount() const;
		void* GetTaskConfiguration() const;
		uint GetTaskConfigurationLength() const;
		scheduler::schedulingModel GetSchedulingModel() const;
		pmTaskExecStats& GetTaskExecStats();
		const pmTaskInfo& GetTaskInfo() const;
        pmSubtaskInfo GetPreSubscriptionSubtaskInfo(ulong pSubtaskId, pmSplitInfo* pSplitInfo) const;
		pmSubscriptionManager& GetSubscriptionManager();

    #ifdef SUPPORT_SPLIT_SUBTASKS
        pmSubtaskSplitter& GetSubtaskSplitter();
    #endif
    
        pmReducer* GetReducer();
        pmRedistributor* GetRedistributor(const pmAddressSpace* pAddressSpace);
		bool HasSubtaskExecutionFinished();
		pmStatus IncrementSubtasksExecuted(ulong pSubtaskCount, ulong pTotalSplitCount);
		ulong GetSubtasksExecuted();
        ulong GetTotalSplitCount(ulong& pSubtasksSplitted);
		bool DoSubtasksNeedShadowMemory(const pmAddressSpace* pAddressSpace) const;
        bool CanForciblyCancelSubtasks();
        bool CanSplitCpuSubtasks();
        bool CanSplitGpuSubtasks();
        bool ShouldSuppressTaskLogs();
    
        bool IsReducible(const pmAddressSpace* pAddressSpace) const;
        bool IsRedistributable(const pmAddressSpace* pAddressSpace) const;
    
        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();

        virtual void MarkUserSideTaskCompletion() = 0;

        void RecordStubWillSendCancellationMessage();
        void MarkAllStubsScannedForCancellationMessages();
        void RegisterStubCancellationMessage();
    
        void RecordStubWillSendShadowMemCommitMessage();
        void MarkAllStubsScannedForShadowMemCommitMessages();
        void RegisterStubShadowMemCommitMessage();
    
        void* CheckOutSubtaskMemory(size_t pLength, uint pAddressSpaceIndex);
        void RepoolCheckedOutSubtaskMemory(uint pAddressSpaceIndex, void* pMem);

        void LockAddressSpaces();
        void UnlockMemories();
    
    #ifdef ENABLE_TWO_LEVEL_STEALING
        const std::vector<const pmMachine*>& GetStealListForDevice(const pmProcessingElement* pDevice);
    #else
        const std::vector<const pmProcessingElement*>& GetStealListForDevice(const pmProcessingElement* pDevice);
    #endif
    
        ulong GetSequenceNumber() const;
        void SetSequenceNumber(ulong pSequenceNumber);
    
        void FlushMemoryOwnerships();
        bool IsMultiAssignEnabled();
    
    #ifdef SUPPORT_CUDA
        bool IsCudaCacheEnabled();
    #endif
    
        bool HasStarted();
    
        void MarkRedistributionFinished(uint pOriginalAddressSpaceIndex, pmAddressSpace* pRedistributedAddressSpace = NULL);

        bool ShouldOverlapComputeCommunication() const;
        bool HasDisjointReadWritesAcrossSubtasks(const pmAddressSpace* pAddressSpace) const;

        pmSubscriptionVisibilityType GetAddressSpaceSubscriptionVisibility(const pmAddressSpace* pAddressSpace, const pmExecutionStub* pStub) const;
        pmMemType GetMemType(const pmAddressSpace* pAddressSpace) const;
    
        void* GetLastReductionScratchBuffer() const;

        bool IsReadOnly(const pmAddressSpace* pAddressSpace) const;
        bool IsWritable(const pmAddressSpace* pAddressSpace) const;
        bool IsWriteOnly(const pmAddressSpace* pAddressSpace) const;
        bool IsReadWrite(const pmAddressSpace* pAddressSpace) const;
        bool IsLazy(const pmAddressSpace* pAddressSpace) const;
        bool IsLazyWriteOnly(const pmAddressSpace* pAddressSpace) const;
        bool IsLazyReadWrite(const pmAddressSpace* pAddressSpace) const;

        bool IsOpenCLTask() const;
    
        bool HasReadOnlyLazyAddressSpace() const;
    
    #ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler* GetTaskProfiler();
    #endif
    
    #ifdef USE_STEAL_AGENT_PER_NODE
        pmStealAgent* GetStealAgent();
    #endif
    
        pmAffinityCriterion GetAffinityCriterion() const;

        void SetAffinityMappings(std::vector<ulong>&& pLogicalToPhysical, std::vector<ulong>&& pPhysicalToLogical);
        ulong GetPhysicalSubtaskId(ulong pLogicalSubtaskId);
        ulong GetLogicalSubtaskId(ulong pPhysicalSubtaskId);

    private:
		void BuildTaskInfo();
        void BuildPreSubscriptionSubtaskInfo();
    
        template<typename T>
        void RandomizeData(T& pData);
    
        void PrepareForStart();
    
        pmPoolAllocator& GetPoolAllocator(uint pAddressSpaceIndex, size_t pIndividualAllocationSize, size_t pMaxAllocations);
    
		/* Constant properties -- no updates, locking not required */
		ulong mTaskId;
		const pmCallbackUnit* mCallbackUnit;
		ulong mSubtaskCount;
		const pmMachine* mOriginatingHost;
		const pmCluster* mCluster;
		ushort mPriority;
		void* mTaskConf;
		uint mTaskConfLength;
		scheduler::schedulingModel mSchedulingModel;
		pmTaskInfo mTaskInfo;
        pmSubtaskInfo mPreSubscriptionSubtaskInfo;
		pmSubscriptionManager mSubscriptionManager;
		pmTaskExecStats mTaskExecStats;
        ulong mSequenceNumber;  // Sequence Id of task on originating host (This along with originating machine is the global unique identifier for a task)
        ushort mTaskFlags;
        bool mStarted;

    #ifdef SUPPORT_SPLIT_SUBTASKS
        pmSubtaskSplitter mSubtaskSplitter;
    #endif

    #ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler mTaskProfiler;
    #endif
    
		/* Updating properties require locking */
		ulong mSubtasksExecuted;
        ulong mTotalSplitCount;
        ulong mSubtasksSplitted;
		bool mSubtaskExecutionFinished;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mExecLock;

        finalize_ptr<pmReducer> mReducer;
        std::list<pmRedistributor> mRedistributorsList;
        std::map<const pmAddressSpace*, std::list<pmRedistributor>::iterator> mRedistributorsMap;
    
        uint mCompletedRedistributions; // How many address spaces have finished redistribution
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mRedistributionLock;

        bool mAllStubsScannedForCancellationMessages, mAllStubsScannedForShadowMemCommitMessages;
        ulong mOutstandingStubsForCancellationMessages, mOutstandingStubsForShadowMemCommitMessages;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTaskCompletionLock;

    #ifdef ENABLE_TWO_LEVEL_STEALING
        std::map<const pmProcessingElement*, std::vector<const pmMachine*>> mStealListForDevice;
    #else
        std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>> mStealListForDevice;
    #endif

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mStealListLock;
    
        std::vector<pmMemInfo> mPreSubscriptionMemInfoForSubtasks;  // Used for lazy memory

        std::map<uint, pmPoolAllocator> mPoolAllocatorMap;  // address space index vs pool allocator
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mPoolAllocatorMapLock;
    
        bool mTaskHasReadWriteAddressSpaceWithNonDisjointSubscriptions;

        std::vector<pmTaskMemory> mTaskMemVector;
        std::vector<pmAddressSpace*> mAddressSpaces;
        std::map<const pmAddressSpace*, size_t> mAddressSpaceTaskMemIndexMap;

    #ifdef USE_STEAL_AGENT_PER_NODE
        std::unique_ptr<pmStealAgent> mStealAgentPtr;
    #endif

        pmAffinityCriterion mAffinityCriterion;

        std::vector<ulong> mLogicalToPhysicalSubtaskMappings, mPhysicalToLogicalSubtaskMappings;
    
    protected:
        void Start();
        void CreateReducerAndRedistributors();
        bool DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() const;
        bool RegisterRedistributionCompletion();    // Returns true when all address spaces finish redistribution
        void ReplaceTaskAddressSpace(uint pAddressSpaceIndex, pmAddressSpace* pNewAddressSpace);
        const std::vector<ulong>& GetLogicalToPhysicalSubtaskMappings() const;
    
		uint mAssignedDeviceCount;
        void* mLastReductionScratchBuffer;
};

class pmLocalTask : public pmTask
{
	public:
        pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, int pTaskTimeOutInSecs, const pmMachine* pOriginatingHost = PM_LOCAL_MACHINE, const pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, ushort pTaskFlags = DEFAULT_TASK_FLAGS_VAL, pmAffinityCriterion pAffinityCriterion = MAX_AFFINITY_CRITERION, const std::set<const pmMachine*>& pRestrictToMachinesVector = std::set<const pmMachine*>());

        virtual ~pmLocalTask();

        const std::vector<const pmProcessingElement*>& FindCandidateProcessingElements(std::set<const pmMachine*>& pMachines, const std::set<const pmMachine*>& pRestrictToMachinesVector = std::set<const pmMachine*>());
        std::vector<const pmProcessingElement*> SelectMaxCpuDevicesPerHost(const std::vector<const pmProcessingElement*>& pDevicesVector, size_t pMaxCpuDevicesPerHost);

		void WaitForCompletion();
		double GetExecutionTimeInSecs();

		void MarkTaskStart();
		void MarkTaskEnd(pmStatus pStatus);

        void DoPostInternalCompletion();
        virtual void MarkUserSideTaskCompletion();
    
        void SaveFinalReducedOutput(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void AllReductionsDone(pmExecutionStub* pLastStub, ulong pLastSubtaskId, pmSplitInfo* pLastSplitInfo);
    
        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();
    
        void RegisterInternalTaskCompletionMessage();
        void UserDeleteTask();
    
        void SetPreprocessorTask(pmLocalTask* pLocalTask);
        const pmLocalTask* GetPreprocessorTask() const;
    
        pmAddressSpace* GetAffinityAddressSpace() const;
    
        void ComputeAffinityData(pmAddressSpace* pAffinityAddressSpace);
        void StartScheduling();
    
        void SetTaskCompletionCallback(pmTaskCompletionCallback pCallback);
    
		pmStatus GetStatus();

        const std::vector<const pmProcessingElement*>& GetAssignedDevices() const;
        const std::set<const pmMachine*>& GetAssignedMachines() const;
    
    #ifdef CENTRALIZED_AFFINITY_COMPUTATION
        const std::vector<const pmMachine*>& GetAssignedMachinesInOrder() const;
    #endif

		pmStatus InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel);
		pmSubtaskManager* GetSubtaskManager();
    
        ulong GetTaskTimeOutTriggerTime();
    
	private:
        pmCommandPtr mTaskCommand;
		finalize_ptr<pmSubtaskManager> mSubtaskManager;
        finalize_ptr<pmAffinityTable> mAffinityTable;
		std::vector<const pmProcessingElement*> mDevices;
        std::set<const pmMachine*> mMachines;
    #ifdef CENTRALIZED_AFFINITY_COMPUTATION
        std::vector<const pmMachine*> mMachinesInOrder;
    #endif
    
        ulong mTaskTimeOutTriggerTime;
        ulong mPendingCompletions;
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfCancellations, mLocalStubsFreeOfShadowMemCommits;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;
    
        pmLocalTask* mPreprocessorTask;
    
        static ulong& GetSequenceId();   // Task number at the originating host
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetSequenceLock();
};

class pmRemoteTask : public pmTask
{
	public:
        pmRemoteTask(finalize_ptr<char, deleteArrayDeallocator<char>>& pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, const pmMachine* pOriginatingHost, ulong pSequenceNumber, std::vector<const pmProcessingElement*>&& pDevices, const pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, ushort pTaskFlags = DEFAULT_TASK_FLAGS_VAL, pmAffinityCriterion pAffinityCriterion = MAX_AFFINITY_CRITERION);

        virtual ~pmRemoteTask();

        const std::vector<const pmProcessingElement*>& GetAssignedDevices() const;
        const std::set<const pmMachine*>& GetAssignedMachines() const;

        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();
    
        void DoPostInternalCompletion();
        virtual void MarkUserSideTaskCompletion();
        void MarkReductionFinished();
    
        void ReceiveAffinityData(std::vector<ulong>&& pLogicalToPhysicalSubtaskMapping, pmAddressSpace* pAffinityAddressSpace);
        pmAddressSpace* GetAffinityAddressSpace();
    
	private:
        finalize_ptr<char, deleteArrayDeallocator<char>> mTaskConfAutoPtr;
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfCancellations, mLocalStubsFreeOfShadowMemCommits;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;

        const std::vector<const pmProcessingElement*> mDevices;	// Only maintained for pull scheduling policy or if reduction is defined
        std::set<const pmMachine*> mMachines;
        pmAddressSpace* mAffinityAddressSpace;
    
    #ifdef USE_AFFINITY_IN_STEAL
        bool mAffinityAddressSpaceFetched;
    #endif
};

} // end namespace pm

#endif

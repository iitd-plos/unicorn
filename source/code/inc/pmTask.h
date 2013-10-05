
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
class pmMemSection;
class pmMachine;
class pmCluster;
class pmReducer;
class pmRedistributor;

extern pmMachine* PM_LOCAL_MACHINE;
extern pmCluster* PM_GLOBAL_CLUSTER;

/**
 * \brief The representation of a parallel task.
 */

class pmTask : public pmBase
{
	protected:
		pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, const pmCluster* pCluster, ushort pPriority, scheduler::schedulingModel pSchedulingModel, ushort pTaskFlags);

	public:
		virtual ~pmTask();

        pmMemSection* GetMemSection(size_t pIndex) const;
        size_t GetMemSectionCount() const;
        std::vector<pmMemSection*>& GetMemSections();
        const std::vector<pmMemSection*>& GetMemSections() const;
        uint GetMemSectionIndex(const pmMemSection* pMemSection) const;

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
        pmRedistributor* GetRedistributor(const pmMemSection* pMemSection);
		bool HasSubtaskExecutionFinished();
		pmStatus IncrementSubtasksExecuted(ulong pSubtaskCount);
		ulong GetSubtasksExecuted();
		bool DoSubtasksNeedShadowMemory(const pmMemSection* pMemSection) const;
        bool CanForciblyCancelSubtasks();
        bool CanSplitCpuSubtasks();
        bool CanSplitGpuSubtasks();
    
        bool IsReducible(const pmMemSection* pMemSection) const;
        bool IsRedistributable(const pmMemSection* pMemSection) const;
    
        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();
    
        void RecordStubWillSendCancellationMessage();
        void MarkAllStubsScannedForCancellationMessages();
        void RegisterStubCancellationMessage();
    
        void RecordStubWillSendShadowMemCommitMessage();
        void MarkAllStubsScannedForShadowMemCommitMessages();
        void RegisterStubShadowMemCommitMessage();
    
        void* CheckOutSubtaskMemory(size_t pLength, uint pMemSectionIndex);
        void RepoolCheckedOutSubtaskMemory(uint pMemSectionIndex, void* pMem);

        void UnlockMemories();
    
        std::vector<const pmProcessingElement*>& GetStealListForDevice(const pmProcessingElement* pDevice);
    
        ulong GetSequenceNumber();
        pmStatus SetSequenceNumber(ulong pSequenceNumber);
    
        pmStatus FlushMemoryOwnerships();
        bool IsMultiAssignEnabled();
    
        bool HasDisjointReadWritesAcrossSubtasks() const;
        bool ShouldOverlapComputeCommunication() const;

#ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler* GetTaskProfiler();
#endif
    
	private:
        void CreateReducerAndRedistributors();
		void BuildTaskInfo();
        void BuildPreSubscriptionSubtaskInfo();
        void RandomizeDevices(std::vector<const pmProcessingElement*>& pDevices);
    
        pmPoolAllocator& GetPoolAllocator(uint pMemSectionIndex, size_t pIndividualAllocationSize, size_t pMaxAllocations);

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
        bool mMultiAssignEnabled;
        bool mDisjointReadWritesAcrossSubtasks;  // for RW memory
        bool mOverlapComputeCommunication;
        bool mCanForciblyCancelSubtasks;
        bool mCanSplitCpuSubtasks;
        bool mCanSplitGpuSubtasks;

    #ifdef SUPPORT_SPLIT_SUBTASKS
        pmSubtaskSplitter mSubtaskSplitter;
    #endif

    #ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler mTaskProfiler;
    #endif
    
		/* Updating properties require locking */
		ulong mSubtasksExecuted;
		bool mSubtaskExecutionFinished;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mExecLock;

        finalize_ptr<pmReducer> mReducer;    
        std::map<const pmMemSection*, pmRedistributor> mRedistributorsMap;
    
        uint mCompletedRedistributions; // How many mem sections have finished redistribution
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mRedistributionLock;

        bool mAllStubsScannedForCancellationMessages, mAllStubsScannedForShadowMemCommitMessages;
        ulong mOutstandingStubsForCancellationMessages, mOutstandingStubsForShadowMemCommitMessages;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTaskCompletionLock;

        std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*> > mStealListForDevice;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mStealListLock;
    
        std::vector<pmMemInfo> mPreSubscriptionMemInfoForSubtasks;  // Used for lazy memory

        std::map<uint, pmPoolAllocator> mPoolAllocatorMap;  // mem section index vs pool allocator
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mPoolAllocatorMapLock;
    
        bool mTaskHasReadWriteMemSectionWithDisjointSubscriptions;

    protected:
        bool DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() const;
        bool RegisterRedistributionCompletion();    // Returns true when all mem sections finish redistribution
    
        std::vector<pmMemSection*> mMemSections;
		uint mAssignedDeviceCount;
};

class pmLocalTask : public pmTask
{
	public:
		pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, int pTaskTimeOutInSecs, const pmMachine* pOriginatingHost = PM_LOCAL_MACHINE, const pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, ushort pTaskFlags = DEFAULT_TASK_FLAGS_VAL);

        virtual ~pmLocalTask();

        const std::vector<const pmProcessingElement*>& FindCandidateProcessingElements(std::set<const pmMachine*>& pMachines);

		void WaitForCompletion();
		double GetExecutionTimeInSecs();

		void MarkTaskStart();
		void MarkTaskEnd(pmStatus pStatus);

        void DoPostInternalCompletion();
        void MarkUserSideTaskCompletion();
    
        void TaskRedistributionDone(uint pOriginalMemSectionIndex, pmMemSection* pRedistributedMemSection);
        void SaveFinalReducedOutput(pmExecutionStub* pStub, pmMemSection* pMemSection, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();
    
        void RegisterInternalTaskCompletionMessage();
        void UserDeleteTask();
    
		pmStatus GetStatus();

		std::vector<const pmProcessingElement*>& GetAssignedDevices();

		pmStatus InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel);
		pmSubtaskManager* GetSubtaskManager();
    
        ulong GetTaskTimeOutTriggerTime();
    
	private:
        pmCommandPtr mTaskCommand;
		finalize_ptr<pmSubtaskManager> mSubtaskManager;
		std::vector<const pmProcessingElement*> mDevices;
        ulong mTaskTimeOutTriggerTime;

        ulong mPendingCompletions;
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfCancellations, mLocalStubsFreeOfShadowMemCommits;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;
    
        static ulong& GetSequenceId();   // Task number at the originating host
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetSequenceLock();
};

class pmRemoteTask : public pmTask
{
	public:
        pmRemoteTask(finalize_ptr<char, deleteArrayDeallocator<char> >& pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, ulong pSequenceNumber, const pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, ushort pTaskFlags = DEFAULT_TASK_FLAGS_VAL);

        virtual ~pmRemoteTask();

        void AddAssignedDevice(const pmProcessingElement* pDevice);
		std::vector<const pmProcessingElement*>& GetAssignedDevices();

        virtual void MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfCancellations();
        virtual void MarkLocalStubsFreeOfShadowMemCommits();
    
        void DoPostInternalCompletion();
        void MarkUserSideTaskCompletion();
        void MarkReductionFinished();
        void MarkRedistributionFinished(uint pOriginalMemSectionIndex, pmMemSection* pRedistributedMemSection = NULL);

	private:
        finalize_ptr<char, deleteArrayDeallocator<char> > mTaskConfAutoPtr;
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfCancellations, mLocalStubsFreeOfShadowMemCommits;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;

        std::vector<const pmProcessingElement*> mDevices;	// Only maintained for pull scheduling policy or if reduction is defined
};

} // end namespace pm

#endif


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

#ifdef ENABLE_TASK_PROFILING
    #include "pmTaskProfiler.h"
#endif

#include <map>
#include <set>
#include <vector>

namespace pm
{

class pmCallbackUnit;
class pmTaskCommand;
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
		pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, pmMemInfo pInputMemInfo, pmMemInfo pOutputMemInfo, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, pmMachine* pOriginatingHost, pmCluster* pCluster, ushort pPriority, scheduler::schedulingModel pSchedulingModel, bool pMultiAssignEnabled);

	public:		
		virtual ~pmTask();

		ulong GetTaskId();
		pmMemSection* GetMemSectionRO();
		pmMemSection* GetMemSectionRW();
		pmCallbackUnit* GetCallbackUnit();
		ulong GetSubtaskCount();
		pmMachine* GetOriginatingHost();
		pmCluster* GetCluster();
		ushort GetPriority();
		uint GetAssignedDeviceCount();
		void* GetTaskConfiguration();
		uint GetTaskConfigurationLength();
		scheduler::schedulingModel GetSchedulingModel();
		pmTaskExecStats& GetTaskExecStats();
		pmTaskInfo& GetTaskInfo();
		pmStatus GetSubtaskInfo(pmExecutionStub* pStub, ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo, bool& pOutputMemWriteOnly);
		pmSubscriptionManager& GetSubscriptionManager();
        pmReducer* GetReducer();
        pmRedistributor* GetRedistributor();
		bool HasSubtaskExecutionFinished();
		pmStatus IncrementSubtasksExecuted(ulong pSubtaskCount);
		ulong GetSubtasksExecuted();
		bool DoSubtasksNeedShadowMemory();
    
        virtual pmStatus MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfTask();
    
        void RecordStubWillSendCancellationMessage();
        void MarkAllStubsScannedForCancellationMessages();
        void RegisterStubFreeOfTask();
    
        void UnlockMemories();
    
        std::vector<pmProcessingElement*>& GetStealListForDevice(pmProcessingElement* pDevice);
    
        ulong GetSequenceNumber();
        pmStatus SetSequenceNumber(ulong pSequenceNumber);
    
        pmStatus FlushMemoryOwnerships();
        bool IsMultiAssignEnabled();

#ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler* GetTaskProfiler();
#endif
    
    protected:
		pmMemSection* mMemRW;
    
	private:
		pmStatus BuildTaskInfo();
        pmStatus RandomizeDevices(std::vector<pmProcessingElement*>& pDevices);

		/* Constant properties -- no updates, locking not required */
		ulong mTaskId;
		pmMemSection* mMemRO;
		pmCallbackUnit* mCallbackUnit;
		ulong mSubtaskCount;
		pmMachine* mOriginatingHost;
		pmCluster* mCluster;
		ushort mPriority;
		void* mTaskConf;
		uint mTaskConfLength;
		scheduler::schedulingModel mSchedulingModel;
		pmTaskInfo mTaskInfo;
		pmSubscriptionManager mSubscriptionManager;
		pmTaskExecStats mTaskExecStats;
        ulong mSequenceNumber;  // Sequence Id of task on originating host (This along with originating machine is the global unique identifier for a task)
        bool mMultiAssignEnabled;
        void* mReadOnlyMemAddrForSubtasks;  // Stores the read only lazy memory address (if present)
    
#ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler mTaskProfiler;
#endif

		/* Updating properties require locking */
		ulong mSubtasksExecuted;
		bool mSubtaskExecutionFinished;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mExecLock;

        finalize_ptr<pmReducer> mReducer;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mReducerLock;
    
        finalize_ptr<pmRedistributor> mRedistributor;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mRedistributorLock;

        bool mAllStubsScanned;
        ulong mOutstandingStubs;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTaskCompletionLock;

        std::map<pmProcessingElement*, std::vector<pmProcessingElement*> > mStealListForDevice;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mStealListLock;
    
    protected:
		uint mAssignedDeviceCount;
};

class pmLocalTask : public pmTask
{
	public:
		pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, pmMemInfo pInputMemInfo, pmMemInfo pOutputMemInfo, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, int pTaskTimeOutInSecs, pmMachine* pOriginatingHost = PM_LOCAL_MACHINE,	pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, bool pMultiAssignEnabled = true);

		pmStatus FindCandidateProcessingElements(std::set<pmProcessingElement*>& pDevices);

		pmStatus WaitForCompletion();
		double GetExecutionTimeInSecs();

		pmStatus MarkTaskStart();
		pmStatus MarkTaskEnd(pmStatus pStatus);

        void DoPostInternalCompletion();
        void MarkUserSideTaskCompletion();
    
        void TaskRedistributionDone(pmMemSection* pRedistributedMemSection);
        pmStatus SaveFinalReducedOutput(pmExecutionStub* pStub, ulong pSubtaskId);
    
        virtual pmStatus MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfTask();
    
        void RegisterInternalTaskCompletionMessage();
        void UserDeleteTask();
    
		pmStatus GetStatus();

		std::vector<pmProcessingElement*>& GetAssignedDevices();

		pmStatus InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel);
		pmSubtaskManager* GetSubtaskManager();
    
        ulong GetTaskTimeOutTriggerTime();

	private:
        virtual ~pmLocalTask();
    
		pmTaskCommandPtr mTaskCommand;
		finalize_ptr<pmSubtaskManager> mSubtaskManager;
		std::vector<pmProcessingElement*> mDevices;
        ulong mTaskTimeOutTriggerTime;

        ulong mPendingCompletions;
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfTask;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;
    
        static ulong mSequenceId;   // Task number at the originating host
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS mSequenceLock;
};

class pmRemoteTask : public pmTask
{
	public:
		pmRemoteTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, pmMemInfo pInputMemInfo, pmMemInfo pOutputMemInfo, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, pmMachine* pOriginatingHost, ulong pSequenceNumber, pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL, bool pMultiAssignEnabled = true);

		pmStatus AddAssignedDevice(pmProcessingElement* pDevice);
		std::vector<pmProcessingElement*>& GetAssignedDevices();

        virtual pmStatus MarkSubtaskExecutionFinished();
        virtual void TerminateTask();
        virtual void MarkLocalStubsFreeOfTask();
    
        void MarkUserSideTaskCompletion();
        void MarkReductionFinished();
        void MarkRedistributionFinished();

	private:
        virtual ~pmRemoteTask();
    
        bool mUserSideTaskCompleted;
        bool mLocalStubsFreeOfTask;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mCompletionLock;

        std::vector<pmProcessingElement*> mDevices;	// Only maintained for pull scheduling policy or if reduction is defined
};

} // end namespace pm

#endif


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
		pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, pmMachine* pOriginatingHost = PM_LOCAL_MACHINE, pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL, scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL);

	public:		
		virtual ~pmTask();

		typedef struct subtaskShadowMem
		{
			char* addr;
			size_t length;
		} subtaskShadowMem;

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
		pmStatus GetSubtaskInfo(ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo, bool& pOutputMemWriteOnly);
		pmSubscriptionManager& GetSubscriptionManager();
        pmReducer* GetReducer();
        pmRedistributor* GetRedistributor();
		virtual pmStatus MarkSubtaskExecutionFinished();
		bool HasSubtaskExecutionFinished();
		pmStatus IncrementSubtasksExecuted(ulong pSubtaskCount);
		ulong GetSubtasksExecuted();
		pmStatus SaveFinalReducedOutput(ulong pSubtaskId);

		bool DoSubtasksNeedShadowMemory();
		pmStatus CreateSubtaskShadowMem(ulong pSubtaskId);
		pmStatus CreateSubtaskShadowMem(ulong pSubtaskId, char* pMem, size_t pMemLength);
		subtaskShadowMem& GetSubtaskShadowMem(ulong pSubtaskId);
		pmStatus DestroySubtaskShadowMem(ulong pSubtaskId);
    
        std::vector<pmProcessingElement*>& GetStealListForDevice(pmProcessingElement* pDevice);
    
        ulong GetSequenceNumber();
        pmStatus SetSequenceNumber(ulong pSequenceNumber);
    
        bool ShouldStartFaultTolerance();
        pmStatus TaskInternallyFinished();

#ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler* GetTaskProfiler();
#endif
    
	private:
		pmStatus BuildTaskInfo();
        pmStatus RandomizeDevices(std::vector<pmProcessingElement*>& pDevices);

		/* Constant properties -- no updates, locking not required */
		ulong mTaskId;
		pmMemSection* mMemRO;
		pmMemSection* mMemRW;
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
    
#ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler mTaskProfiler;
#endif

		/* Updating properties require locking */
		ulong mSubtasksExecuted;
		bool mSubtaskExecutionFinished;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mExecLock;

		std::map<ulong, subtaskShadowMem> mShadowMemMap;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mShadowMemLock;

        pmReducer* mReducer;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mReducerLock;
    
        pmRedistributor* mRedistributor;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mRedistributorLock;
    
        std::map<pmProcessingElement*, std::vector<pmProcessingElement*> > mStealListForDevice;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mStealListLock;
    
        bool mCanStartFaultTolerance;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mFaultToleranceLock;    

    protected:
		uint mAssignedDeviceCount;
};

class pmLocalTask : public pmTask
{
	public:
		pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, 
			pmMachine* pOriginatingHost = PM_LOCAL_MACHINE,	pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL,
			scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL);

		virtual ~pmLocalTask();

		pmStatus FindCandidateProcessingElements(std::set<pmProcessingElement*>& pDevices);

		pmStatus WaitForCompletion();
		double GetExecutionTimeInSecs();

		pmStatus MarkTaskStart();
		pmStatus MarkTaskEnd(pmStatus pStatus);

		virtual pmStatus MarkSubtaskExecutionFinished();
        pmStatus CompleteTask();

		pmStatus GetStatus();

		std::vector<pmProcessingElement*>& GetAssignedDevices();

		pmStatus InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel);
		pmSubtaskManager* GetSubtaskManager();

	private:
		pmTaskCommandPtr mTaskCommand;
		pmSubtaskManager* mSubtaskManager;
		std::vector<pmProcessingElement*> mDevices;

        static RESOURCE_LOCK_IMPLEMENTATION_CLASS mSequenceLock;
        static ulong mSequenceId;   // Task number at the originating host
};

class pmRemoteTask : public pmTask
{
	public:
		pmRemoteTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, 
			pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, pmMachine* pOriginatingHost, ulong pSequenceNumber,
			pmCluster* pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = DEFAULT_PRIORITY_LEVEL,
			scheduler::schedulingModel pSchedulingModel = DEFAULT_SCHEDULING_MODEL);

		virtual ~pmRemoteTask();

		pmStatus AddAssignedDevice(pmProcessingElement* pDevice);
		std::vector<pmProcessingElement*>& GetAssignedDevices();

		virtual pmStatus MarkSubtaskExecutionFinished();

	private:
		std::vector<pmProcessingElement*> mDevices;	// Only maintained for pull scheduling policy or if reduction is defined
};

} // end namespace pm

#endif

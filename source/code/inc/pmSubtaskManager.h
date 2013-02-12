
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

#ifndef __PM_SUBTASK_MANAGER__
#define __PM_SUBTASK_MANAGER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"

#include <set>
#include <map>

#include <tr1/memory>	// For std::tr1

namespace pm
{

class pmLocalTask;
class pmSubtaskRangeCommand;
class pmProcessingElement;

class pmSubtaskManager : public pmBase
{
	public:
        typedef struct pmUnfinishedPartition
        {
            ulong firstSubtaskIndex;	// inclusive
            ulong lastSubtaskIndex;		// inclusive
            pmProcessingElement* originalAllottee;
            std::vector<pmProcessingElement*> secondaryAllottees;   // used only by originalAllottee (i.e. when originalAllottee is NULL)
            
            pmUnfinishedPartition(ulong pFirstSubtaskIndex, ulong pLastSubtaskIndex, pmProcessingElement* pOriginalAllottee = NULL);
        } pmSubtaskPartition;

        typedef std::tr1::shared_ptr<pmUnfinishedPartition> pmUnfinishedPartitionPtr;
    
        typedef struct partitionSorter : std::binary_function<pmUnfinishedPartitionPtr, pmUnfinishedPartitionPtr, bool>
        {
            bool operator() (const pmUnfinishedPartitionPtr& pPartition1Ptr, const pmUnfinishedPartitionPtr& pPartition2Ptr) const;
        } partitionSorter;
        
        virtual ~pmSubtaskManager();
		
		virtual pmStatus GetTaskExecutionStatus();

		virtual bool HasTaskFinished() = 0;
		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee) = 0;
		virtual pmStatus RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus) = 0;

	protected:
		pmSubtaskManager(pmLocalTask* pLocalTask);

        pmStatus UpdateExecutionProfile(pmProcessingElement* pDevice, ulong pSubtaskCount);
    
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
        pmStatus PrintExecutionProfile();
#endif
    
		pmLocalTask* mLocalTask;
		pmStatus mTaskStatus;

    protected:
        std::map<uint, ulong> mDeviceExecutionProfile;    // Global Device Index versus Subtasks Executed
        std::map<uint, ulong> mMachineExecutionProfile;    // Machine Index versus Subtasks Executed
    
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
        bool mExecutionProfilePrinted;
#endif

    protected:
        typedef struct execCountSorter : std::binary_function<pmProcessingElement*, pmProcessingElement*, bool>
        {
            execCountSorter(std::map<uint, ulong>& pDeviceExecutionProfile);
            
            bool operator() (pmProcessingElement* pDevice1, pmProcessingElement* pDevice2) const;
            
            private:
                std::map<uint, ulong>& mDeviceExecutionProfile;
        } execCountSorter;
        
        execCountSorter mExecCountSorter;
        std::set<pmProcessingElement*, execCountSorter> mOrderedDevices;   // Devices sorted in increasing order of subtasks finished
};

class pmPushSchedulingManager : public pmSubtaskManager
{
	public:
		pmPushSchedulingManager(pmLocalTask* pLocalTask);
		virtual ~pmPushSchedulingManager();

		virtual bool HasTaskFinished();
		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee);
		virtual pmStatus RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus);
    
	private:
		pmStatus FreezeAllocationSize(pmProcessingElement* pDevice, ulong pFreezedSize);
		pmStatus UnfreezeAllocationSize(pmProcessingElement* pDevice);
		bool IsAllocationSizeFreezed(pmProcessingElement* pDevice);
		ulong GetFreezedAllocationSize(pmProcessingElement* pDevice);
		pmStatus SetLastAllocationExecTimeInSecs(pmProcessingElement* pDevice, double pTimeInSecs);
		double GetLastAllocationExecTimeInSecs(pmProcessingElement* pDevice);
		ulong GetNextAssignmentSize(pmProcessingElement* pDevice);
		pmStatus AssignPartition(pmProcessingElement* pDevice, pmUnfinishedPartitionPtr pUnfinishedPartitionPtr, ulong pSubtaskCount);
		pmUnfinishedPartitionPtr FetchNewSubPartition(pmProcessingElement* pDevice, ulong pSubtaskCount);
        bool IsUsefulAllottee(pmProcessingElement* pPotentialAllottee, pmProcessingElement* pOriginalAllottee, std::vector<pmProcessingElement*>& pExistingAllottees);
        pmProcessingElement* SelectMultiAssignAllottee(pmProcessingElement* pDevice);
        void CancelOriginalAllottee(pmProcessingElement* pOriginalAllottee, ulong pSubtaskCount, ulong pStartingSubtask);
        void CancelAllButOneSecondaryAllottee(pmProcessingElement* pOriginalAllottee, pmProcessingElement* pPreserveSecondaryAllottee, ulong pSubtaskCount, ulong pStartingSubtask);
        void UpdateAssignedPartition(pmProcessingElement* pDevice, ulong pStartingSubtask, ulong pLastSubtask);
    
		std::map<pmUnfinishedPartitionPtr, pmProcessingElement*, partitionSorter> mSortedUnassignedPartitions;	// Partitions with more pending subtasks are at the end
		std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, ulong> > mAllottedUnassignedPartition;	// Partition and No. of subtasks allotted to this device (usually a power of 2; unless at partition boundary)
		std::map<pmProcessingElement*, std::pair<double, ulong> > mExecTimeStats;	// Mapping from device to last exec time in secs and freezed allocation size
		std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> > mAssignedPartitions;		// Collection of all assigned partitions and corresponding devices

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmSingleAssignmentSchedulingManager : public pmSubtaskManager
{
public:
    pmSingleAssignmentSchedulingManager(pmLocalTask* pLocalTask);
    virtual ~pmSingleAssignmentSchedulingManager();	
    
    virtual bool HasTaskFinished();
    virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee) = 0;

    virtual pmStatus RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus);
    
#ifdef _DEBUG
    void DumpUnacknowledgedPartitions();
#endif
    
private:
    bool HasTaskFinished_Internal();

    std::set<pmUnfinishedPartitionPtr> mUnacknowledgedPartitions;   // Collection of all unacknowledged partitions
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;    
};
    
class pmPullSchedulingManager : public pmSingleAssignmentSchedulingManager
{
	public:
		pmPullSchedulingManager(pmLocalTask* pLocalTask);
		virtual ~pmPullSchedulingManager();	

		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee);

	private:
		std::set<pmUnfinishedPartitionPtr> mSubtaskPartitions;		// Collection of partitions to be assigned to devices
		std::set<pmUnfinishedPartitionPtr>::iterator mIter;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mAssignmentResourceLock;
};

class pmProportionalSchedulingManager : public pmSingleAssignmentSchedulingManager
{
public:
    pmProportionalSchedulingManager(pmLocalTask* pLocalTask);
    virtual ~pmProportionalSchedulingManager();	
    
    virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee);
    
private:
    pmStatus ReadConfigurationFile(std::vector<pmProcessingElement*>& pDevices);
    uint GetDevicePower(pmProcessingElement* pDevice);
    ulong FindDeviceAssignment(pmProcessingElement* pDevice, ulong pSubtaskCount);

    uint mLocalCpuPower, mRemoteCpuPower;
#ifdef SUPPORT_CUDA
    uint mLocalGpuPower, mRemoteGpuPower;
#endif

    bool mExactMode;
    double mTotalClusterPower;
    
    std::map<uint, ulong> mExactPartitions;
    ulong mExactCount;
    
    std::map<pmProcessingElement*, pmUnfinishedPartitionPtr> mDevicePartitionMap;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mAssignmentResourceLock;
};

} // end namespace pm

#endif

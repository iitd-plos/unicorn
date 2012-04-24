
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

			pmUnfinishedPartition(ulong pFirstSubtaskIndex, ulong pLastSubtaskIndex);
		} pmSubtaskPartition;
    
        typedef std::tr1::shared_ptr<pmUnfinishedPartition> pmUnfinishedPartitionPtr;

		virtual ~pmSubtaskManager();
		
		virtual pmStatus GetTaskExecutionStatus();

		virtual bool HasTaskFinished() = 0;
		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask) = 0;
		virtual pmStatus RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus) = 0;

	protected:
		pmSubtaskManager(pmLocalTask* pLocalTask);

#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
        pmStatus UpdateExecutionProfile(pmProcessingElement* pDevice, ulong pSubtaskCount);
        pmStatus PrintExecutionProfile();
#endif
    
		pmLocalTask* mLocalTask;
		pmStatus mTaskStatus;

    private:
    
#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
        std::map<uint, ulong> mDeviceExecutionProfile;    // Global Device Index versus Subtasks Executed
        std::map<uint, ulong> mMachineExecutionProfile;    // Machine Index versus Subtasks Executed
        bool mExecutionProfilePrinted;
#endif    
};

class pmPushSchedulingManager : public pmSubtaskManager
{
	public:
        typedef struct partitionSorter : std::binary_function<pmPushSchedulingManager::pmUnfinishedPartitionPtr, pmPushSchedulingManager::pmUnfinishedPartitionPtr, bool>
		{
			bool operator() (const pmPushSchedulingManager::pmUnfinishedPartitionPtr& pPartition1Ptr, const pmPushSchedulingManager::pmUnfinishedPartitionPtr& pPartition2Ptr) const;
		} partitionSorter;

		pmPushSchedulingManager(pmLocalTask* pLocalTask);
		virtual ~pmPushSchedulingManager();

		virtual bool HasTaskFinished();

		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask);
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

		std::map<pmUnfinishedPartitionPtr, pmProcessingElement*, partitionSorter> mSortedUnassignedPartitions;	// Partitions with more pending subtasks are at the top
		std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, ulong> > mAllottedUnassignedPartition;	// Partition and No. of subtasks allotted to this device (usually a power of 2; unless at partition boundary)
		std::map<pmProcessingElement*, std::pair<double, ulong> > mExecTimeStats;	// Mapping from device to last exec time in secs and freezed allocation size

		std::set<pmUnfinishedPartitionPtr> mUnassignedPartitions;		// Collection of all unassigned partitions
		std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> > mAssignedPartitions;		// Collection of all assigned partitions and corresponding devices

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmSingleAssignmentSchedulingManager : public pmSubtaskManager
{
public:
    pmSingleAssignmentSchedulingManager(pmLocalTask* pLocalTask);
    virtual ~pmSingleAssignmentSchedulingManager();	
    
    virtual bool HasTaskFinished();
    
    virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask) = 0;
    virtual pmStatus RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus);
    
private:
    bool HasTaskFinished_Internal();

    std::set<pmUnfinishedPartitionPtr> mUnacknowledgedPartitions;		// Collection of all unacknowledged partitions
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;    
};
    
class pmPullSchedulingManager : public pmSingleAssignmentSchedulingManager
{
	public:
		pmPullSchedulingManager(pmLocalTask* pLocalTask);
		virtual ~pmPullSchedulingManager();	

		virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask);

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
    
    virtual pmStatus AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask);
    
private:
    pmStatus ReadConfigurationFile(std::vector<pmProcessingElement*>& pDevices);
    uint GetDevicePower(pmProcessingElement* pDevice);
    ulong FindDeviceAssignment(pmProcessingElement* pDevice, ulong pSubtaskCount);

    uint mLocalCpuPower, mRemoteCpuPower;
#ifdef SUPPORT_CUDA
    uint mLocalGpuPower, mRemoteGpuPower;
#endif
    
    double mTotalClusterPower;
    
    std::map<pmProcessingElement*, pmUnfinishedPartitionPtr> mDevicePartitionMap;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mAssignmentResourceLock;
};

} // end namespace pm

#endif

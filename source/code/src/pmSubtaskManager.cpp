
#include "pmSubtaskManager.h"
#include "pmTask.h"
#include "pmCommand.h"
#include "pmHardware.h"

#include <string.h>

namespace pm
{

/* class pmSubtaskManager */
pmSubtaskManager::pmSubtaskManager(pmLocalTask* pLocalTask)
{
	mLocalTask= pLocalTask;
	mTaskStatus = pmStatusUnavailable;

#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
    mExecutionProfilePrinted = false;
#endif
}

pmSubtaskManager::~pmSubtaskManager()
{
}

// Returns last failure status or pmSuccess
pmStatus pmSubtaskManager::GetTaskExecutionStatus()
{
	if(HasTaskFinished() && mTaskStatus == pmStatusUnavailable)
		mTaskStatus = pmSuccess;

	return mTaskStatus;
}


/* struct pmUnfinishedPartition */
pmSubtaskManager::pmUnfinishedPartition::pmUnfinishedPartition(ulong pFirstSubtaskIndex, ulong pLastSubtaskIndex)
{
	firstSubtaskIndex = pFirstSubtaskIndex;
	lastSubtaskIndex = pLastSubtaskIndex;
    
	if(lastSubtaskIndex < firstSubtaskIndex)
		PMTHROW(pmFatalErrorException());
}

#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
// This method must only be called from RegisterSubtaskCompletion method of the subclasses as that acquires the lock and ensures synchronization
pmStatus pmSubtaskManager::UpdateExecutionProfile(pmProcessingElement* pDevice, ulong pSubtaskCount)
{
    uint lDeviceIndex = pDevice->GetGlobalDeviceIndex();
    uint lMachineIndex = (uint)(*(pDevice->GetMachine()));

    if(mDeviceExecutionProfile.find(lDeviceIndex) == mDeviceExecutionProfile.end())
        mDeviceExecutionProfile[lDeviceIndex] = 0;
    
    if(mMachineExecutionProfile.find(lMachineIndex) == mMachineExecutionProfile.end())
        mMachineExecutionProfile[lMachineIndex] = 0;

    mDeviceExecutionProfile[lDeviceIndex] += pSubtaskCount;
    mMachineExecutionProfile[lMachineIndex] += pSubtaskCount;
    
    return pmSuccess;
}
    
pmStatus pmSubtaskManager::PrintExecutionProfile()
{
    if(mExecutionProfilePrinted)
        return pmSuccess;
    
    mExecutionProfilePrinted = true;
    
    std::map<uint, ulong>::iterator lStart, lEnd;
    lStart = mDeviceExecutionProfile.begin();
    lEnd = mDeviceExecutionProfile.end();
    
    std::cout << "Device Subtask Execution Profile ... " << std::endl;
    for(; lStart != lEnd; ++lStart)
        std::cout << "Device " << lStart->first << " Subtasks " << lStart->second << std::endl;

    std::cout << std::endl;
    
    std::cout << "Machine Subtask Execution Profile ... " << std::endl;
    ulong lTotal = 0;
    lStart = mMachineExecutionProfile.begin();
    lEnd = mMachineExecutionProfile.end();
    for(; lStart != lEnd; ++lStart)
    {
        std::cout << "Machine " << lStart->first << " Subtasks " << lStart->second << std::endl;
        lTotal += lStart->second;
    }
    
    std::cout << std::endl;
    
    std::cout << "Total Acknowledgements Received " << lTotal << std::endl; 
    
    return pmSuccess;
}
#endif


/* class pmPushSchedulingManager */
pmPushSchedulingManager::pmPushSchedulingManager(pmLocalTask* pLocalTask)
	: pmSubtaskManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
	ulong lPartitionCount = mLocalTask->GetAssignedDeviceCount();

	if(lSubtaskCount == 0 || lPartitionCount == 0 || lSubtaskCount < lPartitionCount)
		PMTHROW(pmFatalErrorException());

	ulong lPartitionSize = lSubtaskCount/lPartitionCount;
	ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

	ulong lFirstSubtask = 0;
	ulong lLastSubtask = 0;
	
	std::vector<pmProcessingElement*>& lDevices = mLocalTask->GetAssignedDevices();
	std::vector<pmProcessingElement*>::iterator lIter = lDevices.begin();
	std::vector<pmProcessingElement*>::iterator lEndIter = lDevices.end();

    for(ulong i=0; i<lPartitionCount; ++i, ++lIter)
    {
        if(i < lLeftoverSubtasks)
            lLastSubtask = lFirstSubtask + lPartitionSize;
        else
            lLastSubtask = lFirstSubtask + lPartitionSize - 1;

        pmSubtaskManager::pmUnfinishedPartitionPtr lUnfinishedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(lFirstSubtask, lLastSubtask));

        mSortedUnassignedPartitions[lUnfinishedPartitionPtr] = *lIter;
        mUnassignedPartitions.insert(lUnfinishedPartitionPtr);
        mAllottedUnassignedPartition[*lIter] = std::make_pair(lUnfinishedPartitionPtr, (ulong)0);

        mExecTimeStats[*lIter] = std::pair<double, ulong>(0, 0);

        lFirstSubtask = lLastSubtask + 1;

        assert(lIter != lEndIter);
    }
}

pmPushSchedulingManager::~pmPushSchedulingManager()
{
    mSortedUnassignedPartitions.clear();
    mUnassignedPartitions.clear();
    mAllottedUnassignedPartition.clear();
    mAssignedPartitions.clear();
    mExecTimeStats.clear();
}

bool pmPushSchedulingManager::HasTaskFinished()
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mUnassignedPartitions.empty() && mAssignedPartitions.empty())
    {
#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
        PrintExecutionProfile();
#endif

        return true;
    }
    
    return false;
}

pmStatus pmPushSchedulingManager::AssignPartition(pmProcessingElement* pDevice, pmUnfinishedPartitionPtr pUnfinishedPartitionPtr, ulong pSubtaskCount)
{
	ulong lAvailableSubtasks = pUnfinishedPartitionPtr->lastSubtaskIndex - pUnfinishedPartitionPtr->firstSubtaskIndex + 1;
    ulong lCount = mAllottedUnassignedPartition[pDevice].second;

#ifdef _DEBUG
    if(mUnassignedPartitions.find(pUnfinishedPartitionPtr) == mUnassignedPartitions.end())
        PMTHROW(pmFatalErrorException());
#endif
    
    mAllottedUnassignedPartition.erase(pDevice);
	mUnassignedPartitions.erase(pUnfinishedPartitionPtr);
	mSortedUnassignedPartitions.erase(pUnfinishedPartitionPtr);

	if(lAvailableSubtasks == pSubtaskCount)
	{
		mAllottedUnassignedPartition[pDevice] = std::make_pair(pmSubtaskManager::pmUnfinishedPartitionPtr(), lCount);
	}
	else
	{
        if(lAvailableSubtasks < pSubtaskCount)
            PMTHROW(pmFatalErrorException());
        
		pmUnfinishedPartitionPtr lSubPartitionPtr(new pmUnfinishedPartition(pUnfinishedPartitionPtr->firstSubtaskIndex + pSubtaskCount, pUnfinishedPartitionPtr->lastSubtaskIndex));
        
		mAllottedUnassignedPartition[pDevice] = std::make_pair(lSubPartitionPtr, lCount);
		mUnassignedPartitions.insert(lSubPartitionPtr);
        mSortedUnassignedPartitions[lSubPartitionPtr] = pDevice;
	}

	pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(mLocalTask->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);
	pmUnfinishedPartitionPtr lPartitionPtr(new pmUnfinishedPartition(pUnfinishedPartitionPtr->firstSubtaskIndex, pUnfinishedPartitionPtr->firstSubtaskIndex + pSubtaskCount - 1));

	mAssignedPartitions[pDevice] = std::make_pair(lPartitionPtr, lCommand);
	lCommand->MarkExecutionStart();

	return pmSuccess;
}

pmSubtaskManager::pmUnfinishedPartitionPtr pmPushSchedulingManager::FetchNewSubPartition(pmProcessingElement* pDevice, ulong pSubtaskCount)
{
	if(mSortedUnassignedPartitions.empty())
	   return pmUnfinishedPartitionPtr();

    // Heaviest partition is at the end
	std::map<pmUnfinishedPartitionPtr, pmProcessingElement*, partitionSorter>::reverse_iterator lIter = mSortedUnassignedPartitions.rbegin();

	pmUnfinishedPartitionPtr lMaxPendingPartitionPtr = lIter->first;
	pmProcessingElement* lSlowestDevice = lIter->second;

	if(lSlowestDevice == pDevice)
		PMTHROW(pmFatalErrorException());

	ulong lMaxAvailableSubtasks = lMaxPendingPartitionPtr->lastSubtaskIndex - lMaxPendingPartitionPtr->firstSubtaskIndex + 1;
    ulong lStartSubtask, lEndSubtask;
    
    ulong lSlowestDeviceCount = mAllottedUnassignedPartition[lSlowestDevice].second;

#ifdef _DEBUG
    if(mUnassignedPartitions.find(lMaxPendingPartitionPtr) == mUnassignedPartitions.end())
        PMTHROW(pmFatalErrorException());
#endif

    mAllottedUnassignedPartition.erase(lSlowestDevice);
    mUnassignedPartitions.erase(lMaxPendingPartitionPtr);
    mSortedUnassignedPartitions.erase(lMaxPendingPartitionPtr);

	if(lMaxAvailableSubtasks <= pSubtaskCount)
	{
		mAllottedUnassignedPartition[lSlowestDevice] = std::make_pair(pmUnfinishedPartitionPtr(), lSlowestDeviceCount);

        lStartSubtask = lMaxPendingPartitionPtr->firstSubtaskIndex;
        lEndSubtask = lMaxPendingPartitionPtr->lastSubtaskIndex;
	}
	else
	{        
        pmUnfinishedPartitionPtr lSubPartitionPtr = pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lMaxPendingPartitionPtr->firstSubtaskIndex, lMaxPendingPartitionPtr->lastSubtaskIndex - pSubtaskCount));

        mSortedUnassignedPartitions[lSubPartitionPtr] = lSlowestDevice;
        mAllottedUnassignedPartition[lSlowestDevice] = std::make_pair(lSubPartitionPtr, lSlowestDeviceCount);
        mUnassignedPartitions.insert(lSubPartitionPtr);

        lStartSubtask = lMaxPendingPartitionPtr->lastSubtaskIndex - pSubtaskCount + 1;
        lEndSubtask = lMaxPendingPartitionPtr->lastSubtaskIndex;
	}

    pmUnfinishedPartitionPtr lNewPartitionPtr = pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lStartSubtask, lEndSubtask));
    
    mSortedUnassignedPartitions[lNewPartitionPtr] = pDevice;
    mAllottedUnassignedPartition[pDevice] = std::make_pair(lNewPartitionPtr, pSubtaskCount);
    mUnassignedPartitions.insert(lNewPartitionPtr);
    
    return lNewPartitionPtr;
}

pmStatus pmPushSchedulingManager::FreezeAllocationSize(pmProcessingElement* pDevice, ulong pFreezedSize)
{
	mExecTimeStats[pDevice].second = pFreezedSize;

	return pmSuccess;
}

pmStatus pmPushSchedulingManager::UnfreezeAllocationSize(pmProcessingElement* pDevice)
{
	mExecTimeStats[pDevice].second = 0;

	return pmSuccess;
}

bool pmPushSchedulingManager::IsAllocationSizeFreezed(pmProcessingElement* pDevice)
{
	return (mExecTimeStats[pDevice].second != 0);
}

ulong pmPushSchedulingManager::GetFreezedAllocationSize(pmProcessingElement* pDevice)
{
	return mExecTimeStats[pDevice].second;
}

pmStatus pmPushSchedulingManager::SetLastAllocationExecTimeInSecs(pmProcessingElement* pDevice, double pTimeInSecs)
{
	if(IsAllocationSizeFreezed(pDevice))
	{
		if(pTimeInSecs > SLOW_START_SCHEDULING_UPPER_LIMIT_EXEC_TIME_PER_ALLOCATION
		|| pTimeInSecs < SLOW_START_SCHEDULING_LOWER_LIMIT_EXEC_TIME_PER_ALLOCATION)
			UnfreezeAllocationSize(pDevice);
	}

	mExecTimeStats[pDevice].first = pTimeInSecs;

	return pmSuccess;
}

double pmPushSchedulingManager::GetLastAllocationExecTimeInSecs(pmProcessingElement* pDevice)
{
	return mExecTimeStats[pDevice].first;
}

ulong pmPushSchedulingManager::GetNextAssignmentSize(pmProcessingElement* pDevice)
{
	ulong lCurrentSize = mAllottedUnassignedPartition[pDevice].second;
	if(lCurrentSize != 0)
	{
		if(IsAllocationSizeFreezed(pDevice))
			return GetFreezedAllocationSize(pDevice);

		double lExecTime = GetLastAllocationExecTimeInSecs(pDevice);

		if(lExecTime == SLOW_START_SCHEDULING_UPPER_LIMIT_EXEC_TIME_PER_ALLOCATION)
		{
			FreezeAllocationSize(pDevice, lCurrentSize);
			return lCurrentSize;
		}
		else
		{
			if(lExecTime < SLOW_START_SCHEDULING_UPPER_LIMIT_EXEC_TIME_PER_ALLOCATION)
			{
				return (lCurrentSize << 1);
			}
			else
			{
				ulong lSize = lCurrentSize;

				if(lSize > SLOW_START_SCHEDULING_INITIAL_SUBTASK_COUNT)
					lSize >>= 1;
				else
					lSize = SLOW_START_SCHEDULING_INITIAL_SUBTASK_COUNT;

				FreezeAllocationSize(pDevice, lSize);

				return lSize;
			}
		}
	}

	return SLOW_START_SCHEDULING_INITIAL_SUBTASK_COUNT;
}

pmStatus pmPushSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, ulong> >::iterator lIter1 = mAllottedUnassignedPartition.find(pDevice);
	if(lIter1 == mAllottedUnassignedPartition.end())
		PMTHROW(pmFatalErrorException());

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> >::iterator lIter2 = mAssignedPartitions.find(pDevice);
	if(lIter2 != mAssignedPartitions.end())	// This device already has a partition waiting to be acknowledged
		PMTHROW(pmFatalErrorException());

	mAllottedUnassignedPartition[pDevice].second = GetNextAssignmentSize(pDevice);

	pmUnfinishedPartitionPtr lUnfinishedPartitionPtr = mAllottedUnassignedPartition[pDevice].first;

	if(!lUnfinishedPartitionPtr.get())
	{
		lUnfinishedPartitionPtr = FetchNewSubPartition(pDevice, mAllottedUnassignedPartition[pDevice].second);

		if(!lUnfinishedPartitionPtr.get())
		{
			pSubtaskCount = 0;
			return pmSuccess;
		}
	}

	ulong lAvailableSubtasks = lUnfinishedPartitionPtr->lastSubtaskIndex - lUnfinishedPartitionPtr->firstSubtaskIndex + 1;
	
	if(lAvailableSubtasks > mAllottedUnassignedPartition[pDevice].second)
		lAvailableSubtasks = mAllottedUnassignedPartition[pDevice].second;
	
	pStartingSubtask = mAllottedUnassignedPartition[pDevice].first->firstSubtaskIndex;
	pSubtaskCount = lAvailableSubtasks;

	return AssignPartition(pDevice, lUnfinishedPartitionPtr, pSubtaskCount);
}

pmStatus pmPushSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> >::iterator lIter = mAssignedPartitions.find(pDevice);
	if(lIter == mAssignedPartitions.end())
		PMTHROW(pmFatalErrorException());

	pmUnfinishedPartitionPtr lPartitionPtr = mAssignedPartitions[pDevice].first;
	if(lPartitionPtr->firstSubtaskIndex != pStartingSubtask && (lPartitionPtr->lastSubtaskIndex - lPartitionPtr->firstSubtaskIndex + 1) != pSubtaskCount)
		PMTHROW(pmFatalErrorException());
    
#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
    UpdateExecutionProfile(pDevice, pSubtaskCount);
#endif

	pmSubtaskRangeCommandPtr lCommand = mAssignedPartitions[pDevice].second;
	lCommand->MarkExecutionEnd(pExecStatus, std::tr1::static_pointer_cast<pmCommand>(lCommand));

	SetLastAllocationExecTimeInSecs(pDevice, lCommand->GetExecutionTimeInSecs());

	mAssignedPartitions.erase(pDevice);

	if(pExecStatus != pmSuccess)
		mTaskStatus = pExecStatus;

	return pmSuccess;
}

/* struct partitionSorter */
bool pmPushSchedulingManager::partitionSorter::operator() (const pmSubtaskManager::pmUnfinishedPartitionPtr& pPartition1Ptr, const pmSubtaskManager::pmUnfinishedPartitionPtr& pPartition2Ptr) const
{
	ulong lCount1 = pPartition1Ptr->lastSubtaskIndex - pPartition1Ptr->firstSubtaskIndex;
	ulong lCount2 = pPartition2Ptr->lastSubtaskIndex - pPartition2Ptr->firstSubtaskIndex;

    if(lCount1 == lCount2)
        return pPartition1Ptr->firstSubtaskIndex < pPartition2Ptr->firstSubtaskIndex;
    
    return lCount1 < lCount2;
}


/* class pmSingleAssignmentSchedulingManager */
pmSingleAssignmentSchedulingManager::pmSingleAssignmentSchedulingManager(pmLocalTask* pLocalTask)
    : pmSubtaskManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
    if(lSubtaskCount == 0)
        PMTHROW(pmFatalErrorException());
    
	pmUnfinishedPartitionPtr lUnacknowledgedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(0, lSubtaskCount-1));
	mUnacknowledgedPartitions.insert(lUnacknowledgedPartitionPtr);
}

pmSingleAssignmentSchedulingManager::~pmSingleAssignmentSchedulingManager()
{
    mUnacknowledgedPartitions.clear();
}

bool pmSingleAssignmentSchedulingManager::HasTaskFinished()
{
    FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(HasTaskFinished_Internal())
    {
#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
        PrintExecutionProfile();
#endif
        
        return true;
    }
    
    return false;
}

/* This method must be called with mResourceLock acquired */
bool pmSingleAssignmentSchedulingManager::HasTaskFinished_Internal()
{
    return mUnacknowledgedPartitions.empty();
}

pmStatus pmSingleAssignmentSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
    FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    if(HasTaskFinished_Internal())
        PMTHROW(pmFatalErrorException());
    
    std::set<pmUnfinishedPartitionPtr>::iterator lIter = mUnacknowledgedPartitions.begin();
    std::set<pmUnfinishedPartitionPtr>::iterator lEndIter = mUnacknowledgedPartitions.end();
    
    pmUnfinishedPartitionPtr lTargetPartitionPtr;
    for(; lIter != lEndIter; ++lIter)
    {
        pmUnfinishedPartitionPtr lPartitionPtr = *lIter;
        if(lPartitionPtr->firstSubtaskIndex <= pStartingSubtask && lPartitionPtr->lastSubtaskIndex >= pStartingSubtask + pSubtaskCount - 1)
        {
            lTargetPartitionPtr = lPartitionPtr;
            mUnacknowledgedPartitions.erase(lIter);
            break;
        }
    }
    
    if(!lTargetPartitionPtr.get())
        PMTHROW(pmFatalErrorException());
    
    if(lTargetPartitionPtr->firstSubtaskIndex < pStartingSubtask)
        mUnacknowledgedPartitions.insert(pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lTargetPartitionPtr->firstSubtaskIndex, pStartingSubtask - 1)));
    
    if(lTargetPartitionPtr->lastSubtaskIndex > pStartingSubtask + pSubtaskCount - 1)
        mUnacknowledgedPartitions.insert(pmUnfinishedPartitionPtr(new pmUnfinishedPartition(pStartingSubtask + pSubtaskCount, lTargetPartitionPtr->lastSubtaskIndex)));
    
#ifdef BUILD_SUBTASK_EXECUTION_PROFILE
    UpdateExecutionProfile(pDevice, pSubtaskCount);
#endif
    
    return pmSuccess;
}


/* class pmPullSchedulingManager */
pmPullSchedulingManager::pmPullSchedulingManager(pmLocalTask* pLocalTask)
	: pmSingleAssignmentSchedulingManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
	ulong lPartitionCount = mLocalTask->GetAssignedDeviceCount();

	if(lSubtaskCount == 0 || lPartitionCount == 0 || lSubtaskCount < lPartitionCount)
		PMTHROW(pmFatalErrorException());

	ulong lPartitionSize = lSubtaskCount/lPartitionCount;
	ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

	ulong lFirstSubtask = 0;
	ulong lLastSubtask;
	
    for(ulong i=0; i<lPartitionCount; ++i)
    {				
        if(i < lLeftoverSubtasks)
            lLastSubtask = lFirstSubtask + lPartitionSize;
        else
            lLastSubtask = lFirstSubtask + lPartitionSize - 1;

        pmSubtaskManager::pmUnfinishedPartitionPtr lUnfinishedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(lFirstSubtask, lLastSubtask));
        mSubtaskPartitions.insert(lUnfinishedPartitionPtr);
        
        lFirstSubtask = lLastSubtask + 1;
    }

	mIter = mSubtaskPartitions.begin();
}

pmPullSchedulingManager::~pmPullSchedulingManager()
{
    mSubtaskPartitions.clear();
}

pmStatus pmPullSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask)
{
	FINALIZE_RESOURCE_PTR(dAssignmentResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAssignmentResourceLock, Lock(), Unlock());

    assert(mIter != mSubtaskPartitions.end());
    
	pmUnfinishedPartitionPtr lPartition = *mIter;

	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;

	++mIter;

	return pmSuccess;
}

pmProportionalSchedulingManager::pmProportionalSchedulingManager(pmLocalTask* pLocalTask)
	: pmSingleAssignmentSchedulingManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();

    std::vector<pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();
    ulong lDeviceCount = (ulong)(lDevices.size());
    
    if(lSubtaskCount == 0 || lDeviceCount == 0 || lSubtaskCount < lDeviceCount)
        PMTHROW(pmFatalErrorException());
    
    mExactCount = 0;
    ReadConfigurationFile(lDevices);

    if(mExactMode && (mExactPartitions.size() != lDeviceCount || mExactCount != lSubtaskCount))
        PMTHROW(pmFatalErrorException());
        
    ulong lStartSubtask = 0;
    ulong lSubtasks = 0;
    for(ulong i=0; i<lDeviceCount; ++i)
    {
        pmProcessingElement* lDevice = lDevices[i];
        
        if(i == lDeviceCount-1)
            lSubtasks = lSubtaskCount - lStartSubtask;
        else
            lSubtasks = FindDeviceAssignment(lDevice, lSubtaskCount);
      
        pmSubtaskManager::pmUnfinishedPartitionPtr lUnfinishedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(lStartSubtask, lStartSubtask+lSubtasks-1));
        
        mDevicePartitionMap[lDevice] = lUnfinishedPartitionPtr;
        lStartSubtask += lSubtasks;
    }
}
    
pmProportionalSchedulingManager::~pmProportionalSchedulingManager()
{
    mDevicePartitionMap.clear();
}

pmStatus pmProportionalSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask)
{
	FINALIZE_RESOURCE_PTR(dAssignmentResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAssignmentResourceLock, Lock(), Unlock());

    assert(mDevicePartitionMap.find(pDevice) != mDevicePartitionMap.end());
    
	pmUnfinishedPartitionPtr lPartition = mDevicePartitionMap[pDevice];
    
	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;

    return pmSuccess;
}

pmStatus pmProportionalSchedulingManager::ReadConfigurationFile(std::vector<pmProcessingElement*>& pDevices)
{
    FILE* fp = fopen(PROPORTIONAL_SCHEDULING_CONF_FILE, "r");
    if(!fp)
        PMTHROW(pmConfFileNotFoundException());
    
    char dataStr[128];

    // fscanf is compiled with attribute unwarn_unused_result. To avoid compiler warning receiving it's return value
    int lVal = fscanf(fp, "%s", dataStr);

    if(strcmp(dataStr, "Devices") == 0)
    {
        mExactMode = true;
        
        int lDevicesCount;
        lVal = fscanf(fp, "%d", &lDevicesCount);

        for(int i=0; i<lDevicesCount; ++i)
        {
            uint lDeviceId, lSubtasks;
            lVal = fscanf(fp, " %d=%d", &lDeviceId, &lSubtasks);
            mExactPartitions[lDeviceId] = lSubtasks;
            mExactCount += lSubtasks;
        }
    }
    else
    {
        mExactMode = false;
        
        lVal = fscanf(fp, "%d %d", &mLocalCpuPower, &mRemoteCpuPower);
        
    #ifdef SUPPORT_CUDA
        lVal = fscanf(fp, " %d %d", &mLocalGpuPower, &mRemoteGpuPower);
    #endif

        mTotalClusterPower = 0;
        ulong lDeviceCount = (ulong)(pDevices.size());
        
        for(ulong i=0; i<lDeviceCount; ++i)
            mTotalClusterPower += GetDevicePower(pDevices[i]);
    }
    
    lVal = lVal; // Suppress compiler warning
    
    return pmSuccess;
}
    
uint pmProportionalSchedulingManager::GetDevicePower(pmProcessingElement* pDevice)
{
    pmDeviceTypes lDeviceType = pDevice->GetType();
    bool lIsLocalDevice = (pDevice->GetMachine() == PM_LOCAL_MACHINE);

    if(lDeviceType == CPU)
    {
        if(lIsLocalDevice)
            return mLocalCpuPower;
        else
            return mRemoteCpuPower;
    }
#ifdef SUPPORT_CUDA
    else if(lDeviceType == GPU_CUDA)
    {
        if(lIsLocalDevice)
            return mLocalGpuPower;
        else
            return mRemoteGpuPower;        
    }
#endif
    
    PMTHROW(pmFatalErrorException());
    
    return 0;
}
    
ulong pmProportionalSchedulingManager::FindDeviceAssignment(pmProcessingElement* pDevice, ulong pSubtaskCount)
{
    if(mExactMode)
    {
        uint lDeviceId = pDevice->GetGlobalDeviceIndex();
	if(mExactPartitions.find(lDeviceId) == mExactPartitions.end())
		PMTHROW(pmFatalErrorException());

        return mExactPartitions[lDeviceId];
    }
    else
    {
        uint lPower = GetDevicePower(pDevice);
    
        return (ulong)(((double)lPower/mTotalClusterPower)*pSubtaskCount);
    }
    
    return 0;
}

}





#include "pmSubtaskManager.h"
#include "pmTask.h"
#include "pmCommand.h"
#include "pmHardware.h"

namespace pm
{

/* class pmSubtaskManager */
pmSubtaskManager::pmSubtaskManager(pmLocalTask* pLocalTask)
{
	mLocalTask= pLocalTask;
	mTaskStatus = pmStatusUnavailable;
    
#ifdef _DEBUG
    mAcknowledgementsReceived = 0;
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
#ifdef _DEBUG
        std::cout << "Acknowledgements Received " << mAcknowledgementsReceived << std::endl;
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
    
#ifdef _DEBUG
    mAcknowledgementsReceived += pSubtaskCount;
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


/* class pmPullSchedulingManager */
pmPullSchedulingManager::pmPullSchedulingManager(pmLocalTask* pLocalTask)
	: pmSubtaskManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
	ulong lPartitionCount = mLocalTask->GetAssignedDeviceCount();

	if(lSubtaskCount == 0 || lPartitionCount == 0 || lSubtaskCount < lPartitionCount)
		PMTHROW(pmFatalErrorException());

	ulong lPartitionSize = lSubtaskCount/lPartitionCount;
	ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

	ulong lFirstSubtask = 0;
	ulong lLastSubtask;
	
	pmUnfinishedPartitionPtr lUnacknowledgedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(0, lSubtaskCount-1));
	mUnacknowledgedPartitions.insert(lUnacknowledgedPartitionPtr);

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
    mUnacknowledgedPartitions.clear();
}

void pmPullSchedulingManager::PrintUnacknowledgedPartitions()
{
    std::set<pmUnfinishedPartitionPtr>::iterator lBegin = mUnacknowledgedPartitions.begin();
    std::set<pmUnfinishedPartitionPtr>::iterator lEnd = mUnacknowledgedPartitions.end();
    
    for(; lBegin != lEnd; ++lBegin)
        std::cout << (*lBegin)->firstSubtaskIndex << " " << (*lBegin)->lastSubtaskIndex << std::endl;
}
    
bool pmPullSchedulingManager::HasTaskFinished()
{
	if(mUnacknowledgedPartitions.empty())
    {
#ifdef _DEBUG
        std::cout << "Acknowledgements Received " << mAcknowledgementsReceived << std::endl;
#endif
        
        return true;
    }
    
    return false;
}

pmStatus pmPullSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    assert(mIter != mSubtaskPartitions.end());
    
	pmUnfinishedPartitionPtr lPartition = *mIter;

	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;

	++mIter;

	return pmSuccess;
}

pmStatus pmPullSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(HasTaskFinished())
		PMTHROW(pmFatalErrorException());

	std::set<pmUnfinishedPartitionPtr>::iterator lIter = mUnacknowledgedPartitions.begin();
	std::set<pmUnfinishedPartitionPtr>::iterator lEndIter = mUnacknowledgedPartitions.end();
	
	pmUnfinishedPartitionPtr lTargetPartitionPtr;
	for(; lIter != lEndIter; ++lIter)
	{
		pmUnfinishedPartitionPtr lPartitionPtr = *lIter;
		if(lPartitionPtr->firstSubtaskIndex <= pStartingSubtask && lPartitionPtr->lastSubtaskIndex >= pStartingSubtask + pSubtaskCount - 1)
		{
			mUnacknowledgedPartitions.erase(lIter);
			lTargetPartitionPtr = lPartitionPtr;
			break;
		}
	}

	if(!lTargetPartitionPtr.get())
		PMTHROW(pmFatalErrorException());

    if(lTargetPartitionPtr->firstSubtaskIndex < pStartingSubtask)
        mUnacknowledgedPartitions.insert(pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lTargetPartitionPtr->firstSubtaskIndex, pStartingSubtask - 1)));

    if(lTargetPartitionPtr->lastSubtaskIndex > pStartingSubtask + pSubtaskCount - 1)
        mUnacknowledgedPartitions.insert(pmUnfinishedPartitionPtr(new pmUnfinishedPartition(pStartingSubtask + pSubtaskCount, lTargetPartitionPtr->lastSubtaskIndex)));

#ifdef _DEBUG
    mAcknowledgementsReceived += pSubtaskCount;
#endif

	return pmSuccess;
}

}

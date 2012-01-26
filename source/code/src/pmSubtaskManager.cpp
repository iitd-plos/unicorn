
#include "pmSubtaskManager.h"
#include "pmTask.h"
#include "pmCommand.h"

namespace pm
{

/* class pmSubtaskManager */
pmSubtaskManager::pmSubtaskManager(pmLocalTask* pLocalTask)
{
	mLocalTask= pLocalTask;
	mTaskStatus = pmStatusUnavailable;
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
		throw pmFatalErrorException();
}


/* class pmPushSchedulingManager */
pmPushSchedulingManager::pmPushSchedulingManager(pmLocalTask* pLocalTask)
	: pmSubtaskManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
	ulong lPartitionCount = mLocalTask->GetAssignedDeviceCount();

	if(lSubtaskCount == 0 || lPartitionCount == 0 || lSubtaskCount < lPartitionCount)
		throw pmFatalErrorException();

	ulong lPartitionSize = lSubtaskCount/lPartitionCount;
	ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

	ulong lFirstSubtask = 0;
	ulong lLastSubtask;
	
	pmSubtaskManager::pmUnfinishedPartition* lUnfinishedPartition;

	std::vector<pmProcessingElement*>& lDevices = mLocalTask->GetAssignedDevices();
	std::vector<pmProcessingElement*>::iterator lIter = lDevices.begin();
	std::vector<pmProcessingElement*>::iterator lEndIter = lDevices.end();

	try
	{
		START_DESTROY_ON_EXCEPTION(dBlock)

		for(ulong i=0; i<lPartitionCount; ++i)
		{				
			if(i < lLeftoverSubtasks)
				lLastSubtask = lFirstSubtask + lPartitionSize;
			else
				lLastSubtask = lFirstSubtask + lPartitionSize - 1;

			lFirstSubtask = lLastSubtask + 1;

			DESTROY_PTR_ON_EXCEPTION(dBlock, lUnfinishedPartition, new pmSubtaskManager::pmUnfinishedPartition(lFirstSubtask, lLastSubtask));

			mSortedUnassignedPartitions[lUnfinishedPartition] = *lIter;
			mUnassignedPartitions.insert(lUnfinishedPartition);
			mAllottedUnassignedPartition[*lIter] = std::pair<pmUnfinishedPartition*, ulong>(lUnfinishedPartition, 0);
			mExecTimeStats[*lIter] = std::pair<double, ulong>(0, 0);

			if(lIter == lEndIter)
				throw pmFatalErrorException();

			++lIter;
		}

		END_DESTROY_ON_EXCEPTION(dBlock)
	}
	catch(pmException e)
	{
		mUnassignedPartitions.clear();	// To prevent double free in destructor in case of an exception
		throw(e);
	}
}

pmPushSchedulingManager::~pmPushSchedulingManager()
{
	std::set<pmUnfinishedPartition*>::iterator lIter1 = mUnassignedPartitions.begin();
	std::set<pmUnfinishedPartition*>::iterator lEndIter1 = mUnassignedPartitions.end();
	for(; lIter1 != lEndIter1; ++lIter1)
		delete *lIter1;

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartition*, pmSubtaskRangeCommandPtr> >::iterator lIter2 = mAssignedPartitions.begin();
	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartition*, pmSubtaskRangeCommandPtr> >::iterator lEndIter2 = mAssignedPartitions.end();
	for(; lIter2 != lEndIter2; ++lIter2)
		delete lIter2->second.first;
}

bool pmPushSchedulingManager::HasTaskFinished()
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return (mUnassignedPartitions.empty() && mAssignedPartitions.empty());
}

pmStatus pmPushSchedulingManager::AssignPartition(pmProcessingElement* pDevice, pmUnfinishedPartition* pUnfinishedPartition, ulong pSubtaskCount)
{
	ulong lAvailableSubtasks = pUnfinishedPartition->lastSubtaskIndex - pUnfinishedPartition->firstSubtaskIndex + 1;

	if(lAvailableSubtasks == pSubtaskCount)
	{
		mAllottedUnassignedPartition[pDevice].first = NULL;
	}
	else
	{
		pmUnfinishedPartition* lSubPartition = new pmUnfinishedPartition(pUnfinishedPartition->firstSubtaskIndex + pSubtaskCount, pUnfinishedPartition->lastSubtaskIndex);
		mAllottedUnassignedPartition[pDevice].first = lSubPartition;
		mUnassignedPartitions.insert(lSubPartition);
		mSortedUnassignedPartitions[lSubPartition] = pDevice;
	}

	pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(mLocalTask->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);
	pmUnfinishedPartition* lPartition = new pmUnfinishedPartition(pUnfinishedPartition->firstSubtaskIndex, pUnfinishedPartition->firstSubtaskIndex + pSubtaskCount - 1);

	mAssignedPartitions[pDevice] = std::pair<pmUnfinishedPartition*, pmSubtaskRangeCommandPtr>(lPartition, lCommand);
	lCommand->MarkExecutionStart();

	mUnassignedPartitions.erase(pUnfinishedPartition);
	mSortedUnassignedPartitions.erase(pUnfinishedPartition);

	delete pUnfinishedPartition;
	
	return pmSuccess;
}

pmSubtaskManager::pmUnfinishedPartition* pmPushSchedulingManager::FetchNewSubPartition(pmProcessingElement* pDevice, ulong pSubtaskCount)
{
	if(mSortedUnassignedPartitions.empty())
	   return NULL;

	std::map<pmUnfinishedPartition*, pmProcessingElement*, partitionSorter>::iterator lIter = mSortedUnassignedPartitions.begin();
	pmUnfinishedPartition* lMaxPendingPartition = lIter->first;
	pmProcessingElement* lSlowestDevice = lIter->second;

	if(lSlowestDevice == pDevice)
		throw pmFatalErrorException();

	ulong lMaxAvailableSubtasks = lMaxPendingPartition->lastSubtaskIndex - lMaxPendingPartition->firstSubtaskIndex + 1;

	pmUnfinishedPartition* lNewPartition;

	if(lMaxAvailableSubtasks <= pSubtaskCount)
	{
		mAllottedUnassignedPartition[lSlowestDevice].first = NULL;

		mUnassignedPartitions.erase(lMaxPendingPartition);
		mSortedUnassignedPartitions.erase(lMaxPendingPartition);
		lNewPartition = lMaxPendingPartition;
	}
	else
	{
		lMaxPendingPartition->lastSubtaskIndex -= pSubtaskCount;

		// Resort the map
		mSortedUnassignedPartitions.erase(lMaxPendingPartition);
		mSortedUnassignedPartitions[lMaxPendingPartition] = lSlowestDevice;

		lNewPartition = new pmUnfinishedPartition(lMaxPendingPartition->lastSubtaskIndex + 1, lMaxPendingPartition->lastSubtaskIndex + pSubtaskCount);
	}

	 mSortedUnassignedPartitions[lNewPartition] = pDevice;
	 mAllottedUnassignedPartition[pDevice] = std::pair<pmUnfinishedPartition*, ulong>(lNewPartition, pSubtaskCount);
	 mUnassignedPartitions.insert(lNewPartition);

	 return lNewPartition;
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

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartition*, ulong> >::iterator lIter1 = mAllottedUnassignedPartition.find(pDevice);
	if(lIter1 == mAllottedUnassignedPartition.end())
		throw pmFatalErrorException();

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartition*, pmSubtaskRangeCommandPtr> >::iterator lIter2 = mAssignedPartitions.find(pDevice);
	if(lIter2 != mAssignedPartitions.end())	// This device already has a partition waiting to be acknowledged
		throw pmFatalErrorException();

	mAllottedUnassignedPartition[pDevice].second = GetNextAssignmentSize(pDevice);

	pmUnfinishedPartition* lUnfinishedPartition = mAllottedUnassignedPartition[pDevice].first;
	if(!lUnfinishedPartition)
	{
		lUnfinishedPartition = FetchNewSubPartition(pDevice, mAllottedUnassignedPartition[pDevice].second);

		if(lUnfinishedPartition)
		{
			 mAllottedUnassignedPartition[pDevice].first = lUnfinishedPartition;
		}
		else
		{
			pSubtaskCount = 0;
			return pmSuccess;
		}
	}

	ulong lAvailableSubtasks = lUnfinishedPartition->lastSubtaskIndex - lUnfinishedPartition->firstSubtaskIndex + 1;
	
	if(lAvailableSubtasks > mAllottedUnassignedPartition[pDevice].second)
		lAvailableSubtasks = mAllottedUnassignedPartition[pDevice].second;
	
	pStartingSubtask = mAllottedUnassignedPartition[pDevice].first->firstSubtaskIndex;
	pSubtaskCount = lAvailableSubtasks;

	return AssignPartition(pDevice, lUnfinishedPartition, pSubtaskCount);
}

pmStatus pmPushSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartition*, pmSubtaskRangeCommandPtr> >::iterator lIter = mAssignedPartitions.find(pDevice);
	if(lIter == mAssignedPartitions.end())
		throw pmFatalErrorException();

	pmUnfinishedPartition* lPartition = mAssignedPartitions[pDevice].first;
	if(lPartition->firstSubtaskIndex != pStartingSubtask && (lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1) != pSubtaskCount)
		throw pmFatalErrorException();

	pmSubtaskRangeCommandPtr lCommand = mAssignedPartitions[pDevice].second;
	lCommand->MarkExecutionEnd(pExecStatus);

	SetLastAllocationExecTimeInSecs(pDevice, lCommand->GetExecutionTimeInSecs());

	mAssignedPartitions.erase(pDevice);

	if(pExecStatus != pmSuccess)
		mTaskStatus = pExecStatus;

	delete lPartition;

	return pmSuccess;
}

/* struct partitionSorter */
bool pmPushSchedulingManager::partitionSorter::operator() (pmSubtaskManager::pmUnfinishedPartition* pPartition1, pmSubtaskManager::pmUnfinishedPartition* pPartition2) const
{
	ulong lCount1 = pPartition1->lastSubtaskIndex - pPartition1->firstSubtaskIndex;
	ulong lCount2 = pPartition2->lastSubtaskIndex - pPartition2->firstSubtaskIndex;

	return lCount1 > lCount2;
}


/* class pmPullSchedulingManager */
pmPullSchedulingManager::pmPullSchedulingManager(pmLocalTask* pLocalTask)
	: pmSubtaskManager(pLocalTask)
{
	ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
	ulong lPartitionCount = mLocalTask->GetAssignedDeviceCount();

	if(lSubtaskCount == 0 || lPartitionCount == 0 || lSubtaskCount < lPartitionCount)
		throw pmFatalErrorException();

	ulong lPartitionSize = lSubtaskCount/lPartitionCount;
	ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

	ulong lFirstSubtask = 0;
	ulong lLastSubtask;
	
	pmSubtaskManager::pmUnfinishedPartition* lUnfinishedPartition;

	pmUnfinishedPartition* lUnacknowledgedPartition;

	try
	{
		START_DESTROY_ON_EXCEPTION(dBlock)

		DESTROY_PTR_ON_EXCEPTION(dBlock, lUnacknowledgedPartition, new pmSubtaskManager::pmUnfinishedPartition(0, lSubtaskCount-1));

		for(ulong i=0; i<lPartitionCount; ++i)
		{				
			if(i < lLeftoverSubtasks)
				lLastSubtask = lFirstSubtask + lPartitionSize;
			else
				lLastSubtask = lFirstSubtask + lPartitionSize - 1;

			lFirstSubtask = lLastSubtask + 1;

			DESTROY_PTR_ON_EXCEPTION(dBlock, lUnfinishedPartition, new pmSubtaskManager::pmUnfinishedPartition(lFirstSubtask, lLastSubtask));

			mSubtaskPartitions.insert(lUnfinishedPartition);
		}

		END_DESTROY_ON_EXCEPTION(dBlock)
	}
	catch(pmException e)
	{
		mSubtaskPartitions.clear();		// To prevent double free in destructor in case of an exception
		throw(e);
	}

	mUnacknowledgedPartitions.insert(lUnacknowledgedPartition);
	mIter = mSubtaskPartitions.begin();
}

pmPullSchedulingManager::~pmPullSchedulingManager()
{
	std::set<pmUnfinishedPartition*>::iterator lIter1 = mSubtaskPartitions.begin();
	std::set<pmUnfinishedPartition*>::iterator lEndIter1 = mSubtaskPartitions.end();
	for(; lIter1 != lEndIter1; ++lIter1)
		delete *lIter1;

	std::set<pmUnfinishedPartition*>::iterator lIter2 = mUnacknowledgedPartitions.begin();
	std::set<pmUnfinishedPartition*>::iterator lEndIter2 = mUnacknowledgedPartitions.end();
	for(; lIter2 != lEndIter2; ++lIter2)
		delete *lIter2;
}

bool pmPullSchedulingManager::HasTaskFinished()
{
	return mUnacknowledgedPartitions.empty();
}

pmStatus pmPullSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	pmUnfinishedPartition* lPartition = *mIter;

	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;

	++mIter;

	return pmSuccess;
}

pmStatus pmPullSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(HasTaskFinished())
		throw pmFatalErrorException();

	std::set<pmUnfinishedPartition*>::iterator lIter = mUnacknowledgedPartitions.begin();
	std::set<pmUnfinishedPartition*>::iterator lEndIter = mUnacknowledgedPartitions.end();
	
	pmUnfinishedPartition* lTargetPartition = NULL;
	for(; lIter != lEndIter; ++lIter)
	{
		pmUnfinishedPartition* lPartition = *lIter;
		if(lPartition->firstSubtaskIndex <= pStartingSubtask && lPartition->lastSubtaskIndex >= pStartingSubtask + pSubtaskCount - 1)
		{
			mUnacknowledgedPartitions.erase(lIter);
			lTargetPartition = lPartition;
			break;
		}
	}

	if(!lTargetPartition)
		throw pmFatalErrorException();

	pmUnfinishedPartition *lPartition1 = NULL;
	pmUnfinishedPartition *lPartition2 = NULL;
	
	START_DESTROY_ON_EXCEPTION(dBlock)
		if(lTargetPartition->firstSubtaskIndex < pStartingSubtask)
			DESTROY_PTR_ON_EXCEPTION(dBlock, lPartition1, new pmUnfinishedPartition(lTargetPartition->firstSubtaskIndex, pStartingSubtask - 1))

		if(lTargetPartition->lastSubtaskIndex > pStartingSubtask + pSubtaskCount - 1)
			DESTROY_PTR_ON_EXCEPTION(dBlock, lPartition2, new pmUnfinishedPartition(pStartingSubtask + pSubtaskCount, lTargetPartition->lastSubtaskIndex))
	END_DESTROY_ON_EXCEPTION(dBlock)

	if(lPartition1)
		mUnacknowledgedPartitions.insert(lPartition1);

	if(lPartition2)
		mUnacknowledgedPartitions.insert(lPartition2);

	return pmSuccess;
}

}

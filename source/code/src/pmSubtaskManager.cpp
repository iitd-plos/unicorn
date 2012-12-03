
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

#include "pmSubtaskManager.h"
#include "pmTask.h"
#include "pmCommand.h"
#include "pmDevicePool.h"
#include "pmHardware.h"
#include "pmNetwork.h"

#include <string.h>

namespace pm
{

/* class pmSubtaskManager */
pmSubtaskManager::pmSubtaskManager(pmLocalTask* pLocalTask)
    : mLocalTask(pLocalTask)
	, mTaskStatus(pmStatusUnavailable)
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
    , mExecutionProfilePrinted(false)
#endif
    , mExecCountSorter(mDeviceExecutionProfile)
    , mOrderedDevices(mExecCountSorter)
{
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
pmSubtaskManager::pmUnfinishedPartition::pmUnfinishedPartition(ulong pFirstSubtaskIndex, ulong pLastSubtaskIndex, pmProcessingElement* pOriginalAllottee /* = NULL */)
	: firstSubtaskIndex(pFirstSubtaskIndex)
	, lastSubtaskIndex(pLastSubtaskIndex)
    , originalAllottee(pOriginalAllottee)
{    
	if(lastSubtaskIndex < firstSubtaskIndex)
		PMTHROW(pmFatalErrorException());
}
    
/* struct partitionSorter */
bool pmSubtaskManager::partitionSorter::operator() (const pmSubtaskManager::pmUnfinishedPartitionPtr& pPartition1Ptr, const pmSubtaskManager::pmUnfinishedPartitionPtr& pPartition2Ptr) const
{
	ulong lCount1 = pPartition1Ptr->lastSubtaskIndex - pPartition1Ptr->firstSubtaskIndex;
	ulong lCount2 = pPartition2Ptr->lastSubtaskIndex - pPartition2Ptr->firstSubtaskIndex;
    
    if(lCount1 == lCount2)
        return pPartition1Ptr->firstSubtaskIndex < pPartition2Ptr->firstSubtaskIndex;
    
    return lCount1 < lCount2;
}
    
pmSubtaskManager::execCountSorter::execCountSorter(std::map<uint, ulong>& pDeviceExecutionProfile)
    : mDeviceExecutionProfile(pDeviceExecutionProfile)
{
}

bool pmSubtaskManager::execCountSorter::operator() (pmProcessingElement* pDevice1, pmProcessingElement* pDevice2) const
{
    uint lIndex1 = pDevice1->GetGlobalDeviceIndex();
    uint lIndex2 = pDevice2->GetGlobalDeviceIndex();
    
    if(mDeviceExecutionProfile[lIndex1] == mDeviceExecutionProfile[lIndex2])
    {
        pmMachine* lMachine1 = pDevice1->GetMachine();
        pmMachine* lMachine2 = pDevice2->GetMachine();
    
        if(lMachine1 == PM_LOCAL_MACHINE && lMachine2 != PM_LOCAL_MACHINE)
            return false;

        if(lMachine1 != PM_LOCAL_MACHINE && lMachine2 == PM_LOCAL_MACHINE)
            return true;
    
        return pDevice1 < pDevice2;
    }
    
    return mDeviceExecutionProfile[lIndex1] < mDeviceExecutionProfile[lIndex2];
}

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
    
    mOrderedDevices.erase(pDevice);
    mOrderedDevices.insert(pDevice);    
    
    return pmSuccess;
}
    
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
pmStatus pmSubtaskManager::PrintExecutionProfile()
{
    if(mExecutionProfilePrinted)
        return pmSuccess;
    
    mExecutionProfilePrinted = true;

    std::vector<ulong> lCpuSubtasks(NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount(), 0);
    
    std::map<uint, ulong>::iterator lStart, lEnd;
    lStart = mDeviceExecutionProfile.begin();
    lEnd = mDeviceExecutionProfile.end();
    
    std::cout << "Device Subtask Execution Profile ... " << std::endl;
    for(; lStart != lEnd; ++lStart)
    {
        std::cout << "Device " << lStart->first << " Subtasks " << lStart->second << std::endl;

        pmProcessingElement* lDevice = pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(lStart->first);
        if(lDevice->GetType() == CPU)
            lCpuSubtasks[(uint)(*(lDevice->GetMachine()))] += lStart->second;
    }

    std::cout << std::endl;
    
    std::cout << "Machine Subtask Execution Profile ... " << std::endl;
    ulong lTotal = 0;
    lStart = mMachineExecutionProfile.begin();
    lEnd = mMachineExecutionProfile.end();
    for(; lStart != lEnd; ++lStart)
    {
        std::cout << "Machine " << lStart->first << " Subtasks " << lStart->second << " CPU-Subtasks " << lCpuSubtasks[lStart->first] << std::endl;
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
	std::vector<pmProcessingElement*>::iterator lIter = lDevices.begin(), lEndIter = lDevices.end();

    for(ulong i=0; i<lPartitionCount; ++i, ++lIter)
    {
        if(i < lLeftoverSubtasks)
            lLastSubtask = lFirstSubtask + lPartitionSize;
        else
            lLastSubtask = lFirstSubtask + lPartitionSize - 1;

        pmSubtaskManager::pmUnfinishedPartitionPtr lUnfinishedPartitionPtr(new pmSubtaskManager::pmUnfinishedPartition(lFirstSubtask, lLastSubtask));

        mSortedUnassignedPartitions[lUnfinishedPartitionPtr] = *lIter;
        mAllottedUnassignedPartition[*lIter] = std::make_pair(lUnfinishedPartitionPtr, (ulong)0);

        mExecTimeStats[*lIter] = std::pair<double, ulong>(0, 0);
    
        UpdateExecutionProfile((*lIter), 0);

        lFirstSubtask = lLastSubtask + 1;

        assert(lIter != lEndIter);
    }
}

pmPushSchedulingManager::~pmPushSchedulingManager()
{
    mSortedUnassignedPartitions.clear();
    mAllottedUnassignedPartition.clear();
    mAssignedPartitions.clear();
    mExecTimeStats.clear();
}

bool pmPushSchedulingManager::HasTaskFinished()
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSortedUnassignedPartitions.empty() && mAssignedPartitions.empty())
    {
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
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

    if(!pUnfinishedPartitionPtr->originalAllottee)
    {
    #ifdef _DEBUG
        if(mSortedUnassignedPartitions.find(pUnfinishedPartitionPtr) == mSortedUnassignedPartitions.end())
            PMTHROW(pmFatalErrorException());
    #endif
        
        mAllottedUnassignedPartition.erase(pDevice);
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
            mSortedUnassignedPartitions[lSubPartitionPtr] = pDevice;
        }
    }

	pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(mLocalTask->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);
	pmUnfinishedPartitionPtr lPartitionPtr(new pmUnfinishedPartition(pUnfinishedPartitionPtr->firstSubtaskIndex, pUnfinishedPartitionPtr->firstSubtaskIndex + pSubtaskCount - 1, pUnfinishedPartitionPtr->originalAllottee));

	mAssignedPartitions[pDevice] = std::make_pair(lPartitionPtr, lCommand);
	lCommand->MarkExecutionStart();

	return pmSuccess;
}
    
pmProcessingElement* pmPushSchedulingManager::SelectMultiAssignAllottee(pmProcessingElement* pDevice)
{
    pmProcessingElement* lPossibleAllottee = NULL;
    size_t lSecondaryAllotteeCountOfPossibleAllottee = 0;

    // Traverse from slowest to fastest device
    std::set<pmProcessingElement*, execCountSorter>::iterator lIter = mOrderedDevices.begin(), lEndIter = mOrderedDevices.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if((*lIter == pDevice) || (mAssignedPartitions.find(*lIter) == mAssignedPartitions.end()) || (mAssignedPartitions[*lIter].first->originalAllottee != NULL))
            continue;

        size_t lSize = mAssignedPartitions[*lIter].first->secondaryAllottees.size();
        if(lSize == MAX_SUBTASK_MULTI_ASSIGN_COUNT - 1)
            continue;
    
        if(lSize == 0)
            return *lIter;

        /* The following code prevents reassignment of partition to a device multiple times. This may happen when a secondary allottee gets a partial negotiation from original allottee and wants a new partition to be assigned to it after acknowledging the partially negotiated one. */
        bool lAlreadyAssigned = false;
        std::vector<pmProcessingElement*>::iterator lInnerIter = mAssignedPartitions[*lIter].first->secondaryAllottees.begin(), lInnerEndIter = mAssignedPartitions[*lIter].first->secondaryAllottees.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(*lInnerIter == pDevice)
            {
                lAlreadyAssigned = true;
                break;
            }
        }
        
        if(lAlreadyAssigned)
            continue;

        if(lSize < lSecondaryAllotteeCountOfPossibleAllottee || lSecondaryAllotteeCountOfPossibleAllottee == 0)
        {
            lPossibleAllottee = *lIter;
            lSecondaryAllotteeCountOfPossibleAllottee = lSize;
        }
    }
    
    return lPossibleAllottee;
}
    
pmSubtaskManager::pmUnfinishedPartitionPtr pmPushSchedulingManager::FetchNewSubPartition(pmProcessingElement* pDevice, ulong pSubtaskCount)
{
	if(mSortedUnassignedPartitions.empty())
    {
        if(mLocalTask->IsMultiAssignEnabled() && !mAssignedPartitions.empty())     // multi assign
        {
            pmProcessingElement* lAllottee = SelectMultiAssignAllottee(pDevice);
            if(lAllottee)
            {            
            #ifdef _DEBUG
                if(mAssignedPartitions[lAllottee].first->originalAllottee != NULL)
                    PMTHROW(pmFatalErrorException());
            #endif
        
                std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> lPair = mAssignedPartitions[lAllottee];
                mAssignedPartitions.erase(lAllottee);
            
                lPair.first->secondaryAllottees.push_back(pDevice);
                mAssignedPartitions[lAllottee] = lPair;
                
            #ifdef TRACK_MULTI_ASSIGN
                std::cout << "Multiassign of subtasks [" << lPair.first->firstSubtaskIndex << " - " << lPair.first->lastSubtaskIndex << "] - Device " << pDevice->GetGlobalDeviceIndex() << ", Original Allottee - Device " << lAllottee->GetGlobalDeviceIndex() << ", Secondary allottment count - " << mAssignedPartitions[lAllottee].first->secondaryAllottees.size() << std::endl;
            #endif
            
                return pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lPair.first->firstSubtaskIndex, lPair.first->lastSubtaskIndex, lAllottee));
            }
        }
    
	   return pmUnfinishedPartitionPtr();
    }

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
    if(mSortedUnassignedPartitions.find(lMaxPendingPartitionPtr) == mSortedUnassignedPartitions.end())
        PMTHROW(pmFatalErrorException());
#endif

    mAllottedUnassignedPartition.erase(lSlowestDevice);
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

        lStartSubtask = lMaxPendingPartitionPtr->lastSubtaskIndex - pSubtaskCount + 1;
        lEndSubtask = lMaxPendingPartitionPtr->lastSubtaskIndex;
	}

    pmUnfinishedPartitionPtr lNewPartitionPtr = pmUnfinishedPartitionPtr(new pmUnfinishedPartition(lStartSubtask, lEndSubtask));
    
    mSortedUnassignedPartitions[lNewPartitionPtr] = pDevice;
    mAllottedUnassignedPartition[pDevice] = std::make_pair(lNewPartitionPtr, pSubtaskCount);
    
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

pmStatus pmPushSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mAllottedUnassignedPartition.find(pDevice) == mAllottedUnassignedPartition.end())
		PMTHROW(pmFatalErrorException());

	if(mAssignedPartitions.find(pDevice) != mAssignedPartitions.end())	// This device already has a partition waiting to be acknowledged
    {
        if(mAssignedPartitions[pDevice].first->originalAllottee != NULL)
            PMTHROW(pmFatalErrorException());
    
        pSubtaskCount = 0;
        return pmSuccess;
    }

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
    {
        if(lUnfinishedPartitionPtr->originalAllottee == NULL)
            lAvailableSubtasks = mAllottedUnassignedPartition[pDevice].second;
    }
	
	pStartingSubtask = lUnfinishedPartitionPtr->firstSubtaskIndex;
	pSubtaskCount = lAvailableSubtasks;
    pOriginalAllottee = lUnfinishedPartitionPtr->originalAllottee;

	return AssignPartition(pDevice, lUnfinishedPartitionPtr, pSubtaskCount);
}

void pmPushSchedulingManager::UpdateAssignedPartition(pmProcessingElement* pDevice, ulong pStartingSubtask, ulong pLastSubtask)
{
    bool lStartMatches = (pStartingSubtask == mAssignedPartitions[pDevice].first->firstSubtaskIndex);
    bool lEndMatches = (pLastSubtask == mAssignedPartitions[pDevice].first->lastSubtaskIndex);

    if(lStartMatches && lEndMatches)
        mAssignedPartitions.erase(pDevice);
    else if(lStartMatches)
        mAssignedPartitions[pDevice].first->firstSubtaskIndex = pLastSubtask + 1;
    else if(lEndMatches)
        mAssignedPartitions[pDevice].first->lastSubtaskIndex = pStartingSubtask - 1;        
    else
        PMTHROW(pmFatalErrorException());
}

void pmPushSchedulingManager::CancelOriginalAllottee(pmProcessingElement* pOriginalAllottee, ulong pSubtaskCount, ulong pStartingSubtask)
{
#ifdef _DEBUG
    if(mAssignedPartitions.find(pOriginalAllottee) == mAssignedPartitions.end() || mAssignedPartitions[pOriginalAllottee].first->originalAllottee != NULL
       || mAssignedPartitions[pOriginalAllottee].first->firstSubtaskIndex > pStartingSubtask || mAssignedPartitions[pOriginalAllottee].first->lastSubtaskIndex < pStartingSubtask + pSubtaskCount - 1)
        PMTHROW(pmFatalErrorException());
#endif

    pmSubtaskRange lRange;
    lRange.task = mLocalTask;
    lRange.startSubtask = pStartingSubtask;
    lRange.endSubtask = pStartingSubtask + pSubtaskCount - 1;
    lRange.originalAllottee = NULL;
    
    pmScheduler::GetScheduler()->SendSubtaskRangeCancellationMessage(pOriginalAllottee, lRange);

    UpdateAssignedPartition(pOriginalAllottee, lRange.startSubtask, lRange.endSubtask);
}
    
void pmPushSchedulingManager::CancelAllButOneSecondaryAllottee(pmProcessingElement* pOriginalAllottee, pmProcessingElement* pPreserveSecondaryAllottee, ulong pSubtaskCount, ulong pStartingSubtask)
{
    pmSubtaskRange lRange;
    lRange.task = mLocalTask;
    lRange.startSubtask = pStartingSubtask;
    lRange.endSubtask = pStartingSubtask + pSubtaskCount - 1;
    lRange.originalAllottee = pOriginalAllottee;
    
    std::vector<pmProcessingElement*>::iterator lIter = mAssignedPartitions[pOriginalAllottee].first->secondaryAllottees.begin();
    std::vector<pmProcessingElement*>::iterator lEndIter = mAssignedPartitions[pOriginalAllottee].first->secondaryAllottees.end();
    for(; lIter < lEndIter; ++lIter)
    {
        if(*lIter == pPreserveSecondaryAllottee)
            continue;
    
        /* Check if secondary allottee is still executing the same range (a secondary allottee might win partially over a range negotiation with original allottee. In this case, it will send acknowledgement for the partial range only but the original allottee will still keep the secondary allottee maintained on it's list (so as to not create unnecessarily more secondary allottess) and when the original allottee sends acknowledgement for it's part of the partial range it will ask all secondary allottees to cancel their executions. But, in the meanwhile, the one secondary allottee which finished earlier might get some other altogether different range assigned. */
        if(mAssignedPartitions.find(*lIter) != mAssignedPartitions.end() && mAssignedPartitions[*lIter].first->firstSubtaskIndex <= lRange.startSubtask && mAssignedPartitions[*lIter].first->lastSubtaskIndex >= lRange.endSubtask)
        {
            pmScheduler::GetScheduler()->SendSubtaskRangeCancellationMessage(*lIter, lRange);
            UpdateAssignedPartition(*lIter, lRange.startSubtask, lRange.endSubtask);
        }
    }
}

pmStatus pmPushSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
	FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmProcessingElement*, std::pair<pmUnfinishedPartitionPtr, pmSubtaskRangeCommandPtr> >::iterator lIter = mAssignedPartitions.find(pDevice);
	if(lIter == mAssignedPartitions.end())
        PMTHROW(pmFatalErrorException());

	pmUnfinishedPartitionPtr lPartitionPtr = mAssignedPartitions[pDevice].first;

	if(lPartitionPtr->firstSubtaskIndex != pStartingSubtask && (lPartitionPtr->lastSubtaskIndex - lPartitionPtr->firstSubtaskIndex + 1) != pSubtaskCount)
    {
        if(!mLocalTask->IsMultiAssignEnabled() || lPartitionPtr->firstSubtaskIndex > pStartingSubtask || lPartitionPtr->lastSubtaskIndex < pStartingSubtask + pSubtaskCount - 1)
            PMTHROW(pmFatalErrorException());
    }

    UpdateExecutionProfile(pDevice, pSubtaskCount);

    pmSubtaskRangeCommandPtr lCommand = mAssignedPartitions[pDevice].second;
    lCommand->MarkExecutionEnd(pExecStatus, std::tr1::static_pointer_cast<pmCommand>(lCommand));

    SetLastAllocationExecTimeInSecs(pDevice, lCommand->GetExecutionTimeInSecs());
    
    if(mAssignedPartitions[pDevice].first->originalAllottee == NULL)    // acknowledgement from original allottee
    {
        if(!mAssignedPartitions[pDevice].first->secondaryAllottees.empty())
        {
        #ifdef TRACK_MULTI_ASSIGN
            std::cout << "Multi assign partition [" << pStartingSubtask << " - " << (pStartingSubtask + pSubtaskCount - 1) << "] completed by original allottee - Device " << pDevice->GetGlobalDeviceIndex() << std::endl;
        #endif
        
            CancelAllButOneSecondaryAllottee(pDevice, NULL, pSubtaskCount, pStartingSubtask);
        }

        UpdateAssignedPartition(pDevice, pStartingSubtask, (pStartingSubtask + pSubtaskCount - 1));
    }
    else    // acknowledgement from secondary allottee
    {
    #ifdef TRACK_MULTI_ASSIGN
        std::cout << "Multi assign partition [" << pStartingSubtask << " - " << (pStartingSubtask + pSubtaskCount - 1) << "] completed by secondary allottee - Device " << pDevice->GetGlobalDeviceIndex() << ", Original Allottee: Device " << mAssignedPartitions[pDevice].first->originalAllottee->GetGlobalDeviceIndex() << std::endl;
    #endif

        CancelAllButOneSecondaryAllottee(mAssignedPartitions[pDevice].first->originalAllottee, pDevice, pSubtaskCount, pStartingSubtask);
        CancelOriginalAllottee(mAssignedPartitions[pDevice].first->originalAllottee, pSubtaskCount, pStartingSubtask);

        mAssignedPartitions.erase(pDevice);
    }
    
	if(pExecStatus != pmSuccess)
		mTaskStatus = pExecStatus;

	return pmSuccess;
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

#if 0
    std::vector<pmProcessingElement*>& lDevices = mLocalTask->GetAssignedDevices();
	std::vector<pmProcessingElement*>::iterator lIter = lDevices.begin(), lEndIter = lDevices.end();
    
    for(; lIter != lEndIter; ++lIter)
        UpdateExecutionProfile((*lIter), 0);
#endif
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
#ifdef DUMP_SUBTASK_EXECUTION_PROFILE
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
    
#ifdef _DEBUG
void pmSingleAssignmentSchedulingManager::DumpUnacknowledgedPartitions()
{
    FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::set<pmUnfinishedPartitionPtr>::iterator lIter = mUnacknowledgedPartitions.begin(), lEndIter = mUnacknowledgedPartitions.end();
    for(; lIter != lEndIter; ++lIter)
        std::cout << "Unacknowledged partition [" << (*lIter)->firstSubtaskIndex << ", " << (*lIter)->lastSubtaskIndex << "]" << std::endl;
}
#endif

pmStatus pmSingleAssignmentSchedulingManager::RegisterSubtaskCompletion(pmProcessingElement* pDevice, ulong pSubtaskCount, ulong pStartingSubtask, pmStatus pExecStatus)
{
    FINALIZE_RESOURCE_PTR(dResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    if(HasTaskFinished_Internal())
        PMTHROW(pmFatalErrorException());
    
    pmUnfinishedPartitionPtr lTargetPartitionPtr;
    std::set<pmUnfinishedPartitionPtr>::iterator lIter = mUnacknowledgedPartitions.begin(), lEndIter = mUnacknowledgedPartitions.end();
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
    
    UpdateExecutionProfile(pDevice, pSubtaskCount);
    
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

pmStatus pmPullSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee)
{
	FINALIZE_RESOURCE_PTR(dAssignmentResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAssignmentResourceLock, Lock(), Unlock());

    assert(mIter != mSubtaskPartitions.end());
    
	pmUnfinishedPartitionPtr lPartition = *mIter;

	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;
    pOriginalAllottee = NULL;

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

pmStatus pmProportionalSchedulingManager::AssignSubtasksToDevice(pmProcessingElement* pDevice, ulong& pSubtaskCount, ulong& pStartingSubtask, pmProcessingElement*& pOriginalAllottee)
{
	FINALIZE_RESOURCE_PTR(dAssignmentResource, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAssignmentResourceLock, Lock(), Unlock());

    assert(mDevicePartitionMap.find(pDevice) != mDevicePartitionMap.end());
    
	pmUnfinishedPartitionPtr lPartition = mDevicePartitionMap[pDevice];
    
	pStartingSubtask = lPartition->firstSubtaskIndex;
	pSubtaskCount = lPartition->lastSubtaskIndex - lPartition->firstSubtaskIndex + 1;
    pOriginalAllottee = NULL;

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
    pmDeviceType lDeviceType = pDevice->GetType();
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




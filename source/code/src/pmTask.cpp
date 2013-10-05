
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

#include "pmTask.h"
#include "pmDevicePool.h"
#include "pmCallback.h"
#include "pmCallbackUnit.h"
#include "pmCluster.h"
#include "pmCommand.h"
#include "pmHardware.h"
#include "pmTaskManager.h"
#include "pmSubtaskManager.h"
#include "pmReducer.h"
#include "pmRedistributor.h"
#include "pmMemSection.h"
#include "pmTimedEventManager.h"
#include "pmStubManager.h"
#include "pmMemoryManager.h"
#include "pmHeavyOperations.h"
#include "pmExecutionStub.h"

#include <vector>
#include <algorithm>

namespace pm
{

STATIC_ACCESSOR_INIT(ulong, pmLocalTask, GetSequenceId, 0)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmLocalTask::mSequenceLock"), pmLocalTask, GetSequenceLock)
    
#define SAFE_GET_DEVICE_POOL(x) { x = pmDevicePool::GetDevicePool(); if(!x) PMTHROW(pmFatalErrorException()); }

/* class pmTask */
pmTask::pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, const pmCluster* pCluster, ushort pPriority, scheduler::schedulingModel pSchedulingModel, ushort pTaskFlags)
	: mTaskId(pTaskId)
	, mCallbackUnit(pCallbackUnit)
	, mSubtaskCount(pSubtaskCount)
    , mOriginatingHost(pOriginatingHost)
    , mCluster(pCluster)
    , mPriority((pPriority < MAX_PRIORITY_LEVEL) ? MAX_PRIORITY_LEVEL : pPriority)
	, mTaskConf(pTaskConf)
	, mTaskConfLength(pTaskConfLength)
	, mSchedulingModel(pSchedulingModel)
    , mSubscriptionManager(this)
    , mSequenceNumber(0)
    , mMultiAssignEnabled(pTaskFlags & TASK_MULTI_ASSIGN_FLAG_VAL)
    , mDisjointReadWritesAcrossSubtasks(pTaskFlags & TASK_DISJOINT_READ_WRITES_ACROSS_SUBTASKS_FLAG_VAL)
    , mOverlapComputeCommunication(pTaskFlags & TASK_SHOULD_OVERLAP_COMPUTE_COMMUNICATION_FLAG_VAL)
    , mCanForciblyCancelSubtasks(pTaskFlags & TASK_CAN_FORCIBLY_CANCEL_SUBTASKS_FLAG_VAL)
    , mCanSplitCpuSubtasks(pTaskFlags & TASK_CAN_SPLIT_CPU_SUBTASKS_FLAG_VAL)
    , mCanSplitGpuSubtasks(pTaskFlags & TASK_CAN_SPLIT_GPU_SUBTASKS_FLAG_VAL)
#ifdef SUPPORT_SPLIT_SUBTASKS
    , mSubtaskSplitter(this)
#endif
	, mSubtasksExecuted(0)
	, mSubtaskExecutionFinished(false)
    , mExecLock __LOCK_NAME__("pmTask::mExecLock")
    , mCompletedRedistributions(0)
    , mRedistributionLock __LOCK_NAME__("pmTask::mRedistributionLock")
    , mAllStubsScannedForCancellationMessages(false)
    , mAllStubsScannedForShadowMemCommitMessages(false)
    , mOutstandingStubsForCancellationMessages(0)
    , mOutstandingStubsForShadowMemCommitMessages(0)
    , mTaskCompletionLock __LOCK_NAME__("pmTask::mTaskCompletionLock")
    , mStealListLock __LOCK_NAME__("pmTask::mStealListLock")
    , mPoolAllocatorMapLock __LOCK_NAME__("pmTask::mPoolAllocatorMapLock")
    , mTaskHasReadWriteMemSectionWithDisjointSubscriptions(false)
	, mAssignedDeviceCount(pAssignedDeviceCount)
{
    mMemSections.reserve(pTaskMemCount);

    for(size_t i = 0; i < pTaskMemCount; ++i)
    {
        const pmTaskMemory& lTaskMem = pTaskMemPtr[i];
        pmMemSection* lMemSection = lTaskMem.memSection;
        
        mMemSections.push_back(lMemSection);
        
    #ifdef SUPPORT_LAZY_MEMORY
        if(lMemSection->IsInput() && lMemSection->IsLazy())
        {
            pmMemInfo lMemInfo(lMemSection->GetReadOnlyLazyMemoryMapping(), lMemSection->GetReadOnlyLazyMemoryMapping(), NULL, lMemSection->GetLength());
            mPreSubscriptionMemInfoForSubtasks.push_back(lMemInfo);
        }
        else
    #endif
        {
            pmMemInfo lMemInfo;
            mPreSubscriptionMemInfoForSubtasks.push_back(lMemInfo); // Output mem sections do not have a global lazy protection, rather have at subtask level
        }
        
        lMemSection->Lock(this, lTaskMem.memType);
        
        mTaskHasReadWriteMemSectionWithDisjointSubscriptions |= (lMemSection->IsReadWrite() && HasDisjointReadWritesAcrossSubtasks());
    }

    CreateReducerAndRedistributors();

    BuildTaskInfo();
    BuildPreSubscriptionSubtaskInfo();
}

pmTask::~pmTask()
{
    mSubscriptionManager.DropAllSubscriptions();
}

pmStatus pmTask::FlushMemoryOwnerships()
{
    std::vector<pmMemSection*>::iterator lIter = mMemSections.begin(), lEndIter = mMemSections.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if((*lIter)->IsOutput())
            (*lIter)->FlushOwnerships();
    }

    return pmSuccess;
}
    
void pmTask::UnlockMemories()
{
    std::vector<pmMemSection*>::iterator lIter = mMemSections.begin(), lEndIter = mMemSections.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->Unlock(this);
}
    
bool pmTask::IsMultiAssignEnabled()
{
    return mMultiAssignEnabled;
}
    
bool pmTask::CanForciblyCancelSubtasks()
{
    return mCanForciblyCancelSubtasks;
}

bool pmTask::CanSplitCpuSubtasks()
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    return (mCanSplitCpuSubtasks && mCallbackUnit && mCallbackUnit->GetSubtaskCB() && mCallbackUnit->GetSubtaskCB()->HasBothCpuAndGpuCallbacks());
#else
    return false;
#endif
}

bool pmTask::CanSplitGpuSubtasks()
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    return (mCanSplitGpuSubtasks && !mCanSplitCpuSubtasks && mCallbackUnit && mCallbackUnit->GetSubtaskCB() && mCallbackUnit->GetSubtaskCB()->HasBothCpuAndGpuCallbacks());
#else
    return false;
#endif
}
    
bool pmTask::DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() const
{
    return mTaskHasReadWriteMemSectionWithDisjointSubscriptions;
}

void* pmTask::GetTaskConfiguration() const
{
	return mTaskConf;
}

uint pmTask::GetTaskConfigurationLength() const
{
	return mTaskConfLength;
}

ulong pmTask::GetTaskId() const
{
	return mTaskId;
}

pmMemSection* pmTask::GetMemSection(size_t pIndex) const
{
    if(pIndex >= mMemSections.size())
        PMTHROW(pmFatalErrorException());

	return mMemSections[pIndex];
}
    
size_t pmTask::GetMemSectionCount() const
{
    return mMemSections.size();
}
    
std::vector<pmMemSection*>& pmTask::GetMemSections()
{
    return mMemSections;
}

const std::vector<pmMemSection*>& pmTask::GetMemSections() const
{
    return mMemSections;
}
    
uint pmTask::GetMemSectionIndex(const pmMemSection* pMemSection) const
{
    const std::vector<pmMemSection*>& lMemSectionVector = GetMemSections();

    std::vector<pmMemSection*>::const_iterator lIter = lMemSectionVector.begin(), lEndIter = lMemSectionVector.end();
    for(uint i = 0; lIter != lEndIter; ++lIter, ++i)
    {
        if(*lIter == pMemSection)
            return i;
    }
    
    PMTHROW(pmFatalErrorException());
    return std::numeric_limits<uint>::max();
}

const pmCallbackUnit* pmTask::GetCallbackUnit() const
{
	return mCallbackUnit;
}

ulong pmTask::GetSubtaskCount() const
{
	return mSubtaskCount;
}

const pmMachine* pmTask::GetOriginatingHost() const
{
	return mOriginatingHost;
}

const pmCluster* pmTask::GetCluster() const
{
	return mCluster;
}

ushort pmTask::GetPriority() const
{
	return mPriority;
}

uint pmTask::GetAssignedDeviceCount() const
{
	return mAssignedDeviceCount;
}

scheduler::schedulingModel pmTask::GetSchedulingModel() const
{
	return mSchedulingModel;
}

pmTaskExecStats& pmTask::GetTaskExecStats()
{
	return mTaskExecStats;
}

void pmTask::RandomizeDevices(std::vector<const pmProcessingElement*>& pDevices)
{
	std::random_shuffle(pDevices.begin(), pDevices.end());
}

std::vector<const pmProcessingElement*>& pmTask::GetStealListForDevice(const pmProcessingElement* pDevice)
{
    FINALIZE_RESOURCE_PTR(dStealListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mStealListLock, Lock(), Unlock());

    std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*> >::iterator lIter = mStealListForDevice.find(pDevice);
    if(lIter == mStealListForDevice.end())
    {
        std::vector<const pmProcessingElement*>& lDevices = (dynamic_cast<pmLocalTask*>(this) != NULL) ? (((pmLocalTask*)this)->GetAssignedDevices()) : (((pmRemoteTask*)this)->GetAssignedDevices());

        std::srand((uint)reinterpret_cast<size_t>(pDevice));
        lIter = mStealListForDevice.insert(std::make_pair(pDevice, lDevices)).first;
        RandomizeDevices(lIter->second);
    }
    
    return lIter->second;
}

void pmTask::BuildTaskInfo()
{
	mTaskInfo.taskHandle = (void*)this;
	mTaskInfo.taskConf = GetTaskConfiguration();
	mTaskInfo.taskConfLength = GetTaskConfigurationLength();
	mTaskInfo.taskId = GetTaskId();
	mTaskInfo.subtaskCount = GetSubtaskCount();
	mTaskInfo.priority = GetPriority();
	mTaskInfo.originatingHost = *(GetOriginatingHost());
}

const pmTaskInfo& pmTask::GetTaskInfo() const
{
	return mTaskInfo;
}

void pmTask::BuildPreSubscriptionSubtaskInfo()
{
    for_each_with_index(mPreSubscriptionMemInfoForSubtasks, [&] (const pmMemInfo& pMemInfo, size_t pIndex)
    {
        mPreSubscriptionSubtaskInfo.memInfo[pIndex] = pMemInfo;
    });

    mPreSubscriptionSubtaskInfo.memCount = (uint)mPreSubscriptionMemInfoForSubtasks.size();
}
    
pmSubtaskInfo pmTask::GetPreSubscriptionSubtaskInfo(ulong pSubtaskId, pmSplitInfo* pSplitInfo) const
{
    pmSubtaskInfo lSubtaskInfo = mPreSubscriptionSubtaskInfo;
    lSubtaskInfo.subtaskId = pSubtaskId;
    
    if(pSplitInfo)
        lSubtaskInfo.splitInfo = *pSplitInfo;
    
    return lSubtaskInfo;
}

pmPoolAllocator& pmTask::GetPoolAllocator(uint pMemSectionIndex, size_t pIndividualAllocationSize, size_t pMaxAllocations)
{
    DEBUG_EXCEPTION_ASSERT(GetMemSection(pMemSectionIndex)->IsOutput() && DoSubtasksNeedShadowMemory(GetMemSection(pMemSectionIndex)));
    
    FINALIZE_RESOURCE_PTR(dPoolAllocatorMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mPoolAllocatorMapLock, Lock(), Unlock());

    decltype(mPoolAllocatorMap)::iterator lIter = mPoolAllocatorMap.find(pMemSectionIndex);
    if(lIter == mPoolAllocatorMap.end())
        lIter = mPoolAllocatorMap.emplace(std::piecewise_construct, std::forward_as_tuple(pMemSectionIndex), std::forward_as_tuple(pIndividualAllocationSize, pMaxAllocations, true)).first;
    
    return lIter->second;
}
    
void* pmTask::CheckOutSubtaskMemory(size_t pLength, uint pMemSectionIndex)
{
    pmMemSection* lMemSection = GetMemSection(pMemSectionIndex);

    if(lMemSection->IsReadWrite() && !HasDisjointReadWritesAcrossSubtasks())
        return NULL;    // In this case, system might not have enough memory as memory for all individual subtasks need to be held till the end

    size_t lMaxAllocations = std::min(pmStubManager::GetStubManager()->GetStubCount(), GetSubtaskCount());
    pmPoolAllocator& lPoolAllocator = GetPoolAllocator(pMemSectionIndex, pLength, lMaxAllocations);

    void* lMem = lPoolAllocator.Allocate(pLength);

#ifdef SUPPORT_LAZY_MEMORY
    if(lMem && lMemSection->IsLazy())   // Reset protections because previous pooling of same memory might have set permissions otherwise
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lMem, pLength, true, true);
#endif
    
    return lMem;
}
    
void pmTask::RepoolCheckedOutSubtaskMemory(uint pMemSectionIndex, void* pMem)
{
    pmPoolAllocator& lPoolAllocator = GetPoolAllocator(pMemSectionIndex, 0, 0);

    lPoolAllocator.Deallocate(pMem);
}
    
pmSubscriptionManager& pmTask::GetSubscriptionManager()
{
	return mSubscriptionManager;
}

#ifdef SUPPORT_SPLIT_SUBTASKS
pmSubtaskSplitter& pmTask::GetSubtaskSplitter()
{
    return mSubtaskSplitter;
}
#endif

void pmTask::CreateReducerAndRedistributors()
{
    if(mCallbackUnit->GetDataReductionCB())
        mReducer.reset(new pmReducer(this));

    if(mCallbackUnit->GetDataRedistributionCB())
    {
        filtered_for_each_with_index(GetMemSections(), [this] (const pmMemSection* pMemSection)
        {
            return pMemSection->IsOutput() && this->IsRedistributable(pMemSection);
        },
        [&] (const pmMemSection* pMemSection, size_t pMemSectionIndex, size_t pOutputMemSectionIndex)
        {
            mRedistributorsMap.emplace(std::piecewise_construct, std::forward_as_tuple(pMemSection), std::forward_as_tuple(this, GetMemSectionIndex(pMemSection)));
        });
    }
}

pmReducer* pmTask::GetReducer()
{
	return mReducer.get_ptr();
}

pmRedistributor* pmTask::GetRedistributor(const pmMemSection* pMemSection)
{
    DEBUG_EXCEPTION_ASSERT(mRedistributorsMap.find(pMemSection) != mRedistributorsMap.end());
    DEBUG_EXCEPTION_ASSERT(IsRedistributable(pMemSection));

    if(!mCallbackUnit->GetDataRedistributionCB())
        return NULL;

    return &mRedistributorsMap.find(pMemSection)->second;
}
    
bool pmTask::RegisterRedistributionCompletion()
{
    uint lSize = (uint)mRedistributorsMap.size();
    
    FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mRedistributionLock, Lock(), Unlock());
    
    ++mCompletedRedistributions;
    
    return (mCompletedRedistributions == lSize);
}
    
bool pmTask::IsReducible(const pmMemSection* pMemSection) const
{
    DEBUG_EXCEPTION_ASSERT(mCallbackUnit->GetDataReductionCB());
    
    return true;
}

bool pmTask::IsRedistributable(const pmMemSection* pMemSection) const
{
    DEBUG_EXCEPTION_ASSERT(mCallbackUnit->GetDataRedistributionCB());
    
    return true;
}
    
#ifdef ENABLE_TASK_PROFILING
pmTaskProfiler* pmTask::GetTaskProfiler()
{
    return &mTaskProfiler;
}
#endif

void pmTask::MarkSubtaskExecutionFinished()
{
	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

		mSubtaskExecutionFinished = true;
	}

	if(mCallbackUnit->GetDataReductionCB())
        GetReducer()->CheckReductionFinish();

    if(mCallbackUnit->GetDataRedistributionCB())
        for_each(mRedistributorsMap, [] (decltype(mRedistributorsMap)::value_type& pPair) {pPair.second.SendRedistributionInfo();});
}

bool pmTask::HasSubtaskExecutionFinished()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtaskExecutionFinished;
}

pmStatus pmTask::IncrementSubtasksExecuted(ulong pSubtaskCount)
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	mSubtasksExecuted += pSubtaskCount;

	return pmSuccess;
}

ulong pmTask::GetSubtasksExecuted()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtasksExecuted;
}

bool pmTask::DoSubtasksNeedShadowMemory(const pmMemSection* pMemSection) const
{
    DEBUG_EXCEPTION_ASSERT(pMemSection->IsOutput());
    
	return (pMemSection->IsLazy() || (pMemSection->IsReadWrite() && !HasDisjointReadWritesAcrossSubtasks()) || (mCallbackUnit->GetDataReductionCB() != NULL));
}
    
void pmTask::TerminateTask()
{
}

void pmTask::RecordStubWillSendCancellationMessage()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());

#ifdef _DEBUG
    if(mAllStubsScannedForCancellationMessages)
        PMTHROW(pmFatalErrorException());
#endif

    ++mOutstandingStubsForCancellationMessages;
}
    
void pmTask::MarkAllStubsScannedForCancellationMessages()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());
    
    mAllStubsScannedForCancellationMessages = true;

    if(mOutstandingStubsForCancellationMessages == 0)
        MarkLocalStubsFreeOfCancellations();
}

void pmTask::RegisterStubCancellationMessage()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());

#ifdef _DEBUG
    if(mOutstandingStubsForCancellationMessages == 0)
        PMTHROW(pmFatalErrorException());
#endif
    
    --mOutstandingStubsForCancellationMessages;

    if(mOutstandingStubsForCancellationMessages == 0 && mAllStubsScannedForCancellationMessages)
        MarkLocalStubsFreeOfCancellations();
}

void pmTask::RecordStubWillSendShadowMemCommitMessage()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());

#ifdef _DEBUG
    if(mAllStubsScannedForShadowMemCommitMessages)
        PMTHROW(pmFatalErrorException());
#endif

    ++mOutstandingStubsForShadowMemCommitMessages;
}
    
void pmTask::MarkAllStubsScannedForShadowMemCommitMessages()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());
    
    mAllStubsScannedForShadowMemCommitMessages = true;

    if(mOutstandingStubsForShadowMemCommitMessages == 0)
        MarkLocalStubsFreeOfShadowMemCommits();
}
    
void pmTask::RegisterStubShadowMemCommitMessage()
{
	FINALIZE_RESOURCE_PTR(dTaskCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskCompletionLock, Lock(), Unlock());

#ifdef _DEBUG
    if(mOutstandingStubsForShadowMemCommitMessages == 0)
        PMTHROW(pmFatalErrorException());
#endif
    
    --mOutstandingStubsForShadowMemCommitMessages;

    if(mOutstandingStubsForShadowMemCommitMessages == 0 && mAllStubsScannedForShadowMemCommitMessages)
        MarkLocalStubsFreeOfShadowMemCommits();
}
    
bool pmTask::HasDisjointReadWritesAcrossSubtasks() const
{
    return mDisjointReadWritesAcrossSubtasks;
}

bool pmTask::ShouldOverlapComputeCommunication() const
{
    return mOverlapComputeCommunication;
}

void pmTask::MarkLocalStubsFreeOfCancellations()
{
}

void pmTask::MarkLocalStubsFreeOfShadowMemCommits()
{
}

ulong pmTask::GetSequenceNumber()
{
    return mSequenceNumber;
}

pmStatus pmTask::SetSequenceNumber(ulong pSequenceNumber)
{
    mSequenceNumber = pSequenceNumber;
    
    return pmSuccess;
}


/* class pmLocalTask */
pmLocalTask::pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, int pTaskTimeOutInSecs, const pmMachine* pOriginatingHost /* = PM_LOCAL_MACHINE */, const pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = DEFAULT_PRIORITY_LEVEL */, scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */, ushort pTaskFlags /* DEFAULT_TASK_FLAGS_VAL */)
	: pmTask(pTaskConf, pTaskConfLength, pTaskId, pTaskMemPtr, pTaskMemCount, pSubtaskCount, pCallbackUnit, 0, pOriginatingHost, pCluster, pPriority, pSchedulingModel, pTaskFlags)
    , mTaskTimeOutTriggerTime((ulong)__MAX(int))
    , mPendingCompletions(0)
    , mUserSideTaskCompleted(false)
    , mLocalStubsFreeOfCancellations(false)
    , mLocalStubsFreeOfShadowMemCommits(false)
    , mCompletionLock __LOCK_NAME__("pmLocalTask::mCompletionLock")
{
    ulong lCurrentTime = GetIntegralCurrentTimeInSecs();
    ulong lTaskTimeOutTriggerTime = lCurrentTime + pTaskTimeOutInSecs;
    if(pTaskTimeOutInSecs > 0 && lTaskTimeOutTriggerTime > lCurrentTime && lTaskTimeOutTriggerTime < (ulong)__MAX(int))
        mTaskTimeOutTriggerTime = lTaskTimeOutTriggerTime;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dSequenceLock, GetSequenceLock().Lock(), GetSequenceLock().Unlock());
        SetSequenceNumber(GetSequenceId()++);
    }

    mTaskCommand = pmCommand::CreateSharedPtr(pPriority, 0, NULL);
}

pmLocalTask::~pmLocalTask()
{
}

void pmLocalTask::TerminateTask()
{
    pmTimedEventManager::GetTimedEventManager()->ClearTaskTimeOutEvent(this, GetTaskTimeOutTriggerTime());    
    pmScheduler::GetScheduler()->TerminateTaskEvent(this);
}

void pmLocalTask::RegisterInternalTaskCompletionMessage()
{
    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());
    --mPendingCompletions;

    if(mPendingCompletions == 0)
        DoPostInternalCompletion();
}

void pmLocalTask::MarkSubtaskExecutionFinished()
{
    pmTask::MarkSubtaskExecutionFinished();
   
    const pmCallbackUnit* lCallbackUnit = GetCallbackUnit();
    if(!lCallbackUnit->GetDataReductionCB() && !lCallbackUnit->GetDataRedistributionCB())
        MarkUserSideTaskCompletion();
}

// This method must be called with mCompletionLock acquired
void pmLocalTask::DoPostInternalCompletion()
{
    FlushMemoryOwnerships();
    UnlockMemories();

    MarkTaskEnd(mSubtaskManager.get_ptr() ? mSubtaskManager->GetTaskExecutionStatus() : pmNoCompatibleDevice);
}
    
void pmLocalTask::TaskRedistributionDone(uint pOriginalMemSectionIndex, pmMemSection* pRedistributedMemSection)
{
    mMemSections[pOriginalMemSectionIndex] = pRedistributedMemSection;

    if(RegisterRedistributionCompletion())  // Returns true when all mem sections finish redistributions
        MarkUserSideTaskCompletion();
}

void pmLocalTask::MarkLocalStubsFreeOfCancellations()
{
    DEBUG_EXCEPTION_ASSERT(IsMultiAssignEnabled());
    
    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
    
    mLocalStubsFreeOfCancellations = true;
}

void pmLocalTask::MarkLocalStubsFreeOfShadowMemCommits()
{
    DEBUG_EXCEPTION_ASSERT(DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions());

    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!IsMultiAssignEnabled() || mLocalStubsFreeOfCancellations))
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
    
    mLocalStubsFreeOfShadowMemCommits = true;
}
    
void pmLocalTask::MarkUserSideTaskCompletion()
{
    bool lIsMultiAssign = IsMultiAssignEnabled();
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

        if((!lIsMultiAssign || mLocalStubsFreeOfCancellations) && (!DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
            pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
            
        mUserSideTaskCompleted = true;
    
        if(mPendingCompletions == 0)
            DoPostInternalCompletion();
    }
}

void pmLocalTask::UserDeleteTask()
{
    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    TerminateTask();
}
    
void pmLocalTask::SaveFinalReducedOutput(pmExecutionStub* pStub, pmMemSection* pMemSection, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    DEBUG_EXCEPTION_ASSERT(DoSubtasksNeedShadowMemory(pMemSection));
    
    uint lMemSectionIndex = GetMemSectionIndex(pMemSection);

    pmSubscriptionManager& lSubscriptionManager = GetSubscriptionManager();
    void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(pStub, pSubtaskId, pSplitInfo, lMemSectionIndex);
    
    subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
    lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(pStub, pSubtaskId, pSplitInfo, lMemSectionIndex, lBeginIter, lEndIter);

    pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(pStub, pSubtaskId, pSplitInfo, lMemSectionIndex);

    std::for_each(lBeginIter, lEndIter, [&] (const subscription::subscriptionRecordType::value_type& pPair)
    {
        pMemSection->Update(pPair.first, pPair.second.first, reinterpret_cast<void*>(reinterpret_cast<size_t>(lShadowMem) + pPair.first - lUnifiedSubscriptionInfo.offset));
    });

    ((pmLocalTask*)this)->MarkUserSideTaskCompletion();
}

pmStatus pmLocalTask::InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel)
{
	switch(pSchedulingModel)
	{
		case scheduler::PUSH:
			mSubtaskManager.reset(new pmPushSchedulingManager(this));
			break;

		case scheduler::PULL:
		case scheduler::STATIC_EQUAL:
			mSubtaskManager.reset(new pmPullSchedulingManager(this));
			break;

		case scheduler::STATIC_PROPORTIONAL:
			mSubtaskManager.reset(new pmProportionalSchedulingManager(this));
			break;

		default:
			PMTHROW(pmFatalErrorException());
	}

	return pmSuccess;
}
    
ulong pmLocalTask::GetTaskTimeOutTriggerTime()
{
    return mTaskTimeOutTriggerTime;
}

std::vector<const pmProcessingElement*>& pmLocalTask::GetAssignedDevices()
{
	return mDevices;
}

void pmLocalTask::WaitForCompletion()
{
	mTaskCommand->WaitForFinish();
}

double pmLocalTask::GetExecutionTimeInSecs()
{
	return mTaskCommand->GetExecutionTimeInSecs();
}

void pmLocalTask::MarkTaskStart()
{
	mTaskCommand->MarkExecutionStart();
}

void pmLocalTask::MarkTaskEnd(pmStatus pStatus)
{
	mTaskCommand->MarkExecutionEnd(pStatus, mTaskCommand);
}

pmStatus pmLocalTask::GetStatus()
{
	return mTaskCommand->GetStatus();
}

const std::vector<const pmProcessingElement*>& pmLocalTask::FindCandidateProcessingElements(std::set<const pmMachine*>& pMachines)
{
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

    std::vector<const pmProcessingElement*> lAvailableDevices;
	const pmSubtaskCB* lSubtaskCB = GetCallbackUnit()->GetSubtaskCB();
	if(lSubtaskCB)
	{
		for(uint i = 0; i < MAX_DEVICE_TYPES; ++i)
		{
            pmDeviceType lType = (pmDeviceType)(MAX_DEVICE_TYPES - 1 - i);
			if(lSubtaskCB->IsCallbackDefinedForDevice(lType))
				lDevicePool->GetAllDevicesOfTypeInCluster(lType, GetCluster(), lAvailableDevices);
		}
	}

    const pmDeviceSelectionCB* lDeviceSelectionCB = GetCallbackUnit()->GetDeviceSelectionCB();
    if(lDeviceSelectionCB)
    {
        filtered_for_each(lAvailableDevices, [&] (const pmProcessingElement* pDevice) {return lDeviceSelectionCB->Invoke(this, pDevice);},
                          [&] (const pmProcessingElement* pDevice) {mDevices.push_back(pDevice);});
    }
    else
    {
        mDevices = lAvailableDevices;
    }

	mAssignedDeviceCount = (uint)(mDevices.size());

    if(!mDevices.empty())
    {
        pmProcessingElement::GetMachines(mDevices, pMachines);
        
        FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

        mPendingCompletions = pMachines.size();
        if(pMachines.find(PM_LOCAL_MACHINE) == pMachines.end())
            ++mPendingCompletions;
    }

	return mDevices;
}

pmSubtaskManager* pmLocalTask::GetSubtaskManager()
{
	return mSubtaskManager.get_ptr();
}


/* class pmRemoteTask */
pmRemoteTask::pmRemoteTask(finalize_ptr<char, deleteArrayDeallocator<char> >& pTaskConf, uint pTaskConfLength, ulong pTaskId, pmTaskMemory* pTaskMemPtr, uint pTaskMemCount, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, ulong pSequenceNumber, const pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = DEFAULT_PRIORITY_LEVEL */, scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */, ushort pTaskFlags /* = DEFAULT_TASK_FLAGS_VAL */)
	: pmTask(pTaskConf.get_ptr(), pTaskConfLength, pTaskId, pTaskMemPtr, pTaskMemCount, pSubtaskCount, pCallbackUnit, pAssignedDeviceCount, pOriginatingHost, pCluster, pPriority, pSchedulingModel, pTaskFlags)
    , mTaskConfAutoPtr(std::move(pTaskConf))
    , mUserSideTaskCompleted(false)
    , mLocalStubsFreeOfCancellations(false)
    , mLocalStubsFreeOfShadowMemCommits(false)
    , mCompletionLock __LOCK_NAME__("pmRemoteTask::mCompletionLock")
{
    SetSequenceNumber(pSequenceNumber);
}

pmRemoteTask::~pmRemoteTask()
{
}
    
void pmRemoteTask::DoPostInternalCompletion()
{
    FlushMemoryOwnerships();
    UnlockMemories();    
}

void pmRemoteTask::AddAssignedDevice(const pmProcessingElement* pDevice)
{
	mDevices.push_back(pDevice);

    DEBUG_EXCEPTION_ASSERT((uint)(mDevices.size()) <= GetAssignedDeviceCount());
}

std::vector<const pmProcessingElement*>& pmRemoteTask::GetAssignedDevices()
{
	return mDevices;
}
    
void pmRemoteTask::TerminateTask()
{
    pmScheduler::GetScheduler()->TerminateTaskEvent(this);
}

void pmRemoteTask::MarkLocalStubsFreeOfCancellations()
{
#ifdef _DEBUG
    if(!IsMultiAssignEnabled())
        PMTHROW(pmFatalErrorException());
#endif

    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
    {
        DoPostInternalCompletion();
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
        TerminateTask();
    }
    else
    {
        mLocalStubsFreeOfCancellations = true;
    }
}

void pmRemoteTask::MarkLocalStubsFreeOfShadowMemCommits()
{
    DEBUG_EXCEPTION_ASSERT(DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions());

    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!IsMultiAssignEnabled() || mLocalStubsFreeOfCancellations))
    {
        DoPostInternalCompletion();
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
        TerminateTask();
    }
    else
    {
        mLocalStubsFreeOfShadowMemCommits = true;
    }
}

void pmRemoteTask::MarkUserSideTaskCompletion()
{
    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if((!IsMultiAssignEnabled() || mLocalStubsFreeOfCancellations) && (!DoesTaskHaveReadWriteMemSectionWithDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
    {
        DoPostInternalCompletion();
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
        TerminateTask();
    }
    else
    {
        mUserSideTaskCompleted = true;
    }
}

void pmRemoteTask::MarkReductionFinished()
{
    MarkUserSideTaskCompletion();
}
    
void pmRemoteTask::MarkRedistributionFinished(uint pOriginalMemSectionIndex, pmMemSection* pRedistributedMemSection /* = NULL */)
{
    if(pRedistributedMemSection)
        mMemSections[pOriginalMemSectionIndex] = pRedistributedMemSection;

    if(RegisterRedistributionCompletion())
        MarkUserSideTaskCompletion();
}
    
void pmRemoteTask::MarkSubtaskExecutionFinished()
{
    pmTask::MarkSubtaskExecutionFinished();

    const pmCallbackUnit* lCallbackUnit = GetCallbackUnit();
    if(!lCallbackUnit->GetDataReductionCB() && !lCallbackUnit->GetDataRedistributionCB())
        MarkUserSideTaskCompletion();
}   

};


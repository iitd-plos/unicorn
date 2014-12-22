
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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
#include "pmAddressSpace.h"
#include "pmTimedEventManager.h"
#include "pmStubManager.h"
#include "pmMemoryManager.h"
#include "pmHeavyOperations.h"
#include "pmExecutionStub.h"
#include "pmUtility.h"
#include "pmAffinityTable.h"

#ifdef USE_STEAL_AGENT_PER_NODE
#include "pmStealAgent.h"
#endif

#include <vector>
#include <algorithm>
#include <random>

namespace pm
{

STATIC_ACCESSOR_INIT(ulong, pmLocalTask, GetSequenceId, 0)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmLocalTask::mSequenceLock"), pmLocalTask, GetSequenceLock)
    
#define SAFE_GET_DEVICE_POOL(x) { x = pmDevicePool::GetDevicePool(); if(!x) PMTHROW(pmFatalErrorException()); }

void AddressSpacesLockCallback(const pmCommandPtr& pCountDownCommand)
{
    pmTask* lTask = const_cast<pmTask*>(static_cast<const pmTask*>(pCountDownCommand->GetUserIdentifier()));
    
    lTask->PrepareForStart();
}
    
/* class pmTask */
pmTask::pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount, const pmMachine* pOriginatingHost, const pmCluster* pCluster, ushort pPriority, scheduler::schedulingModel pSchedulingModel, ushort pTaskFlags)
	: mTaskId(pTaskId)
	, mCallbackUnit(pCallbackUnit)
	, mSubtaskCount(pSubtaskCount)
    , mOriginatingHost(pOriginatingHost)
    , mCluster(pCluster)
    , mPriority((pPriority < MAX_PRIORITY_LEVEL) ? MAX_PRIORITY_LEVEL : pPriority)
	, mTaskConf(NULL)
	, mTaskConfLength(pTaskConfLength)
	, mSchedulingModel(pSchedulingModel)
    , mSubscriptionManager(this)
    , mTaskExecStats(this)
    , mSequenceNumber(0)
    , mTaskFlags(pTaskFlags)
    , mStarted(false)
#ifdef SUPPORT_SPLIT_SUBTASKS
    , mSubtaskSplitter(this)
#endif
#ifdef ENABLE_TASK_PROFILING
    , mTaskProfiler(this)
#endif
	, mSubtasksExecuted(0)
    , mTotalSplitCount(0)
    , mSubtasksSplitted(0)
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
    , mTaskHasReadWriteAddressSpaceWithNonDisjointSubscriptions(false)
    , mTaskMemVector(std::move(pTaskMemVector))
#ifdef USE_STEAL_AGENT_PER_NODE
    , mStealAgentPtr((pmScheduler::SchedulingModelSupportsStealing(mSchedulingModel) ? new pmStealAgent(this) : NULL))
#endif
	, mAssignedDeviceCount(pAssignedDeviceCount)
    , mLastReductionScratchBuffer(NULL)
{
    if(mTaskConfLength)
    {
        mTaskConf = malloc(mTaskConfLength);
        memcpy(mTaskConf, pTaskConf, mTaskConfLength);
    }

    mAddressSpaces.reserve(mTaskMemVector.size());

    for_each_with_index(mTaskMemVector, [this] (const pmTaskMemory& pTaskMem, size_t pMemIndex)
    {
        pmAddressSpace* lAddressSpace = pTaskMem.addressSpace;

        mAddressSpaces.push_back(lAddressSpace);
        mAddressSpaceTaskMemIndexMap[lAddressSpace] = pMemIndex;

        mTaskHasReadWriteAddressSpaceWithNonDisjointSubscriptions |= (IsReadWrite(lAddressSpace) && !pTaskMem.disjointReadWritesAcrossSubtasks);
    });
}

pmTask::~pmTask()
{
    if(mTaskConf)
        free(mTaskConf);

    mSubscriptionManager.DropAllSubscriptions();
}
    
void pmTask::LockAddressSpaces()
{
    if(mTaskMemVector.empty())
    {
        PrepareForStart();
    }
    else
    {
        pmCommandPtr lCountDownCommand = pmCountDownCommand::CreateSharedPtr(mTaskMemVector.size(), GetPriority(), 0, AddressSpacesLockCallback, this);
        lCountDownCommand->MarkExecutionStart();

        for_each(mTaskMemVector, [this, &lCountDownCommand] (const pmTaskMemory& pTaskMem)
        {
            pmAddressSpace* lAddressSpace = pTaskMem.addressSpace;
            lAddressSpace->EnqueueForLock(this, pTaskMem.memType, lCountDownCommand);
        });
    }
}
    
void pmTask::PrepareForStart()
{
    EXCEPTION_ASSERT(!mStarted);

    for_each_with_index(mTaskMemVector, [this] (const pmTaskMemory& pTaskMem, size_t pMemIndex)
    {
    #ifdef SUPPORT_LAZY_MEMORY
        pmAddressSpace* lAddressSpace = pTaskMem.addressSpace;

        if(IsReadOnly(lAddressSpace) && IsLazy(lAddressSpace))
        {
            pmMemInfo lMemInfo(lAddressSpace->GetReadOnlyLazyMemoryMapping(), lAddressSpace->GetReadOnlyLazyMemoryMapping(), NULL, lAddressSpace->GetLength());
            mPreSubscriptionMemInfoForSubtasks.push_back(lMemInfo);
        }
        else
    #endif
        {
            pmMemInfo lMemInfo;
            mPreSubscriptionMemInfoForSubtasks.push_back(lMemInfo); // Output address spaces do not have a global lazy protection, rather have at subtask level
        }
    });

    BuildTaskInfo();
    BuildPreSubscriptionSubtaskInfo();

    // For remote task with affinity, we need to wait for affinity data (logical to physical subtask mapping) to arrive before starting the user task
    if(!(mSchedulingModel == scheduler::PULL_WITH_AFFINITY && dynamic_cast<pmRemoteTask*>(this)))
        Start();
}
    
void pmTask::Start()
{
    mStarted = true;

    if(dynamic_cast<pmLocalTask*>(this))
        pmTaskManager::GetTaskManager()->StartTask(static_cast<pmLocalTask*>(this));
    else
        pmTaskManager::GetTaskManager()->StartTask(static_cast<pmRemoteTask*>(this));
}
    
bool pmTask::HasReadOnlyLazyAddressSpace() const
{
#ifdef SUPPORT_LAZY_MEMORY
    for_each_with_index(mTaskMemVector, [this] (const pmTaskMemory& pTaskMem, size_t pMemIndex)
    {
        pmAddressSpace* lAddressSpace = pTaskMem.addressSpace;

        if(IsReadOnly(lAddressSpace) && IsLazy(lAddressSpace))
            return true;
    });
#endif
    
    return false;
}
    
bool pmTask::HasStarted()
{
    return mStarted;
}

void pmTask::FlushMemoryOwnerships()
{
    for_each(mTaskMemVector, [this] (const pmTaskMemory& pTaskMem)
    {
        if(IsWritable(pTaskMem.addressSpace))
            pTaskMem.addressSpace->FlushOwnerships();
    });
}
    
void pmTask::UnlockMemories()
{
    for_each(mTaskMemVector, [this] (const pmTaskMemory& pTaskMem)
    {
        pTaskMem.addressSpace->Unlock(this);
    });
}
    
bool pmTask::IsMultiAssignEnabled()
{
    return (mTaskFlags & TASK_MULTI_ASSIGN_FLAG_VAL);
}
    
#ifdef SUPPORT_CUDA
bool pmTask::IsCudaCacheEnabled()
{
    return (mTaskFlags & TASK_HAS_CUDA_CACHE_ENABLED_FLAG_VAL);
}
#endif
    
bool pmTask::CanForciblyCancelSubtasks()
{
    return (mTaskFlags & TASK_CAN_FORCIBLY_CANCEL_SUBTASKS_FLAG_VAL);
}

bool pmTask::CanSplitCpuSubtasks()
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    return ((mTaskFlags & TASK_CAN_SPLIT_CPU_SUBTASKS_FLAG_VAL) && mCallbackUnit && mCallbackUnit->GetSubtaskCB() && mCallbackUnit->GetSubtaskCB()->HasBothCpuAndGpuCallbacks());
#else
    return false;
#endif
}

bool pmTask::CanSplitGpuSubtasks()
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    return ((mTaskFlags & TASK_CAN_SPLIT_GPU_SUBTASKS_FLAG_VAL) && !(mTaskFlags & TASK_CAN_SPLIT_CPU_SUBTASKS_FLAG_VAL) && mCallbackUnit && mCallbackUnit->GetSubtaskCB() && mCallbackUnit->GetSubtaskCB()->HasBothCpuAndGpuCallbacks());
#else
    return false;
#endif
}
    
bool pmTask::ShouldSuppressTaskLogs()
{
    return (mTaskFlags & TASK_SUPPRESS_LOGS_FLAG_VAL);
}
    
bool pmTask::DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() const
{
    return mTaskHasReadWriteAddressSpaceWithNonDisjointSubscriptions;
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

const std::vector<pmTaskMemory>& pmTask::GetTaskMemVector() const
{
    return mTaskMemVector;
}

pmAddressSpace* pmTask::GetAddressSpace(size_t pIndex) const
{
    DEBUG_EXCEPTION_ASSERT(pIndex < mAddressSpaces.size());

	return mAddressSpaces[pIndex];
}
    
size_t pmTask::GetAddressSpaceCount() const
{
    return mAddressSpaces.size();
}
    
std::vector<pmAddressSpace*>& pmTask::GetAddressSpaces()
{
    return mAddressSpaces;
}

const std::vector<pmAddressSpace*>& pmTask::GetAddressSpaces() const
{
    return mAddressSpaces;
}
    
uint pmTask::GetAddressSpaceIndex(const pmAddressSpace* pAddressSpace) const
{
    const std::vector<pmAddressSpace*>& lAddressSpaceVector = GetAddressSpaces();

    std::vector<pmAddressSpace*>::const_iterator lIter = lAddressSpaceVector.begin(), lEndIter = lAddressSpaceVector.end();
    for(uint i = 0; lIter != lEndIter; ++lIter, ++i)
    {
        if(*lIter == pAddressSpace)
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

template<typename T>
void pmTask::RandomizeData(T& pData)
{
    std::random_device lRandomDevice;
    std::mt19937 lGenerator(lRandomDevice());
    
	std::shuffle(pData.begin(), pData.end(), lGenerator);
}

#ifdef ENABLE_TWO_LEVEL_STEALING
const std::vector<const pmMachine*>& pmTask::GetStealListForDevice(const pmProcessingElement* pDevice)
{
    FINALIZE_RESOURCE_PTR(dStealListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mStealListLock, Lock(), Unlock());

    auto lIter = mStealListForDevice.find(pDevice);
    if(lIter == mStealListForDevice.end())
    {
        std::set<const pmMachine*> lMachines;
        std::vector<const pmMachine*> lMachinesVector;

        std::vector<const pmProcessingElement*>& lDevices = (dynamic_cast<pmLocalTask*>(this) != NULL) ? (((pmLocalTask*)this)->GetAssignedDevices()) : (((pmRemoteTask*)this)->GetAssignedDevices());
        pmProcessingElement::GetMachines(lDevices, lMachines);

        lMachinesVector.reserve(lMachines.size());
        std::copy(lMachines.begin(), lMachines.end(), std::back_inserter(lMachinesVector));
        
        lIter = mStealListForDevice.emplace(pDevice, lMachinesVector).first;
        RandomizeData(lIter->second);
    }
    
    return lIter->second;
}
#else
const std::vector<const pmProcessingElement*>& pmTask::GetStealListForDevice(const pmProcessingElement* pDevice)
{
    FINALIZE_RESOURCE_PTR(dStealListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mStealListLock, Lock(), Unlock());
    
    auto lIter = mStealListForDevice.find(pDevice);
    if(lIter == mStealListForDevice.end())
    {
        std::vector<const pmProcessingElement*>& lDevices = (dynamic_cast<pmLocalTask*>(this) != NULL) ? (((pmLocalTask*)this)->GetAssignedDevices()) : (((pmRemoteTask*)this)->GetAssignedDevices());

    #ifdef SUPPORT_SPLIT_SUBTASKS
        std::vector<std::vector<const pmProcessingElement*>> lDeviceGroups;
        std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>*> lQueryMap;
        ulong lUnsplittedDevices = 0;

        GetSubtaskSplitter().MakeDeviceGroups(lDevices, lDeviceGroups, lQueryMap, lUnsplittedDevices);

        if(!lDeviceGroups.empty())
        {
            std::vector<const pmProcessingElement*> lRepresentativeDevices;
            for_each(lDeviceGroups, [&] (const std::vector<const pmProcessingElement*>& pVector)
            {
                lRepresentativeDevices.emplace_back(pVector[0]);
            });
            
            lIter = mStealListForDevice.emplace(pDevice, lRepresentativeDevices).first;
            RandomizeData(lIter->second);
        }
        else
    #endif
        {
            lIter = mStealListForDevice.emplace(pDevice, lDevices).first;
            RandomizeData(lIter->second);
        }
    }

    return lIter->second;
}
#endif

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
    lSubtaskInfo.subtaskId = GetPhysicalSubtaskId(pSubtaskId);
    
    if(pSplitInfo)
        lSubtaskInfo.splitInfo = *pSplitInfo;
    
    return lSubtaskInfo;
}

pmPoolAllocator& pmTask::GetPoolAllocator(uint pAddressSpaceIndex, size_t pIndividualAllocationSize, size_t pMaxAllocations)
{
    DEBUG_EXCEPTION_ASSERT(IsWritable(GetAddressSpace(pAddressSpaceIndex)));

    FINALIZE_RESOURCE_PTR(dPoolAllocatorMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mPoolAllocatorMapLock, Lock(), Unlock());

    decltype(mPoolAllocatorMap)::iterator lIter = mPoolAllocatorMap.find(pAddressSpaceIndex);
    if(lIter == mPoolAllocatorMap.end())
        lIter = mPoolAllocatorMap.emplace(std::piecewise_construct, std::forward_as_tuple(pAddressSpaceIndex), std::forward_as_tuple(pIndividualAllocationSize, pMaxAllocations, true)).first;
    
    return lIter->second;
}
    
void* pmTask::CheckOutSubtaskMemory(size_t pLength, uint pAddressSpaceIndex)
{
    pmAddressSpace* lAddressSpace = GetAddressSpace(pAddressSpaceIndex);
    
    if(IsReadWrite(lAddressSpace) && !HasDisjointReadWritesAcrossSubtasks(lAddressSpace))
        return NULL;    // In this case, system might not have enough memory as memory for all individual subtasks need to be held till the end

    size_t lMaxAllocations = std::min(pmStubManager::GetStubManager()->GetStubCount(), GetSubtaskCount());
    pmPoolAllocator& lPoolAllocator = GetPoolAllocator(pAddressSpaceIndex, pLength, lMaxAllocations);

    void* lMem = lPoolAllocator.Allocate(pLength);

#ifdef SUPPORT_LAZY_MEMORY
    if(lMem && IsLazy(lAddressSpace))   // Reset protections because previous pooling of same memory might have set permissions otherwise
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lMem, pLength, true, true);
#endif
    
    return lMem;
}
    
void pmTask::RepoolCheckedOutSubtaskMemory(uint pAddressSpaceIndex, void* pMem)
{
    pmPoolAllocator& lPoolAllocator = GetPoolAllocator(pAddressSpaceIndex, 0, 0);

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
        filtered_for_each_with_index(GetAddressSpaces(), [this] (const pmAddressSpace* pAddressSpace)
        {
            return IsWritable(pAddressSpace) && this->IsRedistributable(pAddressSpace);
        },
        [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
        {
            mRedistributorsList.emplace_back(this, GetAddressSpaceIndex(pAddressSpace));
            mRedistributorsMap.emplace(pAddressSpace, (++mRedistributorsList.rbegin()).base());
        });
    }
}

pmReducer* pmTask::GetReducer()
{
	return mReducer.get_ptr();
}

pmRedistributor* pmTask::GetRedistributor(const pmAddressSpace* pAddressSpace)
{
    DEBUG_EXCEPTION_ASSERT(mRedistributorsMap.find(pAddressSpace) != mRedistributorsMap.end());
    DEBUG_EXCEPTION_ASSERT(IsRedistributable(pAddressSpace));

    if(!mCallbackUnit->GetDataRedistributionCB())
        return NULL;

    return &*mRedistributorsMap.find(pAddressSpace)->second;
}
    
bool pmTask::RegisterRedistributionCompletion()
{
    uint lSize = (uint)mRedistributorsMap.size();
    
    FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mRedistributionLock, Lock(), Unlock());
    
    ++mCompletedRedistributions;
    
    return (mCompletedRedistributions == lSize);
}
    
bool pmTask::IsReducible(const pmAddressSpace* pAddressSpace) const
{
    DEBUG_EXCEPTION_ASSERT(mCallbackUnit->GetDataReductionCB());
    
    return true;
}

bool pmTask::IsRedistributable(const pmAddressSpace* pAddressSpace) const
{
    DEBUG_EXCEPTION_ASSERT(mCallbackUnit->GetDataRedistributionCB());
    
    return true;
}
    
#ifdef USE_STEAL_AGENT_PER_NODE
pmStealAgent* pmTask::GetStealAgent()
{
    return mStealAgentPtr.get();
}
#endif
    
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
        for_each(mRedistributorsMap, [] (decltype(mRedistributorsMap)::value_type& pPair) {(*pPair.second).SendRedistributionInfo();});
}

bool pmTask::HasSubtaskExecutionFinished()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtaskExecutionFinished;
}

pmStatus pmTask::IncrementSubtasksExecuted(ulong pSubtaskCount, ulong pTotalSplitCount)
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	mSubtasksExecuted += pSubtaskCount;
    mTotalSplitCount += pTotalSplitCount;

    if(pTotalSplitCount)
        mSubtasksSplitted += pSubtaskCount;

	return pmSuccess;
}

ulong pmTask::GetTotalSplitCount(ulong& pSubtasksSplitted)
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

    pSubtasksSplitted = mSubtasksSplitted;
	return mTotalSplitCount;
}

ulong pmTask::GetSubtasksExecuted()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtasksExecuted;
}

bool pmTask::DoSubtasksNeedShadowMemory(const pmAddressSpace* pAddressSpace) const
{
    DEBUG_EXCEPTION_ASSERT(IsWritable(pAddressSpace));

	return (IsLazy(pAddressSpace) || (IsReadWrite(pAddressSpace) && !HasDisjointReadWritesAcrossSubtasks(pAddressSpace)) || (mCallbackUnit->GetDataReductionCB() != NULL));
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
    
void pmTask::MarkRedistributionFinished(uint pOriginalAddressSpaceIndex, pmAddressSpace* pRedistributedAddressSpace /* = NULL */)
{
    if(pRedistributedAddressSpace)
    {
        pmAddressSpace* lAddressSpace = GetAddressSpace(pOriginalAddressSpaceIndex);

        if(GetOriginatingHost() == PM_LOCAL_MACHINE)
            lAddressSpace->GetUserMemHandle()->Reset(pRedistributedAddressSpace);

        lAddressSpace->Unlock(this);

        mAddressSpaceTaskMemIndexMap.erase(lAddressSpace);
        mAddressSpaceTaskMemIndexMap[pRedistributedAddressSpace] = pOriginalAddressSpaceIndex;
        
        mAddressSpaces[pOriginalAddressSpaceIndex] = pRedistributedAddressSpace;
        mTaskMemVector[pOriginalAddressSpaceIndex].addressSpace = pRedistributedAddressSpace;
        
        auto lIter = mRedistributorsMap.find(lAddressSpace);
        mRedistributorsMap[pRedistributedAddressSpace] = lIter->second;
        mRedistributorsMap.erase(lIter);
        
        lAddressSpace->UserDelete();
    }

    if(RegisterRedistributionCompletion())
        MarkUserSideTaskCompletion();
}
    
void* pmTask::GetLastReductionScratchBuffer() const
{
    return mLastReductionScratchBuffer;
}
    
bool pmTask::HasDisjointReadWritesAcrossSubtasks(const pmAddressSpace* pAddressSpace) const
{
    DEBUG_EXCEPTION_ASSERT(mAddressSpaceTaskMemIndexMap.find(pAddressSpace) != mAddressSpaceTaskMemIndexMap.end());

    return mTaskMemVector[mAddressSpaceTaskMemIndexMap.find(pAddressSpace)->second].disjointReadWritesAcrossSubtasks;
}

bool pmTask::ShouldOverlapComputeCommunication() const
{
    return (mTaskFlags & TASK_SHOULD_OVERLAP_COMPUTE_COMMUNICATION_FLAG_VAL);
}

void pmTask::MarkLocalStubsFreeOfCancellations()
{
}

void pmTask::MarkLocalStubsFreeOfShadowMemCommits()
{
}

ulong pmTask::GetSequenceNumber() const
{
    return mSequenceNumber;
}

void pmTask::SetSequenceNumber(ulong pSequenceNumber)
{
    mSequenceNumber = pSequenceNumber;
}

pmSubscriptionVisibilityType pmTask::GetAddressSpaceSubscriptionVisibility(const pmAddressSpace* pAddressSpace, const pmExecutionStub* pStub) const
{
    DEBUG_EXCEPTION_ASSERT(mAddressSpaceTaskMemIndexMap.find(pAddressSpace) != mAddressSpaceTaskMemIndexMap.end());
    DEBUG_EXCEPTION_ASSERT(mAddressSpaceTaskMemIndexMap.find(pAddressSpace)->second < mTaskMemVector.size());

    pmSubscriptionVisibilityType lType = mTaskMemVector[mAddressSpaceTaskMemIndexMap.find(pAddressSpace)->second].subscriptionVisibilityType;
    DEBUG_EXCEPTION_ASSERT(lType < MAX_SUBSCRIPTION_VISBILITY_TYPE);

    if(lType == SUBSCRIPTION_OPTIMAL)
    {
        if(pStub->GetType() == CPU)
            lType = SUBSCRIPTION_NATURAL;
        else
            lType = SUBSCRIPTION_COMPACT;
    }
    
    DEBUG_EXCEPTION_ASSERT(lType == SUBSCRIPTION_NATURAL || lType == SUBSCRIPTION_COMPACT);

    return lType;
}

pmMemType pmTask::GetMemType(const pmAddressSpace* pAddressSpace) const
{
    DEBUG_EXCEPTION_ASSERT(mAddressSpaceTaskMemIndexMap.find(pAddressSpace) != mAddressSpaceTaskMemIndexMap.end());
    DEBUG_EXCEPTION_ASSERT(mAddressSpaceTaskMemIndexMap.find(pAddressSpace)->second < mTaskMemVector.size());
    
    return mTaskMemVector[mAddressSpaceTaskMemIndexMap.find(pAddressSpace)->second].memType;
}
    
void pmTask::SetAffinityMappings(std::vector<ulong>&& pLogicalToPhysical, std::vector<ulong>&& pPhysicalToLogical)
{
    EXCEPTION_ASSERT(pLogicalToPhysical.size() == pPhysicalToLogical.size() && pPhysicalToLogical.size() == GetSubtaskCount());
    
    mLogicalToPhysicalSubtaskMappings = std::move(pLogicalToPhysical);
    mPhysicalToLogicalSubtaskMappings = std::move(pPhysicalToLogical);
}
    
ulong pmTask::GetPhysicalSubtaskId(ulong pLogicalSubtaskId) const
{
    if(mLogicalToPhysicalSubtaskMappings.empty())
        return pLogicalSubtaskId;
    else
        return mLogicalToPhysicalSubtaskMappings[pLogicalSubtaskId];
}

ulong pmTask::GetLogicalSubtaskId(ulong pPhysicalSubtaskId) const
{
    if(mPhysicalToLogicalSubtaskMappings.empty())
        return pPhysicalSubtaskId;
    else
        return mPhysicalToLogicalSubtaskMappings[pPhysicalSubtaskId];
}

const std::vector<ulong>& pmTask::GetLogicalToPhysicalSubtaskMappings() const
{
    return mLogicalToPhysicalSubtaskMappings;
}

bool pmTask::IsReadOnly(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsReadOnly(lMemType);
}

bool pmTask::IsWritable(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsWritable(lMemType);
}

bool pmTask::IsWriteOnly(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsWriteOnly(lMemType);
}

bool pmTask::IsReadWrite(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsReadWrite(lMemType);
}
    
bool pmTask::IsLazy(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsLazy(lMemType);
}

bool pmTask::IsLazyWriteOnly(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsLazyWriteOnly(lMemType);
}

bool pmTask::IsLazyReadWrite(const pmAddressSpace* pAddressSpace) const
{
    pmMemType lMemType = GetMemType(pAddressSpace);
    return pmUtility::IsLazyReadWrite(lMemType);
}
    
bool pmTask::IsOpenCLTask() const
{
    return mCallbackUnit->GetSubtaskCB()->HasOpenCLCallback();
}


/* class pmLocalTask */
pmLocalTask::pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, int pTaskTimeOutInSecs, const pmMachine* pOriginatingHost /* = PM_LOCAL_MACHINE */, const pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = DEFAULT_PRIORITY_LEVEL */, scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */, ushort pTaskFlags /* DEFAULT_TASK_FLAGS_VAL */)
	: pmTask(pTaskConf, pTaskConfLength, pTaskId, std::move(pTaskMemVector), pSubtaskCount, pCallbackUnit, 0, pOriginatingHost, pCluster, pPriority, pSchedulingModel, pTaskFlags)
    , mTaskTimeOutTriggerTime((ulong)__MAX(int))
    , mPendingCompletions(0)
    , mUserSideTaskCompleted(false)
    , mLocalStubsFreeOfCancellations(false)
    , mLocalStubsFreeOfShadowMemCommits(false)
    , mCompletionLock __LOCK_NAME__("pmLocalTask::mCompletionLock")
    , mPreprocessorTask(NULL)
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
    
    std::set<const pmMachine*> lMachines;
    FindCandidateProcessingElements(lMachines);
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

void pmLocalTask::MarkLocalStubsFreeOfCancellations()
{
    DEBUG_EXCEPTION_ASSERT(IsMultiAssignEnabled());
    
    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
        pmScheduler::GetScheduler()->SendTaskCompleteToTaskOwner(this);
    
    mLocalStubsFreeOfCancellations = true;
}

void pmLocalTask::MarkLocalStubsFreeOfShadowMemCommits()
{
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

        if((!lIsMultiAssign || mLocalStubsFreeOfCancellations) && (!DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
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
    
void pmLocalTask::SetPreprocessorTask(pmLocalTask* pLocalTask)
{
    EXCEPTION_ASSERT(!mPreprocessorTask);
    
    mPreprocessorTask = pLocalTask;
}

const pmLocalTask* pmLocalTask::GetPreprocessorTask() const
{
    return mPreprocessorTask;
}
    
void pmLocalTask::ComputeAffinityData(pmAddressSpace* pAffinityAddressSpace)
{
    std::set<const pmMachine*> lMachinesSet;
    pmProcessingElement::GetMachines(GetAssignedDevices(), lMachinesSet);

    EXCEPTION_ASSERT(!mAffinityTable.get_ptr());
    
    mAffinityTable.reset(new pmAffinityTable(this));
    mAffinityTable->PopulateAffinityTable(pAffinityAddressSpace, lMachinesSet);
}

void pmLocalTask::StartScheduling()
{
    InitializeSubtaskManager(GetSchedulingModel());
    pmScheduler::GetScheduler()->AssignSubtasksToDevices(this);
}

void pmLocalTask::SetTaskCompletionCallback(pmTaskCompletionCallback pCallback)
{
    (const_cast<pmCallbackUnit*>(GetCallbackUnit()))->SetTaskCompletionCB(new pmTaskCompletionCB(pCallback));
}
    
void pmLocalTask::SaveFinalReducedOutput(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    DEBUG_EXCEPTION_ASSERT(DoSubtasksNeedShadowMemory(pAddressSpace));
    
    uint lAddressSpaceIndex = GetAddressSpaceIndex(pAddressSpace);

    pmSubscriptionManager& lSubscriptionManager = GetSubscriptionManager();
    void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(pStub, pSubtaskId, pSplitInfo, lAddressSpaceIndex);
    
    if(lShadowMem)
    {
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(pStub, pSubtaskId, pSplitInfo, lAddressSpaceIndex, lBeginIter, lEndIter);

        pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(pStub, pSubtaskId, pSplitInfo, lAddressSpaceIndex);

        std::for_each(lBeginIter, lEndIter, [&] (const subscription::subscriptionRecordType::value_type& pPair)
        {
            pAddressSpace->Update(pPair.first, pPair.second.first, reinterpret_cast<void*>(reinterpret_cast<size_t>(lShadowMem) + pPair.first - lUnifiedSubscriptionInfo.offset));
        });
    }
}

// This is called by reducer after all address spaces in the task are reduced
void pmLocalTask::AllReductionsDone(pmExecutionStub* pLastStub, ulong pLastSubtaskId, pmSplitInfo* pLastSplitInfo)
{
    pmScheduler::GetScheduler()->ReductionTerminationEvent(this);
    
    pmSubscriptionManager& lSubscriptionManager = GetSubscriptionManager();

    size_t lScratchBufferSize = 0;
    mLastReductionScratchBuffer = lSubscriptionManager.CheckAndGetScratchBuffer(pLastStub, pLastSubtaskId, pLastSplitInfo, REDUCTION_TO_REDUCTION, lScratchBufferSize);
    
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
        case scheduler::PULL_WITH_AFFINITY:
        {
            uint lMaxPercentVariation = 0;

            const char* lVal = getenv("PMLIB_MAX_FIXED_ALLOTMENT_VARIATION_FOR_PULL");
            if(lVal)
            {
                uint lValue = (uint)atoi(lVal);

                if(lValue != 0 && lValue <= 100)
                    lMaxPercentVariation = lValue;
            }
            
			mSubtaskManager.reset(new pmPullSchedulingManager(this, lMaxPercentVariation));
            
            if(pSchedulingModel == scheduler::PULL_WITH_AFFINITY)
            {
                mAffinityTable->CreateSubtaskMappings();

                const std::vector<ulong>& lLogicalToPhysicalSubtaskMappings = GetLogicalToPhysicalSubtaskMappings();
                
                std::set<const pmMachine*> lMachines;
                pmProcessingElement::GetMachines(GetAssignedDevices(), lMachines);
                
                lMachines.erase(PM_LOCAL_MACHINE);
                
                pmScheduler::GetScheduler()->AffinityTransferEvent(this, std::move(lMachines), &lLogicalToPhysicalSubtaskMappings);
            }
            
            break;
        }

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

    const pmTaskCompletionCB* lTaskCompletionCB = GetCallbackUnit()->GetTaskCompletionCB();
    if(lTaskCompletionCB)
        lTaskCompletionCB->Invoke(this);
}

pmStatus pmLocalTask::GetStatus()
{
	return mTaskCommand->GetStatus();
}

// This method is stable i.e. it does not change the order of devices in pDevicesVector
std::vector<const pmProcessingElement*> pmLocalTask::SelectMaxCpuDevicesPerHost(const std::vector<const pmProcessingElement*>& pDevicesVector, size_t pMaxCpuDevicesPerHost)
{
    std::vector<const pmProcessingElement*> lVector;
    
    std::map<const pmMachine*, size_t> lMap;
    for_each(pDevicesVector, [this, &lMap, &lVector, pMaxCpuDevicesPerHost] (const pmProcessingElement* pDevice)
    {
        if(pDevice->GetType() == CPU)
        {
            const pmMachine* lMachine = pDevice->GetMachine();
            
            auto lIter = lMap.find(lMachine);
            if(lIter == lMap.end())
                lIter = lMap.emplace(lMachine, 0).first;

            if(lIter->second < pMaxCpuDevicesPerHost)
                lVector.push_back(pDevice);
            
            ++lIter->second;
        }
        else
        {
            lVector.push_back(pDevice);
        }
    });
    
    return lVector;
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

    #ifdef SUPPORT_CUDA
        if(lSubtaskCB->IsCallbackDefinedForDevice(CPU) && lSubtaskCB->IsCallbackDefinedForDevice(GPU_CUDA))
        {
            size_t lMaxCpuDevicesPerHost = pmStubManager::GetStubManager()->GetMaxCpuDevicesPerHostForCpuPlusGpuTasks();
            if(lMaxCpuDevicesPerHost < pmStubManager::GetStubManager()->GetProcessingElementsCPU())
                lAvailableDevices = SelectMaxCpuDevicesPerHost(lAvailableDevices, lMaxCpuDevicesPerHost);
        }
    #endif
	}
    else
    {
        lDevicePool->GetAllDevicesOfTypeInCluster(CPU, GetCluster(), lAvailableDevices);
    }

    const pmDeviceSelectionCB* lDeviceSelectionCB = GetCallbackUnit()->GetDeviceSelectionCB();
    if(lDeviceSelectionCB)
    {
        filtered_for_each(lAvailableDevices, [&] (const pmProcessingElement* pDevice) {return lDeviceSelectionCB->Invoke(this, pDevice);},
                          [&] (const pmProcessingElement* pDevice) {mDevices.push_back(pDevice);});
    }
    else
    {
        mDevices = std::move(lAvailableDevices);
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

    CreateReducerAndRedistributors();

	return mDevices;
}

pmSubtaskManager* pmLocalTask::GetSubtaskManager()
{
	return mSubtaskManager.get_ptr();
}


/* class pmRemoteTask */
pmRemoteTask::pmRemoteTask(finalize_ptr<char, deleteArrayDeallocator<char>>& pTaskConf, uint pTaskConfLength, ulong pTaskId, std::vector<pmTaskMemory>&& pTaskMemVector, ulong pSubtaskCount, const pmCallbackUnit* pCallbackUnit, const pmMachine* pOriginatingHost, ulong pSequenceNumber, std::vector<const pmProcessingElement*>&& pDevices, const pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = DEFAULT_PRIORITY_LEVEL */, scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */, ushort pTaskFlags /* = DEFAULT_TASK_FLAGS_VAL */)
	: pmTask(pTaskConf.get_ptr(), pTaskConfLength, pTaskId, std::move(pTaskMemVector), pSubtaskCount, pCallbackUnit, (uint)pDevices.size(), pOriginatingHost, pCluster, pPriority, pSchedulingModel, pTaskFlags)
    , mTaskConfAutoPtr(std::move(pTaskConf))
    , mUserSideTaskCompleted(false)
    , mLocalStubsFreeOfCancellations(false)
    , mLocalStubsFreeOfShadowMemCommits(false)
    , mCompletionLock __LOCK_NAME__("pmRemoteTask::mCompletionLock")
    , mDevices(pDevices)
{
    SetSequenceNumber(pSequenceNumber);

    CreateReducerAndRedistributors();
}

pmRemoteTask::~pmRemoteTask()
{
}
    
void pmRemoteTask::DoPostInternalCompletion()
{
    FlushMemoryOwnerships();
    UnlockMemories();    
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
    DEBUG_EXCEPTION_ASSERT(IsMultiAssignEnabled());

    FINALIZE_RESOURCE_PTR(dCompletionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCompletionLock, Lock(), Unlock());

    if(mUserSideTaskCompleted && (!DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
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

    if((!IsMultiAssignEnabled() || mLocalStubsFreeOfCancellations) && (!DoesTaskHaveReadWriteAddressSpaceWithNonDisjointSubscriptions() || mLocalStubsFreeOfShadowMemCommits))
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

void pmRemoteTask::MarkSubtaskExecutionFinished()
{
    pmTask::MarkSubtaskExecutionFinished();

    const pmCallbackUnit* lCallbackUnit = GetCallbackUnit();
    if(!lCallbackUnit->GetDataReductionCB() && !lCallbackUnit->GetDataRedistributionCB())
        MarkUserSideTaskCompletion();
    
    // If no subtask is executed on this machine, then signal the send to machine to continue with reduction
    if(lCallbackUnit->GetDataReductionCB() && GetSubtasksExecuted() == 0)
        GetReducer()->SignalSendToMachineAboutNoLocalReduction();
}

void pmRemoteTask::ReceiveAffinityData(std::vector<ulong>&& pLogicalToPhysicalSubtaskMapping)
{
    ulong lSubtaskCount = GetSubtaskCount();
    EXCEPTION_ASSERT(pLogicalToPhysicalSubtaskMapping.size() == lSubtaskCount);
    
    std::vector<ulong> lPhysicalToLogicalSubtaskMapping(lSubtaskCount);
    for(ulong i = 0; i < lSubtaskCount; ++i)
        lPhysicalToLogicalSubtaskMapping[pLogicalToPhysicalSubtaskMapping[i]] = i;

    SetAffinityMappings(std::move(pLogicalToPhysicalSubtaskMapping), std::move(lPhysicalToLogicalSubtaskMapping));

    Start();
}

};


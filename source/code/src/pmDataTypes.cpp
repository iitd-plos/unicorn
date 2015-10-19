
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

#include "pmBase.h"
#include "pmLogger.h"
#include "pmTimer.h"
#include "pmTask.h"
#include "pmResourceLock.h"
#include "pmExecutionStub.h"
#include "pmTaskProfiler.h"
#include "pmHardware.h"

#ifdef SUPPORT_CUDA
#include "pmMemChunk.h"
#include "pmDispatcherGPU.h"
#include "pmCudaInterface.h"
#endif

#include <iostream>
#include <iomanip>
#include <sstream>

#include <string.h>

namespace pm
{

/* Comparison opeartors for pmSubscriptionInfo */
bool operator==(const pmSubscriptionInfo& pSubscription1, const pmSubscriptionInfo& pSubscription2)
{
    return (pSubscription1.offset == pSubscription2.offset && pSubscription1.length == pSubscription2.length);
}
    
bool operator!=(const pmSubscriptionInfo& pSubscription1, const pmSubscriptionInfo& pSubscription2)
{
    return !(pSubscription1 == pSubscription2);
}


/* Comparison opeartors for pmScatteredSubscriptionInfo */
bool operator==(const pmScatteredSubscriptionInfo& pScatteredSubscription1, const pmScatteredSubscriptionInfo& pScatteredSubscription2)
{
    return (pScatteredSubscription1.offset == pScatteredSubscription2.offset && pScatteredSubscription1.size == pScatteredSubscription2.size
            && pScatteredSubscription1.step == pScatteredSubscription2.step && pScatteredSubscription1.count == pScatteredSubscription2.count);
}
    
bool operator!=(const pmScatteredSubscriptionInfo& pScatteredSubscription1, const pmScatteredSubscriptionInfo& pScatteredSubscription2)
{
    return !(pScatteredSubscription1 == pScatteredSubscription2);
}

    
/* class pmJmpBufAutoPtr */
pmJmpBufAutoPtr::pmJmpBufAutoPtr()
    : mStub(NULL)
    , mHasJumped(false)
{
}

pmJmpBufAutoPtr::~pmJmpBufAutoPtr()
{
    if(mStub)
        mStub->UnsetupJmpBuf(mHasJumped);
}

void pmJmpBufAutoPtr::Reset(sigjmp_buf* pJmpBuf, pmExecutionStub* pStub)
{
    mStub = pStub;
    
    if(mStub)
        mStub->SetupJmpBuf(pJmpBuf);
}

void pmJmpBufAutoPtr::SetHasJumped()
{
    mHasJumped = true;
}
    

/* class pmUserLibraryCodeAutoPtr */
pmSubtaskTerminationCheckPointAutoPtr::pmSubtaskTerminationCheckPointAutoPtr(pmExecutionStub* pStub)
    : mStub(pStub)
{
    mStub->MarkInsideLibraryCode();
}
    
pmSubtaskTerminationCheckPointAutoPtr::~pmSubtaskTerminationCheckPointAutoPtr()
{
    mStub->MarkInsideUserCode();
}


#ifdef ENABLE_TASK_PROFILING
/* class pmRecordProfileEventAutoPtr */
pmRecordProfileEventAutoPtr::pmRecordProfileEventAutoPtr(pmTaskProfiler* pTaskProfiler, taskProfiler::profileType pProfileType)
    : mTaskProfiler(pTaskProfiler)
    , mProfileType(pProfileType)
{
    mTaskProfiler->RecordProfileEvent(pProfileType, true);
}
    
pmRecordProfileEventAutoPtr::~pmRecordProfileEventAutoPtr()
{
    mTaskProfiler->RecordProfileEvent(mProfileType, false);
}
#endif

#ifdef DUMP_EVENT_TIMELINE
    
#ifdef SUPPORT_SPLIT_SUBTASKS
/* class pmSplitSubtaskExecutionTimelineAutoPtr */
pmSplitSubtaskExecutionTimelineAutoPtr::pmSplitSubtaskExecutionTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pSubtaskId, uint pSplitId, uint pSplitCount)
    : mTask(pTask)
    , mEventTimeline(pEventTimeline)
    , mSubtaskId(pSubtaskId)
    , mSplitId(pSplitId)
    , mSplitCount(pSplitCount)
    , mCancelledOrException(true)
{
    mEventTimeline->RecordEvent(mTask, GetEventName(mSubtaskId, mSplitId, mSplitCount, mTask), true);
}

pmSplitSubtaskExecutionTimelineAutoPtr::~pmSplitSubtaskExecutionTimelineAutoPtr()
{
    if(mCancelledOrException)
    {
        mEventTimeline->RenameEvent(mTask, GetEventName(mSubtaskId, mSplitId, mSplitCount, mTask), GetCancelledEventName(mSubtaskId, mSplitId, mSplitCount, mTask));
        mEventTimeline->RecordEvent(mTask, GetCancelledEventName(mSubtaskId, mSplitId, mSplitCount, mTask), false);
    }
    else
    {
        mEventTimeline->RecordEvent(mTask, GetEventName(mSubtaskId, mSplitId, mSplitCount, mTask), false);
    }
}

void pmSplitSubtaskExecutionTimelineAutoPtr::SetGracefulCompletion()
{
    mCancelledOrException = false;
}
    
std::string pmSplitSubtaskExecutionTimelineAutoPtr::GetEventName(ulong pSubtaskId, uint pSplitId, uint pSplitCount, pmTask* pTask)
{
    std::stringstream lEventName;
    lEventName << "Task [" << ((uint)(*(pTask->GetOriginatingHost()))) << ", " << (pTask->GetSequenceNumber()) << "] Subtask " << pTask->GetPhysicalSubtaskId(pSubtaskId) << " (Split " << pSplitId << " of " << pSplitCount << ")";
    
    return lEventName.str();
}

std::string pmSplitSubtaskExecutionTimelineAutoPtr::GetCancelledEventName(ulong pSubtaskId, uint pSplitId, uint pSplitCount, pmTask* pTask)
{
    std::stringstream lEventName;
    lEventName << "Task [" << ((uint)(*(pTask->GetOriginatingHost()))) << ", " << (pTask->GetSequenceNumber()) << "] Subtask " << pTask->GetPhysicalSubtaskId(pSubtaskId) << " (Split " << pSplitId << " of " << pSplitCount << ")_Cancelled";

    return lEventName.str();
}
#endif


/* class pmSubtaskRangeExecutionTimelineAutoPtr */
pmSubtaskRangeExecutionTimelineAutoPtr::pmSubtaskRangeExecutionTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pStartSubtask, ulong pEndSubtask)
    : mTask(pTask)
    , mEventTimeline(pEventTimeline)
    , mStartSubtask(pStartSubtask)
    , mEndSubtask(pEndSubtask)
    , mRangeCancelledOrException(true)
    , mSubtasksInitialized(0)
{}

pmSubtaskRangeExecutionTimelineAutoPtr::~pmSubtaskRangeExecutionTimelineAutoPtr()
{
    if(mRangeCancelledOrException)
    {
        for(ulong i = 0; i < mSubtasksInitialized; ++i)
        {
            mEventTimeline->RenameEvent(mTask, GetEventName(mStartSubtask + i, mTask), GetCancelledEventName(mStartSubtask + i, mTask));
            mEventTimeline->StopEventIfRequired(mTask, GetCancelledEventName(mStartSubtask + i, mTask));
        }
    }
}
    
void pmSubtaskRangeExecutionTimelineAutoPtr::ResetEndSubtask(ulong pEndSubtask)
{
    if(mStartSubtask + mSubtasksInitialized > pEndSubtask + 1)
    {
        for(ulong i = pEndSubtask + 1; i < mStartSubtask + mSubtasksInitialized; ++i)
            mEventTimeline->DropEvent(mTask, GetEventName(i, mTask));
    }

    mSubtasksInitialized -= std::max<long>(0, mStartSubtask + mSubtasksInitialized - 1 - pEndSubtask);
    mEndSubtask = pEndSubtask;
}
    
void pmSubtaskRangeExecutionTimelineAutoPtr::InitializeNextSubtask()
{
    mEventTimeline->RecordEvent(mTask, GetEventName(mStartSubtask + mSubtasksInitialized, mTask), true);
    ++mSubtasksInitialized;
}

void pmSubtaskRangeExecutionTimelineAutoPtr::FinishSubtask(ulong pSubtaskId)
{
    mEventTimeline->RecordEvent(mTask, GetEventName(pSubtaskId, mTask), false);
}

void pmSubtaskRangeExecutionTimelineAutoPtr::SetGracefulCompletion()
{
    mRangeCancelledOrException = false;
}

std::string pmSubtaskRangeExecutionTimelineAutoPtr::GetEventName(ulong pSubtaskId, pmTask* pTask)
{
    std::stringstream lEventName;
    lEventName << "Task [" << ((uint)(*(pTask->GetOriginatingHost()))) << ", " << (pTask->GetSequenceNumber()) << "] Subtask " << pTask->GetPhysicalSubtaskId(pSubtaskId);
    
    return lEventName.str();
}

std::string pmSubtaskRangeExecutionTimelineAutoPtr::GetCancelledEventName(ulong pSubtaskId, pmTask* pTask)
{
    std::stringstream lEventName;
    lEventName << "Task [" << ((uint)(*(pTask->GetOriginatingHost()))) << ", " << (pTask->GetSequenceNumber()) << "] Subtask " << pTask->GetPhysicalSubtaskId(pSubtaskId) << "_Cancelled";

    return lEventName.str();
}
    

/* class pmEventTimelineAutoPtr */
pmEventTimelineAutoPtr::pmEventTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pSubtaskId, const pmSplitData& pSplitData, uint pDeviceId, const std::string& pEventNameSuffix)
    : mTask(pTask)
    , mEventTimeline(pEventTimeline)
    , mEventName(GetEventName(pSubtaskId, pSplitData, pDeviceId, pEventNameSuffix))
{
    mEventTimeline->RecordEvent(mTask, mEventName, true);
}

pmEventTimelineAutoPtr::~pmEventTimelineAutoPtr()
{
    mEventTimeline->RecordEvent(mTask, mEventName, false);
}

std::string pmEventTimelineAutoPtr::GetEventName(ulong pSubtaskId, const pmSplitData& pSplitData, uint pDeviceId, const std::string& pEventNameSuffix)
{
    std::stringstream lEventName;
    
    if(pSplitData.valid)
        lEventName << "Task [" << ((uint)(*(mTask->GetOriginatingHost()))) << ", " << (mTask->GetSequenceNumber()) << "] Subtask " << mTask->GetPhysicalSubtaskId(pSubtaskId) << " Sp" << pSplitData.splitId << " D" << pDeviceId << " Event " << pEventNameSuffix;
    else
        lEventName << "Task [" << ((uint)(*(mTask->GetOriginatingHost()))) << ", " << (mTask->GetSequenceNumber()) << "] Subtask " << mTask->GetPhysicalSubtaskId(pSubtaskId) << " D" << pDeviceId << " Event " << pEventNameSuffix;
    
    return lEventName.str();
}
#endif


/* class pmScopeTimer */
pmScopeTimer::pmScopeTimer(const char* pStr)
: mStr(pStr)
, mStartTime(pmBase::GetCurrentTimeInSecs())
{
}

pmScopeTimer::~pmScopeTimer()
{
    std::cout << "Scope Time for " << mStr << ": " << pmBase::GetCurrentTimeInSecs() - mStartTime << std::endl;
}

    
#ifdef ENABLE_ACCUMULATED_TIMINGS

const uint MAX_FLOAT_WIDTH = 8;
const uint MAX_INT_WIDTH = 8;
const double MIN_ACCUMULATED_TIME = 0.001;

/* class pmAccumulationTimer */
pmAccumulationTimer::pmAccumulationTimer(const std::string& pStr)
: mStr(pStr)
, mMinTime(0)
, mMaxTime(0)
, mAccumulatedTime(0)
, mActualTime(0)
, mExecCount(0)
, mThreadCount(0)
, mTimer(new TIMER_IMPLEMENTATION_CLASS())
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );

    mTimer->Start();
    mTimer->Pause();
}
    
pmAccumulationTimer::~pmAccumulationTimer()
{
    pmAccumulatedTimesSorter::GetAccumulatedTimesSorter()->Insert(mStr, mAccumulatedTime, mMinTime, mMaxTime, mActualTime, mExecCount);

	Lock();
	Unlock();
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
}

void pmAccumulationTimer::RegisterExec()
{
    Lock();

    if(mThreadCount == 0)
    {
        mTimer->Resume();
    }
    else
    {
        mTimer->Pause();
    
        RecordElapsedTime();
    
        mTimer->Reset();
    }

    ++mThreadCount;
    
    Unlock();
}

void pmAccumulationTimer::DeregisterExec(double pTime)
{
    Lock();

    if(mMinTime == 0 || pTime < mMinTime)
        mMinTime = pTime;
    
    if(pTime > mMaxTime)
        mMaxTime = pTime;
    
    mTimer->Pause();
    
    RecordElapsedTime();
    
    mTimer->Reset();
    
    --mThreadCount;

    if(mThreadCount == 0)
        mTimer->Pause();
    
    ++mExecCount;
    
    Unlock();
}
    
void pmAccumulationTimer::Lock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
}

void pmAccumulationTimer::Unlock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );
}

void pmAccumulationTimer::RecordElapsedTime()
{
    double lElapsedTime = mTimer->GetElapsedTimeInSecs();

    mActualTime += lElapsedTime;
    mAccumulatedTime += mThreadCount * lElapsedTime;
}


/* class pmAccumulationTimerHelper */
pmAccumulationTimerHelper::pmAccumulationTimerHelper(pmAccumulationTimer* pAccumulationTimer)
: mAccumulationTimer(pAccumulationTimer)
, mStartTime(pmBase::GetCurrentTimeInSecs())
{
    mAccumulationTimer->RegisterExec();
}

pmAccumulationTimerHelper::~pmAccumulationTimerHelper()
{
    mAccumulationTimer->DeregisterExec(pmBase::GetCurrentTimeInSecs() - mStartTime);
}

    
/* class pmAccumulatedTimesSorter */
pmAccumulatedTimesSorter::pmAccumulatedTimesSorter()
    : mMaxNameLength(0)
    , mLogsFlushed(false)
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
}
    
pmAccumulatedTimesSorter::~pmAccumulatedTimesSorter()
{
    Lock();
    Unlock();

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
}

void pmAccumulatedTimesSorter::FlushLogs()
{
    Lock();

    std::stringstream lStream;
    lStream << "Accumulated Timings ..." << std::endl;

    std::map<std::pair<double, std::string>, accumulatedData>::reverse_iterator lIter = mAccumulatedTimesMap.rbegin(), lEndIter = mAccumulatedTimesMap.rend();
    for(; lIter != lEndIter; ++lIter)
    {
        lStream << std::left << std::setw(mMaxNameLength + 1) << (lIter->first.second.empty() ? "UNNAMED" : lIter->first.second.c_str()) << " => " << std::fixed << "Accumulated Time: " << std::setw(MAX_FLOAT_WIDTH) << lIter->first.first << "s Actual Time: " << std::setw(MAX_FLOAT_WIDTH) << lIter->second.actualTime << "s Min Time: " << std::setw(MAX_FLOAT_WIDTH) << lIter->second.minTime << "s Max Time: " << std::setw(MAX_FLOAT_WIDTH) << lIter->second.maxTime << "s Exec Count: " << std::setw(MAX_INT_WIDTH) << lIter->second.execCount << std::endl;
    }
    
    mLogsFlushed = true;

    Unlock();

    pmLogger::GetLogger()->LogDeferred(pmLogger::MINIMAL, pmLogger::WARNING, lStream.str().c_str(), true);
}

pmAccumulatedTimesSorter* pmAccumulatedTimesSorter::GetAccumulatedTimesSorter()
{
    static pmAccumulatedTimesSorter lAccumulatedTimesSorter;
    return &lAccumulatedTimesSorter;
}
    
void pmAccumulatedTimesSorter::Insert(std::string& pName, double pAccumulatedTime, double pMinTime, double pMaxTime, double pActualTime, uint pExecCount)
{
    if(pAccumulatedTime < MIN_ACCUMULATED_TIME)
        return;
    
    Lock();
    
    const char* lStr = pName.c_str();
    if(strlen(lStr) > mMaxNameLength)
        mMaxNameLength = strlen(lStr);
    
#if 0
    if(mLogsFlushed)
    {
        std::cout << std::left << std::setw(mMaxNameLength + 1) << lStr << " => " << std::fixed << "Accumulated Time: " << std::setw(MAX_FLOAT_WIDTH) << pAccumulatedTime << "s Actual Time: " << std::setw(MAX_FLOAT_WIDTH) << pActualTime << "s Min Time: " << std::setw(MAX_FLOAT_WIDTH) << pMinTime << "s Max Time: " << std::setw(MAX_FLOAT_WIDTH) << pMaxTime << "s Exec Count: " << std::setw(MAX_INT_WIDTH) << pExecCount << std::endl;
    }
    else
#endif
    {
        accumulatedData& lData = mAccumulatedTimesMap[std::make_pair(pAccumulatedTime, pName)];
        lData.minTime = pMinTime;
        lData.maxTime = pMaxTime;
        lData.actualTime = pActualTime;
        lData.execCount = pExecCount;
    }
    
    Unlock();
}

void pmAccumulatedTimesSorter::Lock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
}
    
void pmAccumulatedTimesSorter::Unlock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );
}
#endif

#ifdef SUPPORT_CUDA
/* class pmCudaCacheEvictor */
void pmCudaCacheEvictor::operator() (const std::shared_ptr<pmCudaCacheValue>& pValue)
{
#ifdef DUMP_CUDA_CACHE_STATISTICS
    mStub->RecordCudaCacheEviction(pValue->allocationLength);
#endif

    mStub->GetCudaChunkCollection()->Deallocate(pValue->cudaPtr);
}


/* struct pmCudaMemChunkTraits */
std::shared_ptr<pmMemChunk> pmCudaMemChunkTraits::creator::operator()(size_t pSize)
{
    void* lCudaBuffer = pmCudaInterface::AllocateCudaMem(pSize);
    return std::shared_ptr<pmMemChunk>(new pmMemChunk(lCudaBuffer, pSize));
}

void pmCudaMemChunkTraits::destructor::operator()(const std::shared_ptr<pmMemChunk>& pPtr)
{
    pmCudaInterface::DeallocateCudaMem(pPtr->GetChunk());
}


#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
/* struct pmPinnedMemChunkTraits */
std::shared_ptr<pmMemChunk> pmPinnedMemChunkTraits::creator::operator()(size_t pSize)
{
    void* lPinnedBuffer = pmCudaInterface::AllocatePinnedBuffer(pSize);
    return std::shared_ptr<pmMemChunk>(new pmMemChunk(lPinnedBuffer, pSize));
}

void pmPinnedMemChunkTraits::destructor::operator()(const std::shared_ptr<pmMemChunk>& pPtr)
{
    pmCudaInterface::DeallocatePinnedBuffer(pPtr->GetChunk());
}
#endif

#endif

    
/* class pmDestroyOnException */
pmDestroyOnException::pmDestroyOnException()
{
    mDestroy = true;
}

pmDestroyOnException::~pmDestroyOnException()
{
    if(mDestroy)
    {
        size_t lSize = mFreePtrs.size();
        for(size_t i = 0; i < lSize; ++i)
            free(mFreePtrs[i]);
    }
}

void pmDestroyOnException::AddFreePtr(void* pPtr)
{
    mFreePtrs.push_back(pPtr);
}

void pmDestroyOnException::AddDeletePtr(selective_finalize_base* pDeletePtr)
{
    mDeletePtrs.push_back(pDeletePtr);
}
    
void pmDestroyOnException::SetDestroy(bool pDestroy)
{
    mDestroy = pDestroy;

    if(!mDestroy)
    {
        size_t lSize = mDeletePtrs.size();
        for(size_t i = 0; i < lSize; ++i)
            mDeletePtrs[i]->SetDelete(false);
    }
}

bool pmDestroyOnException::ShouldDelete()
{
    return mDestroy;
}


#ifdef TRACK_MEM_COPIES
pmMemCopyTracker gMemCopyTracker;
    
pmMemCopyTracker::pmMemCopyTracker()
    : mBytesCopied(0)
    , mHostId(std::numeric_limits<uint>::max())
{
}
    
pmMemCopyTracker::~pmMemCopyTracker()
{
    std::cout << "Host " << mHostId << " mem-copied " << mBytesCopied << " bytes !!!" << std::endl;
}

void pmMemCopyTracker::SetHostId(uint pHostId)
{
    mHostId = pHostId;
}

void pmMemCopyTracker::Add(size_t pBytes)
{
    mBytesCopied += (ulong)pBytes;
}
#endif

    
#ifdef SUPPORT_SPLIT_SUBTASKS
bool operator<(const pmSplitInfo& pInfo1, const pmSplitInfo& pInfo2)
{
    if(pInfo1.splitId == pInfo2.splitId)
        return (pInfo1.splitCount < pInfo2.splitCount);
        
    return (pInfo1.splitId < pInfo2.splitId);
}
#endif


/* struct naturalSorter */
std::string naturalSorter::GetNextBlock(const std::string& pStr, size_t& pIndex) const
{
    size_t lLength = pStr.length();
    if(pIndex >= lLength)
        return std::string();
    
    std::stringstream lBlock;
    lBlock << pStr[pIndex];
    
    ++pIndex;
    for(size_t i = pIndex; i < lLength; ++i, ++pIndex)
    {
        bool lDigit1 = std::isdigit(pStr[i]);
        bool lDigit2 = std::isdigit(pStr[i - 1]);
        
        if(lDigit1 == lDigit2)
            lBlock << pStr[i];
        else
            break;
    }
    
    return lBlock.str();
}

bool naturalSorter::operator() (const std::string& pStr1, const std::string& pStr2) const
{
    size_t pIndex1 = 0;
    size_t pIndex2 = 0;
    size_t lLength1 = pStr1.length();
    size_t lLength2 = pStr2.length();
    
    while(pIndex1 < lLength1 || pIndex2 < lLength2)
    {
        std::string lStr1 = GetNextBlock(pStr1, pIndex1);
        std::string lStr2 = GetNextBlock(pStr2, pIndex2);
        
        bool lIsDigit1 = (!lStr1.empty() && std::isdigit(lStr1[0]));
        bool lIsDigit2 = (!lStr2.empty() && std::isdigit(lStr2[0]));
        
        if(lIsDigit1 && lIsDigit2)
        {
            uint lNum1 = atoi(lStr1.c_str());
            uint lNum2 = atoi(lStr2.c_str());
            
            if(lNum1 != lNum2)
                return (lNum1 < lNum2);
        }
        else
        {
            int lResult = lStr1.compare(lStr2);
            
            if(lResult)
                return (lResult < 0);
        }
    }
    
    return false;
}


/* struct stubSorter */
bool stubSorter::operator() (const pmExecutionStub* pStub1, const pmExecutionStub* pStub2) const
{
    return (pStub1->GetProcessingElement()->GetGlobalDeviceIndex() < pStub2->GetProcessingElement()->GetGlobalDeviceIndex());
}


}   // end namespace pm


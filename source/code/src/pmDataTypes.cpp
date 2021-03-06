
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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
    
#define MINIMUM_TIME_TO_REPORT 0.1

STATIC_ACCESSOR_GLOBAL(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("gMemCopyResourceLock"), GetMemCopyResourceLock)

pmMemCopyTracker::pmMemCopyStatistics::pmMemCopyStatistics()
    : mBytesCopied(0)
    , mElapsedTime(0)
    , mStartTime(0)
    , mOngoingMemCopies(0)
{}

void pmMemCopyTracker::pmMemCopyStatistics::BeginRecord(size_t pBytes)
{
    mBytesCopied += (ulong)pBytes;
    
    if(!mOngoingMemCopies)
        mStartTime = pmBase::GetCurrentTimeInSecs();
    
    ++mOngoingMemCopies;
}

void pmMemCopyTracker::pmMemCopyStatistics::EndRecord()
{
    --mOngoingMemCopies;
    
    if(!mOngoingMemCopies)
        mElapsedTime += pmBase::GetCurrentTimeInSecs() - mStartTime;
}

pmMemCopyTracker::pmMemCopyTracker()
    : mHostId(std::numeric_limits<uint>::max())
{
}
    
pmMemCopyTracker::~pmMemCopyTracker()
{
    std::cout << "Host " << mHostId << " took " << mNodeTimer.mElapsedTime << " seconds to mem-copy " << mNodeTimer.mBytesCopied << " bytes !!!";
    
    bool lFirst = true;
    for_each(mBifurcationMap, [&lFirst] (const std::pair<std::string, pmMemCopyStatistics>& pPair)
    {
        if(pPair.second.mBytesCopied && pPair.second.mElapsedTime > (double)MINIMUM_TIME_TO_REPORT)
        {
            if(lFirst)
            {
                std::cout << " [ ";
                lFirst = false;
            }
            
            std::cout << pPair.first << " => (" << pPair.second.mBytesCopied << " bytes, " << pPair.second.mElapsedTime << " secs); ";
        }
    });
    
    if(!lFirst)
        std::cout << "]";
    
    std::cout << std::endl;
}

void pmMemCopyTracker::SetHostId(uint pHostId)
{
    mHostId = pHostId;
}

void pmMemCopyTracker::Begin(size_t pBytes, const std::string& pKey)
{
    FINALIZE_RESOURCE(dCompletionLock, GetMemCopyResourceLock().Lock(), GetMemCopyResourceLock().Unlock());

    mNodeTimer.BeginRecord(pBytes);
    mBifurcationMap[pKey].BeginRecord(pBytes);
}
    
void pmMemCopyTracker::End(const std::string& pKey)
{
    FINALIZE_RESOURCE(dCompletionLock, GetMemCopyResourceLock().Lock(), GetMemCopyResourceLock().Unlock());
    
    mNodeTimer.EndRecord();
    mBifurcationMap[pKey].EndRecord();
}

#endif
    
    
#ifdef DUMP_DATA_COMPRESSION_STATISTICS
void pmCompressionDataRecorder::RecordCompressionData(ulong pUncompresedSize, ulong pCompressedSize, bool pIsDataForNetworkTransfer)
{
    std::stringstream lStream;
    
    if(pIsDataForNetworkTransfer)
        lStream << "Network compression: " << pUncompresedSize << " bytes compressed to " << pCompressedSize << " (" << 100.0 * ((double)(pUncompresedSize - pCompressedSize) / pUncompresedSize) << "% compression achieved)"<< std::endl;
    else
        lStream << "GPU->CPU compression: " << pUncompresedSize << " bytes compressed to " << pCompressedSize << " (" << 100.0 * ((double)(pUncompresedSize - pCompressedSize) / pUncompresedSize) << "% compression achieved)"<< std::endl;
    
    pmLogger::GetLogger()->LogDeferred(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
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


size_t getReductionDataTypeSize(pmReductionDataType pDataType)
{
    switch(pDataType)
    {
        case REDUCE_INTS:
        {
            return sizeof(int);
        }
            
        case REDUCE_UNSIGNED_INTS:
        {
            return sizeof(uint);
        }
            
        case REDUCE_LONGS:
        {
            return sizeof(long);
        }
            
        case REDUCE_UNSIGNED_LONGS:
        {
            return sizeof(ulong);
        }
            
        case REDUCE_FLOATS:
        {
            return sizeof(float);
        }
            
        case REDUCE_DOUBLES:
        {
            return sizeof(double);
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
    
    return 0;
}

void findReductionOpAndDataType(pmDataReductionCallback pCallback, pmReductionOpType& pOpType, pmReductionDataType& pDataType)
{
    pOpType = MAX_REDUCTION_OP_TYPES;
    pDataType = MAX_REDUCTION_DATA_TYPES;

    pOpType = findReductionOpType<REDUCE_INTS>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_INTS;
        return;
    }
    
    pOpType = findReductionOpType<REDUCE_UNSIGNED_INTS>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_UNSIGNED_INTS;
        return;
    }
    
    pOpType = findReductionOpType<REDUCE_LONGS>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_LONGS;
        return;
    }
    
    pOpType = findReductionOpType<REDUCE_UNSIGNED_LONGS>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_UNSIGNED_LONGS;
        return;
    }

    pOpType = findReductionOpType<REDUCE_FLOATS>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_FLOATS;
        return;
    }
    
    pOpType = findReductionOpType<REDUCE_DOUBLES>(pCallback);
    if(pOpType != MAX_REDUCTION_OP_TYPES)
    {
        pDataType = REDUCE_DOUBLES;
        return;
    }
}


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


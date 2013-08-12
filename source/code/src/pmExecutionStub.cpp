
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

#include "pmExecutionStub.h"
#include "pmSignalWait.h"
#include "pmTask.h"
#include "pmDispatcherGPU.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmCommand.h"
#include "pmCallbackUnit.h"
#include "pmScheduler.h"
#include "pmMemSection.h"
#include "pmTaskManager.h"
#include "pmTls.h"
#include "pmLogger.h"
#include "pmMemoryManager.h"
#include "pmReducer.h"
#include "pmRedistributor.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <limits>

#include SYSTEM_CONFIGURATION_HEADER // for sched_setaffinity

#define INVOKE_SAFE_PROPAGATE_ON_FAILURE(objectType, object, function, ...) \
{ \
	pmStatus dStatus = pmSuccess; \
	objectType* dObject = object; \
	if(dObject) \
		dStatus = dObject->function(__VA_ARGS__); \
	if(dStatus != pmSuccess) \
		return dStatus; \
}

#define THROW_IF_PARTIAL_RANGE_OVERLAP(start1, end1, start2, end2) \
{ \
    bool dTotalOverlap = (start1 == start2 && end1 == end2); \
    bool dNoOverlap = (end2 < start1) || (start2 > end1); \
    \
    if(!dTotalOverlap && !dNoOverlap) \
        PMTHROW(pmFatalErrorException()); \
}

namespace pm
{

using namespace execStub;

/* class pmExecutionStub */
pmExecutionStub::pmExecutionStub(uint pDeviceIndexOnMachine)
    : mDeviceIndexOnMachine(pDeviceIndexOnMachine)
    , mExecutingLibraryCode(1)
    , mCurrentSubtaskRangeLock __LOCK_NAME__("pmExecutionStub::mCurrentSubtaskRangeLock")
    , mCurrentSubtaskRangeStats(NULL)
    , mDeferredShadowMemCommitsLock __LOCK_NAME__("pmExecutionStub::mDeferredShadowMemCommitsLock")
{
}

pmExecutionStub::~pmExecutionStub()
{
	#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down stub thread");
	#endif
}

pmProcessingElement* pmExecutionStub::GetProcessingElement()
{
	return pmDevicePool::GetDevicePool()->GetDeviceAtMachineIndex(PM_LOCAL_MACHINE, mDeviceIndexOnMachine);
}
    
pmStatus pmExecutionStub::ThreadBindEvent(size_t pPhysicalMemory, size_t pTotalStubCount)
{
	stubEvent lEvent;
	lEvent.eventId = THREAD_BIND;
    lEvent.bindDetails.physicalMemory = pPhysicalMemory;
    lEvent.bindDetails.totalStubCount = pTotalStubCount;
	SwitchThread(lEvent, MAX_CONTROL_PRIORITY);

    return pmSuccess;
}

#ifdef DUMP_EVENT_TIMELINE
pmStatus pmExecutionStub::InitializeEventTimeline()
{
	stubEvent lEvent;
	lEvent.eventId = INIT_EVENT_TIMELINE;

	return SwitchThread(lEvent, MAX_CONTROL_PRIORITY);
}
#endif

pmStatus pmExecutionStub::Push(pmSubtaskRange& pRange)
{
	if(pRange.endSubtask < pRange.startSubtask)
		PMTHROW(pmFatalErrorException());

	stubEvent lEvent;
	lEvent.execDetails.range = pRange;
	lEvent.execDetails.rangeExecutedOnce = false;
	lEvent.execDetails.lastExecutedSubtaskId = 0;
	lEvent.eventId = SUBTASK_EXEC;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmExecutionStub::ReduceSubtasks(pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2)
{
	stubEvent lEvent;
	lEvent.reduceDetails.task = pTask;
	lEvent.reduceDetails.subtaskId1 = pSubtaskId1;
    lEvent.reduceDetails.stub2 = pStub2;
	lEvent.reduceDetails.subtaskId2 = pSubtaskId2;
	lEvent.eventId = SUBTASK_REDUCE;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmExecutionStub::ProcessNegotiatedRange(pmSubtaskRange& pRange)
{
    stubEvent lEvent;
    lEvent.negotiatedRangeDetails.range = pRange;
    lEvent.eventId = NEGOTIATED_RANGE;
    
	return SwitchThread(lEvent, pRange.task->GetPriority());
}

/* This is an asynchronous call. Current subtask is not cancelled immediately. */
pmStatus pmExecutionStub::CancelAllSubtasks(pmTask* pTask, bool pTaskListeningOnCancellation)
{
    ushort lPriority = pTask->GetPriority();

    // There is atmost one range per task at a time
    stubEvent lTaskEvent;
    DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, pTask, lTaskEvent);

    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

    if(mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pTask)
        CancelCurrentlyExecutingSubtaskRange(pTaskListeningOnCancellation);
    
#ifdef _DEBUG
    if(pTask->IsMultiAssignEnabled() && !pmTaskManager::GetTaskManager()->DoesTaskHavePendingSubtasks(pTask))
    {
        std::map<std::pair<pmTask*, ulong>, std::vector<pmProcessingElement*> >::iterator lIter = mSecondaryAllotteeMap.begin(), lEndIter = mSecondaryAllotteeMap.end();
    
        for(; lIter != lEndIter; ++lIter)
        {
            if(lIter->first.first == pTask)
                PMTHROW(pmFatalErrorException());
        }
    }
#endif

    return pmSuccess;
}
    
/* This is an asynchronous call. Current subtask is not cancelled immediately. */
pmStatus pmExecutionStub::CancelSubtaskRange(pmSubtaskRange& pRange)
{
    ushort lPriority = pRange.task->GetPriority();
    
    stubEvent lTaskEvent;
    bool lFound = (DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, pRange.task, lTaskEvent) == pmSuccess);

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
        
        if(mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pRange.task)
        {
            if(mCurrentSubtaskRangeStats->startSubtaskId >= pRange.startSubtask && mCurrentSubtaskRangeStats->endSubtaskId <= pRange.endSubtask)
                CancelCurrentlyExecutingSubtaskRange(false);
        }
    }
    
    if(lFound && (pRange.endSubtask < lTaskEvent.execDetails.range.startSubtask || pRange.startSubtask > lTaskEvent.execDetails.range.endSubtask))
        SwitchThread(lTaskEvent, lPriority);

    return pmSuccess;
}

pmStatus pmExecutionStub::FreeGpuResources()
{
#ifdef SUPPORT_CUDA
    stubEvent lEvent;
    lEvent.eventId = FREE_GPU_RESOURCES;

    SwitchThread(lEvent, RESERVED_PRIORITY);
#endif

	return pmSuccess;
}
    
void pmExecutionStub::FreeTaskResources(pmTask* pTask)
{
#ifdef SUPPORT_CUDA
    if(dynamic_cast<pmStubCUDA*>(this))
    {
        stubEvent lEvent;
        lEvent.eventId = FREE_TASK_RESOURCES;
        lEvent.freeTaskResourcesDetails.taskOriginatingHost = pTask->GetOriginatingHost();
        lEvent.freeTaskResourcesDetails.taskSequenceNumber = pTask->GetSequenceNumber();

        SwitchThread(lEvent, RESERVED_PRIORITY);
    }
#endif
}
    
void pmExecutionStub::PostHandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus)
{
    stubEvent lEvent;
    lEvent.execCompletionDetails.range = pRange;
    lEvent.execCompletionDetails.execStatus = pExecStatus;
    lEvent.eventId = POST_HANDLE_EXEC_COMPLETION;
    
    SwitchThread(lEvent, pRange.task->GetPriority() - 1);
}
    
void pmExecutionStub::ProcessDeferredShadowMemCommits(pmTask* pTask)
{
    bool lPendingCommits = false;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDeferredShadowMemCommitsLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDeferredShadowMemCommitsLock, Lock(), Unlock());
        lPendingCommits = (mDeferredShadowMemCommits.find(pTask) != mDeferredShadowMemCommits.end());
    }

    if(lPendingCommits)
    {
        pTask->RecordStubWillSendShadowMemCommitMessage();
    
        stubEvent lEvent;
        lEvent.deferredShadowMemCommitsDetails.task = pTask;
        lEvent.eventId = DEFERRED_SHADOW_MEM_COMMITS;
        
        SwitchThread(lEvent, pTask->GetPriority() - 1);
    }
}
    
void pmExecutionStub::ReductionFinishEvent(pmTask* pTask)
{
    stubEvent lEvent;
    lEvent.reductionFinishDetails.task = pTask;
    lEvent.eventId = REDUCTION_FINISH;
    
    SwitchThread(lEvent, pTask->GetPriority());
}
    
void pmExecutionStub::ProcessRedistributionBucket(pmTask* pTask, size_t pBucketIndex)
{
    stubEvent lEvent;
    lEvent.processRedistributionBucketDetails.task = pTask;
    lEvent.processRedistributionBucketDetails.bucketIndex = pBucketIndex;
    lEvent.eventId = PROCESS_REDISTRIBUTION_BUCKET;
    
    SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmExecutionStub::NegotiateRange(pmProcessingElement* pRequestingDevice, pmSubtaskRange& pRange)
{
    pmProcessingElement* lLocalDevice = GetProcessingElement();
    if(pRange.originalAllottee != lLocalDevice)
        PMTHROW(pmFatalErrorException());

    ushort lPriority = pRange.task->GetPriority();
    
    stubEvent lTaskEvent;
    if(pRange.task->IsMultiAssignEnabled())
    {
        if(pRange.task->GetSchedulingModel() == scheduler::PULL)
        {
        #ifdef _DEBUG
            if(pRange.startSubtask != pRange.endSubtask && dynamic_cast<pmStubCPU*>(this))
                PMTHROW(pmFatalErrorException());
        #endif
        
            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
            
            if(mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pRange.task)
            {
                THROW_IF_PARTIAL_RANGE_OVERLAP(mCurrentSubtaskRangeStats->startSubtaskId, mCurrentSubtaskRangeStats->endSubtaskId, pRange.startSubtask, pRange.endSubtask);
                
                if(mCurrentSubtaskRangeStats->startSubtaskId == pRange.startSubtask && mCurrentSubtaskRangeStats->endSubtaskId == pRange.endSubtask && !mCurrentSubtaskRangeStats->reassigned)
                {
                #ifdef _DEBUG
                    if(!mCurrentSubtaskRangeStats->originalAllottee)
                        PMTHROW(pmFatalErrorException());
                #endif

                    std::pair<pmTask*, ulong> lPair(pRange.task, pRange.endSubtask);
                    if(mSecondaryAllotteeMap.find(lPair) != mSecondaryAllotteeMap.end())
                    {
                        std::vector<pmProcessingElement*>& lSecondaryAllottees = mSecondaryAllotteeMap[lPair];

                    #ifdef _DEBUG
                        if(std::find(lSecondaryAllottees.begin(), lSecondaryAllottees.end(), pRequestingDevice) == lSecondaryAllottees.end())
                            PMTHROW(pmFatalErrorException());
                    #endif
                    
                    #ifdef TRACK_MULTI_ASSIGN
                        std::cout << "[Host " << pmGetHostId() << "]: Range negotiation success from device " << GetProcessingElement()->GetGlobalDeviceIndex() << " to device " << pRequestingDevice->GetGlobalDeviceIndex() << "; Negotiated range [" << pRange.startSubtask << ", " << pRange.endSubtask << "]" << std::endl;
                    #endif
                    
                        pmScheduler::GetScheduler()->SendRangeNegotiationSuccess(pRequestingDevice, pRange);
                        mCurrentSubtaskRangeStats->reassigned = true;
                        CancelCurrentlyExecutingSubtaskRange(false);
                                
                        if(mCurrentSubtaskRangeStats->parentRangeStartSubtask != mCurrentSubtaskRangeStats->startSubtaskId)
                        {
                            pmSubtaskRange lCompletedRange;
                            lCompletedRange.task = pRange.task;
                            lCompletedRange.startSubtask = mCurrentSubtaskRangeStats->parentRangeStartSubtask;
                            lCompletedRange.endSubtask = mCurrentSubtaskRangeStats->startSubtaskId - 1;
                            lCompletedRange.originalAllottee = NULL;

                            PostHandleRangeExecutionCompletion(lCompletedRange, pmSuccess);
                        }
                    
                        pmScheduler::GetScheduler()->SendSubtaskRangeCancellationMessage(pRange.originalAllottee, pRange);
                        std::vector<pmProcessingElement*>::iterator lBegin = lSecondaryAllottees.begin();
                        std::vector<pmProcessingElement*>::iterator lEnd = lSecondaryAllottees.end();
                    
                        for(; lBegin < lEnd; ++lBegin)
                        {
                            if(*lBegin != pRequestingDevice)
                                pmScheduler::GetScheduler()->SendSubtaskRangeCancellationMessage(*lBegin, pRange);
                        }

                        mSecondaryAllotteeMap.erase(lPair);
                    }
                }
            }
        }
        else
        {
            pmSubtaskRange lNegotiatedRange;
            bool lSuccessfulNegotiation = false;
            bool lCurrentTransferred = false;
        
            bool lFound = (DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, pRange.task, lTaskEvent) == pmSuccess);

            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
        
            bool lConsiderCurrent = (mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pRange.task && mCurrentSubtaskRangeStats->startSubtaskId >= pRange.startSubtask && mCurrentSubtaskRangeStats->endSubtaskId <= pRange.endSubtask && !mCurrentSubtaskRangeStats->reassigned);
                
            if(lFound)
            {
                if(lTaskEvent.execDetails.range.originalAllottee == NULL)
                {       
                    if(pRange.endSubtask < lTaskEvent.execDetails.range.startSubtask || pRange.startSubtask > lTaskEvent.execDetails.range.endSubtask)
                    {
                        SwitchThread(lTaskEvent, lPriority);
                    }
                    else
                    {
                        ulong lFirstPendingSubtask = (lTaskEvent.execDetails.rangeExecutedOnce ? (lTaskEvent.execDetails.lastExecutedSubtaskId + 1) : lTaskEvent.execDetails.range.startSubtask);
                        ulong lLastPendingSubtask = lTaskEvent.execDetails.range.endSubtask;
                    
                        if(lConsiderCurrent)
                        {
                        #ifdef _DEBUG
                            if(!mCurrentSubtaskRangeStats->originalAllottee)
                                PMTHROW(pmFatalErrorException());

                            if(lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskRangeStats->endSubtaskId)
                                PMTHROW(pmFatalErrorException());
                        #endif
                        
                            lFirstPendingSubtask -=  (mCurrentSubtaskRangeStats->endSubtaskId - mCurrentSubtaskRangeStats->startSubtaskId + 1);
                            lCurrentTransferred = true;
                        }
                    
                        lNegotiatedRange.task = pRange.task;
                        lNegotiatedRange.startSubtask = std::max(pRange.startSubtask, lFirstPendingSubtask);
                        lNegotiatedRange.endSubtask = std::min(pRange.endSubtask, lLastPendingSubtask);
                        lNegotiatedRange.originalAllottee = pRange.originalAllottee;
                    
                        lSuccessfulNegotiation = true;
                    
                    #ifdef _DEBUG
                        if(lNegotiatedRange.startSubtask > lNegotiatedRange.endSubtask || lNegotiatedRange.endSubtask < lLastPendingSubtask)
                            PMTHROW(pmFatalErrorException());
                    #endif
                    
                        if(lNegotiatedRange.startSubtask != lTaskEvent.execDetails.range.startSubtask)  // Entire range not negotiated
                        {
                            // Find range left with original allottee
                            lTaskEvent.execDetails.range.endSubtask = lNegotiatedRange.startSubtask - 1;
                            if(lConsiderCurrent && lTaskEvent.execDetails.range.endSubtask >= mCurrentSubtaskRangeStats->endSubtaskId)
                                lCurrentTransferred = false;   // current subtask still with original allottee
                        
                            bool lCurrentSubtaskInRemainingRange = (lConsiderCurrent && !lCurrentTransferred);

                            if(!lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.range.endSubtask == (lTaskEvent.execDetails.lastExecutedSubtaskId - (lCurrentTransferred ? (mCurrentSubtaskRangeStats->endSubtaskId - mCurrentSubtaskRangeStats->startSubtaskId + 1) : 0)))  // no pending subtask
                            {
                                if(lTaskEvent.execDetails.rangeExecutedOnce)
                                    PostHandleRangeExecutionCompletion(lTaskEvent.execDetails.range, pmSuccess);
                            }
                            else if(lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // only current subtask range pending
                            {
                            #ifdef _DEBUG
                                if(lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskRangeStats->endSubtaskId)
                                    PMTHROW(pmFatalErrorException());
                            #endif
                            
                                mCurrentSubtaskRangeStats->forceAckFlag = true;  // send acknowledgement of the done range after current subtask range finishes
                            }
                            else
                            {   // pending range does not have current subtask or has more subtasks after the current one
                                SwitchThread(lTaskEvent, lPriority);
                            }
                        }
                    }
                }
                else
                {
                    SwitchThread(lTaskEvent, lPriority);
                }
            }
            else if(lConsiderCurrent)
            {
            #ifdef _DEBUG
                if(!mCurrentSubtaskRangeStats->originalAllottee)
                    PMTHROW(pmFatalErrorException());
            #endif
            
                lSuccessfulNegotiation = true;
                lCurrentTransferred = true;
            
                lNegotiatedRange.task = pRange.task;
                lNegotiatedRange.startSubtask = mCurrentSubtaskRangeStats->startSubtaskId;
                lNegotiatedRange.endSubtask = mCurrentSubtaskRangeStats->endSubtaskId;
                lNegotiatedRange.originalAllottee = pRange.originalAllottee;
            
                if(mCurrentSubtaskRangeStats->parentRangeStartSubtask != mCurrentSubtaskRangeStats->startSubtaskId)
                {
                    pmSubtaskRange lCompletedRange;
                    lCompletedRange.task = pRange.task;
                    lCompletedRange.startSubtask = mCurrentSubtaskRangeStats->parentRangeStartSubtask;
                    lCompletedRange.endSubtask = mCurrentSubtaskRangeStats->startSubtaskId - 1;
                    lCompletedRange.originalAllottee = NULL;

                    PostHandleRangeExecutionCompletion(lCompletedRange, pmSuccess);
                }
            }

            if(lSuccessfulNegotiation)
            {
            #ifdef TRACK_MULTI_ASSIGN
                std::cout << "[Host " << pmGetHostId() << "]: Range negotiation success from device " << GetProcessingElement()->GetGlobalDeviceIndex() << " to device " << pRequestingDevice->GetGlobalDeviceIndex() << "; Negotiated range [" << lNegotiatedRange.startSubtask << ", " << lNegotiatedRange.endSubtask << "]" << std::endl;
            #endif
            
                pmScheduler::GetScheduler()->SendRangeNegotiationSuccess(pRequestingDevice, lNegotiatedRange);
            }

            if(lCurrentTransferred)
            {
                mCurrentSubtaskRangeStats->reassigned = true;
                CancelCurrentlyExecutingSubtaskRange(false);
            }
        }
    }
#ifdef _DEBUG
    else
    {
        PMTHROW(pmFatalErrorException());
    }
#endif
    
    return pmSuccess;
}
    
pmStatus pmExecutionStub::StealSubtasks(pmTask* pTask, pmProcessingElement* pRequestingDevice, double pRequestingDeviceExecutionRate)
{
    bool lStealSuccess = false;
    ushort lPriority = pTask->GetPriority();
    pmProcessingElement* lLocalDevice = GetProcessingElement();
    double lLocalRate = pTask->GetTaskExecStats().GetStubExecutionRate(this);
    
    stubEvent lTaskEvent;
    bool lFound = (DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, pTask, lTaskEvent) == pmSuccess);
    
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
    
    if(lFound)
    {
        if(!pTask->IsMultiAssignEnabled() || lTaskEvent.execDetails.range.originalAllottee == NULL)
        {
            ulong lStealCount = 0;
            ulong lAvailableSubtasks = ((lTaskEvent.execDetails.rangeExecutedOnce) ? (lTaskEvent.execDetails.range.endSubtask - lTaskEvent.execDetails.lastExecutedSubtaskId) : (lTaskEvent.execDetails.range.endSubtask - lTaskEvent.execDetails.range.startSubtask + 1));

            if(lAvailableSubtasks)
            {
                double lOverheadTime = 0;	// Add network and other overheads here
                double lTotalExecRate = lLocalRate + pRequestingDeviceExecutionRate;
            
                if(lLocalRate == (double)0.0)
                {
                    lStealCount = lAvailableSubtasks;
                }
                else
                {
                    double lTotalExecutionTimeRequired = lAvailableSubtasks / lTotalExecRate;	// if subtasks are divided between both devices, how much time reqd
                    double lLocalExecutionTimeForAllSubtasks = lAvailableSubtasks / lLocalRate;	// if all subtasks are executed locally, how much time it will take
                    double lDividedExecutionTimeForAllSubtasks = lTotalExecutionTimeRequired + lOverheadTime;
                    
                    if(lLocalExecutionTimeForAllSubtasks > lDividedExecutionTimeForAllSubtasks)
                    {
                        double lTimeDiff = lLocalExecutionTimeForAllSubtasks - lDividedExecutionTimeForAllSubtasks;
                        lStealCount = (ulong)(lTimeDiff * lLocalRate);
                    }
                }
            }
            
            if(lStealCount)
            {                        
                pmSubtaskRange lStolenRange;
                lStolenRange.task = pTask;
                lStolenRange.startSubtask = (lTaskEvent.execDetails.range.endSubtask - lStealCount) + 1;
                lStolenRange.endSubtask = lTaskEvent.execDetails.range.endSubtask;
                lStolenRange.originalAllottee = NULL;
                
                lTaskEvent.execDetails.range.endSubtask -= lStealCount;
                
                bool lCurrentSubtaskInRemainingRange = false;
                if(mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pTask && lTaskEvent.execDetails.range.startSubtask <= mCurrentSubtaskRangeStats->startSubtaskId && mCurrentSubtaskRangeStats->endSubtaskId <= lTaskEvent.execDetails.range.endSubtask)
                {
                #ifdef _DEBUG
                    if(!mCurrentSubtaskRangeStats->originalAllottee || mCurrentSubtaskRangeStats->reassigned)
                        PMTHROW(pmFatalErrorException());
                #endif
                
                    lCurrentSubtaskInRemainingRange = true;
                }
                
                if(!lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // no pending subtask
                {
                    if(lTaskEvent.execDetails.rangeExecutedOnce)
                        PostHandleRangeExecutionCompletion(lTaskEvent.execDetails.range, pmSuccess);
                }
                else if(lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // only current subtask range pending
                {
                #ifdef _DEBUG
                    if(lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskRangeStats->endSubtaskId)
                        PMTHROW(pmFatalErrorException());
                #endif

                    mCurrentSubtaskRangeStats->forceAckFlag = true;  // send acknowledgement of the done range after current subtask range finishes
                }
                else // pending range does not have current subtask or has more subtasks after the current one
                {
                    if(lTaskEvent.execDetails.rangeExecutedOnce || !(lStolenRange.startSubtask == lTaskEvent.execDetails.range.startSubtask && lStolenRange.endSubtask == lTaskEvent.execDetails.range.endSubtask + lStealCount))
                        SwitchThread(lTaskEvent, lPriority);
                }
                
                lStealSuccess = true;
                pmScheduler::GetScheduler()->StealSuccessEvent(pRequestingDevice, lLocalDevice, lStolenRange);
            }
            else
            {
                SwitchThread(lTaskEvent, lPriority);
            }
        }
        else
        {
            SwitchThread(lTaskEvent, lPriority);
        }
    }
    else if(pTask->IsMultiAssignEnabled())
    {
        pmSubtaskRange lStolenRange;

        if(mCurrentSubtaskRangeStats && mCurrentSubtaskRangeStats->task == pTask && mCurrentSubtaskRangeStats->originalAllottee && !mCurrentSubtaskRangeStats->reassigned)
        {
            std::pair<pmTask*, ulong> lPair(pTask, mCurrentSubtaskRangeStats->endSubtaskId);
            if((mSecondaryAllotteeMap.find(lPair) == mSecondaryAllotteeMap.end())
               || (mSecondaryAllotteeMap[lPair].size() < MAX_SUBTASK_MULTI_ASSIGN_COUNT - 1))
            {
                ulong lMultiAssignSubtaskCount = mCurrentSubtaskRangeStats->endSubtaskId - mCurrentSubtaskRangeStats->startSubtaskId + 1;

                bool lLocalRateZero = (lLocalRate == (double)0.0);
                double lTransferOverheadTime = 0;   // add network and other overheads here
                double lExpectedRemoteTimeToExecute = (lMultiAssignSubtaskCount/pRequestingDeviceExecutionRate);
                double lExpectedLocalTimeToFinish = (lLocalRateZero ? 0.0 : ((lMultiAssignSubtaskCount/lLocalRate) - (pmBase::GetCurrentTimeInSecs() - mCurrentSubtaskRangeStats->startTime)));
            
                if(lLocalRateZero || (lExpectedRemoteTimeToExecute + lTransferOverheadTime < lExpectedLocalTimeToFinish))
                {
                    lStolenRange.task = pTask;
                    lStolenRange.startSubtask = mCurrentSubtaskRangeStats->startSubtaskId;
                    lStolenRange.endSubtask = mCurrentSubtaskRangeStats->endSubtaskId;
                    lStolenRange.originalAllottee = lLocalDevice;
                
                    mSecondaryAllotteeMap[lPair].push_back(pRequestingDevice);
                    
                #ifdef TRACK_MULTI_ASSIGN
                    std::cout << "Multiassign of subtask range [" << mCurrentSubtaskRangeStats->startSubtaskId << " - " << mCurrentSubtaskRangeStats->endSubtaskId << "] from range [" << mCurrentSubtaskRangeStats->parentRangeStartSubtask << " - " << mCurrentSubtaskRangeStats->endSubtaskId << "+] - Device " << pRequestingDevice->GetGlobalDeviceIndex() << ", Original Allottee - Device " << lLocalDevice->GetGlobalDeviceIndex() << ", Secondary allottment count - " << mSecondaryAllotteeMap[lPair].size() << std::endl;
                #endif

                    lStealSuccess = true;
                    pmScheduler::GetScheduler()->StealSuccessEvent(pRequestingDevice, lLocalDevice, lStolenRange);
                }
            }
        }
    }

    if(!lStealSuccess)
        pmScheduler::GetScheduler()->StealFailedEvent(pRequestingDevice, lLocalDevice, pTask);
    
    return pmSuccess;
}
    
pmStatus pmExecutionStub::ThreadSwitchCallback(stubEvent& pEvent)
{
	try
	{
		return ProcessEvent(pEvent);
	}
    catch(pmException& e)
    {
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from stub thread");
    }

	return pmSuccess;
}

pmStatus pmExecutionStub::ProcessEvent(stubEvent& pEvent)
{
	switch(pEvent.eventId)
	{
		case THREAD_BIND:
		{
			BindToProcessingElement();
            TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_EXEC_STUB, this);

        #ifdef SUPPORT_CUDA
            #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                pmStubCUDA* lStub = dynamic_cast<pmStubCUDA*>(this);
                if(lStub)
                    lStub->ReservePinnedMemory(pEvent.bindDetails.physicalMemory, pEvent.bindDetails.totalStubCount);
            #endif
        #endif

			break;
		}
        
        #ifdef DUMP_EVENT_TIMELINE
        case INIT_EVENT_TIMELINE:
        {
            mEventTimelineAutoPtr.reset(new pmEventTimeline(GetEventTimelineName()));
        
            break;
        }
        #endif
        
		case SUBTASK_EXEC:	/* Comes from scheduler thread */
		{
			pmSubtaskRange& lRange = pEvent.execDetails.range;
			pmSubtaskRange lCurrentRange(lRange);
			if(pEvent.execDetails.rangeExecutedOnce)
				lCurrentRange.startSubtask = pEvent.execDetails.lastExecutedSubtaskId + 1;

        #ifdef _DEBUG
            if(lCurrentRange.startSubtask > lCurrentRange.endSubtask)
                PMTHROW(pmFatalErrorException());
        #endif

            bool lPrematureTermination = false;
            bool lReassigned = false;
            bool lForceAckFlag = false;
            
			pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(lRange.task->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);
            lCommand->MarkExecutionStart();

            pmStatus lExecStatus = pmStatusUnavailable;
            bool lIsMultiAssignRange = (lCurrentRange.task->IsMultiAssignEnabled() && lCurrentRange.originalAllottee != NULL);

        #ifdef DUMP_EVENT_TIMELINE
            // Timeline Scope
            {
                pmSubtaskRangeExecutionTimelineAutoPtr lRangeExecTimelineAutoPtr(lRange.task, mEventTimelineAutoPtr.get(), lCurrentRange.startSubtask, lCurrentRange.endSubtask);
                lCurrentRange.endSubtask = pmExecutionStub::ExecuteWrapper(lCurrentRange, pEvent, lIsMultiAssignRange, lRangeExecTimelineAutoPtr, lReassigned, lForceAckFlag, lPrematureTermination, lExecStatus);
                
                if(!lPrematureTermination && !lReassigned && !lIsMultiAssignRange)
                    lRangeExecTimelineAutoPtr.SetGracefulCompletion();
            }
        #else
            lCurrentRange.endSubtask = pmExecutionStub::ExecuteWrapper(lCurrentRange, pEvent, lIsMultiAssignRange, lReassigned, lForceAckFlag, lPrematureTermination, lExecStatus);
        #endif

            lCommand->MarkExecutionEnd(lExecStatus, std::tr1::static_pointer_cast<pmCommand>(lCommand));

            if(lPrematureTermination)
            {
            #ifdef TRACK_SUBTASK_EXECUTION_VERBOSE
                std::cout << "[Host " << pmGetHostId() << "]: Prematurely terminated subtask range [" << lCurrentRange.startSubtask << " - " << lCurrentRange.endSubtask << "]" << std::endl;
            #endif
            
                if(lForceAckFlag && lCurrentRange.endSubtask != 0)
                {
                    lRange.endSubtask = lCurrentRange.endSubtask - 1;
                
                    if(lRange.endSubtask >= lRange.startSubtask)
                        HandleRangeExecutionCompletion(lRange, pmSuccess);
                }
            }
            else
            {
                if(lReassigned) // A secondary allottee has finished and negotiated this subtask range and added a POST_HANDLE_EXEC_COMPLETION for the rest of the range
                    break;
                
            #ifdef TRACK_SUBTASK_EXECUTION_VERBOSE
                std::cout << "[Host " << pmGetHostId() << "]: Executed subtask range [" << lCurrentRange.startSubtask << " - " << lCurrentRange.endSubtask << "]" << std::endl;
            #endif

                ulong lCompletedCount = lCurrentRange.endSubtask - lCurrentRange.startSubtask + 1;
                lRange.task->GetTaskExecStats().RecordStubExecutionStats(this, lCompletedCount, lCommand->GetExecutionTimeInSecs());
            
                if(lRange.originalAllottee == NULL)
                    CommonPostNegotiationOnCPU(lRange.task, lCurrentRange.endSubtask, false);
            
                if(lForceAckFlag)
                    lRange.endSubtask = lCurrentRange.endSubtask;
            
                if(lCurrentRange.endSubtask == lRange.endSubtask)
                    HandleRangeExecutionCompletion(lRange, lCommand->GetStatus());
            }
        
			break;
		}

		case SUBTASK_REDUCE:
		{
			DoSubtaskReduction(pEvent.reduceDetails.task, pEvent.reduceDetails.subtaskId1, pEvent.reduceDetails.stub2, pEvent.reduceDetails.subtaskId2);

			break;
		}

        case NEGOTIATED_RANGE:
        {
        #ifdef _DEBUG
            if(pEvent.negotiatedRangeDetails.range.originalAllottee == NULL || pEvent.negotiatedRangeDetails.range.originalAllottee == GetProcessingElement())
                PMTHROW(pmFatalErrorException());
        #endif
        
            pmSubtaskRange& lRange = pEvent.negotiatedRangeDetails.range;
            for(ulong subtaskId = lRange.startSubtask; subtaskId <= lRange.endSubtask; ++subtaskId)
            {
            #ifdef DUMP_EVENT_TIMELINE
                mEventTimelineAutoPtr->RenameEvent(pmSubtaskRangeExecutionTimelineAutoPtr::GetCancelledEventName(subtaskId, lRange.task), pmSubtaskRangeExecutionTimelineAutoPtr::GetEventName(subtaskId, lRange.task));
            #endif

                CommonPostNegotiationOnCPU(lRange.task, subtaskId, true);
            }
        
            if(lRange.task->GetSchedulingModel() == scheduler::PULL)
            {
            #ifdef _DEBUG
                if(lRange.startSubtask != lRange.endSubtask)
                    PMTHROW(pmFatalErrorException());
            #endif
            
            #ifdef TRACK_MULTI_ASSIGN
                std::cout << "Multi assign partition [" << lRange.startSubtask << " - " << lRange.endSubtask << "] completed by secondary allottee - Device " << GetProcessingElement()->GetGlobalDeviceIndex() << ", Original Allottee: Device " << pEvent.negotiatedRangeDetails.range.originalAllottee->GetGlobalDeviceIndex() << std::endl;
            #endif
            }

            CommitRange(lRange, pmSuccess);
        
            break;
        }

		case FREE_GPU_RESOURCES:
		{
	#ifdef SUPPORT_CUDA
			((pmStubGPU*)this)->FreeExecutionResources();
	#endif
			break;
		}
        
        case POST_HANDLE_EXEC_COMPLETION:
        {
            HandleRangeExecutionCompletion(pEvent.execCompletionDetails.range, pEvent.execCompletionDetails.execStatus);
        
            break;
        }
        
        case DEFERRED_SHADOW_MEM_COMMITS:
        {
            pmTask* lTask = pEvent.deferredShadowMemCommitsDetails.task;
            std::vector<ulong>::iterator lIter = mDeferredShadowMemCommits[lTask].begin(), lEnd = mDeferredShadowMemCommits[lTask].end();
            for(; lIter != lEnd; ++lIter)
                CommitSubtaskShadowMem(lTask, *lIter);
        
            mDeferredShadowMemCommits.erase(lTask);
        
            lTask->RegisterStubShadowMemCommitMessage();
            
            break;
        }
        
        case REDUCTION_FINISH:
        {
            pmTask* lTask = pEvent.reductionFinishDetails.task;

            lTask->GetReducer()->HandleReductionFinish();
            
            break;
        }
            
        case PROCESS_REDISTRIBUTION_BUCKET:
        {
            pmTask* lTask = pEvent.processRedistributionBucketDetails.task;
            
            lTask->GetRedistributor()->ProcessRedistributionBucket(pEvent.processRedistributionBucketDetails.bucketIndex);
            
            break;
        }
            
        case FREE_TASK_RESOURCES:
        {
    #ifdef SUPPORT_CUDA
            ((pmStubCUDA*)this)->FreeTaskResources(pEvent.freeTaskResourcesDetails.taskOriginatingHost, pEvent.freeTaskResourcesDetails.taskSequenceNumber);
    #endif
            
            break;
        }
	}

	return pmSuccess;
}
    
void pmExecutionStub::HandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus)
{
    if(pRange.task->IsMultiAssignEnabled())
    {
        if(pRange.originalAllottee != NULL)
        {
            // All secondary allottees must get go ahead from original allottee
            pmScheduler::GetScheduler()->NegotiateSubtaskRangeWithOriginalAllottee(GetProcessingElement(), pRange);
            return;
        }
        else if(pRange.task->GetSchedulingModel() == scheduler::PULL)
        {
            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

            std::pair<pmTask*, ulong> lPair(pRange.task, pRange.endSubtask);
            if(mSecondaryAllotteeMap.find(lPair) != mSecondaryAllotteeMap.end())
            {
                std::vector<pmProcessingElement*>& lSecondaryAllottees = mSecondaryAllotteeMap[lPair];
        
                std::vector<pmProcessingElement*>::iterator lIter = lSecondaryAllottees.begin(), lEnd = lSecondaryAllottees.end();
                for(; lIter != lEnd; ++lIter)
                    pmScheduler::GetScheduler()->SendSubtaskRangeCancellationMessage(*lIter, pRange);

            #ifdef TRACK_MULTI_ASSIGN
                std::cout << "Multi assign partition [" << pRange.startSubtask << " - " << pRange.endSubtask << "] completed by original allottee - Device " << GetProcessingElement()->GetGlobalDeviceIndex() << std::endl;
            #endif

                mSecondaryAllotteeMap.erase(lPair);
            }
        }
    }

    CommitRange(pRange, pExecStatus);
}

void pmExecutionStub::CommitRange(pmSubtaskRange& pRange, pmStatus pExecStatus)
{
    std::map<size_t, size_t> lOwnershipMap;
    pmMemSection* lMemSection = pRange.task->GetMemSectionRW();    

	if(lMemSection && !(pRange.task->GetCallbackUnit()->GetDataReductionCB() || pRange.task->GetCallbackUnit()->GetDataRedistributionCB()))
    {
        subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
        pmSubscriptionManager& lSubscriptionManager = pRange.task->GetSubscriptionManager();
    
        for(ulong subtaskId = pRange.startSubtask; subtaskId <= pRange.endSubtask; ++subtaskId)
        {
            if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, subtaskId, false, lBeginIter, lEndIter))
            {
                for(lIter = lBeginIter; lIter != lEndIter; ++lIter)
                    lOwnershipMap[lIter->first] = lIter->second.first;
            }
        }
    }

    pmScheduler::GetScheduler()->SendAcknowledgement(GetProcessingElement(), pRange, pExecStatus, lOwnershipMap);
}
    
void pmExecutionStub::CommitSubtaskShadowMem(pmTask* pTask, ulong pSubtaskId)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::SHADOW_MEM_COMMIT);
#endif

    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    pmSubscriptionInfo lUnifiedSubscriptionInfo;
    if(!lSubscriptionManager.GetUnifiedOutputMemSubscriptionForSubtask(this, pSubtaskId, lUnifiedSubscriptionInfo))
        PMTHROW(pmFatalErrorException());
    
    subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;

    if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, pSubtaskId, false, lBeginIter, lEndIter))
        lSubscriptionManager.CommitSubtaskShadowMem(this, pSubtaskId, lBeginIter, lEndIter, lUnifiedSubscriptionInfo.offset);
}

void pmExecutionStub::DeferShadowMemCommit(pmTask* pTask, ulong pSubtaskId)
{
    mDeferredShadowMemCommits[pTask].push_back(pSubtaskId);
}

// This method must be called with mCurrentSubtaskRangeLock acquired
void pmExecutionStub::CancelCurrentlyExecutingSubtaskRange(bool pTaskListeningOnCancellation)
{
#ifdef _DEBUG
    if(!mCurrentSubtaskRangeStats)
        PMTHROW(pmFatalErrorException());
#endif
    
    if(!mCurrentSubtaskRangeStats->taskListeningOnCancellation && pTaskListeningOnCancellation)
        mCurrentSubtaskRangeStats->task->RecordStubWillSendCancellationMessage();

    mCurrentSubtaskRangeStats->taskListeningOnCancellation |= pTaskListeningOnCancellation;
    
    if(!mCurrentSubtaskRangeStats->prematureTermination)
    {
        mCurrentSubtaskRangeStats->prematureTermination = true;
        
        if(!mExecutingLibraryCode)
        {
            TerminateUserModeExecution();
        }
        else
        {
            if(mCurrentSubtaskRangeStats->accumulatorCommandPtr)
               (*mCurrentSubtaskRangeStats->accumulatorCommandPtr)->ForceComplete(*mCurrentSubtaskRangeStats->accumulatorCommandPtr);
        }
    }
}

void pmExecutionStub::MarkInsideLibraryCode()
{
    mExecutingLibraryCode = 1;

    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

    CheckTermination();
}

void pmExecutionStub::MarkInsideUserCode()
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

        CheckTermination();
    }
    
    mExecutingLibraryCode = 0;
}
    
void pmExecutionStub::SetupJmpBuf(sigjmp_buf* pJmpBuf)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

        if(mCurrentSubtaskRangeStats->jmpBuf)
            PMTHROW(pmFatalErrorException());
        
        mCurrentSubtaskRangeStats->jmpBuf = pJmpBuf;

        CheckTermination();
    }
    
    mExecutingLibraryCode = 0;
}
    
void pmExecutionStub::UnsetupJmpBuf(bool pHasJumped)
{
    mExecutingLibraryCode = 1;

    if(pHasJumped)
    {
        if(!mCurrentSubtaskRangeStats->jmpBuf)
            PMTHROW(pmFatalErrorException());
        
        mCurrentSubtaskRangeStats->jmpBuf = NULL;
    }
    else
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

        if(!mCurrentSubtaskRangeStats->jmpBuf)
            PMTHROW(pmFatalErrorException());
        
        mCurrentSubtaskRangeStats->jmpBuf = NULL;
    }
}

// This method must be called with mCurrentSubtaskRangeLock acquired
void pmExecutionStub::CheckTermination()
{
    if(!mCurrentSubtaskRangeStats)
        return;

    if(mCurrentSubtaskRangeStats->prematureTermination)
        TerminateCurrentSubtaskRange();
}

// This method must be called with mCurrentSubtaskRangeLock acquired
void pmExecutionStub::TerminateCurrentSubtaskRange()
{
#ifdef _DEBUG
    if(!mCurrentSubtaskRangeStats->jmpBuf)
        PMTHROW(pmFatalErrorException());
#endif

    mExecutingLibraryCode = 1;
    siglongjmp(*(mCurrentSubtaskRangeStats->jmpBuf), 1);
}

bool pmExecutionStub::RequiresPrematureExit()
{
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!mCurrentSubtaskRangeStats)
        PMTHROW(pmFatalErrorException());
#endif

    return mCurrentSubtaskRangeStats->prematureTermination;
}

bool pmExecutionStub::IsHighPriorityEventWaiting(ushort pPriority)
{
	return GetPriorityQueue().IsHighPriorityElementPresent(pPriority);
}

#ifdef DUMP_EVENT_TIMELINE
ulong pmExecutionStub::ExecuteWrapper(const pmSubtaskRange& pCurrentRange, execStub::stubEvent& pEvent, bool pIsMultiAssign, pmSubtaskRangeExecutionTimelineAutoPtr& pRangeExecTimelineAutoPtr, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmStatus& pStatus)
#else
ulong pmExecutionStub::ExecuteWrapper(const pmSubtaskRange& pCurrentRange, execStub::stubEvent& pEvent, bool pIsMultiAssign, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmStatus& pStatus)
#endif
{
    ulong lStartSubtask = pCurrentRange.startSubtask;
    ulong lEndSubtask = std::numeric_limits<ulong>::infinity();

    pmSubtaskRange& lParentRange = pEvent.execDetails.range;

    currentSubtaskRangeTerminus lTerminus(pReassigned, pForceAckFlag, pPrematureTermination, this);

    guarded_scoped_ptr<RESOURCE_LOCK_IMPLEMENTATION_CLASS, currentSubtaskRangeTerminus, currentSubtaskRangeStats> lScopedPtr(&mCurrentSubtaskRangeLock, &lTerminus, &mCurrentSubtaskRangeStats, new currentSubtaskRangeStats(pCurrentRange.task, pCurrentRange.startSubtask, pCurrentRange.endSubtask, !pIsMultiAssign, lParentRange.startSubtask, NULL, pmBase::GetCurrentTimeInSecs()));
    
    bool lSuccess = true;
    
    try
    {
        lEndSubtask = FindCollectivelyExecutableSubtaskRangeEnd(pCurrentRange, pIsMultiAssign);

        if(lEndSubtask < pCurrentRange.startSubtask || lEndSubtask > pCurrentRange.endSubtask)
            PMTHROW(pmFatalErrorException());
        
        if(lEndSubtask != pCurrentRange.endSubtask)
        {
        #ifdef DUMP_EVENT_TIMELINE
            pRangeExecTimelineAutoPtr.ResetEndSubtask(lEndSubtask);
        #endif
            
            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
            mCurrentSubtaskRangeStats->ResetEndSubtaskId(lEndSubtask);
        }

        pEvent.execDetails.rangeExecutedOnce = true;
        pEvent.execDetails.lastExecutedSubtaskId = lEndSubtask;
        if(lEndSubtask != lParentRange.endSubtask)
            SwitchThread(pEvent, lParentRange.task->GetPriority());

        ulong* lPrefetchSubtaskIdPtr = NULL;

    #ifdef SUPPORT_COMPUTE_COMMUNICATION_OVERLAP
        ulong lPrefetchSubtaskId = std::numeric_limits<ulong>::infinity();
        if(pCurrentRange.task->ShouldOverlapComputeCommunication() && lEndSubtask != lParentRange.endSubtask)
        {
            lPrefetchSubtaskIdPtr = &lPrefetchSubtaskId;
            lPrefetchSubtaskId = lEndSubtask + 1;
        }
    #endif

        UnblockSecondaryCommands(); // Allow external operations (steal & range negotiation) on priority queue

        PrepareForSubtaskRangeExecution(pCurrentRange.task, lStartSubtask, lEndSubtask);

        for(ulong lSubtaskId = lStartSubtask; lSubtaskId <= lEndSubtask; ++lSubtaskId)
        {
            TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, &lSubtaskId);
            
        #ifdef DUMP_EVENT_TIMELINE
            pRangeExecTimelineAutoPtr.InitializeNextSubtask();
        #endif
            
            pStatus = Execute(pCurrentRange.task, lSubtaskId, pIsMultiAssign, lPrefetchSubtaskIdPtr);
            if(pStatus != pmSuccess)
                break;
            
            TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, NULL);
        }
    }
    catch(pmPrematureExitException& e)
    {
        lSuccess = false;

        if(e.IsSubtaskLockAcquired())
            lScopedPtr.SetLockAcquired();

        pmSubscriptionManager& lSubscriptionManager = pCurrentRange.task->GetSubscriptionManager();
        lSubscriptionManager.DestroySubtaskRangeShadowMem(this, lStartSubtask, lEndSubtask);
    
    #if defined(LINUX) || defined(MACOS)
        ((pmLinuxMemoryManager*)MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager())->InstallSegFaultHandler();
    #endif
    }
    catch(...)
    {
        lSuccess = false;

        CleanupPostSubtaskRangeExecution(pCurrentRange.task, pIsMultiAssign, lStartSubtask, lEndSubtask, lSuccess);
        throw;
    }

    CleanupPostSubtaskRangeExecution(pCurrentRange.task, pIsMultiAssign, lStartSubtask, lEndSubtask, lSuccess);

    return lEndSubtask;
}

pmStatus pmExecutionStub::FindSubtaskMemDependencies(pmTask* pTask, ulong pSubtaskId)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    if(!lSubscriptionManager.IsSubtaskInitialized(this, pSubtaskId))
    {
    #ifdef ENABLE_TASK_PROFILING
        pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr1(pTask->GetTaskProfiler(), taskProfiler::PRE_SUBTASK_EXECUTION);
    #endif
        
        lSubscriptionManager.InitializeSubtaskDefaults(this, pSubtaskId);
    }
    
    if(!lSubscriptionManager.IsSubtaskReadyForExecution(this, pSubtaskId))
    {
        INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataDistributionCB, pTask->GetCallbackUnit()->GetDataDistributionCB(), Invoke, this, pTask, pSubtaskId);
        
        #ifdef ENABLE_TASK_PROFILING
            pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr2(pTask->GetTaskProfiler(), taskProfiler::PRE_SUBTASK_EXECUTION);
        #endif

        lSubscriptionManager.FreezeSubtaskSubscriptions(this, pSubtaskId);
        lSubscriptionManager.MarkSubtaskReadyForExecution(this, pSubtaskId);
    }
    
    return pmSuccess;
}
    
pmStatus pmExecutionStub::CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, bool pPrefetch)
{
    FindSubtaskMemDependencies(pTask, pSubtaskId);
    
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    lSubscriptionManager.FetchSubtaskSubscriptions(this, pSubtaskId, GetType(), pPrefetch);
    
    if(!pPrefetch)
    {
        if(pTask->DoSubtasksNeedShadowMemory() || (pTask->IsMultiAssignEnabled() && pIsMultiAssign))
            lSubscriptionManager.CreateSubtaskShadowMem(this, pSubtaskId);
    }

	return pmSuccess;
}

pmStatus pmExecutionStub::CommonPostNegotiationOnCPU(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign)
{
	pmCallbackUnit* lCallbackUnit = pTask->GetCallbackUnit();
	pmDataReductionCB* lReduceCallback = lCallbackUnit->GetDataReductionCB();
    
	if(lReduceCallback)
		pTask->GetReducer()->AddSubtask(this, pSubtaskId);
	else
		INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataRedistributionCB, pTask->GetCallbackUnit()->GetDataRedistributionCB(), Invoke, this, pTask, pSubtaskId, pIsMultiAssign);
    
    if(!lReduceCallback && (pTask->DoSubtasksNeedShadowMemory() || (pTask->IsMultiAssignEnabled() && pIsMultiAssign)))
    {
        if(pTask->GetMemSectionRW()->IsReadWrite() && !pTask->HasSameReadWriteSubscription())
            DeferShadowMemCommit(pTask, pSubtaskId);
        else
            CommitSubtaskShadowMem(pTask, pSubtaskId);
    }
    
    return pmSuccess;
}

pmStatus pmExecutionStub::DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2)
{
    TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, &pSubtaskId1);
	pmStatus lStatus = pTask->GetCallbackUnit()->GetDataReductionCB()->Invoke(pTask, this, pSubtaskId1, true, pStub2, pSubtaskId2, true);
    TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, NULL);

	/* Handle Transactions */
	switch(lStatus)
	{
		case pmSuccess:
		{
			pTask->GetSubscriptionManager().DestroySubtaskShadowMem(pStub2, pSubtaskId2);
			pTask->GetReducer()->AddSubtask(this, pSubtaskId1);

			break;
		}

		default:
		{
			pTask->GetSubscriptionManager().DestroySubtaskShadowMem(this, pSubtaskId1);
			pTask->GetSubscriptionManager().DestroySubtaskShadowMem(pStub2, pSubtaskId2);
		}
	}

	return lStatus;
}

void pmExecutionStub::WaitForNetworkFetch(std::vector<pmCommunicatorCommandPtr>& pNetworkCommands)
{
    pmAccumulatorCommandPtr lAccumulatorCommand;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskRangeLock, Lock(), Unlock());
        if(!mCurrentSubtaskRangeStats || mCurrentSubtaskRangeStats->accumulatorCommandPtr != NULL)
            PMTHROW(pmFatalErrorException());

        lAccumulatorCommand = pmAccumulatorCommand::CreateSharedPtr(pNetworkCommands);
        mCurrentSubtaskRangeStats->accumulatorCommandPtr = &lAccumulatorCommand;
    }
    
    guarded_ptr<RESOURCE_LOCK_IMPLEMENTATION_CLASS, pmAccumulatorCommandPtr> lGuardedPtr(&mCurrentSubtaskRangeLock, &(mCurrentSubtaskRangeStats->accumulatorCommandPtr), &lAccumulatorCommand);

    pmStatus lStatus = lAccumulatorCommand->WaitForFinish();
    
    if(RequiresPrematureExit())
        PMTHROW_NODUMP(pmPrematureExitException(false));

    if(lStatus != pmSuccess)
        PMTHROW(pmMemoryFetchException());
}


/* struct currentSubtaskStats */
pmExecutionStub::currentSubtaskRangeStats::currentSubtaskRangeStats(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId, bool pOriginalAllottee, ulong pParentRangeStartSubtask, sigjmp_buf* pJmpBuf, double pStartTime)
    : task(pTask)
    , startSubtaskId(pStartSubtaskId)
    , endSubtaskId(pEndSubtaskId)
    , parentRangeStartSubtask(pParentRangeStartSubtask)
    , originalAllottee(pOriginalAllottee)
    , startTime(pStartTime)
    , reassigned(false)
    , forceAckFlag(false)
    , prematureTermination(false)
    , taskListeningOnCancellation(false)
    , jmpBuf(pJmpBuf)
    , accumulatorCommandPtr(NULL)
{
}

void pmExecutionStub::currentSubtaskRangeStats::ResetEndSubtaskId(ulong pEndSubtaskId)
{
    endSubtaskId = pEndSubtaskId;
}

/* class pmStubCPU */
pmStubCPU::pmStubCPU(size_t pCoreId, uint pDeviceIndexOnMachine)
	: pmExecutionStub(pDeviceIndexOnMachine)
{
	mCoreId = pCoreId;
}

pmStubCPU::~pmStubCPU()
{
}

pmStatus pmStubCPU::BindToProcessingElement()
{
	 return SetProcessorAffinity((int)mCoreId);
}

size_t pmStubCPU::GetCoreId()
{
	return mCoreId;
}

std::string pmStubCPU::GetDeviceName()
{
	// Try to use processor_info or getcpuid system calls
	return std::string();
}

std::string pmStubCPU::GetDeviceDescription()
{
	// Try to use processor_info or getcpuid system calls
	return std::string();
}

pmDeviceType pmStubCPU::GetType()
{
	return CPU;
}
    
ulong pmStubCPU::FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, bool pMultiAssign)
{
    return pSubtaskRange.startSubtask;
}

pmStatus pmStubCPU::Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr)
{
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId, pIsMultiAssign, false));
    
    if(pPreftechSubtaskIdPtr)
        PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, *pPreftechSubtaskIdPtr, pIsMultiAssign, true));
    
    pmSubtaskInfo lSubtaskInfo;
    bool lOutputMemWriteOnly = false;
    pTask->GetSubtaskInfo(this, pSubtaskId, pIsMultiAssign, lSubtaskInfo, lOutputMemWriteOnly);
    
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, this, pTask, pSubtaskId, pIsMultiAssign, pTask->GetTaskInfo(), lSubtaskInfo, lOutputMemWriteOnly);
	
	return pmSuccess;
}

void pmStubCPU::PrepareForSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId)
{
    if(pStartSubtaskId != pEndSubtaskId)
        PMTHROW(pmFatalErrorException());
}
    
void pmStubCPU::CleanupPostSubtaskRangeExecution(pmTask* pTask, bool pIsMultiAssign, ulong pStartSubtaskId, ulong pEndSubtaskId, bool pSuccess)
{
}
    
// This method must be called with mCurrentSubtaskRangeLock (of pmExecutionStub) acquired
void pmStubCPU::TerminateUserModeExecution()
{
    InterruptThread();
}


/* class pmStubGPU */
pmStubGPU::pmStubGPU(uint pDeviceIndexOnMachine)
	: pmExecutionStub(pDeviceIndexOnMachine)
{
}

pmStubGPU::~pmStubGPU()
{
}

#ifdef SUPPORT_CUDA
/* class pmStubCUDA */
pmStubCUDA::pmStubCUDA(size_t pDeviceIndex, uint pDeviceIndexOnMachine)
	: pmStubGPU(pDeviceIndexOnMachine)
    , mDeviceIndex(pDeviceIndex)
    , mDeviceInfoCudaPtr(NULL)
    , mTotalAllocationSize(0)
    , mCudaAllocation(NULL)
    , mMemElements(5)    // Input Mem, Output Mem, Scratch Mem, Status
    , mPendingStreams(0)
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    , mPinnedBuffer(NULL)
    , mPinnedAllocation(NULL)
#else
    , mStatusCopySrc(pmStatusUnavailable)
    , mStatusCopyDest(pmStatusUnavailable)
#endif
{
}

pmStubCUDA::~pmStubCUDA()
{
}

pmStatus pmStubCUDA::FreeResources()
{
    FreeGpuResources();
    return pmSuccess;
}

pmStatus pmStubCUDA::FreeExecutionResources()
{
    if(mCudaAllocation)
        PMTHROW(pmFatalErrorException());
        
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    if(mPinnedAllocation)
        PMTHROW(pmFatalErrorException());
        
    if(mPinnedBuffer)
        pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->DeallocatePinnedBuffer(mPinnedBuffer);
#endif

    if(mDeviceInfoCudaPtr)
        pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->DestroyDeviceInfoCudaPtr(mDeviceInfoCudaPtr);

    pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->FreeLastExecutionResources(mLastExecutionRecord);

    if(!mTaskInfoCudaMap.empty())
        PMTHROW(pmFatalErrorException());
    
    return pmSuccess;
}

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
pmMemChunk* pmStubCUDA::GetPinnedBufferChunk()
{
    return mPinnedBufferChunk.get();
}
#endif
    
void pmStubCUDA::FreeTaskResources(pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    std::pair<pmMachine*, ulong> lPair(pOriginatingHost, pSequenceNumber);

    std::map<std::pair<pmMachine*, ulong>, pmTaskInfo>::iterator lIter = mTaskInfoCudaMap.find(lPair);
    if(lIter != mTaskInfoCudaMap.end())
    {
        if(lIter->second.taskConfLength)
            pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->DestroyTaskConf(lIter->second.taskConf);
    
        mTaskInfoCudaMap.erase(lPair);
    }
}

pmStatus pmStubCUDA::BindToProcessingElement()
{
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->BindToDevice(mDeviceIndex);
}

std::string pmStubCUDA::GetDeviceName()
{
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceName(mDeviceIndex);
}

std::string pmStubCUDA::GetDeviceDescription()
{
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceDescription(mDeviceIndex);
}

pmDeviceType pmStubCUDA::GetType()
{
	return GPU_CUDA;
}
    
ulong pmStubCUDA::FindCollectivelyExecutableSubtaskRangeEnd(const pmSubtaskRange& pSubtaskRange, bool pMultiAssign)
{
    mAllocationOffsets.clear();

    pmSubtaskRange lSubtaskRange(pSubtaskRange);

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    if(!pSubtaskRange.task->ShouldOverlapComputeCommunication())
        lSubtaskRange.endSubtask = lSubtaskRange.startSubtask;
#else
    lSubtaskRange.endSubtask = lSubtaskRange.startSubtask;
    
#endif

    return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->FindCollectivelyExecutableSubtaskRangeEnd(this, lSubtaskRange, pMultiAssign, mAllocationOffsets, mTotalAllocationSize);
}

pmStatus pmStubCUDA::Execute(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong* pPreftechSubtaskIdPtr)
{
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId, pIsMultiAssign, false));

    if(pPreftechSubtaskIdPtr)
        PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, *pPreftechSubtaskIdPtr, pIsMultiAssign, true));

    pmSubtaskInfo lSubtaskInfo;
    bool lOutputMemWriteOnly = false;
    pTask->GetSubtaskInfo(this, pSubtaskId, pIsMultiAssign, lSubtaskInfo, lOutputMemWriteOnly);
    
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    CopyDataToPinnedBuffers(pTask, pSubtaskId, lSubtaskInfo, lOutputMemWriteOnly);
#endif
    
    PopulateMemcpyCommands(pTask, pSubtaskId, lSubtaskInfo, lOutputMemWriteOnly);

    std::pair<pmMachine*, ulong> lPair(pTask->GetOriginatingHost(), pTask->GetSequenceNumber());

    std::map<std::pair<pmMachine*, ulong>, pmTaskInfo>::iterator lIter = mTaskInfoCudaMap.find(lPair);
    if(lIter == mTaskInfoCudaMap.end())
    {
        pmTaskInfo& lTaskInfo = pTask->GetTaskInfo();
        void* lTaskConfCudaPtr = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->CreateTaskConf(lTaskInfo);

        mTaskInfoCudaMap[lPair] = lTaskInfo;
        lIter = mTaskInfoCudaMap.find(lPair);
        
        lIter->second.taskConf = lTaskConfCudaPtr;
    }

	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, this, pTask, pSubtaskId, pIsMultiAssign, lIter->second, lSubtaskInfo, lOutputMemWriteOnly);

	return pmSuccess;
}

void* pmStubCUDA::GetDeviceInfoCudaPtr()
{
    if(!mDeviceInfoCudaPtr)
        mDeviceInfoCudaPtr = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->CreateDeviceInfoCudaPtr(GetProcessingElement()->GetDeviceInfo());
        
    return mDeviceInfoCudaPtr;
}

pmLastCudaExecutionRecord& pmStubCUDA::GetLastExecutionRecord()
{
    return mLastExecutionRecord;
}
    
void pmStubCUDA::PopulateMemcpyCommands(pmTask* pTask, ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo, bool pOutputMemWriteOnly)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    mDeviceToHostCommands.clear();
    mHostToDeviceCommands.clear();
    
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    std::vector<void*>& lPinnedVector = mPinnedPointersMap[pSubtaskId];

    void* lInputPinnedMem = lPinnedVector[0];
    void* lOutputPinnedMem = lPinnedVector[1];
    void* lScratchPinnedMem = lPinnedVector[2];
    void* lStatusPinnedMem = lPinnedVector[4];
#endif
    
    std::vector<void*>& lVector = mCudaPointersMap[pSubtaskId];
    
    void* lInputMem = lVector[0];
    void* lOutputMem = lVector[1];
    void* lScratchMem = lVector[2];
    void* lStatusMem = lVector[4];
    
    if(lInputMem)
    {
        pmSubscriptionInfo lInputMemSubscriptionInfo;
        if(lSubscriptionManager.GetInputMemSubscriptionForSubtask(this, pSubtaskId, lInputMemSubscriptionInfo))
        {
            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedInputMemSubscriptionsForSubtask(this, pSubtaskId, lBegin, lEnd))
            {
                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                    void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputPinnedMem) + lIter->first - lInputMemSubscriptionInfo.offset);
                #else
                    void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.inputMem) + lIter->first - lInputMemSubscriptionInfo.offset);
                #endif
                    
                    void* lDestPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputMem) + lIter->first - lInputMemSubscriptionInfo.offset);

                    mHostToDeviceCommands.push_back(pmCudaMemcpyCommand(lSrcPtr, lDestPtr, lIter->second.first));
                }
            }
        }
    }
    
    if(lOutputMem && !pOutputMemWriteOnly)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo;
        if(lSubscriptionManager.GetUnifiedOutputMemSubscriptionForSubtask(this, pSubtaskId, lUnifiedSubscriptionInfo))
        {
        #ifndef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
            void* lMem = pTask->GetMemSectionRW()->GetMem();
        #endif

            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, pSubtaskId, true, lBegin, lEnd))
            {
                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                    void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputPinnedMem) + lIter->first - lUnifiedSubscriptionInfo.offset);
                #else
                    void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMem) + lIter->first);
                #endif
                    
                    void* lDestPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMem) + lIter->first - lUnifiedSubscriptionInfo.offset);

                    mHostToDeviceCommands.push_back(pmCudaMemcpyCommand(lSrcPtr, lDestPtr, lIter->second.first));
                }
            }
        }
    }

    if(lScratchMem)
    {
        pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
        size_t lScratchBufferSize = 0;
        void* lCpuScratchBuffer = lSubscriptionManager.CheckAndGetScratchBuffer(this, pSubtaskId, lScratchBufferSize, lScratchBufferInfo);
        if(lCpuScratchBuffer && lScratchBufferSize)
        {
            if(lScratchBufferInfo == PRE_SUBTASK_TO_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK)
            {
            #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                mHostToDeviceCommands.push_back(pmCudaMemcpyCommand(lScratchPinnedMem, lScratchMem, lScratchBufferSize));
            #else
                mHostToDeviceCommands.push_back(pmCudaMemcpyCommand(lCpuScratchBuffer, lScratchMem, lScratchBufferSize));
            #endif
            }
        }
    }
    
    if(lStatusMem)
    {
    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        mHostToDeviceCommands.push_back(pmCudaMemcpyCommand(lStatusPinnedMem, lStatusMem, sizeof(pmStatus)));
    #else
        mHostToDeviceCommands.push_back(pmCudaMemcpyCommand((void*)(&mStatusCopySrc), lStatusMem, sizeof(pmStatus)));
    #endif
    }
    
    if(lOutputMem)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo;
        if(lSubscriptionManager.GetUnifiedOutputMemSubscriptionForSubtask(this, pSubtaskId, lUnifiedSubscriptionInfo))
        {
        #ifndef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
            void* lMem = pTask->GetMemSectionRW()->GetMem();
        #endif

            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, pSubtaskId, false, lBegin, lEnd))
            {
                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                    void* lDestPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputPinnedMem) + lIter->first - lUnifiedSubscriptionInfo.offset);
                #else
                    void* lDestPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMem) + lIter->first);
                #endif
                    
                    void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMem) + lIter->first - lUnifiedSubscriptionInfo.offset);

                    mDeviceToHostCommands.push_back(pmCudaMemcpyCommand(lSrcPtr, lDestPtr, lIter->second.first));
                }
            }
        }
    }

    if(lScratchMem)
    {
        pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
        size_t lScratchBufferSize = 0;
        void* lCpuScratchBuffer = lSubscriptionManager.CheckAndGetScratchBuffer(this, pSubtaskId, lScratchBufferSize, lScratchBufferInfo);
        if(lCpuScratchBuffer && lScratchBufferSize)
        {
            if(lScratchBufferInfo == SUBTASK_TO_POST_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK)
            {
            #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
                mDeviceToHostCommands.push_back(pmCudaMemcpyCommand(lScratchMem, lScratchPinnedMem, lScratchBufferSize));
            #else
                mDeviceToHostCommands.push_back(pmCudaMemcpyCommand(lScratchMem, lCpuScratchBuffer, lScratchBufferSize));
            #endif
            }
        }
    }
    
    if(lStatusMem)
    {
    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        mDeviceToHostCommands.push_back(pmCudaMemcpyCommand(lStatusMem, lStatusPinnedMem, sizeof(pmStatus)));
    #else
        mDeviceToHostCommands.push_back(pmCudaMemcpyCommand(lStatusMem, (void*)(&mStatusCopyDest), sizeof(pmStatus)));
    #endif
    }
}
    
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
void pmStubCUDA::ReservePinnedMemory(size_t pPhysicalMemory, size_t pTotalStubCount)
{
    if(mPinnedBuffer)
        PMTHROW(pmFatalErrorException());

    if(pPhysicalMemory && pTotalStubCount)
    {
        // Allocate a pinned buffer of size equal to stub's main mem share or available cuda mem (whichever is less), subject to a maximum of 4 GB
        size_t lCudaMem = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetAvailableCudaMem();
        size_t lBufferSize = std::min((size_t)4 * 1024 * 1024 * 1024, std::min(lCudaMem, pPhysicalMemory / pTotalStubCount));
        mPinnedBuffer = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->AllocatePinnedBuffer(lBufferSize);

        if(!mPinnedBuffer)
            PMTHROW(pmFatalErrorException());
        
        mPinnedBufferChunk.reset(new pmMemChunk(mPinnedBuffer, lBufferSize));
    }
}

void pmStubCUDA::CopyDataToPinnedBuffers(pmTask* pTask, ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo, bool pOutputMemWriteOnly)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    std::vector<void*>& lVector = mPinnedPointersMap[pSubtaskId];
    
    void* lInputMem = lVector[0];
    void* lOutputMem = lVector[1];
    void* lScratchMem = lVector[2];
    void* lStatusMem = lVector[4];

    if(lInputMem)
    {
        pmSubscriptionInfo lInputMemSubscriptionInfo;
        if(lSubscriptionManager.GetInputMemSubscriptionForSubtask(this, pSubtaskId, lInputMemSubscriptionInfo))
        {
            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedInputMemSubscriptionsForSubtask(this, pSubtaskId, lBegin, lEnd))
            {
                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                    void* lPinnedPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputMem) + lIter->first - lInputMemSubscriptionInfo.offset);
                    void* lDataPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.inputMem) + lIter->first - lInputMemSubscriptionInfo.offset);

                    memcpy(lPinnedPtr, lDataPtr, lIter->second.first);
                }
            }
        }
    }
    
    if(lOutputMem && !pOutputMemWriteOnly)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo;
        if(lSubscriptionManager.GetUnifiedOutputMemSubscriptionForSubtask(this, pSubtaskId, lUnifiedSubscriptionInfo))
        {
            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, pSubtaskId, true, lBegin, lEnd))
            {
                void* lMem = pTask->GetMemSectionRW()->GetMem();

                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                    void* lPinnedPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMem) + lIter->first - lUnifiedSubscriptionInfo.offset);
                    void* lDataPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMem) + lIter->first);

                    memcpy(lPinnedPtr, lDataPtr, lIter->second.first);
                }
            }
        }
    }

    if(lScratchMem)
    {
        pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
        size_t lScratchBufferSize = 0;
        void* lCpuScratchBuffer = lSubscriptionManager.CheckAndGetScratchBuffer(this, pSubtaskId, lScratchBufferSize, lScratchBufferInfo);
        if(lCpuScratchBuffer && lScratchBufferSize)
        {
            if(lScratchBufferInfo == PRE_SUBTASK_TO_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK)
                memcpy(lScratchMem, lCpuScratchBuffer, lScratchBufferSize);
        }
    }
    
    if(lStatusMem)
        *((pmStatus*)lStatusMem) = pmStatusUnavailable;
}
    
pmStatus pmStubCUDA::CopyDataFromPinnedBuffers(pmTask* pTask, ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();
    
    std::vector<void*>& lVector = mPinnedPointersMap[pSubtaskId];
    
    void* lOutputMem = lVector[1];
    void* lScratchMem = lVector[2];
    void* lStatusMem = lVector[4];
    
    if(lOutputMem)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo;
        if(lSubscriptionManager.GetUnifiedOutputMemSubscriptionForSubtask(this, pSubtaskId, lUnifiedSubscriptionInfo))
        {
            subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;
            if(lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, pSubtaskId, false, lBegin, lEnd))
            {
                for(lIter = lBegin; lIter != lEnd; ++lIter)
                {
                    void* lPinnedPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMem) + lIter->first - lUnifiedSubscriptionInfo.offset);
                    void* lDataPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.outputMem) + lIter->first - lUnifiedSubscriptionInfo.offset);

                    memcpy(lDataPtr, lPinnedPtr, lIter->second.first);
                }
            }
        }
    }

    if(lScratchMem)
    {
        pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
        size_t lScratchBufferSize = 0;
        void* lCpuScratchBuffer = lSubscriptionManager.CheckAndGetScratchBuffer(this, pSubtaskId, lScratchBufferSize, lScratchBufferInfo);
        if(lCpuScratchBuffer && lScratchBufferSize)
        {
            if(lScratchBufferInfo == SUBTASK_TO_POST_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK)
                memcpy(lCpuScratchBuffer, lScratchMem, lScratchBufferSize);
        }
    }
    
    if(lStatusMem)
        return *((pmStatus*)lStatusMem);
    
    return pmStatusUnavailable;
}
    
#endif

void pmStubCUDA::PrepareForSubtaskRangeExecution(pmTask* pTask, ulong pStartSubtaskId, ulong pEndSubtaskId)
{
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    if(pEndSubtaskId - pStartSubtaskId + 1 != mAllocationOffsets.size())
        PMTHROW(pmFatalErrorException());

    if(mPinnedAllocation)
        PMTHROW(pmFatalErrorException());
    
    mPinnedAllocation = mPinnedBufferChunk->Allocate(mTotalAllocationSize, pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetLogCudaAlignment());
    
    if(!mPinnedAllocation)
        PMTHROW(pmFatalErrorException());
    
    size_t lPinnedBaseAddr = reinterpret_cast<size_t>(mPinnedAllocation);
    mPinnedPointersMap.clear();
#else
    if(pStartSubtaskId != pEndSubtaskId || mAllocationOffsets.size() != 1)
        PMTHROW(pmFatalErrorException());
#endif
    
    if(mCudaAllocation)
        PMTHROW(pmFatalErrorException());

    mCudaAllocation = pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->AllocateCudaMem(mTotalAllocationSize);

    if(!mCudaAllocation)
        PMTHROW(pmFatalErrorException());
    
    size_t lCudaBaseAddr = reinterpret_cast<size_t>(mCudaAllocation);
    mCudaPointersMap.clear();
    
    std::vector<std::vector<std::pair<size_t, size_t> > >::iterator lIter = mAllocationOffsets.begin(), lEndIter = mAllocationOffsets.end();
    for(size_t subtaskId = pStartSubtaskId; lIter != lEndIter; ++lIter, ++subtaskId)
    {
        std::vector<std::pair<size_t, size_t> >& lVector = *lIter;
        
        if(lVector.size() != mMemElements)
            PMTHROW(pmFatalErrorException());
        
        std::vector<std::pair<size_t, size_t> >::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            std::pair<size_t, size_t>& lPair = *lInnerIter;
            
            mCudaPointersMap[subtaskId].push_back(lPair.second ? reinterpret_cast<void*>(lCudaBaseAddr + lPair.first) : NULL);

        #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
            mPinnedPointersMap[subtaskId].push_back(lPair.second ? reinterpret_cast<void*>(lPinnedBaseAddr + lPair.first) : NULL);
        #endif
        }
    }
    
    mPendingStreams = pEndSubtaskId - pStartSubtaskId + 1;
    
    if(mStreamSignalWait.get())
        PMTHROW(pmFatalErrorException());
    
    mStreamSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());
}

void pmStubCUDA::CleanupPostSubtaskRangeExecution(pmTask* pTask, bool pIsMultiAssign, ulong pStartSubtaskId, ulong pEndSubtaskId, bool pSuccess)
{
    if(pSuccess)
    {
        mStreamSignalWait->Wait();

    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        for(ulong i = pStartSubtaskId; i <= pEndSubtaskId; ++i)
        {
            pmSubtaskInfo lSubtaskInfo;
            bool lOutputMemWriteOnly = false;
            pTask->GetSubtaskInfo(this, i, pIsMultiAssign, lSubtaskInfo, lOutputMemWriteOnly);

            CopyDataFromPinnedBuffers(pTask, i, lSubtaskInfo);
        }
    #endif
    }

    mPendingStreams = 0;

    if(mCudaAllocation)
    {
        pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->DeallocateCudaMem(mCudaAllocation);
        mCudaAllocation = NULL;
    }

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    if(mPinnedAllocation)
    {
        mPinnedBufferChunk->Deallocate(mPinnedAllocation);
        mPinnedAllocation = NULL;
    }
#endif
    
    mStreamSignalWait.reset(NULL);
}
    
void pmStubCUDA::StreamFinishCallback()
{
#ifdef _DEBUG
    if(mPendingStreams == 0)
        PMTHROW(pmFatalErrorException());
#endif
    
    --mPendingStreams;
    
    if(!mPendingStreams)
        mStreamSignalWait->Signal();
}

void pmStubCUDA::TerminateUserModeExecution()
{
}
    
#endif


bool execEventMatchFunc(stubEvent& pEvent, void* pCriterion)
{
	if(pEvent.eventId == SUBTASK_EXEC && pEvent.execDetails.range.task == (pmTask*)pCriterion)
		return true;

	return false;
}


/* struct currentSubtaskRnageTerminus */
pmExecutionStub::currentSubtaskRangeTerminus::currentSubtaskRangeTerminus(bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmExecutionStub* pStub)
    : mReassigned(pReassigned)
    , mForceAckFlag(pForceAckFlag)
    , mPrematureTermination(pPrematureTermination)
    , mStub(pStub)
{
}

void pmExecutionStub::currentSubtaskRangeTerminus::Terminating(currentSubtaskRangeStats* pStats)
{
    mPrematureTermination = pStats->prematureTermination;
    mReassigned = pStats->reassigned;
    mForceAckFlag = pStats->forceAckFlag;

    if(pStats->prematureTermination)
    {
        if(pStats->taskListeningOnCancellation)
            pStats->task->RegisterStubCancellationMessage();
    }

#ifdef _DEBUG
    if(mReassigned && !pStats->originalAllottee)
        PMTHROW(pmFatalErrorException());
#endif
}

#ifdef DUMP_EVENT_TIMELINE
std::string pmExecutionStub::GetEventTimelineName()
{
    std::stringstream lStream;
    lStream << "Device " << GetProcessingElement()->GetGlobalDeviceIndex();
    
    return lStream.str();
}
#endif

/* struct stubEvent */
bool execStub::stubEvent::BlocksSecondaryCommands()
{
    return (eventId == SUBTASK_EXEC);
}
    
}




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
#include "pmReducer.h"
#include "pmScheduler.h"
#include "pmMemSection.h"
#include "pmTaskManager.h"
#include "pmTls.h"

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

namespace pm
{

using namespace execStub;

/* class pmExecutionStub */
pmExecutionStub::pmExecutionStub(uint pDeviceIndexOnMachine)
    : mCurrentSubtaskStats(NULL)
{
	mDeviceIndexOnMachine = pDeviceIndexOnMachine;

	stubEvent lEvent;
	lEvent.eventId = THREAD_BIND;
	threadBind lBindDetails;
	lEvent.bindDetails = lBindDetails;
	SwitchThread(lEvent, MAX_CONTROL_PRIORITY);
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

    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

    if(mCurrentSubtaskStats && mCurrentSubtaskStats->task == pTask)
        CancelCurrentlyExecutingSubtask(pTaskListeningOnCancellation);
    
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

    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

    if(mCurrentSubtaskStats && mCurrentSubtaskStats->task == pRange.task && mCurrentSubtaskStats->subtaskId >= pRange.startSubtask && mCurrentSubtaskStats->subtaskId <= pRange.endSubtask)
        CancelCurrentlyExecutingSubtask(false);
    
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
    
void pmExecutionStub::PostHandleRangeExecutionCompletion(pmSubtaskRange& pRange, pmStatus pExecStatus)
{
    stubEvent lEvent;
    lEvent.execCompletionDetails.range = pRange;
    lEvent.execCompletionDetails.execStatus = pExecStatus;
    lEvent.eventId = POST_HANDLE_EXEC_COMPLETION;
    
    SwitchThread(lEvent, pRange.task->GetPriority() - 1);
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
            if(pRange.startSubtask != pRange.endSubtask)
                PMTHROW(pmFatalErrorException());
        #endif
        
            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());
            
            if(mCurrentSubtaskStats && mCurrentSubtaskStats->task == pRange.task && mCurrentSubtaskStats->subtaskId == pRange.startSubtask && !mCurrentSubtaskStats->reassigned)
            {
            #ifdef _DEBUG
                if(!mCurrentSubtaskStats->originalAllottee)
                    PMTHROW(pmFatalErrorException());
            #endif

                std::pair<pmTask*, ulong> lPair(pRange.task, pRange.startSubtask);            
            
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
                    mCurrentSubtaskStats->reassigned = true;
                    CancelCurrentlyExecutingSubtask(false);
                            
                    if(mCurrentSubtaskStats->parentRangeStartSubtask != mCurrentSubtaskStats->subtaskId)
                    {
                        pmSubtaskRange lCompletedRange;
                        lCompletedRange.task = pRange.task;
                        lCompletedRange.startSubtask = mCurrentSubtaskStats->parentRangeStartSubtask;
                        lCompletedRange.endSubtask = mCurrentSubtaskStats->subtaskId - 1;
                        lCompletedRange.originalAllottee = NULL;
std::cout << "Added post range exec completion for range [" << lCompletedRange.startSubtask << ", " << lCompletedRange.endSubtask << "]" << std::endl;
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
        else
        {
            pmSubtaskRange lNegotiatedRange;
            bool lSuccessfulNegotiation = false;
            bool lCurrentTransferred = false;
        
            bool lFound = (DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, pRange.task, lTaskEvent) == pmSuccess);

            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());
        
            bool lConsiderCurrent = (mCurrentSubtaskStats && mCurrentSubtaskStats->task == pRange.task && mCurrentSubtaskStats->subtaskId >= pRange.startSubtask && mCurrentSubtaskStats->subtaskId <= pRange.endSubtask && !mCurrentSubtaskStats->reassigned);
                
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
                        if(!mCurrentSubtaskStats->originalAllottee || lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskStats->subtaskId)
                                PMTHROW(pmFatalErrorException());
                        #endif
                        
                            lFirstPendingSubtask -= 1;
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
                            if(lConsiderCurrent && lTaskEvent.execDetails.range.endSubtask >= mCurrentSubtaskStats->subtaskId)
                                lCurrentTransferred = false;   // current subtask still with original allottee
                        
                            bool lCurrentSubtaskInRemainingRange = (lConsiderCurrent && !lCurrentTransferred);

                            if(!lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.range.endSubtask == (lTaskEvent.execDetails.lastExecutedSubtaskId - (lCurrentTransferred ? 1 : 0)))  // no pending subtask
                            {
                                if(lTaskEvent.execDetails.rangeExecutedOnce)
                                    PostHandleRangeExecutionCompletion(lTaskEvent.execDetails.range, pmSuccess);
                            }
                            else if(lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // only current subtask pending
                            {
                            #ifdef _DEBUG
                                if(lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskStats->subtaskId)
                                    PMTHROW(pmFatalErrorException());
                            #endif
                            
                                mCurrentSubtaskStats->forceAckFlag = true;  // send acknowledgement of the done range after current subtask finishes
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
                if(!mCurrentSubtaskStats->originalAllottee)
                    PMTHROW(pmFatalErrorException());
            #endif
            
                lSuccessfulNegotiation = true;
                lCurrentTransferred = true;
            
                lNegotiatedRange.task = pRange.task;
                lNegotiatedRange.startSubtask = mCurrentSubtaskStats->subtaskId;
                lNegotiatedRange.endSubtask = mCurrentSubtaskStats->subtaskId;
                lNegotiatedRange.originalAllottee = pRange.originalAllottee;
            
                if(mCurrentSubtaskStats->parentRangeStartSubtask != mCurrentSubtaskStats->subtaskId)
                {
                    pmSubtaskRange lCompletedRange;
                    lCompletedRange.task = pRange.task;
                    lCompletedRange.startSubtask = mCurrentSubtaskStats->parentRangeStartSubtask;
                    lCompletedRange.endSubtask = mCurrentSubtaskStats->subtaskId - 1;
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
                mCurrentSubtaskStats->reassigned = true;
                CancelCurrentlyExecutingSubtask(false);
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
    
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());
    
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
                if(mCurrentSubtaskStats && mCurrentSubtaskStats->task == pTask && lTaskEvent.execDetails.range.startSubtask <= mCurrentSubtaskStats->subtaskId && mCurrentSubtaskStats->subtaskId <= lTaskEvent.execDetails.range.endSubtask)
                {
                #ifdef _DEBUG
                    if(!mCurrentSubtaskStats->originalAllottee || mCurrentSubtaskStats->reassigned)
                        PMTHROW(pmFatalErrorException());
                #endif
                
                    lCurrentSubtaskInRemainingRange = true;
                }
                
                if(!lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // no pending subtask
                {
                    if(lTaskEvent.execDetails.rangeExecutedOnce)
                        PostHandleRangeExecutionCompletion(lTaskEvent.execDetails.range, pmSuccess);
                }
                else if(lCurrentSubtaskInRemainingRange && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask) // only current subtask pending
                {
                #ifdef _DEBUG
                    if(lTaskEvent.execDetails.lastExecutedSubtaskId != mCurrentSubtaskStats->subtaskId)
                        PMTHROW(pmFatalErrorException());
                #endif
                    mCurrentSubtaskStats->forceAckFlag = true;  // send acknowledgement of the done range after current subtask finishes
                }
                else
                {   // pending range does not have current subtask or has more subtasks after the current one
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
    else
    {
        if(pTask->IsMultiAssignEnabled())
        {
            pmSubtaskRange lStolenRange;

            if(mCurrentSubtaskStats && mCurrentSubtaskStats->task == pTask && mCurrentSubtaskStats->originalAllottee && !mCurrentSubtaskStats->reassigned)
            {
                std::pair<pmTask*, ulong> lPair(pTask, mCurrentSubtaskStats->subtaskId);                
                if((mSecondaryAllotteeMap.find(lPair) == mSecondaryAllotteeMap.end())
                   || (mSecondaryAllotteeMap[lPair].size() < MAX_SUBTASK_MULTI_ASSIGN_COUNT - 1))
                {
                    bool lLocalRateZero = (lLocalRate == (double)0.0);
                    double lTransferOverheadTime = 0;   // add network and other obverheads here
                    double lExpectedRemoteTimeToExecute = (1.0/pRequestingDeviceExecutionRate);
                    double lExpectedLocalTimeToFinish = (lLocalRateZero ? 0.0 : ((1.0/lLocalRate) - (pmBase::GetCurrentTimeInSecs() - mCurrentSubtaskStats->startTime)));
                
                    if(lLocalRateZero || (lExpectedRemoteTimeToExecute + lTransferOverheadTime < lExpectedLocalTimeToFinish))
                    {
                        lStolenRange.task = pTask;
                        lStolenRange.startSubtask = mCurrentSubtaskStats->subtaskId;
                        lStolenRange.endSubtask = mCurrentSubtaskStats->subtaskId;
                        lStolenRange.originalAllottee = lLocalDevice;
                    
                        mSecondaryAllotteeMap[lPair].push_back(pRequestingDevice);
                        
                    #ifdef TRACK_MULTI_ASSIGN
                        std::cout << "Multiassign of subtask " << mCurrentSubtaskStats->subtaskId << " from range [" << mCurrentSubtaskStats->parentRangeStartSubtask << " - " << mCurrentSubtaskStats->subtaskId << "+] - Device " << pRequestingDevice->GetGlobalDeviceIndex() << ", Original Allottee - Device " << lLocalDevice->GetGlobalDeviceIndex() << ", Secondary allottment count - " << mSecondaryAllotteeMap[lPair].size() << std::endl;
                    #endif

                        lStealSuccess = true;
                        pmScheduler::GetScheduler()->StealSuccessEvent(pRequestingDevice, lLocalDevice, lStolenRange);
                    }
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
    catch(pmException e)
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

			break;
		}
        
		case SUBTASK_EXEC:	/* Comes from scheduler thread */
		{
            ulong lCompletedCount, lLastExecutedSubtaskId;
			pmSubtaskRange& lRange = pEvent.execDetails.range;
        
            bool lIsMultiAssignRange = (lRange.task->IsMultiAssignEnabled() && lRange.originalAllottee != NULL);
			pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(lRange.task->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);

			pmSubtaskRange lCurrentRange;
			if(pEvent.execDetails.rangeExecutedOnce)
			{
				lCurrentRange.task = lRange.task;
				lCurrentRange.endSubtask = lRange.endSubtask;
				lCurrentRange.startSubtask = pEvent.execDetails.lastExecutedSubtaskId + 1;
                lCurrentRange.originalAllottee = lRange.originalAllottee;
			}
			else
			{
				lCurrentRange = lRange;
			}

        #ifdef _DEBUG
            if(lCurrentRange.startSubtask > lCurrentRange.endSubtask)
                PMTHROW(pmFatalErrorException());
        #endif
               
            lLastExecutedSubtaskId = lCurrentRange.startSubtask;
            lCompletedCount = 1;
        
            pEvent.execDetails.rangeExecutedOnce = true;
            pEvent.execDetails.lastExecutedSubtaskId = lLastExecutedSubtaskId;
        
            if(lLastExecutedSubtaskId != lRange.endSubtask)
                SwitchThread(pEvent, lRange.task->GetPriority());
        
            bool lPrematureTermination = false;
            bool lReassigned = false;
            bool lForceAckFlag = false;
            lCommand->MarkExecutionStart();
            pmStatus lExecStatus = pmExecutionStub::ExecuteWrapper(lCurrentRange.task, lLastExecutedSubtaskId, lIsMultiAssignRange, lRange.startSubtask, lReassigned, lForceAckFlag, lPrematureTermination);
            lCommand->MarkExecutionEnd(lExecStatus, std::tr1::static_pointer_cast<pmCommand>(lCommand));
        
            if(lPrematureTermination)
            {
            #ifdef TRACK_SUBTASK_EXECUTION_VERBOSE
                std::cout << "[Host " << pmGetHostId() << "]: Prematurely terminated subtask " << lLastExecutedSubtaskId << std::endl;
            #endif
            
                if(lForceAckFlag && lLastExecutedSubtaskId != 0)
                {
                    lRange.endSubtask = lLastExecutedSubtaskId - 1;
                
                    if(lRange.endSubtask >= lRange.startSubtask)
                        HandleRangeExecutionCompletion(lRange, pmSuccess);
                }
            }
            else
            {
                if(lReassigned) // A secondary allottee has finished and negotiated this subtask and added a POST_HANDLE_EXEC_COMPLETION for the rest of the range
                {
                #ifdef _DEBUG
                    if(!lRange.task->IsMultiAssignEnabled() || lRange.originalAllottee != NULL)
                    {
                    std::cout << "[Device " << GetProcessingElement()->GetGlobalDeviceIndex() << "]: Range - [" << lRange.startSubtask << " - " << lRange.endSubtask << "] Original Allottee: " << lRange.originalAllottee->GetGlobalDeviceIndex() << std::endl;
                        PMTHROW(pmFatalErrorException());
                    }
                #endif
                       
                    break;
                }
            
            #ifdef TRACK_SUBTASK_EXECUTION_VERBOSE
                std::cout << "[Host " << pmGetHostId() << "]: Executed subtask " << lLastExecutedSubtaskId << std::endl;
            #endif

                lRange.task->GetTaskExecStats().RecordStubExecutionStats(this, lCompletedCount, lCommand->GetExecutionTimeInSecs());

                if(!lReassigned && lRange.originalAllottee == NULL)
                    CommonPostNegotiationOnCPU(lRange.task, lLastExecutedSubtaskId);
            
                if(lForceAckFlag)
                    lRange.endSubtask = lLastExecutedSubtaskId;
            
                if(lLastExecutedSubtaskId == lRange.endSubtask)
                    HandleRangeExecutionCompletion(lRange, lExecStatus);
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
                CommonPostNegotiationOnCPU(lRange.task, subtaskId);
        
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
			((pmStubGPU*)this)->FreeLastExecutionResources();
	#endif
			break;
		}
        
        case POST_HANDLE_EXEC_COMPLETION:
        {
            HandleRangeExecutionCompletion(pEvent.execCompletionDetails.range, pEvent.execCompletionDetails.execStatus);
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
            FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

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

	if(lMemSection && !pRange.task->GetCallbackUnit()->GetDataReductionCB())
    {
        pmSubscriptionInfo lSubscriptionInfo;
        subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
        bool lHasShadowMemory = pRange.task->DoSubtasksNeedShadowMemory();
        pmSubscriptionManager& lSubscriptionManager = pRange.task->GetSubscriptionManager();
    
        for(ulong subtaskId = pRange.startSubtask; subtaskId <= pRange.endSubtask; ++subtaskId)
        {
            if(!lSubscriptionManager.GetOutputMemSubscriptionForSubtask(this, subtaskId, lSubscriptionInfo))
                PMTHROW(pmFatalErrorException());
            
            lSubscriptionManager.GetNonConsolidatedOutputMemSubscriptionsForSubtask(this, subtaskId, lBeginIter, lEndIter);

            for(lIter = lBeginIter; lIter != lEndIter; ++lIter)
                lOwnershipMap[lIter->first] = lIter->second.first;
        
            if(lHasShadowMemory)
                lSubscriptionManager.CommitSubtaskShadowMem(this, subtaskId, lBeginIter, lEndIter, lSubscriptionInfo.offset);
        }
    }

    pmScheduler::GetScheduler()->SendAcknowledment(GetProcessingElement(), pRange, pExecStatus, lOwnershipMap);
}

// This method must be called with mCurrentSubtaskLock acquired
void pmExecutionStub::CancelCurrentlyExecutingSubtask(bool pTaskListeningOnCancellation)
{
#ifdef _DEBUG
    if(!mCurrentSubtaskStats)
        PMTHROW(pmFatalErrorException());
#endif
    
    if(!mCurrentSubtaskStats->taskListeningOnCancellation && pTaskListeningOnCancellation)
        mCurrentSubtaskStats->task->RecordStubWillSendCancellationMessage();

    mCurrentSubtaskStats->taskListeningOnCancellation |= pTaskListeningOnCancellation;
    
    if(!mCurrentSubtaskStats->prematureTermination)
    {
        mCurrentSubtaskStats->prematureTermination = true;
        
        if(!mCurrentSubtaskStats->executingLibraryCode)
            RaiseCurrentSubtaskTerminationSignalInThread();
    }
}

bool pmExecutionStub::IsInsideLibraryCode(bool& pPastCancellationStage)
{
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

    pPastCancellationStage = (mCurrentSubtaskStats == NULL);
    return (!mCurrentSubtaskStats || mCurrentSubtaskStats->executingLibraryCode);
}
    
void pmExecutionStub::MarkInsideLibraryCode(ulong pSubtaskId)
{
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!mCurrentSubtaskStats || mCurrentSubtaskStats->subtaskId != pSubtaskId || mCurrentSubtaskStats->executingLibraryCode)
        PMTHROW(pmFatalErrorException());
#endif
    
    mCurrentSubtaskStats->executingLibraryCode = true;
}

void pmExecutionStub::MarkInsideUserCode(ulong pSubtaskId)
{
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!mCurrentSubtaskStats || mCurrentSubtaskStats->subtaskId != pSubtaskId || !mCurrentSubtaskStats->executingLibraryCode)
        PMTHROW(pmFatalErrorException());
#endif
    
    mCurrentSubtaskStats->executingLibraryCode = false;
}
    
void pmExecutionStub::TerminateCurrentSubtask()
{
    longjmp(*(mCurrentSubtaskStats->jmpBuf), 1);
}

// This method must be called with mCurrentSubtaskLock acquired
void pmExecutionStub::RaiseCurrentSubtaskTerminationSignalInThread()
{
    InterruptThread();
}

// Must be called on stub thread
void pmExecutionStub::CheckForSubtaskTermination(ulong pSubtaskId)
{
    bool lPrematureTermination = false;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());
        lPrematureTermination = mCurrentSubtaskStats->prematureTermination;
    
    #ifdef _DEBUG
        if(mCurrentSubtaskStats->subtaskId != pSubtaskId)
            PMTHROW(pmFatalErrorException());
    #endif
    }
    
    if(lPrematureTermination)
        TerminateCurrentSubtask();
}

bool pmExecutionStub::RequiresPrematureExit(ulong pSubtaskId)
{
    FINALIZE_RESOURCE_PTR(dCurrentSubtaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCurrentSubtaskLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!mCurrentSubtaskStats || mCurrentSubtaskStats->subtaskId != pSubtaskId)
        PMTHROW(pmFatalErrorException());
#endif

    return mCurrentSubtaskStats->prematureTermination;
}

bool pmExecutionStub::IsHighPriorityEventWaiting(ushort pPriority)
{
	return GetPriorityQueue().IsHighPriorityElementPresent(pPriority);
}

pmStatus pmExecutionStub::ExecuteWrapper(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong pParentRangeStartSubtask, bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination)
{
    currentSubtaskTerminus lTerminus(pReassigned, pForceAckFlag, pPrematureTermination, this);

    TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, &pSubtaskId);
    pmStatus lStatus = ExecuteWrapperInternal(pTask, pSubtaskId, pIsMultiAssign, pParentRangeStartSubtask, lTerminus);
    TLS_IMPLEMENTATION_CLASS::GetTls()->SetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID, NULL);
    
    return lStatus;
}
    
pmStatus pmExecutionStub::ExecuteWrapperInternal(pmTask* pTask, ulong pSubtaskId, bool pIsMultiAssign, ulong pParentRangeStartSubtask, currentSubtaskTerminus& pTerminus)
{
    pmStatus lStatus = pmStatusUnavailable;
    jmp_buf lJmpBuf;
    
    guarded_scoped_ptr<RESOURCE_LOCK_IMPLEMENTATION_CLASS, currentSubtaskTerminus, currentSubtaskStats> lScopedPtr(&mCurrentSubtaskLock, &pTerminus, &mCurrentSubtaskStats, new currentSubtaskStats(pTask, pSubtaskId, !pIsMultiAssign, pParentRangeStartSubtask, &lJmpBuf, pmBase::GetCurrentTimeInSecs()));
    
    if(!setjmp(lJmpBuf))
    {
        UnblockSecondaryCommands(); // Allows external operations (steal & range negotiation) on priority queue
        lStatus = Execute(pTask, pSubtaskId);
    }
    
    return lStatus;
}
    
pmStatus pmExecutionStub::CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId)
{
    subscription::pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(this, pSubtaskId);
    
    try
    {
    #ifdef ENABLE_TASK_PROFILING
        pmTaskProfiler* lTaskProfiler = pTask->GetTaskProfiler();

        lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_PARTITIONING, true);
        pTask->GetSubscriptionManager().InitializeSubtaskDefaults(this, pSubtaskId);
        INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataDistributionCB, pTask->GetCallbackUnit()->GetDataDistributionCB(), Invoke, this, pTask, pSubtaskId);
        lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_PARTITIONING, false);

        pTask->GetSubscriptionManager().FetchSubtaskSubscriptions(this, pSubtaskId, GetType());
        
        if(pTask->DoSubtasksNeedShadowMemory())
            pTask->GetSubscriptionManager().CreateSubtaskShadowMem(this, pSubtaskId);

        lTaskProfiler->RecordProfileEvent(pmTaskProfiler::SUBTASK_EXECUTION, true);
    #else
        pTask->GetSubscriptionManager().InitializeSubtaskDefaults(this, pSubtaskId);
        INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataDistributionCB, pTask->GetCallbackUnit()->GetDataDistributionCB(), Invoke, this, pTask, pSubtaskId);
        pTask->GetSubscriptionManager().FetchSubtaskSubscriptions(this, pSubtaskId, GetType());

        if(pTask->DoSubtasksNeedShadowMemory())
            pTask->GetSubscriptionManager().CreateSubtaskShadowMem(this, pSubtaskId);
    #endif
    }
    catch(pmPrematureExitException&)
    {
    }
	
	return pmSuccess;
}

pmStatus pmExecutionStub::CommonPostExecuteOnCPU(pmTask* pTask, ulong pSubtaskId)
{
#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_EXECUTION, false);
#endif
    
    return pmSuccess;
}
    
pmStatus pmExecutionStub::CommonPostNegotiationOnCPU(pmTask* pTask, ulong pSubtaskId)
{
	pmCallbackUnit* lCallbackUnit = pTask->GetCallbackUnit();
	pmDataReductionCB* lReduceCallback = lCallbackUnit->GetDataReductionCB();
    
	if(lReduceCallback)
		pTask->GetReducer()->AddSubtask(this, pSubtaskId);
	else
		INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataRedistributionCB, pTask->GetCallbackUnit()->GetDataRedistributionCB(), Invoke, this, pTask, pSubtaskId);
    
    return pmSuccess;
}

pmStatus pmExecutionStub::DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2)
{
	pmStatus lStatus = pTask->GetCallbackUnit()->GetDataReductionCB()->Invoke(pTask, this, pSubtaskId1, pStub2, pSubtaskId2);

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


/* struct currentSubtaskStats */
pmExecutionStub::currentSubtaskStats::currentSubtaskStats(pmTask* pTask, ulong pSubtaskId, bool pOriginalAllottee, ulong pParentRangeStartSubtask, jmp_buf* pJmpBuf, double pStartTime)
    : task(pTask)
    , subtaskId(pSubtaskId)
    , parentRangeStartSubtask(pParentRangeStartSubtask)
    , originalAllottee(pOriginalAllottee)
    , startTime(pStartTime)
    , reassigned(false)
    , forceAckFlag(false)
    , executingLibraryCode(true)
    , prematureTermination(false)
    , taskListeningOnCancellation(false)
    , jmpBuf(pJmpBuf)
{
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

pmStatus pmStubCPU::Execute(pmTask* pTask, ulong pSubtaskId)
{
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId));
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, this, pTask, pSubtaskId, mCoreId);
	PROPAGATE_FAILURE_RET_STATUS(CommonPostExecuteOnCPU(pTask, pSubtaskId));
	
	return pmSuccess;
}


/* class pmStubGPU */
pmStubGPU::pmStubGPU(uint pDeviceIndexOnMachine)
	: pmExecutionStub(pDeviceIndexOnMachine)
{
}

pmStubGPU::~pmStubGPU()
{
}

/* class pmStubCUDA */
pmStubCUDA::pmStubCUDA(size_t pDeviceIndex, uint pDeviceIndexOnMachine)
	: pmStubGPU(pDeviceIndexOnMachine)
{
	mDeviceIndex = pDeviceIndex;
}

pmStubCUDA::~pmStubCUDA()
{
}

pmStatus pmStubCUDA::FreeResources()
{
#ifdef SUPPORT_CUDA
    FreeGpuResources();
#endif
    return pmSuccess;
}

pmStatus pmStubCUDA::FreeLastExecutionResources()
{
#ifdef SUPPORT_CUDA
    pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->FreeLastExecutionResources(mDeviceIndex);
#endif
    return pmSuccess;
}

pmStatus pmStubCUDA::BindToProcessingElement()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->BindToDevice(mDeviceIndex);
#else
	return pmSuccess;
#endif
}

std::string pmStubCUDA::GetDeviceName()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceName(mDeviceIndex);
#else
	return std::string();
#endif
}

std::string pmStubCUDA::GetDeviceDescription()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceDescription(mDeviceIndex);
#else
	return std::string();
#endif
}

pmDeviceType pmStubCUDA::GetType()
{
#ifdef SUPPORT_CUDA
	return GPU_CUDA;
#else
	PMTHROW(pmFatalErrorException());
	return MAX_DEVICE_TYPES;
#endif
}

pmStatus pmStubCUDA::Execute(pmTask* pTask, ulong pSubtaskId)
{
#ifdef SUPPORT_CUDA
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId));
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, this, pTask, pSubtaskId, mDeviceIndex);
	PROPAGATE_FAILURE_RET_STATUS(CommonPostExecuteOnCPU(pTask, pSubtaskId));
#endif

	return pmSuccess;
}


bool execEventMatchFunc(stubEvent& pEvent, void* pCriterion)
{
	if(pEvent.eventId == SUBTASK_EXEC && pEvent.execDetails.range.task == (pmTask*)pCriterion)
		return true;

	return false;
}


/* struct currentSubtaskTerminus */
pmExecutionStub::currentSubtaskTerminus::currentSubtaskTerminus(bool& pReassigned, bool& pForceAckFlag, bool& pPrematureTermination, pmExecutionStub* pStub)
    : mReassigned(pReassigned)
    , mForceAckFlag(pForceAckFlag)
    , mPrematureTermination(pPrematureTermination)
    , mStub(pStub)
{
}

void pmExecutionStub::currentSubtaskTerminus::Terminating(currentSubtaskStats* pStats)
{
    mPrematureTermination = pStats->prematureTermination;
    mReassigned = pStats->reassigned;
    mForceAckFlag = pStats->forceAckFlag;

    if(pStats->prematureTermination)
    {
        if(pStats->taskListeningOnCancellation)
            pStats->task->RegisterStubFreeOfTask();
    }
}

/* struct stubEvent */
bool execStub::stubEvent::BlocksSecondaryCommands()
{
    return (eventId == SUBTASK_EXEC);
}
    
};

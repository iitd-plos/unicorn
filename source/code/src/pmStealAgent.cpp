
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

#include "pmStealAgent.h"
#include "pmTask.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"

#ifdef USE_DYNAMIC_AFFINITY
#include "pmAffinityTable.h"
#endif

#ifdef USE_STEAL_AGENT_PER_NODE

namespace pm
{

using namespace stealAgent;
    
pmStealAgent::pmStealAgent(pmTask* pTask)
: mTask(pTask)
, mStubSink(pmStubManager::GetStubManager()->GetStubCount())
#ifdef ENABLE_DYNAMIC_AGGRESSION
, mDynamicAggressionStubSink(pmStubManager::GetStubManager()->GetStubCount())
#endif
{
    DEBUG_EXCEPTION_ASSERT(pmScheduler::SchedulingModelSupportsStealing(mTask->GetSchedulingModel()));
}

void pmStealAgent::RegisterPendingSubtasks(pmExecutionStub* pStub, ulong pPendingSubtasks)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    lStubData.subtasksPendingInStubQueue += pPendingSubtasks;
}

void pmStealAgent::DeregisterPendingSubtasks(pmExecutionStub* pStub, ulong pSubtasks)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    EXCEPTION_ASSERT(pSubtasks);
    EXCEPTION_ASSERT(lStubData.subtasksPendingInStubQueue >= pSubtasks);
    lStubData.subtasksPendingInStubQueue -= pSubtasks;
}

void pmStealAgent::RegisterExecutingSubtasks(pmExecutionStub* pStub, ulong pExecutingSubtasks)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    lStubData.subtasksPendingInPipeline += pExecutingSubtasks;
}

void pmStealAgent::DeregisterExecutingSubtasks(pmExecutionStub* pStub, ulong pSubtasks)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    EXCEPTION_ASSERT(pSubtasks && lStubData.subtasksPendingInPipeline >= pSubtasks);
    lStubData.subtasksPendingInPipeline -= pSubtasks;
}

void pmStealAgent::ClearExecutingSubtasks(pmExecutionStub* pStub)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    lStubData.subtasksPendingInPipeline = 0;
}

void pmStealAgent::SetStubMultiAssignment(pmExecutionStub* pStub, bool pCanMultiAssign)
{
    auto& lStubData = mStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lStubData.stubLock, Lock(), Unlock());

    lStubData.isMultiAssigning = pCanMultiAssign;
}

// This method operates without lock, so may work with stale data. But that is fine here.
pmExecutionStub* pmStealAgent::GetStubWithMaxStealLikelihood(bool pConsiderMultiAssign, pmExecutionStub* pIgnoreStub /* = NULL */)
{
    size_t lIgnoreStubIndex = pIgnoreStub ? pIgnoreStub->GetProcessingElement()->GetDeviceIndexInMachine() : 0;

    std::map<ulong, size_t> lSubtasksMap;   // subtasks versus stub
    for_each_with_index(mStubSink, [&] (stubData& pData, size_t pStubIndex)
    {
        if(pData.subtasksPendingInStubQueue && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
            lSubtasksMap.emplace(pData.subtasksPendingInStubQueue, pStubIndex);
    });

    if(!lSubtasksMap.empty())
        return pmStubManager::GetStubManager()->GetStub((uint)lSubtasksMap.rbegin()->second);
    
    for_each_with_index(mStubSink, [&] (stubData& pData, size_t pStubIndex)
    {
        if(pData.subtasksPendingInPipeline && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
            lSubtasksMap.emplace(pData.subtasksPendingInPipeline, pStubIndex);
    });

    if(!lSubtasksMap.empty())
        return pmStubManager::GetStubManager()->GetStub((uint)lSubtasksMap.rbegin()->second);

    if(pConsiderMultiAssign)
    {
        std::vector<size_t> lMultiAssigningStubs;
        lMultiAssigningStubs.reserve(mStubSink.size());

        std::vector<size_t> lPreferredMultiAssigningStubs;
        lPreferredMultiAssigningStubs.reserve(mStubSink.size());

        for_each_with_index(mStubSink, [&] (stubData& pData, size_t pStubIndex)
        {
            if(pData.isMultiAssigning && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
            {
                // For local requests (i.e. pIgnoreStub != NULL), multi-assign from a stub of different type preferrably
                if(!pIgnoreStub || pIgnoreStub->GetType() == pmStubManager::GetStubManager()->GetStub((uint)pStubIndex)->GetType())
                    lMultiAssigningStubs.emplace_back(pStubIndex);
                else
                    lPreferredMultiAssigningStubs.emplace_back(pStubIndex);
            }
        });

        if(!lPreferredMultiAssigningStubs.empty())
            return pmStubManager::GetStubManager()->GetStub((uint)lPreferredMultiAssigningStubs[rand() % lPreferredMultiAssigningStubs.size()]);

        if(!lMultiAssigningStubs.empty())
            return pmStubManager::GetStubManager()->GetStub((uint)lMultiAssigningStubs[rand() % lMultiAssigningStubs.size()]);
    }
        
    return NULL;
}

bool pmStealAgent::HasAnotherStubToStealFrom(pmExecutionStub* pStub, bool pCanMultiAssign)
{
    return GetStubWithMaxStealLikelihood(pCanMultiAssign, pStub);
}

#ifdef USE_DYNAMIC_AFFINITY
void pmStealAgent::HibernateSubtasks(const std::vector<ulong>& pSubtasksVector)
{
    EXCEPTION_ASSERT(mTask->GetAffinityCriterion() != MAX_AFFINITY_CRITERION);
    
    FINALIZE_RESOURCE_PTR(dHibernationLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mHibernationLock, Lock(), Unlock());

    std::copy(pSubtasksVector.begin(), pSubtasksVector.end(), std::inserter(mHibernatedSubtasks, mHibernatedSubtasks.end()));
}
    
ulong pmStealAgent::GetNextHibernatedSubtask(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dHibernationLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mHibernationLock, Lock(), Unlock());

    return pmAffinityTable::GetSubtaskWithBestAffinity(mTask, pStub, mHibernatedSubtasks, mAffinityData, true);
}
#endif
    
#ifdef ENABLE_DYNAMIC_AGGRESSION
void pmStealAgent::RecordStealRequestIssue(pmExecutionStub* pStub)
{
    auto& lDynamicAggressionStubData = mDynamicAggressionStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lDynamicAggressionStubData.stubLock, Lock(), Unlock());

    // This exception got triggered a few times. A lock is needed over pmExecutionStub::mStealRequestIssuedMap
    EXCEPTION_ASSERT(lDynamicAggressionStubData.stealStartTime == 0);
    lDynamicAggressionStubData.stealStartTime = pmBase::GetCurrentTimeInSecs();
}
    
void pmStealAgent::RecordSuccessfulSteal(pmExecutionStub* pStub)
{
    auto& lDynamicAggressionStubData = mDynamicAggressionStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lDynamicAggressionStubData.stubLock, Lock(), Unlock());

    lDynamicAggressionStubData.totalStealWaitTime += (pmBase::GetCurrentTimeInSecs() - lDynamicAggressionStubData.stealStartTime);
    ++lDynamicAggressionStubData.totalSteals;
    
    lDynamicAggressionStubData.stealStartTime = 0;
}

double pmStealAgent::GetAverageStealWaitTime(pmExecutionStub* pStub)
{
    auto& lDynamicAggressionStubData = mDynamicAggressionStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lDynamicAggressionStubData.stubLock, Lock(), Unlock());

    return ((lDynamicAggressionStubData.totalSteals == 0) ? std::numeric_limits<double>::max() : (lDynamicAggressionStubData.totalStealWaitTime / lDynamicAggressionStubData.totalSteals));
}
    
void pmStealAgent::RecordSubtaskSubscriptionFetchTime(pmExecutionStub* pStub, double pTime)
{
    auto& lDynamicAggressionStubData = mDynamicAggressionStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lDynamicAggressionStubData.stubLock, Lock(), Unlock());

    /* There is a loss in pre-fetching when a subtask is stolen (even when pipeline continuation is done.
       This is because there is no subtask to start pre-fetching data for before steal actually completes.
       By stealing early, we hope to cover this loss.
       The subscription fetch time sent to this function is the time a subtask waits on its subscriptions
       and all the time overlapped in pre-fetch is not accounted. For this reason, we consider maximum pre-fetch
       time below rather than average.
     */

//    lDynamicAggressionStubData.totalSubscriptionFetchTime += pTime;
//    ++lDynamicAggressionStubData.totalSubscriptionsFetched;

    if(pTime > lDynamicAggressionStubData.totalSubscriptionFetchTime)
        lDynamicAggressionStubData.totalSubscriptionFetchTime = pTime;

    lDynamicAggressionStubData.totalSubscriptionsFetched = 1;
}

double pmStealAgent::GetAverageSubtaskSubscriptionFetchTime(pmExecutionStub* pStub)
{
    auto& lDynamicAggressionStubData = mDynamicAggressionStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    FINALIZE_RESOURCE_PTR(dStubLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lDynamicAggressionStubData.stubLock, Lock(), Unlock());
    
    return ((lDynamicAggressionStubData.totalSubscriptionsFetched == 0) ? std::numeric_limits<double>::max() : (lDynamicAggressionStubData.totalSubscriptionFetchTime / lDynamicAggressionStubData.totalSubscriptionsFetched));
}
#endif
    
} // end namespace pm

#endif




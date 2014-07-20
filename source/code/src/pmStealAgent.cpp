
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

#ifdef USE_STEAL_AGENT_PER_NODE

namespace pm
{

using namespace stealAgent;
    
pmStealAgent::pmStealAgent(pmTask* pTask)
: mTask(pTask)
, mLockFreeStubSink(pmStubManager::GetStubManager()->GetStubCount())
{
    DEBUG_EXCEPTION_ASSERT(mTask->GetSchedulingModel() == scheduler::PULL);
}

void pmStealAgent::RegisterPendingSubtasks(pmExecutionStub* pStub, ulong pPendingSubtasks)
{
    mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()].subtasksPendingInStubQueue = pPendingSubtasks;
}

void pmStealAgent::DeregisterPendingSubtasks(pmExecutionStub* pStub, ulong pSubtasks)
{
    auto& lStubData = mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    EXCEPTION_ASSERT(pSubtasks && lStubData.subtasksPendingInStubQueue >= pSubtasks);
    lStubData.subtasksPendingInStubQueue -= pSubtasks;
}

void pmStealAgent::RegisterExecutingSubtasks(pmExecutionStub* pStub, ulong pExecutingSubtasks)
{
    mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()].subtasksPendingInPipeline = pExecutingSubtasks;
}

void pmStealAgent::DeregisterExecutingSubtasks(pmExecutionStub* pStub, ulong pSubtasks)
{
    auto& lStubData = mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

    EXCEPTION_ASSERT(pSubtasks && lStubData.subtasksPendingInPipeline >= pSubtasks);
    lStubData.subtasksPendingInPipeline -= pSubtasks;
}

void pmStealAgent::ClearExecutingSubtasks(pmExecutionStub* pStub)
{
    mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()].subtasksPendingInStubQueue = 0;
}

void pmStealAgent::SetStubMultiAssignment(pmExecutionStub* pStub, bool pCanMultiAssign)
{
    mLockFreeStubSink[pStub->GetProcessingElement()->GetDeviceIndexInMachine()].isMultiAssigning = pCanMultiAssign;
}

pmExecutionStub* pmStealAgent::GetStubWithMaxStealLikelihood(bool pConsiderMultiAssign, pmExecutionStub* pIgnoreStub /* = NULL */)
{
    size_t lIgnoreStubIndex = pIgnoreStub ? pIgnoreStub->GetProcessingElement()->GetDeviceIndexInMachine() : 0;

    std::map<ulong, size_t> lSubtasksMap;   // subtasks versus stub
    for_each_with_index(mLockFreeStubSink, [&] (stubData& pData, size_t pStubIndex)
    {
        if(pData.subtasksPendingInStubQueue && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
            lSubtasksMap.emplace(pData.subtasksPendingInStubQueue, pStubIndex);
    });

    if(!lSubtasksMap.empty())
        return pmStubManager::GetStubManager()->GetStub((uint)lSubtasksMap.rbegin()->second);
    
    for_each_with_index(mLockFreeStubSink, [&] (stubData& pData, size_t pStubIndex)
    {
        if(pData.subtasksPendingInPipeline && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
            lSubtasksMap.emplace(pData.subtasksPendingInPipeline, pStubIndex);
    });

    if(!lSubtasksMap.empty())
        return pmStubManager::GetStubManager()->GetStub((uint)lSubtasksMap.rbegin()->second);

    if(pConsiderMultiAssign)
    {
        std::vector<size_t> lMultiAssigningStubs;
        lMultiAssigningStubs.reserve(mLockFreeStubSink.size());

        for_each_with_index(mLockFreeStubSink, [&] (stubData& pData, size_t pStubIndex)
        {
            if(pData.isMultiAssigning && (!pIgnoreStub || lIgnoreStubIndex != pStubIndex))
                lMultiAssigningStubs.emplace_back(pStubIndex);
        });

        if(!lMultiAssigningStubs.empty())
            return pmStubManager::GetStubManager()->GetStub((uint)lMultiAssigningStubs[rand() % lMultiAssigningStubs.size()]);
    }
        
    return NULL;
}

bool pmStealAgent::HasAnotherStubToStealFrom(pmExecutionStub* pStub, bool pCanMultiAssign)
{
    return GetStubWithMaxStealLikelihood(pCanMultiAssign, pStub);
}
    
} // end namespace pm

#endif

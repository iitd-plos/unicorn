
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

#include "pmSubtaskSplitter.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmSubscriptionManager.h"
#include "pmTask.h"

#include <limits>

#ifdef SUPPORT_SPLIT_SUBTASKS

namespace pm
{

using namespace splitter;

pmSubtaskSplitter::pmSubtaskSplitter(pmTask* pTask)
    : mTask(pTask)
    , mSplittingType(MAX_DEVICE_TYPES)
{
    pmStubManager* lStubManager = pmStubManager::GetStubManager();

    if(mTask->CanSplitCpuSubtasks())    // Create one split group per CPU NUMA domain
    {
        const auto& lNumaDomainsVector = lStubManager->GetCpuNumaDomains();
        size_t lNumaDomains = (size_t)lStubManager->GetCpuNumaDomainsCount();
        size_t lMaxCpuDevicesPerHost = lStubManager->GetMaxCpuDevicesPerHostForCpuPlusGpuTasks();
        size_t lTotalCpuDevices = lStubManager->GetProcessingElementsCPU();
        size_t lExcessCpuDevices = ((lMaxCpuDevicesPerHost < lTotalCpuDevices) ? (lTotalCpuDevices - lMaxCpuDevicesPerHost) : 0);
        size_t lExcessCpuDevicesPerDomain = lExcessCpuDevices / lNumaDomains;
        size_t lLeftoverExcessCpuDevices = lExcessCpuDevices - lExcessCpuDevicesPerDomain * lNumaDomains;
        
        // If max CPU devices to be used are less than the max avaiable, then try to impact all domains equally
        size_t lStubsProcessed = 0;
        for_each_with_index(lNumaDomainsVector, [&] (const std::vector<pmExecutionStub*>& pDomain, size_t pIndex)
        {
            size_t lDomainStubs = pDomain.size();
            if(lDomainStubs > lExcessCpuDevicesPerDomain)
            {
                size_t lUsableStubs = lDomainStubs - lExcessCpuDevicesPerDomain;
                if(lLeftoverExcessCpuDevices)
                {
                    size_t lRemainingDomains = lNumaDomains - pIndex - 1;
                    size_t lMinStubsToBeRemoved = (lLeftoverExcessCpuDevices / (lRemainingDomains + 1));
                    
                    EXCEPTION_ASSERT(lUsableStubs >= lMinStubsToBeRemoved);
                    
                    lUsableStubs -= lMinStubsToBeRemoved;
                    lLeftoverExcessCpuDevices -= lMinStubsToBeRemoved;
                    
                    if(lUsableStubs && lLeftoverExcessCpuDevices)
                    {
                        size_t lRemainingStubs = lTotalCpuDevices - lStubsProcessed - lDomainStubs;
                        EXCEPTION_ASSERT(!lRemainingStubs || lLeftoverExcessCpuDevices <= lRemainingStubs + lUsableStubs);
                        
                        if(lRemainingStubs < lLeftoverExcessCpuDevices)
                        {
                            size_t lMoreStubsToBeRemoved = lLeftoverExcessCpuDevices - lRemainingStubs;
                            EXCEPTION_ASSERT(lMoreStubsToBeRemoved <= lUsableStubs);
                            
                            lUsableStubs -= lMoreStubsToBeRemoved;
                            lLeftoverExcessCpuDevices -= lMoreStubsToBeRemoved;
                        }
                    }
                }
                
                if(lUsableStubs)
                {
                    size_t lSplitGroupIndex = mSplitGroupVector.size();

                    mSplitGroupVector.emplace_back(this, std::vector<pmExecutionStub*>(pDomain.begin(), pDomain.begin() + lUsableStubs));
                    
                    std::for_each(pDomain.begin(), pDomain.begin() + lUsableStubs, [&] (pmExecutionStub* pStub)
                    {
                        mSplitGroupMap.emplace(pStub, lSplitGroupIndex);
                    });
                }
            }
            else if(lDomainStubs < lExcessCpuDevicesPerDomain)
            {
                lLeftoverExcessCpuDevices += lExcessCpuDevicesPerDomain - lDomainStubs;
            }
            
            lStubsProcessed += lDomainStubs;
        });
        
        mSplittingType = CPU;
    }
    else if(mTask->CanSplitGpuSubtasks())   // All GPUs are added to the same split group
    {
        size_t lCount = lStubManager->GetProcessingElementsGPU();
        
        if(lCount)
        {
            std::vector<pmExecutionStub*> lDomainStubs;
            lDomainStubs.reserve(lCount);
            
            size_t lSplitGroupIndex = mSplitGroupVector.size();

            for(size_t i = 0; i < lCount; ++i)
            {
                pmExecutionStub* lStub = lStubManager->GetGpuStub((uint)i);

                lDomainStubs.emplace_back(lStub);
                mSplitGroupMap.emplace(lStub, lSplitGroupIndex);
            }
            
            mSplitGroupVector.emplace_back(this, std::move(lDomainStubs));

            mSplittingType = GPU_CUDA;
        }
    }
}

pmDeviceType pmSubtaskSplitter::GetSplittingType()
{
    return mSplittingType;
}

bool pmSubtaskSplitter::IsSplitting(pmDeviceType pDeviceType)
{
    return (mSplittingType == pDeviceType);
}
    
std::unique_ptr<pmSplitSubtask> pmSubtaskSplitter::GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub)
{
    return mSplitGroupVector[mSplitGroupMap[pSourceStub]].GetPendingSplit(pSubtaskId, pSourceStub);
}
    
void pmSubtaskSplitter::FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination, double pExecTime)
{
    mSplitGroupVector[mSplitGroupMap[pStub]].FinishedSplitExecution(pSubtaskId, pSplitId, pStub, pPrematureTermination, pExecTime);
}

bool pmSubtaskSplitter::Negotiate(pmExecutionStub* pStub, ulong pSubtaskId, std::vector<pmExecutionStub*>& pStubsToBeCancelled, pmExecutionStub*& pSourceStub)
{
    return mSplitGroupVector[mSplitGroupMap[pStub]].Negotiate(pSubtaskId, pStubsToBeCancelled, pSourceStub);
}
    
void pmSubtaskSplitter::StubHasProcessedDummyEvent(pmExecutionStub* pStub)
{
    mSplitGroupVector[mSplitGroupMap[pStub]].StubHasProcessedDummyEvent(pStub);
}

void pmSubtaskSplitter::FreezeDummyEvents()
{
    for_each(mSplitGroupVector, [] (pmSplitGroup& pSplitGroup)
    {
        pSplitGroup.FreezeDummyEvents();
    });
}
    
void pmSubtaskSplitter::PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    mSplitGroupVector[mSplitGroupMap[pStub]].PrefetchSubscriptionsForUnsplittedSubtask(pStub, pSubtaskId);
}

void pmSubtaskSplitter::MakeDeviceGroups(const std::vector<const pmProcessingElement*>& pDevices, std::vector<std::vector<const pmProcessingElement*>>& pDeviceGroups, std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>*>& pQueryMap, ulong& pUnsplittedDevices)
{
    pmDeviceType lSplittingType = GetSplittingType();

    if(lSplittingType != MAX_DEVICE_TYPES)
    {
        struct mapKey
        {
            const pmMachine* machine;
            pmDeviceType type;
            ushort numaDomain;
            
            bool operator<(const mapKey& pKey) const
            {
                return std::tie(machine, type, numaDomain) < std::tie(pKey.machine, pKey.type, pKey.numaDomain);
            }
        };
        
        std::map<mapKey, std::vector<const pmProcessingElement*>*> lMachineInfo;

        for_each(pDevices, [&] (const pmProcessingElement* pProcessingElement)
        {
            pmDeviceType lDeviceType = pProcessingElement->GetType();
            if(lDeviceType == lSplittingType)
            {
                mapKey lKey{pProcessingElement->GetMachine(), lDeviceType, pProcessingElement->GetNumaDomainId()};
                
                auto lMachineInfoIter = lMachineInfo.find(lKey);
                if(lMachineInfoIter == lMachineInfo.end())
                {
                    pDeviceGroups.emplace_back(1, pProcessingElement);
                    lMachineInfo.emplace(lKey, &pDeviceGroups.back());
                }
                else
                {
                    lMachineInfoIter->second->emplace_back(pProcessingElement);
                }
            }
            else
            {
                pDeviceGroups.emplace_back(1, pProcessingElement);
                ++pUnsplittedDevices;
            }
        });
        
        for_each(pDeviceGroups, [&] (std::vector<const pmProcessingElement*>& pGroup)
        {
            for_each(pGroup, [&] (const pmProcessingElement* pProcessingElement)
            {
                pQueryMap.emplace(pProcessingElement, &pGroup);
            });
        });
    }
}

std::vector<std::pair<std::vector<const pmProcessingElement*>, std::pair<ulong, ulong>>> pmSubtaskSplitter::MakeInitialSchedulingAllotments(pmLocalTask* pLocalTask)
{
    std::vector<std::pair<std::vector<const pmProcessingElement*>, std::pair<ulong, ulong>>> lDeviceGroupAndAllotmentVector;
	std::vector<const pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();
    std::vector<std::vector<const pmProcessingElement*>> lDeviceGroups;
    std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>*> lQueryMap;
    ulong lUnsplittedDevices = 0, lFirstSubtask = 0, lLastSubtask = 0;
    ulong lSubtaskCount = pLocalTask->GetSubtaskCount();

    MakeDeviceGroups(lDevices, lDeviceGroups, lQueryMap, lUnsplittedDevices);

    if(!lDeviceGroups.empty())
    {
        ulong lUnsplittedGroups = lUnsplittedDevices;
        ulong lSplittedGroups = lDeviceGroups.size() - lUnsplittedGroups;
        
        ulong lSplittedGroupSubtasks = 0, lUnsplittedGroupSubtasks = 0;

    #ifdef FORCE_START_WITH_ONE_SUBTASK_PER_SPLIT_GROUP
        if(pLocalTask->GetSchedulingModel() != scheduler::STATIC_EQUAL)
        {
            if(lSubtaskCount < lUnsplittedGroups)
            {
                lUnsplittedGroupSubtasks = lSubtaskCount;
            }
            else
            {
                if(lSubtaskCount > lDeviceGroups.size())
                {
                    lSplittedGroupSubtasks = lSplittedGroups;
                    lUnsplittedGroupSubtasks = lSubtaskCount - lSplittedGroupSubtasks;
                }
                else
                {
                    lUnsplittedGroupSubtasks = lUnsplittedGroups;
                    lSplittedGroupSubtasks = lSubtaskCount - lUnsplittedGroupSubtasks;
                }
            }
        }
        else
    #endif
        {
            ulong lTotalGroups = lSplittedGroups + lUnsplittedGroups;
            ulong lPartitionCount = std::min(lSubtaskCount, lTotalGroups);
            ulong lPartitionSize = lSubtaskCount / lPartitionCount;
            ulong lLeftoverSubtasks = lSubtaskCount - lPartitionSize * lPartitionCount;

            lUnsplittedGroupSubtasks = std::min(lSubtaskCount, lLeftoverSubtasks + lPartitionSize * lUnsplittedGroups);
            lSplittedGroupSubtasks = lSubtaskCount - lUnsplittedGroupSubtasks;
        }
        
        ulong lSplittedPartitionSize = lSplittedGroupSubtasks / lSplittedGroups;
        ulong lUnsplittedPartitionSize = lUnsplittedGroupSubtasks / lUnsplittedGroups;
        
        ulong lSplittedLeftoverSubtasks = lSplittedGroupSubtasks - lSplittedPartitionSize * lSplittedGroups;
        ulong lUnsplittedLeftoverSubtasks = lUnsplittedGroupSubtasks - lUnsplittedPartitionSize * lUnsplittedGroups;

        std::vector<std::vector<const pmProcessingElement*>>::iterator lGroupIter = lDeviceGroups.begin(), lGroupEndIter = lDeviceGroups.end();
        for(; lGroupIter != lGroupEndIter; ++lGroupIter)
        {
            size_t lCount = 0;

            if((*lGroupIter).size() > 1)   // Splitting Group
            {
                if(!lSplittedGroupSubtasks)
                    continue;
                
                lCount = lSplittedPartitionSize;
                if(lSplittedLeftoverSubtasks)
                {
                    ++lCount;
                    --lSplittedLeftoverSubtasks;
                }
                
                if(lSplittedGroupSubtasks < lCount)
                    lCount = lSplittedGroupSubtasks;

                lSplittedGroupSubtasks -= lCount;
            }
            else    // Unsplitting Group
            {
                if(!lUnsplittedGroupSubtasks)
                    continue;

                lCount = lUnsplittedPartitionSize;
                if(lUnsplittedLeftoverSubtasks)
                {
                    ++lCount;
                    --lUnsplittedLeftoverSubtasks;
                }
                
                if(lUnsplittedGroupSubtasks < lCount)
                    lCount = lUnsplittedGroupSubtasks;

                lUnsplittedGroupSubtasks -= lCount;
            }
            
            if(!lCount)
                continue;

            lLastSubtask = lFirstSubtask + lCount - 1;
            
            lDeviceGroupAndAllotmentVector.emplace_back(std::move(*lGroupIter), std::make_pair(lFirstSubtask, lLastSubtask));

            lFirstSubtask = lLastSubtask + 1;
        }
        
        EXCEPTION_ASSERT(!lSplittedGroupSubtasks && !lUnsplittedGroupSubtasks);
    }
    
    return lDeviceGroupAndAllotmentVector;
}


/* class pmSplitGroup */
std::unique_ptr<pmSplitSubtask> pmSplitGroup::GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub)
{
    std::shared_ptr<splitRecord> lSplitRecord;
    uint lSplitId = std::numeric_limits<uint>::max();

    // Auto lock/unlock scope
    {
        std::shared_ptr<splitRecord> lModifiableSplitRecord;

        FINALIZE_RESOURCE_PTR(dSplitRecordListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSplitRecordListLock, Lock(), Unlock());
     
        if(!mSplitRecordList.empty())
        {
            lModifiableSplitRecord = mSplitRecordList.back();
            
            if(lModifiableSplitRecord->splitId == lModifiableSplitRecord->splitCount || lModifiableSplitRecord->reassigned)
                lModifiableSplitRecord.reset();
        }
        
        if(!lModifiableSplitRecord.get())
        {
            if(!pSubtaskId)
                return std::unique_ptr<pmSplitSubtask>();
         
            mSplitRecordList.emplace_back(std::make_shared<splitRecord>(pSourceStub, *pSubtaskId, mSplitFactorCalculator.GetNextSplitFactor()));
            mSplitRecordMap.emplace(std::piecewise_construct, std::forward_as_tuple(*pSubtaskId), std::forward_as_tuple(--mSplitRecordList.end()));
            
            lModifiableSplitRecord = mSplitRecordList.back();
        }
        
        lSplitId = lModifiableSplitRecord->splitId;

        ++(lModifiableSplitRecord->splitId);
        lModifiableSplitRecord->assignedStubs.push_back(std::make_pair(pSourceStub, false));
        
        lSplitRecord = lModifiableSplitRecord;
    }

    AddDummyEventToRequiredStubs();

    return std::unique_ptr<pmSplitSubtask>(new pmSplitSubtask(mSubtaskSplitter->mTask, lSplitRecord->sourceStub, lSplitRecord->subtaskId, lSplitId, lSplitRecord->splitCount));
}
    
void pmSplitGroup::FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination, double pExecTime)
{
    bool lCompleted = false;
    std::shared_ptr<splitter::splitRecord> lSplitRecord;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dSplitRecordListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSplitRecordListLock, Lock(), Unlock());
        
        const auto lMapIter = mSplitRecordMap.find(pSubtaskId);
        
        EXCEPTION_ASSERT(lMapIter != mSplitRecordMap.end());

        const std::list<std::shared_ptr<splitRecord>>::iterator lIter = lMapIter->second;
        
        EXCEPTION_ASSERT(lIter != mSplitRecordList.end() && (*lIter)->pendingCompletions);
        EXCEPTION_ASSERT((*lIter)->assignedStubs[pSplitId].first == pStub);

        --((*lIter)->pendingCompletions);
        (*lIter)->assignedStubs[pSplitId].second = true;
        
        (*lIter)->subtaskExecTime += pExecTime;
        
        if(pPrematureTermination)
            (*lIter)->reassigned = true;

        if((*lIter)->pendingCompletions == 0)
        {
            if(!(*lIter)->reassigned)
            {
                lSplitRecord = *lIter;
                lCompleted = true;
                
                mSplitFactorCalculator.RegisterSubtaskCompletion(lSplitRecord->splitCount, pExecTime);
            }
            
            mSplitRecordList.erase(lIter);
            mSplitRecordMap.erase(lMapIter);
        }
    }
    
    if(lCompleted)
    {
        for_each_with_index(lSplitRecord->assignedStubs, [&] (const std::pair<pmExecutionStub*, bool>& pPair, size_t pIndex)
        {
            pmSplitInfo lSplitInfo((uint)pIndex, lSplitRecord->splitCount);

            pPair.first->CommonPostNegotiationOnCPU(mSubtaskSplitter->mTask, pSubtaskId, false, &lSplitInfo);
        });
        
        lSplitRecord->sourceStub->HandleSplitSubtaskExecutionCompletion(mSubtaskSplitter->mTask, *lSplitRecord.get(), pmSuccess);
    }
}

void pmSplitGroup::StubHasProcessedDummyEvent(pmExecutionStub* pStub)
{
    bool lPendingSplits = false;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dSplitRecordListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSplitRecordListLock, Lock(), Unlock());
    
        lPendingSplits = !mSplitRecordList.empty();
    }

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDummyEventLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDummyEventLock, Lock(), Unlock());
        
        if(!lPendingSplits || mDummyEventsFreezed)
        {
            mStubsWithDummyEvent.erase(pStub);
            return;
        }
    }
    
    pStub->SplitSubtaskCheckEvent(mSubtaskSplitter->mTask);
}
    
void pmSplitGroup::AddDummyEventToRequiredStubs()
{
    FINALIZE_RESOURCE_PTR(dDummyEventLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDummyEventLock, Lock(), Unlock());
    
    if(mDummyEventsFreezed)
        return;
    
    std::vector<pmExecutionStub*>::iterator lIter = mConcernedStubs.begin(), lEndIter = mConcernedStubs.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if(mStubsWithDummyEvent.find(*lIter) == mStubsWithDummyEvent.end())
            AddDummyEventToStub(*lIter);
    }
}

/* This method must be called with pmSplitGroup's mDummyEventLock acquired */
void pmSplitGroup::AddDummyEventToStub(pmExecutionStub* pStub)
{
    pStub->SplitSubtaskCheckEvent(mSubtaskSplitter->mTask);
    mStubsWithDummyEvent.insert(pStub);
}
    
void pmSplitGroup::FreezeDummyEvents()
{
    std::set<pmExecutionStub*> lPendingStubs;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDummyEventLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDummyEventLock, Lock(), Unlock());
        
        mDummyEventsFreezed = true;
        lPendingStubs = mStubsWithDummyEvent;
    }
    
    std::set<pmExecutionStub*>::iterator lIter = lPendingStubs.begin(), lEndIter = lPendingStubs.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->RemoveSplitSubtaskCheckEvent(mSubtaskSplitter->mTask);
}

bool pmSplitGroup::Negotiate(ulong pSubtaskId, std::vector<pmExecutionStub*>& pStubsToBeCancelled, pmExecutionStub*& pSourceStub)
{
    bool lRetVal = false;
    std::vector<std::pair<pmExecutionStub*, bool>> lStubVector;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dSplitRecordListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSplitRecordListLock, Lock(), Unlock());
        
        const auto lMapIter = mSplitRecordMap.find(pSubtaskId);
        
        if(lMapIter != mSplitRecordMap.end())
        {
            const std::list<std::shared_ptr<splitRecord>>::iterator lIter = lMapIter->second;
            
            EXCEPTION_ASSERT(lIter != mSplitRecordList.end());
            
            DEBUG_EXCEPTION_ASSERT((*lIter)->subtaskId == pSubtaskId);

            if(!(*lIter)->reassigned)
            {
                pSourceStub = (*lIter)->sourceStub;
                lStubVector = (*lIter)->assignedStubs;
                (*lIter)->reassigned = true;
                lRetVal = true;
            }
        }
    }
    
    if(lRetVal)
    {
        filtered_for_each(lStubVector, [] (const std::pair<pmExecutionStub*, bool>& pPair) {return !pPair.second;},
                [&] (const std::pair<pmExecutionStub*, bool>& pPair)
                {
                    pStubsToBeCancelled.push_back(pPair.first);
                });
    }
    
    return lRetVal;
}
    
void pmSplitGroup::PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    FINALIZE_RESOURCE_PTR(dSplitRecordListLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSplitRecordListLock, Lock(), Unlock());

    const auto lMapIter = mSplitRecordMap.find(pSubtaskId);
    
    EXCEPTION_ASSERT(lMapIter != mSplitRecordMap.end());
    
    auto lIter = lMapIter->second;
    
    EXCEPTION_ASSERT(lIter != mSplitRecordList.end());

    DEBUG_EXCEPTION_ASSERT((*lIter)->subtaskId == pSubtaskId);

    if(!(*lIter)->prefetched)
    {
        pmSubscriptionManager& lSubscriptionManager = mSubtaskSplitter->mTask->GetSubscriptionManager();
        
        lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL);
        lSubscriptionManager.FetchSubtaskSubscriptions(pStub, pSubtaskId, NULL, pStub->GetType(), true);
        
        (*lIter)->prefetched = true;
    }
}


/* class mSplitFactorCalculator */
void pmSplitFactorCalculator::RegisterSubtaskCompletion(uint pSplitFactor, double pTime)
{
#ifdef ENABLE_DYNAMIC_SPLITTING
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    auto lIter = mMap.find(pSplitFactor);
    if(lIter == mMap.end())
    {
        mMap.emplace(std::piecewise_construct, std::forward_as_tuple(pSplitFactor), std::forward_as_tuple(pTime, 1));
    }
    else
    {
        lIter->second.first += pTime;
        ++lIter->second.second;
    }
#else
#endif
}
    
uint pmSplitFactorCalculator::GetNextSplitFactor()
{
#ifdef ENABLE_DYNAMIC_SPLITTING
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    size_t lSize = mMap.size();

    if(lSize == 0)
        return mFirstSplitFactor;
    else if(lSize == 1)
        return mSecondSplitFactor;
    
    double lFirstTime = GetAverageTimeForSplitFactor(mFirstSplitFactor);
    double lSecondTime = GetAverageTimeForSplitFactor(mSecondSplitFactor);
    
    if(lFirstTime <= lSecondTime)
        std::swap(mSecondSplitFactor, mThirdSplitFactor);
    else
        std::swap(mSecondSplitFactor, mFirstSplitFactor);
    
    mSecondSplitFactor = std::max((uint)1, (mFirstSplitFactor + mThirdSplitFactor)/2);
    if(mSecondSplitFactor % 2 != 0 && mSecondSplitFactor + 1 <= mMaxSplitFactor)
    {
        double lTime1 = GetAverageTimeForSplitFactor(mSecondSplitFactor + 1);
        double lTime2 = GetAverageTimeForSplitFactor(mSecondSplitFactor);
        
        if(lTime1 < lTime2 && lTime1 != std::numeric_limits<double>::max() && lTime2 != std::numeric_limits<double>::max())
            ++mSecondSplitFactor;
    }
    
    return mSecondSplitFactor;
#else
    return mMaxSplitFactor;
#endif
}

#ifdef ENABLE_DYNAMIC_SPLITTING
// This method must be called with mResourceLock acquired
double pmSplitFactorCalculator::GetAverageTimeForSplitFactor(uint pSplitFactor)
{
    auto lIter = mMap.find(pSplitFactor);
    if(lIter == mMap.end())
        return std::numeric_limits<double>::max();
    
    return (lIter->second.first / (double)lIter->second.second);
}
#endif

}

#endif


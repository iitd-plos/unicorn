
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

#include "pmSubscriptionManager.h"
#include "pmCommunicator.h"
#include "pmTaskManager.h"
#include "pmMemoryManager.h"
#include "pmAddressSpace.h"
#include "pmTask.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmCallbackUnit.h"
#include "pmCallback.h"
#include "pmHardware.h"
#include "pmLogger.h"

#ifdef ENABLE_DYNAMIC_AGGRESSION
#include "pmStealAgent.h"
#endif

#include <string.h>
#include <sstream>

#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace pm
{

using namespace subscription;

#ifdef SUPPORT_SPLIT_SUBTASKS

#define GET_SUBTASK(subtaskVar, stub, subtaskId, splitInfo) \
uint dDeviceIndex = stub->GetProcessingElement()->GetDeviceIndexInMachine(); \
RESOURCE_LOCK_IMPLEMENTATION_CLASS& dLock = splitInfo ? mSplitSubtaskMapVector[dDeviceIndex].second : mSubtaskMapVector[dDeviceIndex].second; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dLock, Lock(), Unlock()); \
subscription::pmSubtask* dSubtaskPtr = NULL; \
if(splitInfo) \
{ \
    splitSubtaskMapType& lMap = mSplitSubtaskMapVector[dDeviceIndex].first; \
    splitSubtaskMapType::iterator dIter = lMap.find(std::make_pair(subtaskId, *splitInfo)); \
    if(dIter == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr = &dIter->second; \
} \
else \
{ \
    subtaskMapType& lMap = mSubtaskMapVector[dDeviceIndex].first; \
    subtaskMapType::iterator dIter = lMap.find(subtaskId); \
    if(dIter == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr = &dIter->second; \
} \
subscription::pmSubtask& subtaskVar = *dSubtaskPtr;

#define GET_SUBTASKS(subtaskVar1, subtaskVar2, stub, subtaskId1, subtaskId2, splitInfo1, splitInfo2) \
if((splitInfo1 && !splitInfo2) || (!splitInfo1 && splitInfo2)) \
    PMTHROW(pmFatalErrorException()); \
uint dDeviceIndex = stub->GetProcessingElement()->GetDeviceIndexInMachine(); \
RESOURCE_LOCK_IMPLEMENTATION_CLASS& dLock = splitInfo1 ? mSplitSubtaskMapVector[dDeviceIndex].second : mSubtaskMapVector[dDeviceIndex].second; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dLock, Lock(), Unlock()); \
subscription::pmSubtask* dSubtaskPtr1 = NULL; \
subscription::pmSubtask* dSubtaskPtr2 = NULL; \
if(splitInfo1) \
{ \
    splitSubtaskMapType& lMap = mSplitSubtaskMapVector[dDeviceIndex].first; \
    splitSubtaskMapType::iterator dIter1 = lMap.find(std::make_pair(subtaskId1, *splitInfo1)); \
    if(dIter1 == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    splitSubtaskMapType::iterator dIter2 = lMap.find(std::make_pair(subtaskId2, *splitInfo2)); \
    if(dIter2 == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr1 = &dIter1->second; \
    dSubtaskPtr2 = &dIter2->second; \
} \
else \
{ \
    subtaskMapType& lMap = mSubtaskMapVector[dDeviceIndex].first; \
    subtaskMapType::iterator dIter1 = lMap.find(subtaskId1); \
    if(dIter1 == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    subtaskMapType::iterator dIter2 = lMap.find(subtaskId2); \
    if(dIter2 == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr1 = &dIter1->second; \
    dSubtaskPtr2 = &dIter2->second; \
} \
subscription::pmSubtask& subtaskVar1 = *dSubtaskPtr1; \
subscription::pmSubtask& subtaskVar2 = *dSubtaskPtr2;

#define GET_SUBTASK2(subtaskVar, stub, subtaskId, splitInfo) \
uint dDeviceIndex2 = stub->GetProcessingElement()->GetDeviceIndexInMachine(); \
RESOURCE_LOCK_IMPLEMENTATION_CLASS& dLock2 = splitInfo ? mSplitSubtaskMapVector[dDeviceIndex2].second : mSubtaskMapVector[dDeviceIndex2].second; \
FINALIZE_RESOURCE_PTR(dResourceLock2, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dLock2, Lock(), Unlock()); \
subscription::pmSubtask* dSubtaskPtr2 = NULL; \
if(splitInfo) \
{ \
    splitSubtaskMapType& lMap = mSplitSubtaskMapVector[dDeviceIndex2].first; \
    splitSubtaskMapType::iterator dIter = lMap.find(std::make_pair(subtaskId, *splitInfo)); \
    if(dIter == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr = &dIter->second; \
} \
else \
{ \
    subtaskMapType& lMap = mSubtaskMapVector[dDeviceIndex2].first; \
    subtaskMapType::iterator dIter = lMap.find(subtaskId); \
    if(dIter == lMap.end()) \
        PMTHROW(pmFatalErrorException()); \
    dSubtaskPtr = &dIter->second; \
} \
subscription::pmSubtask& subtaskVar = *dSubtaskPtr2;
    
#else
    
#define GET_SUBTASK(subtaskVar, stub, subtaskId, splitInfo) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair.second, Lock(), Unlock()); \
subtaskMapType::iterator dIter = dPair.first.find(subtaskId); \
if(dIter == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar = dIter->second;

#define GET_SUBTASKS(subtaskVar1, subtaskVar2, stub, subtaskId1, subtaskId2, splitInfo1, splitInfo2) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair.second, Lock(), Unlock()); \
subtaskMapType::iterator dIter1 = dPair.first.find(subtaskId1); \
if(dIter1 == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subtaskMapType::iterator dIter2 = dPair.first.find(subtaskId2); \
if(dIter2 == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar1 = dIter1->second; \
subscription::pmSubtask& subtaskVar2 = dIter2->second;

#define GET_SUBTASK2(subtaskVar, stub, subtaskId, splitInfo) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair2 = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock2, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair2.second, Lock(), Unlock()); \
subtaskMapType::iterator dIter2 = dPair.first.find(subtaskId); \
if(dIter2 == dPair2.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar = dIter2->second;

#endif

#ifdef SUPPORT_LAZY_MEMORY
STATIC_ACCESSOR(pmSubscriptionManager::shadowMemMapType, pmSubscriptionManager, GetShadowMemMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmSubscriptionManager::mShadowMemLock"), pmSubscriptionManager, GetShadowMemLock)
#endif

pmSubscriptionManager::pmSubscriptionManager(pmTask* pTask)
	: mTask(pTask)
    , mSubtaskMapVector(pmStubManager::GetStubManager()->GetStubCount())
#ifdef SUPPORT_SPLIT_SUBTASKS
    , mSplitSubtaskMapVector(pmStubManager::GetStubManager()->GetStubCount())
#endif
{
}

void pmSubscriptionManager::DropAllSubscriptions()
{
    for_each(mSubtaskMapVector, [] (decltype(mSubtaskMapVector)::value_type& pPair)
    {
        pPair.first.clear();
    });

#ifdef SUPPORT_SPLIT_SUBTASKS
    for_each(mSplitSubtaskMapVector, [] (decltype(mSplitSubtaskMapVector)::value_type& pPair)
    {
        pPair.first.clear();
    });
#endif
}
    
void pmSubscriptionManager::MoveConstantSubtaskDataToPreprocessorTaskHoldings()
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    #ifdef _DEBUG
        for_each(mSplitSubtaskMapVector, [] (decltype(mSplitSubtaskMapVector)::value_type& pPair)
        {
            EXCEPTION_ASSERT(pPair.first.empty());
        });
    #endif
#endif
    
    for_each(mSubtaskMapVector, [&] (decltype(mSubtaskMapVector)::value_type& pPair)
    {
        for_each(pPair.first, [&] (subtaskMapType::value_type& pMapPair)
        {
            mPreprocessorTaskHoldings.emplace(mTask->GetLogicalSubtaskId(pMapPair.first), std::move(pMapPair.second));
        });
    });

    DropAllSubscriptions();
}
    
bool pmSubscriptionManager::HasSubtask(pm::pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    if(pSplitInfo)
    {
        std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSplitSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
        return (lPair.first.find(std::make_pair(pSubtaskId, *pSplitInfo)) != lPair.first.end());
    }
    else
#endif
    {
        std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

        return (lPair.first.find(pSubtaskId) != lPair.first.end());
    }
}

void pmSubscriptionManager::EraseSubtask(pm::pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    if(pSplitInfo)
    {
        std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSplitSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
        lPair.first.erase(std::make_pair(pSubtaskId, *pSplitInfo));
    }
    else
#endif
    {
        std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

        lPair.first.erase(pSubtaskId);
    }
}

void pmSubscriptionManager::InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    if(pSplitInfo)
    {
        std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSplitSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

        std::pair<ulong, pmSplitInfo> lDataPair(pSubtaskId, *pSplitInfo);
        if(lPair.first.find(lDataPair) != lPair.first.end())
            PMTHROW(pmFatalErrorException());
        
        lPair.first.emplace(std::piecewise_construct, std::forward_as_tuple(lDataPair), std::forward_as_tuple(pmSubtask(mTask)));
    }
    else
#endif
    {
        std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

        if(lPair.first.find(pSubtaskId) != lPair.first.end())
            PMTHROW(pmFatalErrorException());
        
        lPair.first.emplace(pSubtaskId, pmSubtask(mTask));
    }
}
    
void pmSubscriptionManager::FindSubtaskMemDependencies(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pNoJmpBuf /* = false */)
{
#ifdef SUPPORT_SPLIT_SUBTASKS
    if(pSplitInfo)
    {
        std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSplitSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

        std::pair<ulong, pmSplitInfo> lDataPair(pSubtaskId, *pSplitInfo);

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
          
            splitSubtaskMapType::iterator lIter = lPair.first.find(lDataPair);
            if(lIter == lPair.first.end())
                lIter = lPair.first.emplace(std::piecewise_construct, std::forward_as_tuple(lDataPair), std::forward_as_tuple(pmSubtask(mTask))).first;
            
            if(lIter->second.mReadyForExecution)
                return;
        }
        
        const pmDataDistributionCB* lCallback = mTask->GetCallbackUnit()->GetDataDistributionCB();
        if(lCallback)
        {
            if(pNoJmpBuf)
                lCallback->InvokeDirect(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status
            else
                lCallback->Invoke(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status
        }
        
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
            
            pmSubtask& lSubtask = lPair.first.find(lDataPair)->second;

            if(!lSubtask.mReadyForExecution)
            {
                InsertScatteredSubscriptionsToSubtaskMapsInternal(lSubtask);
                lSubtask.mReadyForExecution = true;
            }
        }
    }
    else
#endif
    {
        std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

        subtaskMapType::iterator lPreprocessorTaskHoldingsIterator;
        if(mTask->GetSchedulingModel() == scheduler::PULL_WITH_AFFINITY && pStub->GetType() == pmDeviceType::CPU && ((lPreprocessorTaskHoldingsIterator = mPreprocessorTaskHoldings.find(pSubtaskId)) != mPreprocessorTaskHoldings.end()) && lPreprocessorTaskHoldingsIterator->second.mValid)
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
            
            subtaskMapType::iterator lIter = lPair.first.find(pSubtaskId);
            if(lIter == lPair.first.end())
                lIter = lPair.first.emplace(pSubtaskId, std::move(lPreprocessorTaskHoldingsIterator->second)).first;    // The move constructor sets mValid (of the subtask in mPreprocessorTaskHoldings) to false, preventing its further use

            lIter->second.mReadyForExecution = true;
        }
        else
        {
            // Auto lock/unlock scope
            {
                FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

                subtaskMapType::iterator lIter = lPair.first.find(pSubtaskId);
                if(lIter == lPair.first.end())
                    lIter = lPair.first.emplace(pSubtaskId, pmSubtask(mTask)).first;
                
                if(lIter->second.mReadyForExecution)
                    return;
            }

            const pmDataDistributionCB* lCallback = mTask->GetCallbackUnit()->GetDataDistributionCB();
            if(lCallback)
            {
                if(pNoJmpBuf)
                    lCallback->InvokeDirect(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status
                else
                    lCallback->Invoke(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status
            }

            // Auto lock/unlock scope
            {
                FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

                pmSubtask& lSubtask = lPair.first.find(pSubtaskId)->second;

                if(!lSubtask.mReadyForExecution)
                {
                    InsertScatteredSubscriptionsToSubtaskMapsInternal(lSubtask);
                    lSubtask.mReadyForExecution = true;
                }
            }
        }
    }
}

void pmSubscriptionManager::RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo)
{
    if(!pSubscriptionInfo.length)
        return;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    if(pSubscriptionType == READ_WRITE_SUBSCRIPTION)
    {
        RegisterSubscriptionInternal(lSubtask, pMemIndex, READ_SUBSCRIPTION, pSubscriptionInfo);
        RegisterSubscriptionInternal(lSubtask, pMemIndex, WRITE_SUBSCRIPTION, pSubscriptionInfo);
    }
    else
    {
        RegisterSubscriptionInternal(lSubtask, pMemIndex, pSubscriptionType, pSubscriptionInfo);
    }
}

void pmSubscriptionManager::RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
{
    if(!pScatteredSubscriptionInfo.size || !pScatteredSubscriptionInfo.count)
        return;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    if(pSubscriptionType == READ_WRITE_SUBSCRIPTION)
    {
        RegisterSubscriptionInternal(lSubtask, pMemIndex, READ_SUBSCRIPTION, pScatteredSubscriptionInfo);
        RegisterSubscriptionInternal(lSubtask, pMemIndex, WRITE_SUBSCRIPTION, pScatteredSubscriptionInfo);
    }
    else
    {
        RegisterSubscriptionInternal(lSubtask, pMemIndex, pSubscriptionType, pScatteredSubscriptionInfo);
    }
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::RegisterSubscriptionInternal(pmSubtask& pSubtask, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo)
{
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(pMemIndex);
    CheckAppropriateSubscription(lAddressSpace, pSubscriptionType);

    pmSubtaskAddressSpaceData& lAddressSpaceData = pSubtask.mAddressSpacesData[pMemIndex];
    pmSubtaskSubscriptionData& lSubscriptionData = (IsReadSubscription(pSubscriptionType) ? lAddressSpaceData.mReadSubscriptionData : lAddressSpaceData.mWriteSubscriptionData);

    if(IsReadSubscription(pSubscriptionType))
        lAddressSpaceData.mReadSubscriptionInfoVector.emplace_back(pSubscriptionInfo);
    else
        lAddressSpaceData.mWriteSubscriptionInfoVector.emplace_back(pSubscriptionInfo);

    subscriptionRecordType& lMap = lSubscriptionData.mSubscriptionRecords;
    pmSubscriptionInfo& lConsolidatedSubscription = lSubscriptionData.mConsolidatedSubscriptions;
    
    subscriptionRecordType::iterator lIter = lMap.find(pSubscriptionInfo.offset);
    if(lIter != lMap.end() && lIter->second.first == pSubscriptionInfo.length)
        return;     // Subscription information already present

    if(lMap.empty())
    {
        lConsolidatedSubscription = pSubscriptionInfo;
    }
    else
    {
        size_t lExistingOffset = lConsolidatedSubscription.offset;
        size_t lExistingLength = lConsolidatedSubscription.length;
        size_t lExistingSpan = lExistingOffset + lExistingLength;
        size_t lNewOffset = pSubscriptionInfo.offset;
        size_t lNewLength = pSubscriptionInfo.length;
        size_t lNewSpan = lNewOffset + lNewLength;
        
        lConsolidatedSubscription.offset = std::min(lExistingOffset, lNewOffset);
        lConsolidatedSubscription.length = (std::max(lExistingSpan, lNewSpan) - lConsolidatedSubscription.offset);
    }
    
    AddSubscriptionRecordToMap(pSubscriptionInfo, lMap);
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::RegisterSubscriptionInternal(pmSubtask& pSubtask, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
{
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(pMemIndex);
    CheckAppropriateSubscription(lAddressSpace, pSubscriptionType);
    
    if(lAddressSpace->GetAddressSpaceType() == ADDRESS_SPACE_2D && pScatteredSubscriptionInfo.size > lAddressSpace->GetCols())
        PMTHROW(pmFatalErrorException());

    pmSubtaskAddressSpaceData& lAddressSpaceData = pSubtask.mAddressSpacesData[pMemIndex];
    pmSubtaskSubscriptionData& lSubscriptionData = (IsReadSubscription(pSubscriptionType) ? lAddressSpaceData.mReadSubscriptionData : lAddressSpaceData.mWriteSubscriptionData);
    pmSubscriptionInfo& lConsolidatedSubscription = lSubscriptionData.mConsolidatedSubscriptions;

    if(IsReadSubscription(pSubscriptionType))
        lAddressSpaceData.mScatteredReadSubscriptionInfoVector.emplace_back(pScatteredSubscriptionInfo);
    else
        lAddressSpaceData.mScatteredWriteSubscriptionInfoVector.emplace_back(pScatteredSubscriptionInfo);

    if(lConsolidatedSubscription.length == 0)
    {
        lConsolidatedSubscription = pmSubscriptionInfo(pScatteredSubscriptionInfo.offset, pScatteredSubscriptionInfo.step * pScatteredSubscriptionInfo.count);
    }
    else
    {
        size_t lExistingOffset = lConsolidatedSubscription.offset;
        size_t lExistingLength = lConsolidatedSubscription.length;
        size_t lExistingSpan = lExistingOffset + lExistingLength;
        size_t lNewOffset = pScatteredSubscriptionInfo.offset;
        size_t lNewLength = pScatteredSubscriptionInfo.step * pScatteredSubscriptionInfo.count;
        size_t lNewSpan = lNewOffset + lNewLength;
        
        lConsolidatedSubscription.offset = std::min(lExistingOffset, lNewOffset);
        lConsolidatedSubscription.length = (std::max(lExistingSpan, lNewSpan) - lConsolidatedSubscription.offset);
    }
}
    
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::InsertScatteredSubscriptionsToSubtaskMapsInternal(subscription::pmSubtask& pSubtask)
{
    auto lSameStepLambda = [] (const std::vector<pmScatteredSubscriptionInfo>& pVector) -> bool
    {
        auto lIter = pVector.begin(), lEndIter = pVector.end();
        
        unsigned long lCurrentStep = lIter->step;
        ++lIter;
        
        for(; lIter != lEndIter; ++lIter)
        {
            if(lIter->step != lCurrentStep)
                return false;   // return from lambda
        }
        
        return true;
    };
    
    auto lExhaustiveLambda = [&] (subscriptionRecordType& pMap, const std::vector<pmScatteredSubscriptionInfo>& pVector)
    {
        for_each(pVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
        {
            for(size_t i = 0; i < pScatteredSubscriptionInfo.count; ++i)
                AddSubscriptionRecordToMap(pmSubscriptionInfo(pScatteredSubscriptionInfo.offset + i * pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.size), pMap);
        });
    };

    auto lFastLambda = [&] (subscriptionRecordType& pMap, const std::vector<pmScatteredSubscriptionInfo>& pVector)
    {
        namespace bg = boost::geometry;
        namespace bgi = boost::geometry::index;

        bgi::rtree<bg::model::box<bg::model::point<ulong, 2, bg::cs::cartesian>>, bgi::quadratic<16>> lRtree;

        auto lIter = pVector.begin(), lEndIter = pVector.end();
        for(; lIter != lEndIter; ++lIter)
        {
            const pmScatteredSubscriptionInfo& lScatteredSubscriptionInfo = *lIter;

            bg::model::point<ulong, 2, bg::cs::cartesian> lPoint1(lScatteredSubscriptionInfo.offset % lScatteredSubscriptionInfo.step, lScatteredSubscriptionInfo.offset / lScatteredSubscriptionInfo.step);
            bg::model::point<ulong, 2, bg::cs::cartesian> lPoint2(lPoint1.get<0>() + lScatteredSubscriptionInfo.size - 1, lPoint1.get<1>() + lScatteredSubscriptionInfo.count - 1);
            
            bg::model::box<bg::model::point<ulong, 2, bg::cs::cartesian>> lBox(lPoint1, lPoint2);

            std::vector<bg::model::box<bg::model::point<ulong, 2, bg::cs::cartesian>>> lOverlappingSubscriptions;
            lRtree.query(bgi::intersects(lBox), std::back_inserter(lOverlappingSubscriptions));

            if(lOverlappingSubscriptions.empty())
            {
                lRtree.insert(lBox);
            }
            else
            {
                // This code needs to be improved by breaking overlapping rects into non-overlapping ones
                lExhaustiveLambda(pMap, pVector);
                return;
            }
        }
        
        std::vector<bg::model::box<bg::model::point<ulong, 2, bg::cs::cartesian>>> lResolvedSubscriptions;
        bg::model::box<bg::model::point<ulong, 2, bg::cs::cartesian>> lBox(bg::model::point<ulong, 2, bg::cs::cartesian>(0, 0), bg::model::point<ulong, 2, bg::cs::cartesian>(std::numeric_limits<ulong>::max(), std::numeric_limits<ulong>::max()));
        lRtree.query(bgi::intersects(lBox), std::back_inserter(lResolvedSubscriptions));

        unsigned long lStep = pVector.begin()->step;
        for_each(lResolvedSubscriptions, [&] (const bg::model::box<bg::model::point<unsigned long, 2, bg::cs::cartesian>>& pBox)
        {
            auto lPoint1 = pBox.min_corner();
            auto lPoint2 = pBox.max_corner();
            
            pmScatteredSubscriptionInfo lScatteredSubscription(lPoint1.get<1>() * lStep + lPoint1.get<0>(), lPoint2.get<0>() - lPoint1.get<0>() + 1, lStep, lPoint2.get<1>() - lPoint1.get<1>() + 1);
            
            for(size_t i = 0; i < lScatteredSubscription.count; ++i)
                pMap[lScatteredSubscription.offset + i * lScatteredSubscription.step].first = lScatteredSubscription.size;
        });
    };
    
    multi_for_each(mTask->GetAddressSpaces(), pSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
    {
        if(!pAddressSpaceData.mScatteredReadSubscriptionInfoVector.empty())
        {
            if(pAddressSpaceData.mReadSubscriptionInfoVector.empty() && lSameStepLambda(pAddressSpaceData.mScatteredReadSubscriptionInfoVector))
                lFastLambda(pAddressSpaceData.mReadSubscriptionData.mSubscriptionRecords, pAddressSpaceData.mScatteredReadSubscriptionInfoVector);
            else
                lExhaustiveLambda(pAddressSpaceData.mReadSubscriptionData.mSubscriptionRecords, pAddressSpaceData.mScatteredReadSubscriptionInfoVector);
        }

        if(!pAddressSpaceData.mScatteredWriteSubscriptionInfoVector.empty())
        {
            if(pAddressSpaceData.mWriteSubscriptionInfoVector.empty() && lSameStepLambda(pAddressSpaceData.mScatteredWriteSubscriptionInfoVector))
                lFastLambda(pAddressSpaceData.mWriteSubscriptionData.mSubscriptionRecords, pAddressSpaceData.mScatteredWriteSubscriptionInfoVector);
            else
                lExhaustiveLambda(pAddressSpaceData.mWriteSubscriptionData.mSubscriptionRecords, pAddressSpaceData.mScatteredWriteSubscriptionInfoVector);
        }
    });
}
    
pmSubscriptionFormat pmSubscriptionManager::GetSubscriptionFormat(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    pmSubtaskAddressSpaceData& lData = lSubtask.mAddressSpacesData[pMemIndex];
    
    size_t lScatteredSize = lData.mScatteredReadSubscriptionInfoVector.size() + lData.mScatteredWriteSubscriptionInfoVector.size();
    size_t lNonScatteredSize = lData.mReadSubscriptionInfoVector.size() + lData.mWriteSubscriptionInfoVector.size();
    
    if(lScatteredSize && lNonScatteredSize)
        return SUBSCRIPTION_GENERAL;
    else if(lNonScatteredSize == 1)
        return SUBSCRIPTION_CONTIGUOUS;
    else if(lScatteredSize)
        return SUBSCRIPTION_SCATTERED;

    return SUBSCRIPTION_GENERAL;
}

std::vector<pmScatteredSubscriptionInfo> pmSubscriptionManager::GetUnifiedScatteredSubscriptionInfoVector(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    DEBUG_EXCEPTION_ASSERT(GetSubscriptionFormat(pStub, pSubtaskId, pSplitInfo, pMemIndex) == SUBSCRIPTION_SCATTERED);

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    pmSubtaskAddressSpaceData& lAddressSpaceData = lSubtask.mAddressSpacesData[pMemIndex];
    std::vector<pmScatteredSubscriptionInfo>& lReadVector = lAddressSpaceData.mScatteredReadSubscriptionInfoVector;
    std::vector<pmScatteredSubscriptionInfo>& lWriteVector = lAddressSpaceData.mScatteredWriteSubscriptionInfoVector;

    std::vector<pmScatteredSubscriptionInfo> lVector(lReadVector.begin(), lReadVector.end());
    std::copy(lWriteVector.begin(), lWriteVector.end(), std::back_inserter(lVector));
    
    return lVector;
}

void pmSubscriptionManager::ProcessScatteredWriteSubscriptionsInfoVector(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, const std::function<void (const pmScatteredSubscriptionInfo&)>& pFunc)
{
    DEBUG_EXCEPTION_ASSERT(GetSubscriptionFormat(pStub, pSubtaskId, pSplitInfo, pMemIndex) == SUBSCRIPTION_SCATTERED);

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    pmSubtaskAddressSpaceData& lAddressSpaceData = lSubtask.mAddressSpacesData[pMemIndex];
    for_each(lAddressSpaceData.mScatteredWriteSubscriptionInfoVector, [&pFunc] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
    {
        pFunc(pScatteredSubscriptionInfo);
    });
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
subscriptionRecordType pmSubscriptionManager::GetNonConsolidatedReadWriteSubscriptionsAsMapInternal(pmSubtask& pSubtask, uint pMemIndex)
{
    subscriptionRecordType lMap;

    subscriptionRecordType& lReadMap = pSubtask.mAddressSpacesData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords;
    subscriptionRecordType& lWriteMap = pSubtask.mAddressSpacesData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords;
    
    if(!lReadMap.empty() && !lWriteMap.empty())
    {
        lMap.insert(lReadMap.begin(), lReadMap.end());
        
        for_each(lWriteMap, [this, &lMap] (subscriptionRecordType::value_type& pPair)
        {
            AddSubscriptionRecordToMap(pmSubscriptionInfo(pPair.first, pPair.second.first), lMap);
        });
    }
    else
    {
        if(!lReadMap.empty())
            return lReadMap;
        else
            return lWriteMap;
    }

    return lMap;
}
    
std::vector<pmSubscriptionInfo> pmSubscriptionManager::GetNonConsolidatedReadWriteSubscriptions(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    std::vector<pmSubscriptionInfo> lVector;
    
    subscriptionRecordType* lTargetMap = NULL;
    subscriptionRecordType lMap;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    subscriptionRecordType& lReadMap = lSubtask.mAddressSpacesData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords;
    subscriptionRecordType& lWriteMap = lSubtask.mAddressSpacesData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords;
    
    if(!lReadMap.empty() && !lWriteMap.empty())
    {
        lMap.insert(lReadMap.begin(), lReadMap.end());
        
        for_each(lWriteMap, [this, &lMap] (subscriptionRecordType::value_type& pPair)
        {
            AddSubscriptionRecordToMap(pmSubscriptionInfo(pPair.first, pPair.second.first), lMap);
        });
        
        lTargetMap = &lMap;
    }
    else
    {
        if(!lReadMap.empty())
            lTargetMap = &lReadMap;
        else
            lTargetMap = &lWriteMap;
    }

    lVector.reserve(lTargetMap->size());
    for_each(*lTargetMap, [&lVector] (subscriptionRecordType::value_type& pPair)
    {
        lVector.emplace_back(pPair.first, pPair.second.first);
    });
    
    return lVector;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::AddSubscriptionRecordToMap(const pmSubscriptionInfo& pSubscriptionInfo, subscriptionRecordType& pMap)
{
    /* Only add the region which is yet not subscribed */
    subscriptionRecordType::iterator lStartIter, lEndIter;
    subscriptionRecordType::iterator* lStartIterAddr = &lStartIter;
    subscriptionRecordType::iterator* lEndIterAddr = &lEndIter;
    
    size_t lFirstAddr = pSubscriptionInfo.offset;
    size_t lLastAddr = pSubscriptionInfo.offset + pSubscriptionInfo.length - 1;
    
    FIND_FLOOR_ELEM(subscriptionRecordType, pMap, lFirstAddr, lStartIterAddr);
    FIND_FLOOR_ELEM(subscriptionRecordType, pMap, lLastAddr, lEndIterAddr);
    
    if(!lStartIterAddr && !lEndIterAddr)
    {
        pMap[lFirstAddr].first = pSubscriptionInfo.length;
    }
    else
    {
        std::vector<std::pair<size_t, size_t> > lRangesToBeAdded;
        if(!lStartIterAddr)
        {
            lStartIter = pMap.begin();
            lRangesToBeAdded.push_back(std::make_pair(lFirstAddr, lStartIter->first - 1));
            lFirstAddr = lStartIter->first;
        }
        
        bool lStartInside = ((lFirstAddr >= lStartIter->first) && (lFirstAddr < (lStartIter->first + lStartIter->second.first)));
        bool lEndInside = ((lLastAddr >= lEndIter->first) && (lLastAddr < (lEndIter->first + lEndIter->second.first)));

        if(lStartIter == lEndIter)
        {
            if(lStartInside && lEndInside)
            {
            }
            else if(lStartInside && !lEndInside)
            {
                lRangesToBeAdded.push_back(std::make_pair(lStartIter->first + lStartIter->second.first, lLastAddr));
            }
            else
            {
                lRangesToBeAdded.push_back(std::make_pair(lFirstAddr, lLastAddr));
            }
        }
        else
        {
            if(!lStartInside)
            {
                ++lStartIter;
                lRangesToBeAdded.push_back(std::make_pair(lFirstAddr, lStartIter->first-1));
            }
            
            if(!lEndInside)
            {
                lRangesToBeAdded.push_back(std::make_pair(lEndIter->first + lEndIter->second.first, lLastAddr));
            }
            
            if(lStartIter != lEndIter)
            {
                for(subscriptionRecordType::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
                {
                    subscriptionRecordType::iterator lNextIter = lTempIter;
                    ++lNextIter;
                    lRangesToBeAdded.push_back(std::make_pair(lTempIter->first + lTempIter->second.first, lNextIter->first-1));
                }
            }
        }
        
        std::vector<std::pair<size_t, size_t> >::iterator lInnerIter = lRangesToBeAdded.begin(), lInnerEndIter = lRangesToBeAdded.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            size_t lRangeOffset = (*lInnerIter).first;
            size_t lRangeLastAddr = (*lInnerIter).second;
            size_t lRangeLength = (*lInnerIter).second - (*lInnerIter).first + 1;
            if(lRangeLength)
            {
                subscriptionRecordType::iterator lPrevIter, lNextIter;
                subscriptionRecordType::iterator* lPrevIterAddr = &lPrevIter;
                subscriptionRecordType::iterator* lNextIterAddr = &lNextIter;

                // Combine with previous range
                FIND_FLOOR_ELEM(subscriptionRecordType, pMap, lRangeOffset - 1, lPrevIterAddr);
                if(lPrevIterAddr && (lPrevIter->first + lPrevIter->second.first == lRangeOffset))
                {
                    lRangeOffset = lPrevIter->first;
                    lRangeLength = lRangeLastAddr - lRangeOffset + 1;
                    pMap.erase(lPrevIter);
                }
                
                // Combine with following range
                FIND_FLOOR_ELEM(subscriptionRecordType, pMap, lRangeLastAddr + 1, lNextIterAddr);
                if(lNextIterAddr && (lNextIter->first == lRangeLastAddr + 1))
                {
                    lRangeLastAddr = lNextIter->first + lNextIter->second.first - 1;
                    lRangeLength = lRangeLastAddr - lRangeOffset + 1;
                    pMap.erase(lNextIter);
                }

                pMap[lRangeOffset].first = lRangeLength;
            }
        }
    }
}

void pmSubscriptionManager::SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
#ifdef SUPPORT_CUDA
    EXCEPTION_ASSERT(pStub->GetType() == GPU_CUDA);
#else
    PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mCudaLaunchConf = pCudaLaunchConf;
}

void pmSubscriptionManager::RegisterRedistribution(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmRedistributor& pRedistributor, ulong pOffset, ulong pLength, uint pOrder)
{
    if(!pLength)
        return;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    pmSubscriptionInfo lSubscriptionInfo = GetConsolidatedWriteSubscriptionInternal(pStub, lSubtask, pRedistributor.GetAddressSpaceIndex());
    EXCEPTION_ASSERT(lSubscriptionInfo.length);

    size_t lGlobalOffset = lSubscriptionInfo.offset + pOffset;
    EXCEPTION_ASSERT(lGlobalOffset < mTask->GetAddressSpace(pRedistributor.GetAddressSpaceIndex())->GetLength());

    auto& lOrderData = lSubtask.mAddressSpacesData[pRedistributor.GetAddressSpaceIndex()].mRedistributionData[pOrder];
    lOrderData.entryVector.emplace_back(lGlobalOffset, pLength);
    lOrderData.orderLength += pLength;
}

void pmSubscriptionManager::ConsolidateRedistributionRecords(pmRedistributor& pRedistributor, redistribution::localRedistributionData& pData)
{
    auto lLambda = [&] (std::map<uint, pmAddressSpaceRedistributionData>& pSubtaskData)
    {
        for_each(pSubtaskData, [&] (std::pair<const uint, pmAddressSpaceRedistributionData>& pPair)
        {
            std::move(pPair.second.entryVector.begin(), pPair.second.entryVector.end(), std::back_inserter(pData.mLocalRedistributionMap[pPair.first]));
        });
    };
    
    std::map<uint, std::pair<size_t, ulong>> lCollectiveOrderData;  // order no. versus pair of (no. of redistributions registered and total length of redistributions)

#ifdef SUPPORT_SPLIT_SUBTASKS
    for_each(mSplitSubtaskMapVector, [&] (decltype(mSplitSubtaskMapVector)::value_type& pPair)
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pPair.second, Lock(), Unlock());
        
        for_each(pPair.first, [&] (decltype(pPair.first)::value_type& pSubtaskMapPair)
        {
            for_each(pSubtaskMapPair.second.mAddressSpacesData[pRedistributor.GetAddressSpaceIndex()].mRedistributionData, [&] (std::pair<const uint, pmAddressSpaceRedistributionData>& pInternalPair)
            {
                auto lIter = lCollectiveOrderData.find(pInternalPair.first);
                if(lIter == lCollectiveOrderData.end())
                    lIter = lCollectiveOrderData.emplace(std::piecewise_construct, std::forward_as_tuple(pInternalPair.first), std::forward_as_tuple(0, 0)).first;
                
                ++lIter->second.first;
                lIter->second.second += pInternalPair.second.orderLength;
            });
        });
    });
#endif

    for_each(mSubtaskMapVector, [&] (decltype(mSubtaskMapVector)::value_type& pPair)
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pPair.second, Lock(), Unlock());
        
        for_each(pPair.first, [&] (decltype(pPair.first)::value_type& pSubtaskMapPair)
        {
            for_each(pSubtaskMapPair.second.mAddressSpacesData[pRedistributor.GetAddressSpaceIndex()].mRedistributionData, [&] (std::pair<const uint, pmAddressSpaceRedistributionData>& pInternalPair)
            {
                auto lIter = lCollectiveOrderData.find(pInternalPair.first);
                if(lIter == lCollectiveOrderData.end())
                    lIter = lCollectiveOrderData.emplace(std::piecewise_construct, std::forward_as_tuple(pInternalPair.first), std::forward_as_tuple(0, 0)).first;
                
                ++lIter->second.first;
                lIter->second.second += pInternalPair.second.orderLength;
            });
        });
    });
    
    pData.mLocalRedistributionVector.reserve(lCollectiveOrderData.size());
    
    for_each(lCollectiveOrderData, [&] (typename decltype(lCollectiveOrderData)::value_type& pPair)
    {
        pData.mLocalRedistributionVector.emplace_back(pPair.first, pPair.second.second);
        pData.mLocalRedistributionMap[pPair.first].reserve(pPair.second.first);
    });

#ifdef SUPPORT_SPLIT_SUBTASKS
    for_each(mSplitSubtaskMapVector, [&] (decltype(mSplitSubtaskMapVector)::value_type& pPair)
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pPair.second, Lock(), Unlock());
        
        for_each(pPair.first, [&] (decltype(pPair.first)::value_type& pSubtaskMapPair)
        {
            lLambda(pSubtaskMapPair.second.mAddressSpacesData[pRedistributor.GetAddressSpaceIndex()].mRedistributionData);
        });
    });
#endif

    for_each(mSubtaskMapVector, [&] (decltype(mSubtaskMapVector)::value_type& pPair)
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pPair.second, Lock(), Unlock());
        
        for_each(pPair.first, [&] (decltype(pPair.first)::value_type& pSubtaskMapPair)
        {
            lLambda(pSubtaskMapPair.second.mAddressSpacesData[pRedistributor.GetAddressSpaceIndex()].mRedistributionData);
        });
    });
}
    
void pmSubscriptionManager::ReserveCudaGlobalMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
#ifdef SUPPORT_CUDA
    EXCEPTION_ASSERT(pStub->GetType() == GPU_CUDA);
#else
        PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mReservedCudaGlobalMemSize = pSize;
}

pmCudaLaunchConf& pmSubscriptionManager::GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

	return lSubtask.mCudaLaunchConf;
}
    
size_t pmSubscriptionManager::GetReservedCudaGlobalMemSize(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

	return lSubtask.mReservedCudaGlobalMemSize;
}

void pmSubscriptionManager::DeleteScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    lSubtask.mScratchBuffers.erase(pScratchBufferType);
}

void* pmSubscriptionManager::GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    EXCEPTION_ASSERT(((size_t)(lSubtask.mScratchBuffers.find(PRE_SUBTASK_TO_SUBTASK) != lSubtask.mScratchBuffers.end()) + (size_t)(lSubtask.mScratchBuffers.find(SUBTASK_TO_POST_SUBTASK) != lSubtask.mScratchBuffers.end()) + (size_t)(lSubtask.mScratchBuffers.find(PRE_SUBTASK_TO_POST_SUBTASK) != lSubtask.mScratchBuffers.end())) <= 1);    // Only one of these can exist (The CUDA code currently does not copy more than one in and out of the device

    auto lIter = lSubtask.mScratchBuffers.find(pScratchBufferType);
    if(lIter == lSubtask.mScratchBuffers.end())
    {
        if(!pBufferSize)
            return NULL;    // Calling this function with a zero scratch buffer size serves as a user check if the scratch buffer already exists or not
            
        lIter = lSubtask.mScratchBuffers.emplace(std::piecewise_construct, std::forward_as_tuple(pScratchBufferType), std::forward_as_tuple(new char[pBufferSize], pBufferSize)).first;
    }
    
    return lIter->second.mScratchBuffer.get_ptr();
}

void* pmSubscriptionManager::CheckAndGetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType, size_t& pScratchBufferSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    auto lIter = lSubtask.mScratchBuffers.find(pScratchBufferType);
    if(lIter == lSubtask.mScratchBuffers.end())
        return NULL;

	char* lScratchBuffer = lIter->second.mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
        return NULL;

    pScratchBufferSize = lIter->second.mScratchBufferSize;

    return lScratchBuffer;
}
    
void pmSubscriptionManager::DropScratchBuffersNotRequiredPostSubtaskExec(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    lSubtask.mScratchBuffers.erase(PRE_SUBTASK_TO_SUBTASK);
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType)
{
    if(pStub1 == pStub2)
        return SubtasksHaveMatchingSubscriptionsCommonStub(pStub1, pSubtaskId1, pSplitInfo1, pSubtaskId2, pSplitInfo2, pSubscriptionType);

    return SubtasksHaveMatchingSubscriptionsDifferentStubs(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pSubscriptionType);
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsCommonStub(pmExecutionStub* pStub, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType)
{
    GET_SUBTASKS(lSubtask1, lSubtask2, pStub, pSubtaskId1, pSubtaskId2, pSplitInfo1, pSplitInfo2);
    
    return SubtasksHaveMatchingSubscriptionsInternal(lSubtask1, lSubtask2, pSubscriptionType);
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsDifferentStubs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType)
{
    GET_SUBTASK(lSubtask1, pStub1, pSubtaskId1, pSplitInfo1);
    GET_SUBTASK2(lSubtask2, pStub2, pSubtaskId2, pSplitInfo2);
    
    return SubtasksHaveMatchingSubscriptionsInternal(lSubtask1, lSubtask2, pSubscriptionType);
}

/* Must be called with mSubtaskMapVector stub's lock acquired for both subtasks */
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsInternal(const pmSubtask& pSubtask1, const pmSubtask& pSubtask2, pmSubscriptionType pSubscriptionType) const
{
#ifdef _DEBUG
    if(pSubtask1.mAddressSpacesData.size() != pSubtask2.mAddressSpacesData.size())
        PMTHROW(pmFatalErrorException());   // both subtasks belong to same task; so must have same address space entries
#endif
    
    std::vector<pmSubtaskAddressSpaceData>::const_iterator lIter1 = pSubtask1.mAddressSpacesData.begin(), lEndIter1 = pSubtask2.mAddressSpacesData.end();
    std::vector<pmSubtaskAddressSpaceData>::const_iterator lIter2 = pSubtask2.mAddressSpacesData.begin();
    
    for(; lIter1 != lEndIter1; ++lIter1, ++lIter2)
    {
        if(!AddressSpacesHaveMatchingSubscriptionsInternal((*lIter1), (*lIter2)))
            return false;
    }
    
    return true;
}
    
float pmSubscriptionManager::FindPercentLocalInputDataForSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    ulong lTotalInputSubscriptionSize = 0;
    ulong lLocalDataSize = 0;

    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace) { return mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace); },
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pFilteredIndex)
    {
        uint lMemIndex = (uint)pAddressSpaceIndex;

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, lMemIndex, lBeginIter, lEndIter);
        
        for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
        {
            lTotalInputSubscriptionSize += lIter->second.first;
            lLocalDataSize += pAddressSpace->FindLocalDataSizeUnprotected(lIter->first, lIter->second.first);
        }
    });

    EXCEPTION_ASSERT(lLocalDataSize <= lTotalInputSubscriptionSize);
    return ((float)lLocalDataSize / (float)lTotalInputSubscriptionSize * 100.0);
}
    
ulong pmSubscriptionManager::FindLocalInputDataSizeForSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    ulong lLocalDataSize = 0;

    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace) { return mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace); },
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pFilteredIndex)
    {
        uint lMemIndex = (uint)pAddressSpaceIndex;

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, lMemIndex, lBeginIter, lEndIter);
        
        for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
            lLocalDataSize += pAddressSpace->FindLocalDataSizeUnprotected(lIter->first, lIter->second.first);
    });

    return lLocalDataSize;
}

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
void pmSubscriptionManager::FindLocalInputDataSizeForSubtaskOnMachines(pmExecutionStub* pStub, ulong pSubtaskId, const std::vector<const pmMachine*>& pMachinesVector, ulong* pDataArray, size_t pStepSizeInBytes /* = 0 */)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace) { return mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace); },
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pFilteredIndex)
    {
        uint lMemIndex = (uint)pAddressSpaceIndex;

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, lMemIndex, lBeginIter, lEndIter);
        
        for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
            pAddressSpace->FindLocalDataSizeOnMachinesUnprotected(lIter->first, lIter->second.first, pMachinesVector, pDataArray, pStepSizeInBytes);
    });
}
#endif
    
uint pmSubscriptionManager::FindRemoteDataSourcesForSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    uint lMaxRemoteSources = pmGetHostCount() - 1;
    std::set<const pmMachine*> lRemoteDataSources;

    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace) { return mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace); },
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pFilteredIndex)
    {
        uint lMemIndex = (uint)pAddressSpaceIndex;

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, lMemIndex, lBeginIter, lEndIter);
        
        for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
        {
            std::set<const pmMachine*> lSet = pAddressSpace->FindRemoteDataSourcesUnprotected(lIter->first, lIter->second.first);
            
            for_each(lSet, [&] (const pmMachine* pMachine)
            {
                lRemoteDataSources.emplace(pMachine);
            });
            
            if(lRemoteDataSources.size() == lMaxRemoteSources)
                return;     // return from lambda
        }
    });

    return (uint)lRemoteDataSources.size();
}

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
void pmSubscriptionManager::FindRemoteDataSourcesForSubtaskOnMachines(pmExecutionStub* pStub, ulong pSubtaskId, const std::vector<const pmMachine*>& pMachinesVector, uint* pDataArray, size_t pStepSizeInBytes /* = 0 */)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace) { return mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace); },
    [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pFilteredIndex)
    {
        uint lMemIndex = (uint)pAddressSpaceIndex;

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, lMemIndex, lBeginIter, lEndIter);
        
        for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
            pAddressSpace->FindRemoteDataSourcesOnMachinesUnprotected(lIter->first, lIter->second.first, pMachinesVector, pDataArray, pStepSizeInBytes);
    });
}
#endif
    
ulong pmSubscriptionManager::FindRemoteTransferEventsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    ulong lRemoteTransferEvents = 0;

    multi_for_each(mTask->GetAddressSpaces(), lSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
    {
        if(mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace))
        {
            // Fetch exact scattered subscriptions (these subscriptions are packed at the sender side before transfer and then unpacked by the receiver)
            for_each(pAddressSpaceData.mScatteredReadSubscriptionInfoVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
            {
                lRemoteTransferEvents += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetScatteredMemoryFetchEvents(pAddressSpace, pScatteredSubscriptionInfo);
            });
            
            // For non-scattered subscriptions, we fetch entire memory pages (no packing/unpacking happens here)
            for_each(pAddressSpaceData.mReadSubscriptionInfoVector, [&] (const pmSubscriptionInfo& pSubscriptionInfo)
            {
                lRemoteTransferEvents += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetMemoryFetchEvents(pAddressSpace, pSubscriptionInfo.offset, pSubscriptionInfo.length);
            });
        }
    });

    return lRemoteTransferEvents;
}

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
void pmSubscriptionManager::FindRemoteTransferEventsForSubtaskOnMachines(pmExecutionStub* pStub, ulong pSubtaskId, const std::vector<const pmMachine*>& pMachinesVector, ulong* pDataArray, size_t pStepSizeInBytes /* = 0 */)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    for_each_with_index(pMachinesVector, [&] (const pmMachine* pMachine, size_t pIndex)
    {
        ulong lRemoteTransferEvents = 0;

        multi_for_each(mTask->GetAddressSpaces(), lSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
        {
            if(mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace))
            {
                // Fetch exact scattered subscriptions (these subscriptions are packed at the sender side before transfer and then unpacked by the receiver)
                for_each(pAddressSpaceData.mScatteredReadSubscriptionInfoVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
                {
                    lRemoteTransferEvents += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetScatteredMemoryFetchEvents(pAddressSpace, pScatteredSubscriptionInfo);
                });
                
                // For non-scattered subscriptions, we fetch entire memory pages (no packing/unpacking happens here)
                for_each(pAddressSpaceData.mReadSubscriptionInfoVector, [&] (const pmSubscriptionInfo& pSubscriptionInfo)
                {
                    lRemoteTransferEvents += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetMemoryFetchEvents(pAddressSpace, pSubscriptionInfo.offset, pSubscriptionInfo.length);
                });
            }
        });

        *((ulong*)((char*)pDataArray + (pStepSizeInBytes * pIndex))) = lRemoteTransferEvents;
    });
}
#endif
    
float pmSubscriptionManager::FindRemoteTransferEstimateForSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
    const double lLatencyPerNonScatteredPageFetch = 1.0;
    const double lLatencyPerScatteredPageFetch = 1.3;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    double lScatteredPages = 0;
    double lNonScatteredPages = 0;

    multi_for_each(mTask->GetAddressSpaces(), lSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
    {
        if(mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace))
        {
            // Fetch exact scattered subscriptions (these subscriptions are packed at the sender side before transfer and then unpacked by the receiver)
            for_each(pAddressSpaceData.mScatteredReadSubscriptionInfoVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
            {
                lScatteredPages += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetScatteredMemoryFetchPages(pAddressSpace, pScatteredSubscriptionInfo);
            });
            
            // For non-scattered subscriptions, we fetch entire memory pages (no packing/unpacking happens here)
            for_each(pAddressSpaceData.mReadSubscriptionInfoVector, [&] (const pmSubscriptionInfo& pSubscriptionInfo)
            {
                lNonScatteredPages += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetMemoryFetchPages(pAddressSpace, pSubscriptionInfo.offset, pSubscriptionInfo.length);
            });
        }
    });

    return (float)(lScatteredPages * lLatencyPerScatteredPageFetch + lNonScatteredPages * lLatencyPerNonScatteredPageFetch);
}

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
void pmSubscriptionManager::FindRemoteTransferEstimateForSubtaskOnMachines(pmExecutionStub* pStub, ulong pSubtaskId, const std::vector<const pmMachine*>& pMachinesVector, float* pDataArray, size_t pStepSizeInBytes /* = 0 */)
{
    const double lLatencyPerNonScatteredPageFetch = 1.0;
    const double lLatencyPerScatteredPageFetch = 1.3;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, (pmSplitInfo*)NULL);

    for_each_with_index(pMachinesVector, [&] (const pmMachine* pMachine, size_t pIndex)
    {
        double lScatteredPages = 0;
        double lNonScatteredPages = 0;

        multi_for_each(mTask->GetAddressSpaces(), lSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
        {
            if(mTask->IsReadOnly(pAddressSpace) || mTask->IsReadWrite(pAddressSpace))
            {
                // Fetch exact scattered subscriptions (these subscriptions are packed at the sender side before transfer and then unpacked by the receiver)
                for_each(pAddressSpaceData.mScatteredReadSubscriptionInfoVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
                {
                    lScatteredPages += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetScatteredMemoryFetchPages(pAddressSpace, pScatteredSubscriptionInfo);
                });
                
                // For non-scattered subscriptions, we fetch entire memory pages (no packing/unpacking happens here)
                for_each(pAddressSpaceData.mReadSubscriptionInfoVector, [&] (const pmSubscriptionInfo& pSubscriptionInfo)
                {
                    lNonScatteredPages += MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetMemoryFetchPages(pAddressSpace, pSubscriptionInfo.offset, pSubscriptionInfo.length);
                });
            }
        });

        *((float*)((char*)pDataArray + (pStepSizeInBytes * pIndex))) = (float)(lScatteredPages * lLatencyPerScatteredPageFetch + lNonScatteredPages * lLatencyPerNonScatteredPageFetch);
    });
}
#endif
    
/* Must be called with mSubtaskMapVector stub's lock acquired for both subtasks */
bool pmSubscriptionManager::AddressSpacesHaveMatchingSubscriptionsInternal(const pmSubtaskAddressSpaceData& pAddressSpaceData1, const pmSubtaskAddressSpaceData& pAddressSpaceData2) const
{
    return (pAddressSpaceData1.mReadSubscriptionData == pAddressSpaceData2.mReadSubscriptionData && pAddressSpaceData1.mWriteSubscriptionData == pAddressSpaceData2.mWriteSubscriptionData);
}

void pmSubscriptionManager::GetNonConsolidatedReadSubscriptions(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    GetNonConsolidatedReadSubscriptionsInternal(lSubtask, pMemIndex, pBegin, pEnd);
}

void pmSubscriptionManager::GetNonConsolidatedWriteSubscriptions(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    GetNonConsolidatedWriteSubscriptionsInternal(lSubtask, pMemIndex, pBegin, pEnd);
}

const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedReadSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    return GetConsolidatedReadSubscriptionInternal(pStub, lSubtask, pMemIndex);
}
    
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    return GetConsolidatedWriteSubscriptionInternal(pStub, lSubtask, pMemIndex);
}
    
const pmSubscriptionInfo& pmSubscriptionManager::GetUnifiedReadWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    return GetUnifiedReadWriteSubscriptionInternal(pStub, lSubtask, pMemIndex);
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::GetNonConsolidatedReadSubscriptionsInternal(pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	pBegin = pSubtask.mAddressSpacesData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords.begin();
	pEnd = pSubtask.mAddressSpacesData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords.end();
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::GetNonConsolidatedWriteSubscriptionsInternal(pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	pBegin = pSubtask.mAddressSpacesData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords.begin();
	pEnd = pSubtask.mAddressSpacesData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords.end();
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedReadSubscriptionInternal(const pmExecutionStub* pStub, pmSubtask& pSubtask, uint pMemIndex)
{
    DEBUG_EXCEPTION_ASSERT(mTask->GetAddressSpaceSubscriptionVisibility(mTask->GetAddressSpace(pMemIndex), pStub) != SUBSCRIPTION_COMPACT);

    return pSubtask.mAddressSpacesData[pMemIndex].mReadSubscriptionData.mConsolidatedSubscriptions;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedWriteSubscriptionInternal(const pmExecutionStub* pStub, pmSubtask& pSubtask, uint pMemIndex)
{
    DEBUG_EXCEPTION_ASSERT(mTask->GetAddressSpaceSubscriptionVisibility(mTask->GetAddressSpace(pMemIndex), pStub) != SUBSCRIPTION_COMPACT);

    return pSubtask.mAddressSpacesData[pMemIndex].mWriteSubscriptionData.mConsolidatedSubscriptions;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
const pmSubscriptionInfo& pmSubscriptionManager::GetUnifiedReadWriteSubscriptionInternal(const pmExecutionStub* pStub, pmSubtask& pSubtask, uint pMemIndex)
{
    DEBUG_EXCEPTION_ASSERT(mTask->GetAddressSpaceSubscriptionVisibility(mTask->GetAddressSpace(pMemIndex), pStub) != SUBSCRIPTION_COMPACT);

    pmSubtaskAddressSpaceData& lAddressSpaceData = pSubtask.mAddressSpacesData[pMemIndex];
    
    if(!lAddressSpaceData.mUnifiedSubscription.get_ptr())
    {
        size_t lReadOffset = lAddressSpaceData.mReadSubscriptionData.mConsolidatedSubscriptions.offset;
        size_t lReadLength = lAddressSpaceData.mReadSubscriptionData.mConsolidatedSubscriptions.length;
        size_t lReadSpan = lReadOffset + lReadLength;
        size_t lWriteOffset = lAddressSpaceData.mWriteSubscriptionData.mConsolidatedSubscriptions.offset;
        size_t lWriteLength = lAddressSpaceData.mWriteSubscriptionData.mConsolidatedSubscriptions.length;
        size_t lWriteSpan = lWriteOffset + lWriteLength;
        
        if(!lReadLength && !lWriteLength)
            lAddressSpaceData.mUnifiedSubscription.reset(new pmSubscriptionInfo(0, 0));
        else if(lReadLength && !lWriteLength)
            lAddressSpaceData.mUnifiedSubscription.reset(new pmSubscriptionInfo(lReadOffset, lReadLength));
        else if(!lReadLength && lWriteLength)
            lAddressSpaceData.mUnifiedSubscription.reset(new pmSubscriptionInfo(lWriteOffset, lWriteLength));
        else
            lAddressSpaceData.mUnifiedSubscription.reset(new pmSubscriptionInfo(std::min(lReadOffset, lWriteOffset), std::max(lReadSpan, lWriteSpan) - std::min(lReadOffset, lWriteOffset)));
    }

    return *lAddressSpaceData.mUnifiedSubscription.get_ptr();
}

void pmSubscriptionManager::CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pMem /* = NULL */, size_t pMemLength /* = 0 */, size_t pWriteOnlyUnprotectedRanges /* = 0 */, uint* pUnprotectedRanges /* = NULL */)
{
    DEBUG_EXCEPTION_ASSERT(!GetSubtaskShadowMem(pStub, pSubtaskId, pSplitInfo, pMemIndex));

    const pmSubscriptionVisibilityType lVisibilityType = mTask->GetAddressSpaceSubscriptionVisibility(mTask->GetAddressSpace(pMemIndex), pStub);

    DEBUG_EXCEPTION_ASSERT(mTask->IsWritable(mTask->GetAddressSpace(pMemIndex)) || lVisibilityType == SUBSCRIPTION_COMPACT);

    size_t lTotalLength = 0;
    if(lVisibilityType == SUBSCRIPTION_NATURAL)
        lTotalLength = GetUnifiedReadWriteSubscription(pStub, pSubtaskId, pSplitInfo, pMemIndex).length;
    else
        lTotalLength = GetCompactedSubscription(pStub, pSubtaskId, pSplitInfo, pMemIndex).subscriptionInfo.length;

    if(lTotalLength)
    {
        bool lExplicitAllocation = false;
        char* lShadowMem = reinterpret_cast<char*>(mTask->CheckOutSubtaskMemory(lTotalLength, pMemIndex));

        if(!lShadowMem)
        {
            lShadowMem = reinterpret_cast<char*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateCheckOutMemory(lTotalLength));
            lExplicitAllocation = true;
        }
        
        EXCEPTION_ASSERT(lShadowMem);

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem " << (void*)lShadowMem << " allocated for device/subtask " << pStub << "/" << pSubtaskId << " " << std::endl;
    #endif

        // Auto lock/unlock scope
        {
            GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

            lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.reset(lShadowMem);

            if(lExplicitAllocation)
                lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.GetDeallocator().SetExplicitAllocation();

            lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.GetDeallocator().SetTaskAndAddressSpaceIndex(mTask, pMemIndex);
        }
        
        if(lVisibilityType == SUBSCRIPTION_NATURAL)
            InitializeSubtaskShadowMemNaturalView(pStub, pSubtaskId, pSplitInfo, pMemIndex, lShadowMem, pMem, pMemLength, pWriteOnlyUnprotectedRanges, pUnprotectedRanges);
        else
            InitializeSubtaskShadowMemCompactView(pStub, pSubtaskId, pSplitInfo, pMemIndex, lShadowMem, pMem, pMemLength, pWriteOnlyUnprotectedRanges, pUnprotectedRanges);
    }
}

void pmSubscriptionManager::InitializeSubtaskShadowMemNaturalView(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pShadowMem, void* pMem, size_t pMemLength, size_t pWriteOnlyUnprotectedRanges, uint* pUnprotectedRanges)
{
    pmSubscriptionInfo lUnifiedSubscriptionInfo = GetUnifiedReadWriteSubscription(pStub, pSubtaskId, pSplitInfo, pMemIndex);

    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(pMemIndex);
    bool lIsLazyMem = (mTask->IsLazy(lAddressSpace) && pStub->GetType() == CPU && !pMem);

    char* lShadowMem = static_cast<char*>(pShadowMem);
    
    if(pMem)
    {
    #ifdef SUPPORT_LAZY_MEMORY
        if(mTask->IsLazyWriteOnly(lAddressSpace))
        {
            GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

            size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
            char* lSrcMem = (char*)pMem;

            typedef std::map<size_t, size_t> mapType;
            mapType& lMap = lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;

            lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount = 0;
            
            for(size_t i = 0; i < pWriteOnlyUnprotectedRanges; ++i)
            {
                uint lStartPage = pUnprotectedRanges[2 * i];
                uint lCount = pUnprotectedRanges[2 * i + 1];

                lMap[lStartPage] = lCount;
                lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount += lCount;
                
                size_t lMemSize = std::min(lCount * lPageSize, lUnifiedSubscriptionInfo.length - lStartPage * lPageSize);
                PMLIB_MEMCPY(lShadowMem + lStartPage * lPageSize, lSrcMem, lMemSize, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemNaturalView1"));

                lSrcMem += lMemSize;
            }
        }
        else
    #endif
        {
            void* lCurrPtr = pMem;
            subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
            GetNonConsolidatedWriteSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);

            for(; lBeginIter != lEndIter; ++lBeginIter)
            {
                PMLIB_MEMCPY(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lCurrPtr, lBeginIter->second.first, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemNaturalView2"));
                lCurrPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lCurrPtr) + lBeginIter->second.first);
            }
        }
    }
    else if(!lIsLazyMem && pStub->GetType() == CPU)     // no need to copy for GPU; it will be copied to GPU memory directly and after kernel executes results will be put inside shadow memory
    {
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);

        char* lMem = (char*)(lAddressSpace->GetMem());
        for(; lBeginIter != lEndIter; ++lBeginIter)
            PMLIB_MEMCPY(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lMem + lBeginIter->first, lBeginIter->second.first, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemNaturalView3"));
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(lIsLazyMem)
    {
        // Lazy protect read subscriptions
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
    
        if(mTask->IsReadWrite(lAddressSpace))
        {
            GetNonConsolidatedReadSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);
            for(; lBeginIter != lEndIter; ++lBeginIter)
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lBeginIter->second.first, false, false);
        }
        else
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lShadowMem, lUnifiedSubscriptionInfo.length, false, false);
        }

        FINALIZE_RESOURCE(dShadowMemLock, GetShadowMemLock().Lock(), GetShadowMemLock().Unlock());

        GetShadowMemMap().emplace(std::piecewise_construct, std::forward_as_tuple((void*)lShadowMem), std::forward_as_tuple(lUnifiedSubscriptionInfo, lAddressSpace, mTask));
    }
#endif
}
    
void pmSubscriptionManager::InitializeSubtaskShadowMemCompactView(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pShadowMem, void* pMem, size_t pMemLength, size_t pWriteOnlyUnprotectedRanges, uint* pUnprotectedRanges)
{
    const pmCompactViewData& lCompactViewData = GetCompactedSubscription(pStub, pSubtaskId, pSplitInfo, pMemIndex);

    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(pMemIndex);
    bool lIsLazyMem = (mTask->IsLazy(lAddressSpace) && pStub->GetType() == CPU && !pMem);

    char* lShadowMem = static_cast<char*>(pShadowMem);
    
    if(pMem)
    {
    #ifdef SUPPORT_LAZY_MEMORY
        if(mTask->IsLazyWriteOnly(lAddressSpace))
        {
            GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

            size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
            char* lSrcMem = (char*)pMem;

            typedef std::map<size_t, size_t> mapType;
            mapType& lMap = lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;

            lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount = 0;
            
            for(size_t i = 0; i < pWriteOnlyUnprotectedRanges; ++i)
            {
                uint lStartPage = pUnprotectedRanges[2 * i];
                uint lCount = pUnprotectedRanges[2 * i + 1];

                lMap[lStartPage] = lCount;
                lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount += lCount;

                size_t lMemSize = std::min(lCount * lPageSize, lCompactViewData.subscriptionInfo.length - lStartPage * lPageSize);
                PMLIB_MEMCPY(lShadowMem + lStartPage * lPageSize, lSrcMem, lMemSize, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemCompactView1"));

                lSrcMem += lMemSize;
            }
        }
        else
    #endif
        {
            void* lCurrPtr = pMem;
            subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
            GetNonConsolidatedWriteSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);

            auto lOffsetsIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
            DEBUG_EXCEPTION_ASSERT(std::distance(lOffsetsIter, lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.end()) == std::distance(lBeginIter, lEndIter));

            for(lIter = lBeginIter; lIter != lEndIter; ++lIter, ++lOffsetsIter)
            {
                PMLIB_MEMCPY(lShadowMem + (*lOffsetsIter), lCurrPtr, lIter->second.first, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemCompactView2"));
                lCurrPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lCurrPtr) + lIter->second.first);
            }
        }
    }
    else if(!lIsLazyMem && pStub->GetType() == CPU)     // no need to copy for GPU; it will be copied to GPU memory directly and after kernel executes results will be put inside shadow memory
    {
        subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);
        
        auto lOffsetsIter = lCompactViewData.nonConsolidatedReadSubscriptionOffsets.begin();
        DEBUG_EXCEPTION_ASSERT(std::distance(lOffsetsIter, lCompactViewData.nonConsolidatedReadSubscriptionOffsets.end()) == std::distance(lBeginIter, lEndIter));

        char* lMem = (char*)(lAddressSpace->GetMem());
        for(lIter = lBeginIter; lIter != lEndIter; ++lIter, ++lOffsetsIter)
            PMLIB_MEMCPY(lShadowMem + (*lOffsetsIter), lMem + lIter->first, lIter->second.first, std::string("pmSubscriptionManager::InitializeSubtaskShadowMemCompactView3"));
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(lIsLazyMem)
    {
        // Lazy protect read subscriptions
        subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
    
        if(mTask->IsReadWrite(lAddressSpace))
        {
            GetNonConsolidatedReadSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);
            
            auto lOffsetsIter = lCompactViewData.nonConsolidatedReadSubscriptionOffsets.begin();
            DEBUG_EXCEPTION_ASSERT(std::distance(lOffsetsIter, lCompactViewData.nonConsolidatedReadSubscriptionOffsets.end()) == std::distance(lBeginIter, lEndIter));
            
            for(; lBeginIter != lEndIter; ++lBeginIter, ++lOffsetsIter)
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lShadowMem + (*lOffsetsIter), lBeginIter->second.first, false, false);
        }
        else
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lShadowMem, lCompactViewData.subscriptionInfo.length, false, false);
        }

        FINALIZE_RESOURCE(dShadowMemLock, GetShadowMemLock().Lock(), GetShadowMemLock().Unlock());

        GetShadowMemMap().emplace(std::piecewise_construct, std::forward_as_tuple((void*)lShadowMem), std::forward_as_tuple(lCompactViewData.subscriptionInfo, lAddressSpace, mTask));
    }
#endif
}

void* pmSubscriptionManager::GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

#ifdef DUMP_SHADOW_MEM
    if(pSplitInfo)
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " (Split " << pSplitInfo->splitId << " of " << pSplitInfo->splitCount << ") Mem Index " << pMemIndex << " " << (void*)(lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr)() << std::endl;
    else
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr)() << std::endl;
#endif

	return (void*)(lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr());
}

void pmSubscriptionManager::DestroySubtaskRangeShadowMem(pmExecutionStub* pStub, ulong pStartSubtaskId, ulong pEndSubtaskId, uint pMemIndex)
{
    for(ulong lSubtaskId = pStartSubtaskId; lSubtaskId < pEndSubtaskId; ++lSubtaskId)
        DestroySubtaskShadowMem(pStub, lSubtaskId, NULL, pMemIndex);
}

void pmSubscriptionManager::DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

	DestroySubtaskShadowMemInternal(lSubtask, pStub, pSubtaskId, pSplitInfo, pMemIndex);
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::DestroySubtaskShadowMemInternal(pmSubtask& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
#ifdef DUMP_SHADOW_MEM
    if(pSplitInfo)
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " (Split " << pSplitInfo->splitId << " of " << pSplitInfo->splitCount << ") Mem Index " << pMemIndex << " " << (void*)(pSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
    else
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(pSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
#endif

	pSubtask.mAddressSpacesData[pMemIndex].mShadowMem.reset(NULL);
}

void pmSubscriptionManager::CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::SHADOW_MEM_COMMIT);
#endif

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

#ifdef DUMP_SHADOW_MEM
    std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem committed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
#endif
    
    char* lShadowMem = (char*)(lSubtask.mAddressSpacesData[pMemIndex].mShadowMem.get_ptr());
    EXCEPTION_ASSERT(lShadowMem);

    const pmSubscriptionVisibilityType lVisibilityType = mTask->GetAddressSpaceSubscriptionVisibility(mTask->GetAddressSpace(pMemIndex), pStub);
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(pMemIndex);
    char* lMem = (char*)(lAddressSpace->GetMem());

    subscription::subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
    GetNonConsolidatedWriteSubscriptionsInternal(lSubtask, pMemIndex, lBeginIter, lEndIter);

    if(lVisibilityType == SUBSCRIPTION_NATURAL)
    {
        const pmSubscriptionInfo& lUnifiedSubscriptionInfo = GetUnifiedReadWriteSubscriptionInternal(pStub, lSubtask, pMemIndex);
        
        for(lIter = lBeginIter; lIter != lEndIter; ++lIter)
            PMLIB_MEMCPY(lMem + lIter->first, lShadowMem + (lIter->first - lUnifiedSubscriptionInfo.offset), lIter->second.first, std::string("pmSubscriptionManager::CommitSubtaskShadowMem1"));
    }
    else    // SUBSCRIPTION_COMPACT
    {
        const auto& lCompactViewData = *lSubtask.mAddressSpacesData[pMemIndex].mCompactedSubscription.get_ptr();

        auto lOffsetsIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
        DEBUG_EXCEPTION_ASSERT(std::distance(lOffsetsIter, lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.end()) == std::distance(lBeginIter, lEndIter));

        for(lIter = lBeginIter; lIter != lEndIter; ++lIter, ++lOffsetsIter)
            PMLIB_MEMCPY(lMem + lIter->first, lShadowMem + (*lOffsetsIter), lIter->second.first, std::string("pmSubscriptionManager::CommitSubtaskShadowMem2"));
    }

    DestroySubtaskShadowMemInternal(lSubtask, pStub, pSubtaskId, pSplitInfo, pMemIndex);
}
    
#ifdef SUPPORT_LAZY_MEMORY
pmAddressSpace* pmSubscriptionManager::FindAddressSpaceContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr, pmTask*& pTask)
{
    ACCUMULATION_TIMER(Timer_ACC, "FindAddressSpaceContainingShadowAddr");

    char* lAddress = static_cast<char*>(pAddr);

    typedef std::map<void*, subscription::shadowMemDetails> mapType;
    mapType::iterator lStartIter;
    mapType::iterator* lStartIterAddr = &lStartIter;

    char* lMemAddress = NULL;
    subscription::shadowMemDetails* lShadowMemDetails = NULL;

    shadowMemMapType& lShadowMemMap = GetShadowMemMap();

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dShadowMemLock, GetShadowMemLock().Lock(), GetShadowMemLock().Unlock());
    
        FIND_FLOOR_ELEM(mapType, lShadowMemMap, lAddress, lStartIterAddr);
        
        if(!lStartIterAddr)
            return NULL;
    
        lMemAddress = static_cast<char*>(lStartIter->first);
        lShadowMemDetails = &(lStartIter->second);
    }
    
    size_t lLength = lShadowMemDetails->subscriptionInfo.length;
    
    if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
    {
        pShadowMemOffset = lShadowMemDetails->subscriptionInfo.offset;
        pShadowMemBaseAddr = lMemAddress;
        pTask = lShadowMemDetails->task;

        return lShadowMemDetails->addressSpace;
    }
    
    return NULL;
}
#endif

void pmSubscriptionManager::FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmDeviceType pDeviceType, bool pPrefetch)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

#ifdef ENABLE_DYNAMIC_AGGRESSION
    double lFetchStartTime = pmBase::GetCurrentTimeInSecs();
#endif

    ushort lPriority = (mTask->GetPriority() + (pPrefetch ? 1 : 0));    // Prefetch at slightly low priority

    std::vector<pmCommandPtr> lCommandVector;

    multi_for_each(mTask->GetAddressSpaces(), lSubtask.mAddressSpacesData, [&] (pmAddressSpace* pAddressSpace, pmSubtaskAddressSpaceData& pAddressSpaceData)
    {
        if(!mTask->IsLazy(pAddressSpace) || pDeviceType != CPU)
        {
        #if 0
            for_each(pAddressSpaceData.mReadSubscriptionData.mSubscriptionRecords, [&] (subscriptionRecordType::value_type& pPair)
            {
                size_t lOffset = pPair.first;
                size_t lLength = pPair.second.first;

                pAddressSpace->GetPageAlignedAddresses(lOffset, lLength);
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(pAddressSpace, lPriority, lOffset, lLength, lCommandVector);
            });
        #else
            // Fetch exact scattered subscriptions (these subscriptions are packed at the sender side before transfer and then unpacked by the receiver)
            for_each(pAddressSpaceData.mScatteredReadSubscriptionInfoVector, [&] (const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
            {
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchScatteredMemoryRegion(pAddressSpace, lPriority, pScatteredSubscriptionInfo, lCommandVector);
            });
            
            // For non-scattered subscriptions, we fetch entire memory pages (no packing/unpacking happens here)
            for_each(pAddressSpaceData.mReadSubscriptionInfoVector, [&] (const pmSubscriptionInfo& pSubscriptionInfo)
            {
                size_t lOffset = pSubscriptionInfo.offset;
                size_t lLength = pSubscriptionInfo.length;

                pAddressSpace->GetPageAlignedAddresses(lOffset, lLength);
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(pAddressSpace, lPriority, lOffset, lLength, lCommandVector);
            });
        #endif
        }
    });

    if(!pPrefetch)
    {
        WaitForSubscriptions(lSubtask, pStub, pDeviceType, lCommandVector);

    #ifdef ENABLE_DYNAMIC_AGGRESSION
        if(pmScheduler::SchedulingModelSupportsStealing(mTask->GetSchedulingModel()))
            mTask->GetStealAgent()->RecordSubtaskSubscriptionFetchTime(pStub, pmBase::GetCurrentTimeInSecs() - lFetchStartTime);
    #endif
    }
}
    
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::WaitForSubscriptions(pmSubtask& pSubtask, pmExecutionStub* pStub, pmDeviceType pDeviceType, const std::vector<pmCommandPtr>& pCommandVector)
{
    pStub->WaitForNetworkFetch(pCommandVector);

#ifdef SUPPORT_CUDA
    #ifdef SUPPORT_LAZY_MEMORY
        for_each_with_index(mTask->GetAddressSpaces(), [&pSubtask, pDeviceType, this] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex)
        {
            ClearInputMemLazyProtectionForCuda(pSubtask, pAddressSpace, (uint)pAddressSpaceIndex, pDeviceType);
        });
    #endif
#endif
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::CheckAppropriateSubscription(pmAddressSpace* pAddressSpace, pmSubscriptionType pSubscriptionType) const
{
    // Read Write subscriptions are not collectively handled by this class. Instead do two subscriptions - one read and other write
    DEBUG_EXCEPTION_ASSERT(pSubscriptionType != READ_WRITE_SUBSCRIPTION);
    
    if(mTask->IsReadOnly(pAddressSpace) && pSubscriptionType != READ_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
    
    if(mTask->IsWriteOnly(pAddressSpace) && pSubscriptionType != WRITE_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
}
    
bool pmSubscriptionManager::IsReadSubscription(pmSubscriptionType pSubscriptionType) const
{
    return (pSubscriptionType == READ_SUBSCRIPTION);
}

bool pmSubscriptionManager::IsWriteSubscription(pmSubscriptionType pSubscriptionType) const
{
    return (pSubscriptionType == WRITE_SUBSCRIPTION);
}

const pmSubtaskInfo& pmSubscriptionManager::GetSubtaskInfo(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    if(!lSubtask.mSubtaskInfo.get_ptr())
    {
        for_each_with_index(mTask->GetAddressSpaces(), [&] (pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex)
        {
            pmMemInfo lMemInfo;
            uint lMemIndex = (uint)pAddressSpaceIndex;

            lMemInfo.visibilityType = mTask->GetAddressSpaceSubscriptionVisibility(pAddressSpace, pStub);

            if(lMemInfo.visibilityType == SUBSCRIPTION_NATURAL)
            {
                if(mTask->IsReadOnly(pAddressSpace))
                {
                    void* lAddr = pAddressSpace->GetMem();

                #ifdef SUPPORT_LAZY_MEMORY
                    if(mTask->IsLazy(pAddressSpace))
                        lAddr = pAddressSpace->GetReadOnlyLazyMemoryMapping();
                #endif
                    
                    const pmSubscriptionInfo& lSubscriptionInfo = GetConsolidatedReadSubscriptionInternal(pStub, lSubtask, lMemIndex);
                    if(lSubscriptionInfo.length)
                    {
                        lMemInfo.readPtr = lMemInfo.ptr = (reinterpret_cast<char*>(lAddr) + lSubscriptionInfo.offset);
                        lMemInfo.length = lSubscriptionInfo.length;
                    }
                }
                else
                {
                    pmSubscriptionInfo lUnifiedSubscriptionInfo = GetUnifiedReadWriteSubscriptionInternal(pStub, lSubtask, lMemIndex);
                    if(lUnifiedSubscriptionInfo.length)
                    {
                        lMemInfo.ptr = (lSubtask.mAddressSpacesData[lMemIndex].mShadowMem.get_ptr());
                        if(!lMemInfo.ptr)
                            lMemInfo.ptr = (reinterpret_cast<char*>(pAddressSpace->GetMem()) + lUnifiedSubscriptionInfo.offset);
                        
                        lMemInfo.length = lUnifiedSubscriptionInfo.length;

                        if(mTask->IsWriteOnly(pAddressSpace))
                        {
                            lMemInfo.writePtr = lMemInfo.ptr;
                            //lMemInfo.writeLength = lUnifiedSubscriptionInfo.length;
                        }
                        else
                        {
                            const pmSubscriptionInfo& lReadSubscriptionInfo = GetConsolidatedReadSubscriptionInternal(pStub, lSubtask, lMemIndex);
                            const pmSubscriptionInfo& lWriteSubscriptionInfo = GetConsolidatedWriteSubscriptionInternal(pStub, lSubtask, lMemIndex);
                        
                            lMemInfo.readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lReadSubscriptionInfo.offset - lUnifiedSubscriptionInfo.offset);
                            lMemInfo.writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lWriteSubscriptionInfo.offset - lUnifiedSubscriptionInfo.offset);

                            //lMemInfo.readLength = lReadSubscriptionInfo.length;
                            //lMemInfo.writeLength = lWriteSubscriptionInfo.length;
                        }
                    }
                }
            }
            else    // SUBSCRIPTION_COMPACT
            {
                const pmCompactViewData& lCompactViewData = *lSubtask.mAddressSpacesData[lMemIndex].mCompactedSubscription.get_ptr();

                lMemInfo.ptr = (lSubtask.mAddressSpacesData[lMemIndex].mShadowMem.get_ptr());
                
                // For GPU subtasks, shadow mem is not created unless absolutely required (as GPU mem is itself a shadow mem)
                EXCEPTION_ASSERT(pStub->GetType() != CPU || !lMemInfo.length || lMemInfo.ptr);
                
                lMemInfo.length = lCompactViewData.subscriptionInfo.length;

                if(lMemInfo.ptr)
                {
                    if(!lCompactViewData.nonConsolidatedReadSubscriptionOffsets.empty())
                        lMemInfo.readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lCompactViewData.nonConsolidatedReadSubscriptionOffsets[0]);

                    if(!lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.empty())
                        lMemInfo.writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lCompactViewData.nonConsolidatedWriteSubscriptionOffsets[0]);
                }
            }
            
            lSubtask.mMemInfo.push_back(lMemInfo);
        });
        
        lSubtask.mSubtaskInfo.reset(new pmSubtaskInfo(pSubtaskId, &lSubtask.mMemInfo[0], (uint)(lSubtask.mMemInfo.size())));
    }
        
    lSubtask.mSubtaskInfo->gpuContext.scratchBuffer = NULL;

    if(pSplitInfo)
        lSubtask.mSubtaskInfo->splitInfo = *pSplitInfo;

	return *lSubtask.mSubtaskInfo.get_ptr();
}
    
const pmCompactViewData& pmSubscriptionManager::GetCompactedSubscription(pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    pmSubtaskAddressSpaceData& lData = lSubtask.mAddressSpacesData[pMemIndex];
    if(!lData.mCompactedSubscription.get_ptr())
    {
        pmCompactViewData lCompactViewData;
        subscriptionRecordType lTempMap = GetNonConsolidatedReadWriteSubscriptionsAsMapInternal(lSubtask, pMemIndex);

        subscriptionRecordType::const_iterator lBeginIter1, lEndIter1, lBeginIter2, lEndIter2;

        GetNonConsolidatedReadSubscriptionsInternal(lSubtask, pMemIndex, lBeginIter1, lEndIter1);
        GetNonConsolidatedWriteSubscriptionsInternal(lSubtask, pMemIndex, lBeginIter2, lEndIter2);

        if(!lTempMap.empty())
        {
            lCompactViewData.nonConsolidatedReadSubscriptionOffsets.reserve(std::distance(lBeginIter1, lEndIter1));
            lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.reserve(std::distance(lBeginIter2, lEndIter2));
            
            subscriptionRecordType::const_iterator lTempIter = lTempMap.begin(), lTempEndIter = lTempMap.end();
            size_t lCoveredLength = 0;

            auto lLambda = [lCoveredLength, lTempIter, &lTempEndIter] (const subscriptionRecordType::value_type& pPair, std::vector<size_t>& pVector) mutable
            {
                while(pPair.first < lTempIter->first || pPair.first >= lTempIter->first + lTempIter->second.first)
                {
                    EXCEPTION_ASSERT(lTempIter != lTempEndIter);

                    lCoveredLength += lTempIter->second.first;
                    ++lTempIter;
                }
                
                EXCEPTION_ASSERT(pPair.first >= lTempIter->first && pPair.first + pPair.second.first <= lTempIter->first + lTempIter->second.first);
                
                pVector.push_back(lCoveredLength + pPair.first - lTempIter->first);
            };
            
            for_each_with_arg(lBeginIter1, lEndIter1, lLambda, lCompactViewData.nonConsolidatedReadSubscriptionOffsets);
            for_each_with_arg(lBeginIter2, lEndIter2, lLambda, lCompactViewData.nonConsolidatedWriteSubscriptionOffsets);
            
            for_each(lTempMap, [&lCoveredLength] (const subscriptionRecordType::value_type& pPair)
            {
                lCoveredLength += pPair.second.first;
            });
            
            lCompactViewData.subscriptionInfo = pmSubscriptionInfo(lTempMap.begin()->first, lCoveredLength);
        }

        lData.mCompactedSubscription.reset(new pmCompactViewData(lCompactViewData));
    }

    return *lData.mCompactedSubscription.get_ptr();
}
    
std::vector<pmCompactPageInfo> pmSubscriptionManager::GetReadSubscriptionPagesForCompactViewPage(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pCompactViewPageOffset, size_t pPageSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    std::vector<pmCompactPageInfo> lPageVector;
    const pmCompactViewData& lCompactViewData = *lSubtask.mAddressSpacesData[pMemIndex].mCompactedSubscription.get_ptr();
    
    auto lLambda = [&pCompactViewPageOffset, &pPageSize, &lPageVector, &lCompactViewData] (const subscriptionRecordType::value_type& pPair, size_t pCompactViewSubscriptionOffset)
    {
        if(pCompactViewSubscriptionOffset >= pCompactViewPageOffset + pPageSize)
            return;

        if(pCompactViewSubscriptionOffset <= (pCompactViewPageOffset + pPageSize) && (pCompactViewSubscriptionOffset + pPair.second.first) >= pCompactViewPageOffset)
        {
            pmCompactPageInfo lInfo;

            if(pCompactViewPageOffset <= pCompactViewSubscriptionOffset)
            {
                lInfo.compactViewOffset = pCompactViewSubscriptionOffset - lCompactViewData.subscriptionInfo.offset;
                lInfo.addressSpaceOffset = pPair.first;
                lInfo.length = std::min((pCompactViewSubscriptionOffset + pPair.second.first), (pCompactViewPageOffset + pPageSize)) - pCompactViewSubscriptionOffset;
            }
            else
            {
                lInfo.compactViewOffset = 0;
                lInfo.addressSpaceOffset = pPair.first + (pCompactViewPageOffset - pCompactViewSubscriptionOffset);
                lInfo.length = std::min((pCompactViewSubscriptionOffset + pPair.second.first), (pCompactViewPageOffset + pPageSize)) - pCompactViewPageOffset;
            }
            
            lPageVector.push_back(lInfo);
        }
    };

    subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
    GetNonConsolidatedReadSubscriptionsInternal(lSubtask, pMemIndex, lBeginIter, lEndIter);

    multi_for_each(lBeginIter, lEndIter, lCompactViewData.nonConsolidatedReadSubscriptionOffsets.begin(), lCompactViewData.nonConsolidatedReadSubscriptionOffsets.end(), lLambda);
    
    return lPageVector;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
size_t pmSubscriptionManager::GetAddressSpaceOffsetFromCompactViewOffsetInternal(pmSubtask& pSubtask, uint pMemIndex, size_t pCompactViewOffset)
{
    size_t lAddressSpaceOffset = 0;
    size_t* lAddressSpaceOffsetPtr = NULL;
    
    auto lLambda = [&pCompactViewOffset, &lAddressSpaceOffset, &lAddressSpaceOffsetPtr] (const subscriptionRecordType::value_type& pPair, size_t pCompactViewSubscriptionOffset)
    {
        if(pCompactViewSubscriptionOffset <= pCompactViewOffset && (pCompactViewSubscriptionOffset + pPair.second.first) >= pCompactViewOffset)
        {
            lAddressSpaceOffset = pPair.first + pCompactViewOffset - pCompactViewSubscriptionOffset;
            lAddressSpaceOffsetPtr = &lAddressSpaceOffset;
        }
    };
    
    const pmCompactViewData& lCompactViewData = *pSubtask.mAddressSpacesData[pMemIndex].mCompactedSubscription.get_ptr();

    subscriptionRecordType::const_iterator lBeginIter1, lEndIter1, lBeginIter2, lEndIter2;
    GetNonConsolidatedReadSubscriptionsInternal(pSubtask, pMemIndex, lBeginIter1, lEndIter1);
    multi_for_each(lBeginIter1, lEndIter1, lCompactViewData.nonConsolidatedReadSubscriptionOffsets.begin(), lCompactViewData.nonConsolidatedReadSubscriptionOffsets.end(), lLambda);

    if(lAddressSpaceOffsetPtr)
        return lAddressSpaceOffset;

    GetNonConsolidatedWriteSubscriptionsInternal(pSubtask, pMemIndex, lBeginIter2, lEndIter2);
    multi_for_each(lBeginIter2, lEndIter2, lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin(), lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.end(), lLambda);
    
    EXCEPTION_ASSERT(lAddressSpaceOffsetPtr);

    return lAddressSpaceOffset;
}
    
#ifdef SUPPORT_LAZY_MEMORY
void pmSubscriptionManager::SetWriteOnlyLazyDefaultValue(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, char* pVal, size_t pLength)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyDefaultValue.clear();

    lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyDefaultValue.insert(lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyDefaultValue.end(), pVal, pVal + pLength);
}

void pmSubscriptionManager::InitializeWriteOnlyLazyMemory(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmTask* pTask, pmAddressSpace* pAddressSpace, size_t pOffsetFromBase, void* pLazyPageAddr, size_t pLength)
{
    ACCUMULATION_TIMER(Timer_ACC, "InitializeWriteOnlyLazyMemory");

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    size_t lDataLength = lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyDefaultValue.size();
    char* lData = &(lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyDefaultValue[0]);
    char lDefaultValue = 0;
    
    if(!lDataLength)
    {
        lDataLength = 1;
        lData = &lDefaultValue;
    }
    
    if(lDataLength == 1)
    {
        ACCUMULATION_TIMER(Timer_ACC_Internal, "memset");
        memset(pLazyPageAddr, *lData, pLength);
    }
    else
    {
        if(pTask->GetAddressSpaceSubscriptionVisibility(pAddressSpace, pStub) == SUBSCRIPTION_COMPACT)
            pOffsetFromBase = GetAddressSpaceOffsetFromCompactViewOffsetInternal(lSubtask, pMemIndex, pOffsetFromBase);
        
        size_t lIndex = pOffsetFromBase % lDataLength;

        char* lAddr = (char*)pLazyPageAddr;
        char* lLastAddr = lAddr + pLength;
        while(lAddr != lLastAddr)
        {
            *lAddr = lData[lIndex];
            ++lAddr;
            ++lIndex;
            if(lIndex >= lDataLength)
                lIndex = 0;
        }
    }
}

void pmSubscriptionManager::AddWriteOnlyLazyUnprotection(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pPageNum)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    typedef std::map<size_t, size_t> mapType;
    mapType& lMap = lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;
    
    mapType::iterator lIter, lNextIter, lEndIter = lMap.end();
    mapType::iterator *lIterAddr = &lIter;
    FIND_FLOOR_ELEM(mapType, lMap, pPageNum, lIterAddr);

    size_t lStartPage = pPageNum;
    size_t lEndPage = pPageNum;
    
    if(lIterAddr)
    {
        if(lIter->first <= pPageNum && pPageNum <= lIter->first + lIter->second - 1)
            PMTHROW(pmFatalErrorException());
        
        lNextIter = lIter;
        ++lNextIter;

        if(lIter->first + lIter->second == pPageNum)
        {
            lStartPage = lIter->first;
            lMap.erase(lIter);
        }
        
        if(lNextIter != lEndIter && lNextIter->first == pPageNum + 1)
        {
            lEndPage = lNextIter->first + lNextIter->second - 1;
            lMap.erase(lNextIter);
        }
    }
    else
    {
        size_t lNextPage = pPageNum + 1;
        mapType::iterator lNextPageIter;
        mapType::iterator* lNextPageIterAddr = &lNextPageIter;
        
        FIND_FLOOR_ELEM(mapType, lMap, lNextPage, lNextPageIterAddr);
        
        if(lNextPageIterAddr)
        {
            if(lNextPageIter->first == lNextPage)
            {
                lEndPage = lNextPageIter->first + lNextPageIter->second - 1;
                lMap.erase(lNextPageIter);
            }
        }
    }

    lMap[lStartPage] = lEndPage - lStartPage + 1;
    
    ++lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount;

#ifdef _DEBUG
    if(lMap.size() > 1)
    {
        mapType::iterator lDebugIter = lMap.begin(), lDebugEndIter = lMap.end();
        mapType::iterator lDebugPenultimateIter = lDebugEndIter;

        --lDebugPenultimateIter;
        
        for(; lDebugIter != lDebugPenultimateIter; ++lDebugIter)
        {
            mapType::iterator lDebugNextIter = lDebugIter;
            ++lDebugNextIter;

            if(lDebugIter->first + lDebugIter->second == lDebugNextIter->first)
            {
                std::cout << pPageNum << " " << lDebugIter->first << " " << lDebugIter->second << " " << lDebugNextIter->first << " " << lDebugNextIter->second << std::endl;
                PMTHROW(pmFatalErrorException());
            }
        }
    }
#endif
}
    
size_t pmSubscriptionManager::GetWriteOnlyLazyUnprotectedPagesCount(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    return lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount;
}
    
const std::map<size_t, size_t>& pmSubscriptionManager::GetWriteOnlyLazyUnprotectedPageRanges(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    return lSubtask.mAddressSpacesData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;
}

#ifdef SUPPORT_CUDA
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::ClearInputMemLazyProtectionForCuda(pmSubtask& pSubtask, pmAddressSpace* pAddressSpace, uint pAddressSpaceIndex, pmDeviceType pDeviceType)
{
    if(pDeviceType == GPU_CUDA && mTask->IsReadOnly(pAddressSpace) && mTask->IsLazy(pAddressSpace))
    {
        size_t lLazyMemAddr = reinterpret_cast<size_t>(pAddressSpace->GetReadOnlyLazyMemoryMapping());

        subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(pSubtask, pAddressSpaceIndex, lBeginIter, lEndIter);

        for(lIter = lBeginIter; lIter != lEndIter; ++lIter)
        {
            void* lAddr = reinterpret_cast<void*>(lLazyMemAddr + lIter->first);
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lAddr, lIter->second.first, true, true);
        }
    }
}
#endif
#endif

    
/* struct pmSubtask */
pmSubtask::pmSubtask(pmTask* pTask)
    : mAddressSpacesData(pTask->GetAddressSpaceCount())
    , mReadyForExecution(false)
    , mReservedCudaGlobalMemSize(0)
    , mValid(true)
{
    mMemInfo.reserve(pTask->GetAddressSpaceCount());
}


namespace subscription
{

/* Comparison operators for subscriptionRecordType */
bool operator==(const subscriptionRecordType& pRecord1, const subscriptionRecordType& pRecord2)
{
    if(pRecord1.size() != pRecord2.size())
        return false;
    
    subscriptionRecordType::const_iterator lIter1 = pRecord1.begin(), lEndIter1 = pRecord1.end();
    subscriptionRecordType::const_iterator lIter2 = pRecord2.begin();

    for(; lIter1 != lEndIter1; ++lIter1, ++lIter2)
    {
        if(lIter1->first != lIter2->first || lIter1->second.first != lIter2->second.first)
            return false;
    }
    
    return true;
}

bool operator!=(const subscriptionRecordType& pRecord1, const subscriptionRecordType& pRecord2)
{
    return !(pRecord1 == pRecord2);
}

}


/* shadowMemDeallocator */
void shadowMemDeallocator::operator()(void* pMem)
{
    if(pMem)
    {
    #ifdef SUPPORT_LAZY_MEMORY
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE(dShadowMemLock, pmSubscriptionManager::GetShadowMemLock().Lock(), pmSubscriptionManager::GetShadowMemLock().Unlock());

            pmSubscriptionManager::GetShadowMemMap().erase(pMem);
        }
    #endif

        if(mExplicitAllocation)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(pMem);

            #ifdef DUMP_SHADOW_MEM
                std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem deallocated " << (void*)pMem << std::endl;
            #endif
        }
        else
        {
            mTask->RepoolCheckedOutSubtaskMemory(mMemIndex, pMem);
        }
    }
}
    
}


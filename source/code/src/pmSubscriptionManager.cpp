
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

#include "pmSubscriptionManager.h"
#include "pmCommunicator.h"
#include "pmTaskManager.h"
#include "pmMemoryManager.h"
#include "pmMemSection.h"
#include "pmTask.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmCallbackUnit.h"
#include "pmCallback.h"
#include "pmHardware.h"
#include "pmLogger.h"

#include <string.h>
#include <sstream>

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

pmSubscriptionManager::~pmSubscriptionManager()
{
}
    
void pmSubscriptionManager::DropAllSubscriptions()
{
    mSubtaskMapVector.clear();

#ifdef SUPPORT_SPLIT_SUBTASKS
    mSplitSubtaskMapVector.clear();
#endif
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

void pmSubscriptionManager::FindSubtaskMemDependencies(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
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
            lCallback->Invoke(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status
        
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
            
            lPair.first.find(lDataPair)->second.mReadyForExecution = true;
        }
    }
    else
#endif
    {
        std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];

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
            lCallback->Invoke(pStub, mTask, pSubtaskId, pSplitInfo);    // Check return status

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

            lPair.first.find(pSubtaskId)->second.mReadyForExecution = true;
        }
    }
}
    
void pmSubscriptionManager::RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo)
{
    if(!pSubscriptionInfo.length)
        return;

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    pmMemSection* lMemSection = mTask->GetMemSection(pMemIndex);
    CheckAppropriateSubscription(lMemSection, pSubscriptionType);

    pmSubtaskMemSectionData& lMemSectionData = lSubtask.mMemSectionsData[pMemIndex];
    pmSubtaskSubscriptionData& lSubscriptionData = (IsReadSubscription(pSubscriptionType) ? lMemSectionData.mReadSubscriptionData : lMemSectionData.mWriteSubscriptionData);

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

    /* Only add the region which is yet not subscribed */
    subscriptionRecordType::iterator lStartIter, lEndIter;
    subscriptionRecordType::iterator* lStartIterAddr = &lStartIter;
    subscriptionRecordType::iterator* lEndIterAddr = &lEndIter;
    
    size_t lFirstAddr = pSubscriptionInfo.offset;
    size_t lLastAddr = pSubscriptionInfo.offset + pSubscriptionInfo.length - 1;
    
    FIND_FLOOR_ELEM(subscriptionRecordType, lMap, lFirstAddr, lStartIterAddr);
    FIND_FLOOR_ELEM(subscriptionRecordType, lMap, lLastAddr, lEndIterAddr);
    
    if(!lStartIterAddr && !lEndIterAddr)
    {
        lMap[lFirstAddr].first = pSubscriptionInfo.length;
    }
    else
    {
        std::vector<std::pair<size_t, size_t> > lRangesToBeAdded;
        if(!lStartIterAddr)
        {
            lStartIter = lMap.begin();
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
                FIND_FLOOR_ELEM(subscriptionRecordType, lMap, lRangeOffset - 1, lPrevIterAddr);
                if(lPrevIterAddr && (lPrevIter->first + lPrevIter->second.first == lRangeOffset))
                {
                    lRangeOffset = lPrevIter->first;
                    lRangeLength = lRangeLastAddr - lRangeOffset + 1;
                    lMap.erase(lPrevIter);
                }
                
                // Combine with following range
                FIND_FLOOR_ELEM(subscriptionRecordType, lMap, lRangeLastAddr + 1, lNextIterAddr);
                if(lNextIterAddr && (lNextIter->first == lRangeLastAddr + 1))
                {
                    lRangeLastAddr = lNextIter->first + lNextIter->second.first - 1;
                    lRangeLength = lRangeLastAddr - lRangeOffset + 1;
                    lMap.erase(lNextIter);
                }

                lMap[lRangeOffset].first = lRangeLength;
            }
        }
    }
}

pmStatus pmSubscriptionManager::SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
#ifdef SUPPORT_CUDA
    if(pStub->GetType() != GPU_CUDA)
		PMTHROW(pmFatalErrorException());
#else
        PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mCudaLaunchConf = pCudaLaunchConf;

	return pmSuccess;
}

pmStatus pmSubscriptionManager::ReserveCudaGlobalMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pSize)
{
    if(!mTask->GetCallbackUnit()->GetSubtaskCB()->HasCustomGpuCallback())
        return pmIgnorableError;
    
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
#ifdef SUPPORT_CUDA
    if(pStub->GetType() != GPU_CUDA)
		PMTHROW(pmFatalErrorException());
#else
        PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mReservedCudaGlobalMemSize = pSize;

	return pmSuccess;
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
    
void* pmSubscriptionManager::GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    if(!lSubtask.mScratchBuffer.get_ptr())
    {
        lSubtask.mScratchBuffer.reset(new char[pBufferSize]);
        lSubtask.mScratchBufferSize = pBufferSize;
        lSubtask.mScratchBufferType = pScratchBufferType;
    }
    
    return lSubtask.mScratchBuffer.get_ptr();
}

void* pmSubscriptionManager::CheckAndGetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t& pScratchBufferSize, pmScratchBufferType& pScratchBufferType)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
	char* lScratchBuffer = lSubtask.mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
        return NULL;

    pScratchBufferSize = lSubtask.mScratchBufferSize;
    pScratchBufferType = lSubtask.mScratchBufferType;

    return lScratchBuffer;
}
    
void pmSubscriptionManager::DropScratchBufferIfNotRequiredPostSubtaskExec(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
	char* lScratchBuffer = lSubtask.mScratchBuffer.get_ptr();
    if(lScratchBuffer && lSubtask.mScratchBufferType == PRE_SUBTASK_TO_SUBTASK)
    {
        lSubtask.mScratchBuffer.reset(NULL);
        lSubtask.mScratchBufferSize = 0;
    }
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
    if(pSubtask1.mMemSectionsData.size() != pSubtask2.mMemSectionsData.size())
        PMTHROW(pmFatalErrorException());   // both subtasks belong to same task; so must have same mem section entries
#endif
    
    std::vector<pmSubtaskMemSectionData>::const_iterator lIter1 = pSubtask1.mMemSectionsData.begin(), lEndIter1 = pSubtask2.mMemSectionsData.end();
    std::vector<pmSubtaskMemSectionData>::const_iterator lIter2 = pSubtask2.mMemSectionsData.begin();
    
    for(; lIter1 != lEndIter1; ++lIter1, ++lIter2)
    {
        if(!MemSectionsHaveMatchingSubscriptionsInternal((*lIter1), (*lIter2)))
            return false;
    }
    
    return true;
}
    
/* Must be called with mSubtaskMapVector stub's lock acquired for both subtasks */
bool pmSubscriptionManager::MemSectionsHaveMatchingSubscriptionsInternal(const pmSubtaskMemSectionData& pMemSectionData1, const pmSubtaskMemSectionData& pMemSectionData2) const
{
    return (pMemSectionData1.mReadSubscriptionData == pMemSectionData2.mReadSubscriptionData && pMemSectionData1.mWriteSubscriptionData == pMemSectionData2.mWriteSubscriptionData);
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
    
    return GetConsolidatedReadSubscriptionInternal(lSubtask, pMemIndex);
}
    
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    return GetConsolidatedWriteSubscriptionInternal(lSubtask, pMemIndex);
}
    
pmSubscriptionInfo pmSubscriptionManager::GetUnifiedReadWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    return GetUnifiedReadWriteSubscriptionInternal(lSubtask, pMemIndex);
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::GetNonConsolidatedReadSubscriptionsInternal(pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	pBegin = pSubtask.mMemSectionsData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords.begin();
	pEnd = pSubtask.mMemSectionsData[pMemIndex].mReadSubscriptionData.mSubscriptionRecords.end();
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::GetNonConsolidatedWriteSubscriptionsInternal(pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	pBegin = pSubtask.mMemSectionsData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords.begin();
	pEnd = pSubtask.mMemSectionsData[pMemIndex].mWriteSubscriptionData.mSubscriptionRecords.end();
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedReadSubscriptionInternal(pmSubtask& pSubtask, uint pMemIndex)
{
    return pSubtask.mMemSectionsData[pMemIndex].mReadSubscriptionData.mConsolidatedSubscriptions;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
const pmSubscriptionInfo& pmSubscriptionManager::GetConsolidatedWriteSubscriptionInternal(pmSubtask& pSubtask, uint pMemIndex)
{
    return pSubtask.mMemSectionsData[pMemIndex].mWriteSubscriptionData.mConsolidatedSubscriptions;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
pmSubscriptionInfo pmSubscriptionManager::GetUnifiedReadWriteSubscriptionInternal(pmSubtask& pSubtask, uint pMemIndex)
{
    pmSubtaskMemSectionData& lMemSectionData = pSubtask.mMemSectionsData[pMemIndex];

    size_t lReadOffset = lMemSectionData.mReadSubscriptionData.mConsolidatedSubscriptions.offset;
    size_t lReadLength = lMemSectionData.mReadSubscriptionData.mConsolidatedSubscriptions.length;
    size_t lReadSpan = lReadOffset + lReadLength;
    size_t lWriteOffset = lMemSectionData.mWriteSubscriptionData.mConsolidatedSubscriptions.offset;
    size_t lWriteLength = lMemSectionData.mWriteSubscriptionData.mConsolidatedSubscriptions.length;
    size_t lWriteSpan = lWriteOffset + lWriteLength;
    
    if(!lReadLength && !lWriteLength)
        return pmSubscriptionInfo(0, 0);
    else if(lReadLength && !lWriteLength)
        return pmSubscriptionInfo(lReadOffset, lReadLength);
    else if(!lReadLength && lWriteLength)
        return pmSubscriptionInfo(lWriteOffset, lWriteLength);
        
    size_t lOffset = std::min(lReadOffset, lWriteOffset);
    size_t lLength = (std::max(lReadSpan, lWriteSpan) - lOffset);

    return pmSubscriptionInfo(lOffset, lLength);
}

pmStatus pmSubscriptionManager::CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pMem /* = NULL */, size_t pMemLength /* = 0 */, size_t pWriteOnlyUnprotectedRanges /* = 0 */, uint* pUnprotectedRanges /* = NULL */)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

        if(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr() != NULL)
            PMTHROW(pmFatalErrorException());
    }
#endif

    pmSubscriptionInfo lUnifiedSubscriptionInfo = GetUnifiedReadWriteSubscription(pStub, pSubtaskId, pSplitInfo, pMemIndex);
    
    pmMemSection* lMemSection = mTask->GetMemSection(pMemIndex);
    bool lIsLazyMem = (lMemSection->IsLazy() && pStub->GetType() == CPU && !pMem);

    bool lExplicitAllocation = false;
    char* lShadowMem = reinterpret_cast<char*>(mTask->CheckOutSubtaskMemory(lUnifiedSubscriptionInfo.length, pMemIndex));

    if(!lShadowMem)
    {
        lShadowMem = reinterpret_cast<char*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateCheckOutMemory(lUnifiedSubscriptionInfo.length));
        lExplicitAllocation = true;
    }
    
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());
    
#ifdef DUMP_SHADOW_MEM
    std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem " << (void*)lShadowMem << " allocated for device/subtask " << pStub << "/" << pSubtaskId << " " << std::endl;
#endif

    // Auto lock/unlock scope
    {
        GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

        lSubtask.mMemSectionsData[pMemIndex].mShadowMem.reset(lShadowMem);

        if(lExplicitAllocation)
            lSubtask.mMemSectionsData[pMemIndex].mShadowMem.GetDeallocator().SetExplicitAllocation();

        lSubtask.mMemSectionsData[pMemIndex].mShadowMem.GetDeallocator().SetTaskAndMemSectionIndex(mTask, pMemIndex);
    }

    if(pMem)
    {
    #ifdef SUPPORT_LAZY_MEMORY
        if(lMemSection->IsLazyWriteOnly())
        {
            GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

            size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
            char* lSrcMem = (char*)pMem;

            typedef std::map<size_t, size_t> mapType;
            mapType& lMap = lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;

            lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount = 0;
            
            for(size_t i = 0; i < pWriteOnlyUnprotectedRanges; ++i)
            {
                uint lStartPage = pUnprotectedRanges[2 * i];
                uint lCount = pUnprotectedRanges[2 * i +1];

                lMap[lStartPage] = lCount;
                lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount += lCount;
                
                size_t lMemSize = std::min(lCount * lPageSize, lUnifiedSubscriptionInfo.length - lStartPage * lPageSize);
                memcpy(lShadowMem + lStartPage * lPageSize, lSrcMem, lMemSize);

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
                memcpy(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lCurrPtr, lBeginIter->second.first);
                lCurrPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lCurrPtr) + lBeginIter->second.first);
            }
        }
    }
    else if(!lIsLazyMem && pStub->GetType() == CPU)     // no need to copy for GPU; it will be copied to GPU memory directly and after kernel executes results will be put inside shadow memory
    {
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptions(pStub, pSubtaskId, pSplitInfo, pMemIndex, lBeginIter, lEndIter);

        char* lMem = (char*)(lMemSection->GetMem());
        for(; lBeginIter != lEndIter; ++lBeginIter)
            memcpy(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lMem + lBeginIter->first, lBeginIter->second.first);
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(lIsLazyMem)
    {
        // Lazy protect read subscriptions
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
    
        if(lMemSection->IsReadWrite())
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
        shadowMemMapType& lShadowMemMap = GetShadowMemMap();
        lShadowMemMap[(void*)lShadowMem].subscriptionInfo = lUnifiedSubscriptionInfo;
        lShadowMemMap[(void*)lShadowMem].memSection = lMemSection;
    }
#endif
    
	return pmSuccess;
}
    
void* pmSubscriptionManager::GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

#ifdef DUMP_SHADOW_MEM
    if(pSplitInfo)
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " (Split " << pSplitInfo->splitId << " of " << pSplitInfo->splitCount << ") Mem Index " << pMemIndex << " " << (void*)(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr)() << std::endl;
    else
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr)() << std::endl;
#endif

	return (void*)(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr());
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
pmStatus pmSubscriptionManager::DestroySubtaskShadowMemInternal(pmSubtask& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
#ifdef DUMP_SHADOW_MEM
    if(pSplitInfo)
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " (Split " << pSplitInfo->splitId << " of " << pSplitInfo->splitCount << ") Mem Index " << pMemIndex << " " << (void*)(pSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
    else
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(pSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
#endif

	pSubtask.mMemSectionsData[pMemIndex].mShadowMem.reset(NULL);

	return pmSuccess;
}

void pmSubscriptionManager::CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBeginIter, subscription::subscriptionRecordType::const_iterator& pEndIter, ulong pShadowMemOffset)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::SHADOW_MEM_COMMIT);
#endif

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem committed for device/subtask " << pStub << "/" << pSubtaskId << " Mem Index " << pMemIndex << " " << (void*)(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr()) << std::endl;
    #endif
    
    char* lShadowMem = (char*)(lSubtask.mMemSectionsData[pMemIndex].mShadowMem.get_ptr());
    
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());

    pmMemSection* lMemSection = mTask->GetMemSection(pMemIndex);
    char* lMem = (char*)(lMemSection->GetMem());
    
    subscription::subscriptionRecordType::const_iterator lIter = pBeginIter;
    for(; lIter != pEndIter; ++lIter)
        memcpy(lMem + lIter->first, lShadowMem + (lIter->first - pShadowMemOffset), lIter->second.first);
    
    DestroySubtaskShadowMemInternal(lSubtask, pStub, pSubtaskId, pSplitInfo, pMemIndex);
}

#ifdef SUPPORT_LAZY_MEMORY
pmMemSection* pmSubscriptionManager::FindMemSectionContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr)
{
    ACCUMULATION_TIMER(Timer_ACC, "FindMemSectionContainingShadowAddr");

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
        return lShadowMemDetails->memSection;
    }
    
    return NULL;
}
#endif

void pmSubscriptionManager::FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmDeviceType pDeviceType, bool pPrefetch)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    ushort lPriority = (mTask->GetPriority() + (pPrefetch ? 1 : 0));    // Prefetch at slightly low priority

    std::vector<pmMemSection*>& lMemSections = mTask->GetMemSections();
    std::vector<pmMemSection*>::const_iterator lMemSectionIter = lMemSections.begin();
    
    std::vector<pmSubtaskMemSectionData>::iterator lMemSectionDataIter = lSubtask.mMemSectionsData.begin(), lMemSectionDataEndIter = lSubtask.mMemSectionsData.end();
    for(; lMemSectionDataIter != lMemSectionDataEndIter; ++lMemSectionDataIter, ++lMemSectionIter)
    {
        pmMemSection* lMemSection = (*lMemSectionIter);
        if(!lMemSection->IsLazy() || pDeviceType != CPU)
        {
            pmSubtaskMemSectionData& lMemSectionData = (*lMemSectionDataIter);
            subscriptionRecordType& lMap = lMemSectionData.mReadSubscriptionData.mSubscriptionRecords;
            
            subscriptionRecordType::iterator lIter = lMap.begin(), lEndIter = lMap.end();
            for(; lIter != lEndIter; ++lIter)
            {
                size_t lOffset = lIter->first;
                size_t lLength = lIter->second.first;

                lMemSection->GetPageAlignedAddresses(lOffset, lLength);
                MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, lPriority, lOffset, lLength, lIter->second.second.receiveCommandVector);
            }
        }
    }

    if(!pPrefetch)
        WaitForSubscriptions(lSubtask, pStub, pDeviceType);
}
    
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::WaitForSubscriptions(pmSubtask& pSubtask, pmExecutionStub* pStub, pmDeviceType pDeviceType)
{
    std::vector<pmMemSection*>& lMemSections = mTask->GetMemSections();
    std::vector<pmMemSection*>::const_iterator lMemSectionIter = lMemSections.begin();

    std::vector<pmSubtaskMemSectionData>::const_iterator lMemSectionDataIter = pSubtask.mMemSectionsData.begin(), lMemSectionDataEndIter = pSubtask.mMemSectionsData.end();
    for(uint lMemSectionIndex = 0; lMemSectionDataIter != lMemSectionDataEndIter; ++lMemSectionDataIter, ++lMemSectionIter, ++lMemSectionIndex)
    {
        const pmSubtaskMemSectionData& lMemSectionData = (*lMemSectionDataIter);
        const subscriptionRecordType& lMap = lMemSectionData.mReadSubscriptionData.mSubscriptionRecords;

        subscriptionRecordType::const_iterator lIter = lMap.begin(), lEndIter = lMap.end();
        for(; lIter != lEndIter; ++lIter)
            pStub->WaitForNetworkFetch(lIter->second.second.receiveCommandVector);

    #ifdef SUPPORT_CUDA
        #ifdef SUPPORT_LAZY_MEMORY
            ClearInputMemLazyProtectionForCuda(pSubtask, *lMemSectionIter, lMemSectionIndex, pDeviceType);
        #endif
    #endif
	}
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::CheckAppropriateSubscription(pmMemSection* pMemSection, pmSubscriptionType pSubscriptionType) const
{
#ifdef _DEBUG
    // Read Write subscriptions are not collectively handled by this class. Instead do two subscriptions - one read and other write
    if(pSubscriptionType == OUTPUT_MEM_READ_WRITE_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
#endif
    
    if(pMemSection->IsInput() && pSubscriptionType != INPUT_MEM_READ_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
    
    if(pMemSection->IsOutput() && pSubscriptionType != OUTPUT_MEM_READ_SUBSCRIPTION && pSubscriptionType != OUTPUT_MEM_WRITE_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
    
    if(pMemSection->IsWriteOnly() && pSubscriptionType != OUTPUT_MEM_WRITE_SUBSCRIPTION)
        PMTHROW(pmFatalErrorException());
}
    
bool pmSubscriptionManager::IsReadSubscription(pmSubscriptionType pSubscriptionType) const
{
    return (pSubscriptionType == INPUT_MEM_READ_SUBSCRIPTION || pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION);
}

bool pmSubscriptionManager::IsWriteSubscription(pmSubscriptionType pSubscriptionType) const
{
    return (pSubscriptionType == OUTPUT_MEM_WRITE_SUBSCRIPTION);
}
    
const pmSubtaskInfo& pmSubscriptionManager::GetSubtaskInfo(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    if(!lSubtask.mSubtaskInfo.get_ptr())
    {
        const std::vector<pmMemSection*>& lMemSections = mTask->GetMemSections();

        std::vector<pmMemSection*>::const_iterator lIter = lMemSections.begin(), lEndIter = lMemSections.end();
        for(uint lMemIndex = 0; lIter != lEndIter; ++lIter, ++lMemIndex)
        {
            pmMemSection* lMemSection = (*lIter);
            
            pmMemInfo lMemInfo;
            if(lMemSection->IsInput())
            {
                void* lAddr = lMemSection->GetMem();

            #ifdef SUPPORT_LAZY_MEMORY
                if(lMemSection->IsLazy())
                    lAddr = lMemSection->GetReadOnlyLazyMemoryMapping();
            #endif
                
                const pmSubscriptionInfo& lSubscriptionInfo = GetConsolidatedReadSubscriptionInternal(lSubtask, lMemIndex);
                if(lSubscriptionInfo.length)
                {
                    lMemInfo.readPtr = lMemInfo.ptr = (reinterpret_cast<char*>(lAddr) + lSubscriptionInfo.offset);
                    lMemInfo.length = lSubscriptionInfo.length;
                }
            }
            else
            {
                pmSubscriptionInfo lUnifiedSubscriptionInfo = GetUnifiedReadWriteSubscriptionInternal(lSubtask, lMemIndex);
                if(lUnifiedSubscriptionInfo.length)
                {
                    lMemInfo.ptr = (lSubtask.mMemSectionsData[lMemIndex].mShadowMem.get_ptr());
                    if(!lMemInfo.ptr)
                        lMemInfo.ptr = (reinterpret_cast<char*>(lMemSection->GetMem()) + lUnifiedSubscriptionInfo.offset);
                    
                    lMemInfo.length = lUnifiedSubscriptionInfo.length;

                    if(lMemSection->IsWriteOnly())
                    {
                        lMemInfo.writePtr = lMemInfo.ptr;
                        //lMemInfo.writeLength = lUnifiedSubscriptionInfo.length;
                    }
                    else
                    {
                        const pmSubscriptionInfo& lReadSubscriptionInfo = GetConsolidatedReadSubscriptionInternal(lSubtask, lMemIndex);
                        const pmSubscriptionInfo& lWriteSubscriptionInfo = GetConsolidatedWriteSubscriptionInternal(lSubtask, lMemIndex);
                    
                        lMemInfo.readPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lReadSubscriptionInfo.offset - lUnifiedSubscriptionInfo.offset);
                        lMemInfo.writePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lMemInfo.ptr) + lWriteSubscriptionInfo.offset - lUnifiedSubscriptionInfo.offset);

                        //lMemInfo.readLength = lReadSubscriptionInfo.length;
                        //lMemInfo.writeLength = lWriteSubscriptionInfo.length;
                    }
                }
            }
            
            lSubtask.mMemInfo.push_back(lMemInfo);
        }
        
        lSubtask.mSubtaskInfo.reset(new pmSubtaskInfo(pSubtaskId, &lSubtask.mMemInfo[0], (uint)(lSubtask.mMemInfo.size())));
    }
        
    lSubtask.mSubtaskInfo->gpuContext.scratchBuffer = NULL;

    if(pSplitInfo)
        lSubtask.mSubtaskInfo->splitInfo = *pSplitInfo;

	return *lSubtask.mSubtaskInfo.get_ptr();
}
    
#ifdef SUPPORT_LAZY_MEMORY
pmStatus pmSubscriptionManager::SetWriteOnlyLazyDefaultValue(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, char* pVal, size_t pLength)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyDefaultValue.clear();
    
    lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyDefaultValue.insert(lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyDefaultValue.end(), pVal, pVal + pLength);
    
    return pmSuccess;
}

void pmSubscriptionManager::InitializeWriteOnlyLazyMemory(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pOffsetFromBase, void* pLazyPageAddr, size_t pLength)
{
    ACCUMULATION_TIMER(Timer_ACC, "InitializeWriteOnlyLazyMemory");

    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);
    
    size_t lDataLength = lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyDefaultValue.size();
    char* lData = &(lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyDefaultValue[0]);
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
    mapType& lMap = lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;
    
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
    
    ++lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount;

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

    return lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageCount;
}
    
const std::map<size_t, size_t>& pmSubscriptionManager::GetWriteOnlyLazyUnprotectedPageRanges(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId, pSplitInfo);

    return lSubtask.mMemSectionsData[pMemIndex].mWriteOnlyLazyUnprotectedPageRangesMap;
}

#ifdef SUPPORT_CUDA
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::ClearInputMemLazyProtectionForCuda(pmSubtask& pSubtask, pmMemSection* pMemSection, uint pMemSectionIndex, pmDeviceType pDeviceType)
{
    if(pDeviceType == GPU_CUDA && pMemSection->IsInput() && pMemSection->IsLazy())
    {
        size_t lLazyMemAddr = reinterpret_cast<size_t>(pMemSection->GetReadOnlyLazyMemoryMapping());

        subscriptionRecordType::const_iterator lIter, lBeginIter, lEndIter;
        GetNonConsolidatedReadSubscriptionsInternal(pSubtask, pMemSectionIndex, lBeginIter, lEndIter);

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
    : mScratchBufferSize(0)
    , mScratchBufferType(SUBTASK_TO_POST_SUBTASK)
    , mMemSectionsData(pTask->GetMemSectionCount())
    , mReadyForExecution(false)
    , mReservedCudaGlobalMemSize(0)
{
    mMemInfo.reserve(pTask->GetMemSectionCount());
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


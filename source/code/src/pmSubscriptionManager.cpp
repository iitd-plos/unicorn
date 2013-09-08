
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

#define GET_SUBTASK(subtaskVar, stub, subtaskId) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair.second, Lock(), Unlock()); \
if(dPair.first.find(subtaskId) == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar = dPair.first[subtaskId];

#define GET_SUBTASKS(subtaskVar1, subtaskVar2, stub, subtaskId1, subtaskId2) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair.second, Lock(), Unlock()); \
if(dPair.first.find(subtaskId1) == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
if(dPair.first.find(subtaskId2) == dPair.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar1 = dPair.first[subtaskId1]; \
subscription::pmSubtask& subtaskVar2 = dPair.first[subtaskId2];

#define GET_SUBTASK2(subtaskVar, stub, subtaskId) \
std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& dPair2 = mSubtaskMapVector[stub->GetProcessingElement()->GetDeviceIndexInMachine()]; \
FINALIZE_RESOURCE_PTR(dResourceLock2, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &dPair2.second, Lock(), Unlock()); \
if(dPair2.first.find(subtaskId) == dPair2.first.end()) \
    PMTHROW(pmFatalErrorException()); \
subscription::pmSubtask& subtaskVar = dPair2.first[subtaskId];


STATIC_ACCESSOR(pmSubscriptionManager::shadowMemMapType, pmSubscriptionManager, GetShadowMemMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmSubscriptionManager::mShadowMemLock"), pmSubscriptionManager, GetShadowMemLock)

pmSubscriptionManager::pmSubscriptionManager(pmTask* pTask)
	: mTask(pTask)
{
    mSubtaskMapVector.resize(pmStubManager::GetStubManager()->GetStubCount());
}

pmSubscriptionManager::~pmSubscriptionManager()
{
}
    
void pmSubscriptionManager::DropAllSubscriptions()
{
    mSubtaskMapVector.clear();
}
    
pmStatus pmSubscriptionManager::EraseSubscription(pm::pmExecutionStub* pStub, ulong pSubtaskId)
{
    std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
	
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());
    lPair.first.erase(pSubtaskId);
    
    return pmSuccess;
}

pmStatus pmSubscriptionManager::InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId)
{
    std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
	
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

	if(lPair.first.find(pSubtaskId) != lPair.first.end())
		PMTHROW(pmFatalErrorException());
    
	lPair.first[pSubtaskId].Initialize(mTask);

	return pmSuccess;
}
    
bool pmSubscriptionManager::IsSubtaskInitialized(pmExecutionStub* pStub, ulong pSubtaskId)
{
    std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS>& lPair = mSubtaskMapVector[pStub->GetProcessingElement()->GetDeviceIndexInMachine()];
	
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lPair.second, Lock(), Unlock());

	return !(lPair.first.find(pSubtaskId) == lPair.first.end());
}

pmStatus pmSubscriptionManager::RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionType pSubscriptionType, pmSubscriptionInfo pSubscriptionInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    bool lIsInputMem = (pSubscriptionType == INPUT_MEM_READ_SUBSCRIPTION);
	if((lIsInputMem && !mTask->GetMemSectionRO()) || (!lIsInputMem && !mTask->GetMemSectionRW()))
        PMTHROW(pmFatalErrorException());

    subscriptionRecordType& lMap = lIsInputMem ? lSubtask.mInputMemSubscriptions : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? lSubtask.mOutputMemReadSubscriptions : lSubtask.mOutputMemWriteSubscriptions);
    pmSubscriptionInfo& lConsolidatedSubscription = lIsInputMem ? lSubtask.mConsolidatedInputMemSubscription : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? lSubtask.mConsolidatedOutputMemReadSubscription : lSubtask.mConsolidatedOutputMemWriteSubscription);
    
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

	return pmSuccess;
}

pmStatus pmSubscriptionManager::SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
#ifdef SUPPORT_CUDA
    if(pStub->GetType() != GPU_CUDA)
		PMTHROW(pmFatalErrorException());
#else
        PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mCudaLaunchConf = pCudaLaunchConf;

	return pmSuccess;
}

pmStatus pmSubscriptionManager::ReserveCudaGlobalMem(pmExecutionStub* pStub, ulong pSubtaskId, size_t pSize)
{
    if(!mTask->GetCallbackUnit()->GetSubtaskCB()->HasCustomGpuCallback())
        return pmIgnorableError;
    
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
#ifdef SUPPORT_CUDA
    if(pStub->GetType() != GPU_CUDA)
		PMTHROW(pmFatalErrorException());
#else
        PMTHROW(pmFatalErrorException());
#endif

	lSubtask.mReservedCudaGlobalMemSize = pSize;

	return pmSuccess;
}
    
pmStatus pmSubscriptionManager::SetWriteOnlyLazyDefaultValue(pmExecutionStub* pStub, ulong pSubtaskId, char* pVal, size_t pLength)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
    lSubtask.mWriteOnlyLazyDefaultValue.clear();
    
    lSubtask.mWriteOnlyLazyDefaultValue.insert(lSubtask.mWriteOnlyLazyDefaultValue.end(), pVal, pVal + pLength);
    
    return pmSuccess;
}

void pmSubscriptionManager::InitializeWriteOnlyLazyMemory(pmExecutionStub* pStub, ulong pSubtaskId, size_t pOffsetFromBase, void* pLazyPageAddr, size_t pLength)
{
    ACCUMULATION_TIMER(Timer_ACC, "InitializeWriteOnlyLazyMemory");

    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
    size_t lDataLength = lSubtask.mWriteOnlyLazyDefaultValue.size();
    char* lData = &(lSubtask.mWriteOnlyLazyDefaultValue[0]);
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

pmCudaLaunchConf& pmSubscriptionManager::GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	return lSubtask.mCudaLaunchConf;
}
    
size_t pmSubscriptionManager::GetReservedCudaGlobalMemSize(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	return lSubtask.mReservedCudaGlobalMemSize;
}
    
void* pmSubscriptionManager::GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmScratchBufferInfo pScratchBufferInfo, size_t pBufferSize)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
	char* lScratchBuffer = lSubtask.mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
    {
        lScratchBuffer = new char[pBufferSize];

        lSubtask.mScratchBuffer.reset(lScratchBuffer);
        lSubtask.mScratchBufferSize = pBufferSize;
        lSubtask.mScratchBufferInfo = pScratchBufferInfo;
    }
    
    return lScratchBuffer;
}

void* pmSubscriptionManager::CheckAndGetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, size_t& pScratchBufferSize, pmScratchBufferInfo& pScratchBufferInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
	char* lScratchBuffer = lSubtask.mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
        return NULL;

    pScratchBufferSize = lSubtask.mScratchBufferSize;
    pScratchBufferInfo = lSubtask.mScratchBufferInfo;
    
    return lScratchBuffer;
}
    
void pmSubscriptionManager::DropScratchBufferIfNotRequiredPostSubtaskExec(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
	char* lScratchBuffer = lSubtask.mScratchBuffer.get_ptr();
    if(lScratchBuffer && lSubtask.mScratchBufferInfo == PRE_SUBTASK_TO_SUBTASK)
    {
        lSubtask.mScratchBuffer.reset(NULL);
        lSubtask.mScratchBufferSize = 0;
    }
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType)
{
    if(pStub1 == pStub2)
        return SubtasksHaveMatchingSubscriptionsCommonStub(pStub1, pSubtaskId1, pSubtaskId2, pSubscriptionType);

    return SubtasksHaveMatchingSubscriptionsDifferentStubs(pStub1, pSubtaskId1, pStub2, pSubtaskId2, pSubscriptionType);
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsCommonStub(pmExecutionStub* pStub, ulong pSubtaskId1, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType)
{
    GET_SUBTASKS(lSubtask1, lSubtask2, pStub, pSubtaskId1, pSubtaskId2);
    
    return SubtasksHaveMatchingSubscriptionsInternal(lSubtask1, lSubtask2, pSubscriptionType);
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsDifferentStubs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType)
{
    GET_SUBTASK(lSubtask1, pStub1, pSubtaskId1);
    GET_SUBTASK2(lSubtask2, pStub2, pSubtaskId2);
    
    return SubtasksHaveMatchingSubscriptionsInternal(lSubtask1, lSubtask2, pSubscriptionType);
}

/* Must be called with mSubtaskMapVector stub's lock acquired for both subtasks */
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptionsInternal(pmSubtask& pSubtask1, pmSubtask& pSubtask2, pmSubscriptionType pSubscriptionType)
{
    bool lIsInputMem = (pSubscriptionType == INPUT_MEM_READ_SUBSCRIPTION);
	if((lIsInputMem && !mTask->GetMemSectionRO()) || (!lIsInputMem && !mTask->GetMemSectionRW()))
		PMTHROW(pmFatalErrorException());
    
	pmSubscriptionInfo& lConsolidatedSubscription1 = lIsInputMem ? pSubtask1.mConsolidatedInputMemSubscription : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? pSubtask1.mConsolidatedOutputMemReadSubscription : pSubtask1.mConsolidatedOutputMemWriteSubscription);
	pmSubscriptionInfo& lConsolidatedSubscription2 = lIsInputMem ? pSubtask2.mConsolidatedInputMemSubscription : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? pSubtask2.mConsolidatedOutputMemReadSubscription : pSubtask2.mConsolidatedOutputMemWriteSubscription);
    
    if(lConsolidatedSubscription1 != lConsolidatedSubscription2)
        return false;

    subscriptionRecordType& lSubscriptions1 = lIsInputMem ? pSubtask1.mInputMemSubscriptions : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? pSubtask1.mOutputMemReadSubscriptions : pSubtask1.mOutputMemWriteSubscriptions);
    subscriptionRecordType& lSubscriptions2 = lIsInputMem ? pSubtask2.mInputMemSubscriptions : ((pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION) ? pSubtask2.mOutputMemReadSubscriptions : pSubtask2.mOutputMemWriteSubscriptions);
    
    if(lSubscriptions1.size() != lSubscriptions2.size())
        return false;
    
    subscriptionRecordType::iterator lIter1 = lSubscriptions1.begin(), lEndIter1 = lSubscriptions1.end();
    subscriptionRecordType::iterator lIter2 = lSubscriptions2.begin();
    
    for(; lIter1 != lEndIter1; ++lIter1, ++lIter2)
    {
        if(lIter1->first != lIter2->first || lIter1->second.first != lIter2->second.first)
            return false;
    }
    
    return true;
}
    
bool pmSubscriptionManager::GetNonConsolidatedInputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
	if(!mTask->GetMemSectionRO())
		return false;
    
	pBegin = lSubtask.mInputMemSubscriptions.begin();
	pEnd = lSubtask.mInputMemSubscriptions.end();
    
	return true;    
}

bool pmSubscriptionManager::GetNonConsolidatedOutputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, bool pReadSubscription, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);
    
	if(!mTask->GetMemSectionRW() || (pReadSubscription && mTask->GetMemSectionRW()->IsWriteOnly()))
		return false;
    
	pBegin = pReadSubscription ? lSubtask.mOutputMemReadSubscriptions.begin() : lSubtask.mOutputMemWriteSubscriptions.begin();
	pEnd = pReadSubscription ? lSubtask.mOutputMemReadSubscriptions.end() : lSubtask.mOutputMemWriteSubscriptions.end();
    
	return true;    
}

bool pmSubscriptionManager::GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	if(!mTask->GetMemSectionRO())
		return false;

	pSubscriptionInfo = lSubtask.mConsolidatedInputMemSubscription;

	return true;
}

bool pmSubscriptionManager::GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	if(!mTask->GetMemSectionRW() || (pReadSubscription && mTask->GetMemSectionRW()->IsWriteOnly()))
		return false;

	pSubscriptionInfo = pReadSubscription ? lSubtask.mConsolidatedOutputMemReadSubscription : lSubtask.mConsolidatedOutputMemWriteSubscription;

	return true;
}

bool pmSubscriptionManager::GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	if(!mTask->GetMemSectionRW())
		return false;

    if(mTask->GetMemSectionRW()->IsWriteOnly())
    {
        pSubscriptionInfo = lSubtask.mConsolidatedOutputMemWriteSubscription;
    }
    else
    {
        size_t lReadOffset = lSubtask.mConsolidatedOutputMemReadSubscription.offset;
        size_t lReadLength = lSubtask.mConsolidatedOutputMemReadSubscription.length;
        size_t lReadSpan = lReadOffset + lReadLength;
        size_t lWriteOffset = lSubtask.mConsolidatedOutputMemWriteSubscription.offset;
        size_t lWriteLength = lSubtask.mConsolidatedOutputMemWriteSubscription.length;
        size_t lWriteSpan = lWriteOffset + lWriteLength;
        
        pSubscriptionInfo.offset = std::min(lReadOffset, lWriteOffset);
        pSubscriptionInfo.length = (std::max(lReadSpan, lWriteSpan) - pSubscriptionInfo.offset);
    }
    
	return true;
}

pmStatus pmSubscriptionManager::CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, void* pMem /* = NULL */, size_t pMemLength /* = 0 */, size_t pWriteOnlyUnprotectedRanges /* = 0 */, uint* pUnprotectedRanges /* = NULL */)
{
    pmSubscriptionInfo lUnifiedSubscriptionInfo;
    if(!GetUnifiedOutputMemSubscriptionForSubtask(pStub, pSubtaskId, lUnifiedSubscriptionInfo))
        PMTHROW(pmFatalErrorException());

#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        GET_SUBTASK(lSubtask, pStub, pSubtaskId);

        if(lSubtask.mShadowMem.get_ptr() != NULL)
            PMTHROW(pmFatalErrorException());
    }
#endif
    
    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    bool lIsLazyMem = (lMemSection->IsLazy() && pStub->GetType() == CPU && !pMem);

    bool lExplicitAllocation = false;
    char* lShadowMem = reinterpret_cast<char*>(mTask->CheckOutSubtaskMemory(lUnifiedSubscriptionInfo.length, (pMem != NULL)));

    if(!lShadowMem)
    {
#if 0
        std::cout << "Additional shadow mem allocation for " << lUnifiedSubscriptionInfo.length << " bytes" << std::endl;
#endif

        lShadowMem = reinterpret_cast<char*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateCheckOutMemory(lUnifiedSubscriptionInfo.length, lIsLazyMem));
        lExplicitAllocation = true;
    }
    
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());
    
#ifdef DUMP_SHADOW_MEM
    std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem " << (void*)lShadowMem << " allocated for device/subtask " << pStub << "/" << pSubtaskId << " " << std::endl;
#endif

    // Auto lock/unlock scope
    {
        GET_SUBTASK(lSubtask, pStub, pSubtaskId);

        lSubtask.mShadowMem.reset(lShadowMem);

        if(lExplicitAllocation)
            lSubtask.mShadowMem.GetDeallocator().SetExplicitAllocation();

        lSubtask.mShadowMem.GetDeallocator().SetTask(mTask);
    }

    if(pMem)
    {
        if(lMemSection->IsLazyWriteOnly())
        {
            GET_SUBTASK(lSubtask, pStub, pSubtaskId);

            size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
            char* lSrcMem = (char*)pMem;

            typedef std::map<size_t, size_t> mapType;
            mapType& lMap = lSubtask.mWriteOnlyLazyUnprotectedPageRangesMap;

            lSubtask.mWriteOnlyLazyUnprotectedPageCount = 0;
            
            for(size_t i = 0; i < pWriteOnlyUnprotectedRanges; ++i)
            {
                uint lStartPage = pUnprotectedRanges[2 * i];
                uint lCount = pUnprotectedRanges[2 * i +1];

                lMap[lStartPage] = lCount;
                lSubtask.mWriteOnlyLazyUnprotectedPageCount += lCount;
                
                size_t lMemSize = std::min(lCount * lPageSize, lUnifiedSubscriptionInfo.length - lStartPage * lPageSize);
                memcpy(lShadowMem + lStartPage * lPageSize, lSrcMem, lMemSize);

                lSrcMem += lMemSize;
            }
        }
        else
        {
            subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
            if(GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskId, false, lBeginIter, lEndIter))
            {
                void* lCurrPtr = pMem;
                for(; lBeginIter != lEndIter; ++lBeginIter)
                {
                    memcpy(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lCurrPtr, lBeginIter->second.first);
                    lCurrPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lCurrPtr) + lBeginIter->second.first);
                }
            }
        }
    }
    else if(!lIsLazyMem && pStub->GetType() == CPU)     // no need to copy for GPU; it will be copied to GPU memory directly and after kernel executes results will be put inside shadow memory
    {
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        if(GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskId, true, lBeginIter, lEndIter))
        {
            char* lMem = (char*)(lMemSection->GetMem());
            for(; lBeginIter != lEndIter; ++lBeginIter)
                memcpy(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lMem + lBeginIter->first, lBeginIter->second.first);
        }
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(lIsLazyMem)
    {
        // Lazy protect read subscriptions
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
    
        if(lMemSection->IsReadWrite())
        {
            if(GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskId, true, lBeginIter, lEndIter))
            {
                for(; lBeginIter != lEndIter; ++lBeginIter)
                    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lShadowMem + (lBeginIter->first - lUnifiedSubscriptionInfo.offset), lBeginIter->second.first, false, false);
            }
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
    
void pmSubscriptionManager::MarkSubtaskReadyForExecution(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    if(lSubtask.mReadyForExecution)
        PMTHROW(pmFatalErrorException());
    
    lSubtask.mReadyForExecution = true;
}

bool pmSubscriptionManager::IsSubtaskReadyForExecution(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    return lSubtask.mReadyForExecution;
}

void* pmSubscriptionManager::GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(lSubtask.mShadowMem.get_ptr)() << std::endl;
    #endif

	return (void*)(lSubtask.mShadowMem.get_ptr());
}

void pmSubscriptionManager::DestroySubtaskRangeShadowMem(pmExecutionStub* pStub, ulong pStartSubtaskId, ulong pEndSubtaskId)
{
    for(ulong lSubtaskId = pStartSubtaskId; lSubtaskId < pEndSubtaskId; ++lSubtaskId)
        DestroySubtaskShadowMem(pStub, lSubtaskId);
}

void pmSubscriptionManager::DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

	DestroySubtaskShadowMemInternal(lSubtask, pStub, pSubtaskId);
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
pmStatus pmSubscriptionManager::DestroySubtaskShadowMemInternal(pmSubtask& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId)
{
    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(pSubtask.mShadowMem.get_ptr()) << std::endl;
    #endif

	pSubtask.mShadowMem.reset(NULL);

	return pmSuccess;
}

void pmSubscriptionManager::CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBeginIter, subscription::subscriptionRecordType::const_iterator& pEndIter, ulong pShadowMemOffset)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem committed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(lSubtask.mShadowMem.get_ptr()) << std::endl;
    #endif
    
    char* lShadowMem = (char*)(lSubtask.mShadowMem.get_ptr());
    
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());

    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    char* lMem = (char*)(lMemSection->GetMem());
    
    subscription::subscriptionRecordType::const_iterator lIter = pBeginIter;
    for(; lIter != pEndIter; ++lIter)
        memcpy(lMem + lIter->first, lShadowMem + (lIter->first - pShadowMemOffset), lIter->second.first);
    
    DestroySubtaskShadowMemInternal(lSubtask, pStub, pSubtaskId);
}

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

pmStatus pmSubscriptionManager::FreezeSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    if(mTask->GetMemSectionRO())
    {
        subscriptionRecordType& lMap = lSubtask.mInputMemSubscriptions;
        if(lMap.empty())
        {
            size_t lOffset = lSubtask.mConsolidatedInputMemSubscription.offset;
            lMap[lOffset].first = lSubtask.mConsolidatedInputMemSubscription.length;
        }
    }
    
    pmMemSection* lMem = mTask->GetMemSectionRW();
	if(lMem)
    {
        if(lMem->IsReadWrite())
        {
            subscriptionRecordType& lMap = lSubtask.mOutputMemReadSubscriptions;
            if(lMap.empty())
            {
                size_t lOffset = lSubtask.mConsolidatedOutputMemReadSubscription.offset;
                lMap[lOffset].first = lSubtask.mConsolidatedOutputMemReadSubscription.length;
            }
        }

        subscriptionRecordType& lMap = lSubtask.mOutputMemWriteSubscriptions;
        if(lMap.empty())
        {
            size_t lOffset = lSubtask.mConsolidatedOutputMemWriteSubscription.offset;
            lMap[lOffset].first = lSubtask.mConsolidatedOutputMemWriteSubscription.length;
        }
    }
    
    return pmSuccess;
}

pmStatus pmSubscriptionManager::FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType, bool pPrefetch)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

#ifdef DUMP_SUBTASK_SUBSCRIPTION_LENGTH
    size_t lReadSubscriptionsLength = 0;
    size_t lWriteSubscriptionsLength = 0;
#endif
    
    ushort lPriority = (mTask->GetPriority() + (pPrefetch ? 1 : 0));
    
	if(mTask->GetMemSectionRO())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        subscriptionRecordType& lMap = lSubtask.mInputMemSubscriptions;
        
        lIter = lMap.begin();
        lEndIter = lMap.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
        #ifdef DUMP_SUBTASK_SUBSCRIPTION_LENGTH
            lReadSubscriptionsLength += lSubscription.length;
        #endif
            
            FetchInputMemSubscription(lSubtask, pDeviceType, lSubscription, lPriority, lIter->second.second);
        }
    }

    pmMemSection* lMem = mTask->GetMemSectionRW();
	if(lMem && lMem->IsReadWrite())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        subscriptionRecordType& lMap = lSubtask.mOutputMemReadSubscriptions;

        lIter = lMap.begin();
        lEndIter = lMap.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
        #ifdef DUMP_SUBTASK_SUBSCRIPTION_LENGTH
            lWriteSubscriptionsLength += lSubscription.length;
        #endif
            
            FetchOutputMemSubscription(lSubtask, pDeviceType, lSubscription, lPriority, lIter->second.second);
        }
    }

#ifdef DUMP_SUBTASK_SUBSCRIPTION_LENGTH
    std::stringstream lStream;
    lStream << "Subtask Id: " << pSubtaskId << "; Input Mem Read Subscriptions Length: " << lReadSubscriptionsLength << "; Output Mem Read Subscriptions Length: " << lWriteSubscriptionsLength ;
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif
    
    if(!pPrefetch)
    {
        WaitForSubscriptions(lSubtask, pStub);
        
    #ifdef SUPPORT_CUDA
        #ifdef SUPPORT_LAZY_MEMORY
            ClearInputMemLazyProtectionForCuda(lSubtask, pDeviceType);
        #endif
    #endif
    }
    
    return pmSuccess;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
pmStatus pmSubscriptionManager::FetchInputMemSubscription(pmSubtask& pSubtask, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, ushort pPriority, subscriptionData& pData)
{
	pmMemSection* lMemSection = mTask->GetMemSectionRO();
    bool lIsLazy = lMemSection->IsLazy();

    if(!lIsLazy || pDeviceType != CPU)
    {   
        std::vector<pmCommunicatorCommandPtr> lReceiveVector;
        size_t lOffset = pSubscriptionInfo.offset;
        size_t lLength = pSubscriptionInfo.length;

        lMemSection->GetPageAlignedAddresses(lOffset, lLength);
        
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, pPriority, lOffset, lLength, lReceiveVector);
        pData.receiveCommandVector.insert(pData.receiveCommandVector.end(), lReceiveVector.begin(), lReceiveVector.end());
    }
    
	return pmSuccess;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
pmStatus pmSubscriptionManager::FetchOutputMemSubscription(pmSubtask& pSubtask, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, ushort pPriority, subscriptionData& pData)
{
	pmMemSection* lMemSection = mTask->GetMemSectionRW();
    bool lIsLazy = lMemSection->IsLazy();
    
    if(!lMemSection->IsWriteOnly() && (!lIsLazy || pDeviceType != CPU))
    {   
        std::vector<pmCommunicatorCommandPtr> lReceiveVector;
        size_t lOffset = pSubscriptionInfo.offset;
        size_t lLength = pSubscriptionInfo.length;
        
        lMemSection->GetPageAlignedAddresses(lOffset, lLength);
        
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, pPriority, lOffset, lLength, lReceiveVector);
        pData.receiveCommandVector.insert(pData.receiveCommandVector.end(), lReceiveVector.begin(), lReceiveVector.end());
    }

	return pmSuccess;
}

/* Must be called with mSubtaskMapVector stub's lock acquired */
pmStatus pmSubscriptionManager::WaitForSubscriptions(pmSubtask& pSubtask, pmExecutionStub* pStub)
{
	if(mTask->GetMemSectionRO())
	{
        subscriptionRecordType::iterator lIter, lEndIter;
        
        subscriptionRecordType& lMap = pSubtask.mInputMemSubscriptions;
        lIter = lMap.begin();
        lEndIter = lMap.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = lIter->second.second.receiveCommandVector;
            pStub->WaitForNetworkFetch(lCommandVector);
		}
	}

	if(mTask->GetMemSectionRW())
	{
        subscriptionRecordType::iterator lIter, lEndIter;
        
        subscriptionRecordType& lMap = pSubtask.mOutputMemReadSubscriptions;
        lIter = lMap.begin();
        lEndIter = lMap.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = lIter->second.second.receiveCommandVector;
            pStub->WaitForNetworkFetch(lCommandVector);
		}
	}

	return pmSuccess;
}
    
#ifdef SUPPORT_LAZY_MEMORY
void pmSubscriptionManager::AddWriteOnlyLazyUnprotection(pmExecutionStub* pStub, ulong pSubtaskId, size_t pPageNum)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    typedef std::map<size_t, size_t> mapType;
    mapType& lMap = lSubtask.mWriteOnlyLazyUnprotectedPageRangesMap;
    
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
    
    ++lSubtask.mWriteOnlyLazyUnprotectedPageCount;

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
    
size_t pmSubscriptionManager::GetWriteOnlyLazyUnprotectedPagesCount(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    return lSubtask.mWriteOnlyLazyUnprotectedPageCount;
}
    
#ifdef SUPPORT_LAZY_MEMORY
const std::map<size_t, size_t>& pmSubscriptionManager::GetWriteOnlyLazyUnprotectedPageRanges(pmExecutionStub* pStub, ulong pSubtaskId)
{
    GET_SUBTASK(lSubtask, pStub, pSubtaskId);

    return lSubtask.mWriteOnlyLazyUnprotectedPageRangesMap;
}
#endif

#ifdef SUPPORT_CUDA
/* Must be called with mSubtaskMapVector stub's lock acquired */
void pmSubscriptionManager::ClearInputMemLazyProtectionForCuda(pmSubtask& pSubtask, pmDeviceType pDeviceType)
{
	pmMemSection* lMemSection = mTask->GetMemSectionRO();
    
    if(lMemSection && lMemSection->IsLazy() && pDeviceType == GPU_CUDA)
    {
        size_t lLazyMemAddr = reinterpret_cast<size_t>(lMemSection->GetReadOnlyLazyMemoryMapping());
        subscriptionRecordType::iterator lIter, lEndIter;
        
        subscriptionRecordType& lMap = pSubtask.mInputMemSubscriptions;
        
        lIter = lMap.begin();
        lEndIter = lMap.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscriptionInfo;
            lSubscriptionInfo.offset = lIter->first;
            lSubscriptionInfo.length = lIter->second.first;
        
            void* lAddr = reinterpret_cast<void*>(lLazyMemAddr + lSubscriptionInfo.offset);
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(lAddr, lSubscriptionInfo.length, true, true);
        }
    }
}
#endif
#endif


/* struct pmSubtask */    
pmStatus pmSubtask::Initialize(pmTask* pTask)
{
	pmMemSection* lInputMemSection = pTask->GetMemSectionRO();
	pmMemSection* lOutputMemSection = pTask->GetMemSectionRW();

	if(lInputMemSection)
	{
		mConsolidatedInputMemSubscription.offset = 0;
		mConsolidatedInputMemSubscription.length = lInputMemSection->GetLength();
	}

	if(lOutputMemSection)
	{
        mConsolidatedOutputMemReadSubscription.offset = 0;
        mConsolidatedOutputMemReadSubscription.length = lOutputMemSection->GetLength();

        mConsolidatedOutputMemWriteSubscription.offset = 0;
        mConsolidatedOutputMemWriteSubscription.length = lOutputMemSection->GetLength();
	}
    
    mScratchBufferSize = 0;
    mScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
    
    mWriteOnlyLazyUnprotectedPageCount = 0;
    mReadyForExecution = false;
    mReservedCudaGlobalMemSize = 0;
    
	return pmSuccess;
}

bool operator==(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2)
{
    return (pSubscription1.offset == pSubscription2.offset && pSubscription1.length == pSubscription2.length);
}
    
bool operator!=(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2)
{
    return !(pSubscription1 == pSubscription2);
}


/* shadowMemDeallocator */
void shadowMemDeallocator::operator()(void* pMem)
{
    if(pMem)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE(dShadowMemLock, pmSubscriptionManager::GetShadowMemLock().Lock(), pmSubscriptionManager::GetShadowMemLock().Unlock());

            pmSubscriptionManager::GetShadowMemMap().erase(pMem);
        }

        if(mExplicitAllocation)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(pMem);

            #ifdef DUMP_SHADOW_MEM
                std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem deallocated " << (void*)pMem << std::endl;
            #endif
        }
        else
        {
            mTask->RepoolCheckedOutSubtaskMemory(pMem);
        }
    }
}

}



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
#include "pmExecutionStub.h"
#include "pmCallbackUnit.h"
#include "pmCallback.h"
#include "pmHardware.h"

#include <string.h>

namespace pm
{

using namespace subscription;

std::map<void*, subscription::shadowMemDetails> pmSubscriptionManager::mShadowMemMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmSubscriptionManager::mShadowMemLock;

pmSubscriptionManager::pmSubscriptionManager(pmTask* pTask)
{
	mTask = pTask;
}

pmSubscriptionManager::~pmSubscriptionManager()
{
}

pmStatus pmSubscriptionManager::InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) != mSubtaskMap.end())
        PMTHROW(pmFatalErrorException());

	mSubtaskMap[lPair].Initialize(mTask);

	return pmSuccess;
}

pmStatus pmSubscriptionManager::RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if((pIsInputMem && !mTask->GetMemSectionRO()) || (!pIsInputMem && !mTask->GetMemSectionRW()))
        PMTHROW(pmFatalErrorException());

    subscriptionRecordType& lMap = pIsInputMem ? mSubtaskMap[lPair].mInputMemSubscriptions : mSubtaskMap[lPair].mOutputMemSubscriptions;
    pmSubscriptionInfo& lConsolidatedSubscription = pIsInputMem ? mSubtaskMap[lPair].mConsolidatedInputMemSubscription : mSubtaskMap[lPair].mConsolidatedOutputMemSubscription;
    
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
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
#ifdef SUPPORT_CUDA
    if(pStub->GetType() != GPU_CUDA)
		PMTHROW(pmFatalErrorException());
#else
        PMTHROW(pmFatalErrorException());
#endif

	mSubtaskMap[lPair].mCudaLaunchConf = pCudaLaunchConf;

	return pmSuccess;
}

pmCudaLaunchConf& pmSubscriptionManager::GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	return mSubtaskMap[lPair].mCudaLaunchConf;
}
    
void* pmSubscriptionManager::GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, size_t pBufferSize)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	char* lScratchBuffer = mSubtaskMap[lPair].mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
    {
        lScratchBuffer = new char[pBufferSize];
        mSubtaskMap[lPair].mScratchBuffer.reset(lScratchBuffer);
    }
    
    return lScratchBuffer;
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2, bool pIsInputMem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair1(pStub1, pSubtaskId1);
    std::pair<pmExecutionStub*, ulong> lPair2(pStub2, pSubtaskId2);
    
    if(mSubtaskMap.find(lPair1) == mSubtaskMap.end() || mSubtaskMap.find(lPair2) == mSubtaskMap.end())
        PMTHROW(pmFatalErrorException());

	if((pIsInputMem && !mTask->GetMemSectionRO()) || (!pIsInputMem && !mTask->GetMemSectionRW()))
		PMTHROW(pmFatalErrorException());
    
	pmSubscriptionInfo& lConsolidatedSubscription1 = pIsInputMem ? mSubtaskMap[lPair1].mConsolidatedInputMemSubscription : mSubtaskMap[lPair1].mConsolidatedOutputMemSubscription;
	pmSubscriptionInfo& lConsolidatedSubscription2 = pIsInputMem ? mSubtaskMap[lPair2].mConsolidatedInputMemSubscription : mSubtaskMap[lPair2].mConsolidatedOutputMemSubscription;
    
    if(lConsolidatedSubscription1 != lConsolidatedSubscription2)
        return false;

    subscriptionRecordType& lSubscriptions1 = pIsInputMem ? mSubtaskMap[lPair1].mInputMemSubscriptions : mSubtaskMap[lPair1].mOutputMemSubscriptions;
    subscriptionRecordType& lSubscriptions2 = pIsInputMem ? mSubtaskMap[lPair2].mInputMemSubscriptions : mSubtaskMap[lPair2].mOutputMemSubscriptions;
    
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
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	if(!mTask->GetMemSectionRO())
		return false;
    
	pBegin = mSubtaskMap[lPair].mInputMemSubscriptions.begin();
	pEnd = mSubtaskMap[lPair].mInputMemSubscriptions.end();
    
	return true;    
}
    
bool pmSubscriptionManager::GetNonConsolidatedOutputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	if(!mTask->GetMemSectionRW())
		return false;
    
	pBegin = mSubtaskMap[lPair].mOutputMemSubscriptions.begin();
	pEnd = mSubtaskMap[lPair].mOutputMemSubscriptions.end();
    
	return true;    
}

bool pmSubscriptionManager::GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRO())
		return false;

	pSubscriptionInfo = mSubtaskMap[lPair].mConsolidatedInputMemSubscription;

	return true;
}

bool pmSubscriptionManager::GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRW())
		return false;

	pSubscriptionInfo = mSubtaskMap[lPair].mConsolidatedOutputMemSubscription;

	return true;
}
    
pmStatus pmSubscriptionManager::CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, char* pMem /* = NULL */, size_t pMemLength /* = 0 */)
{
    pmSubscriptionInfo lSubscriptionInfo;
    if(!GetOutputMemSubscriptionForSubtask(pStub, pSubtaskId, lSubscriptionInfo))
        PMTHROW(pmFatalErrorException());
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);

#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        if(mSubtaskMap[lPair].mShadowMem.get_ptr() != NULL)
            PMTHROW(pmFatalErrorException());
    }
#endif
    
    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    bool lIsLazyMem = (lMemSection->IsLazy() && pStub->GetType() == CPU && !pMem);

    char* lShadowMem = (char*)(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateCheckOutMemory(lSubscriptionInfo.length, lIsLazyMem));
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem " << (void*)lShadowMem << " allocated for device/subtask " << pStub << "/" << pSubtaskId << " " << std::endl;
    #endif
    
        mSubtaskMap[lPair].mShadowMem.reset(lShadowMem);
    }
    
    if(pMem)
    {
        std::cout << "To be done: reduction non-consolidated shadow mem" << std::endl;
        if(pMemLength <= lSubscriptionInfo.length)
            memcpy(lShadowMem, pMem, pMemLength);
    }
    else if(!lIsLazyMem && pStub->GetType() == CPU)     // no need to copy for GPU; it will be copied to GPU memory directly and after kernel executes results will be put inside shadow memory
    {
        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskId, lBeginIter, lEndIter);
    
        char* lMem = (char*)(lMemSection->GetMem());
        for(; lBeginIter != lEndIter; ++lBeginIter)
            memcpy(lShadowMem + (lBeginIter->first - lSubscriptionInfo.offset), lMem + lBeginIter->first, lBeginIter->second.first);
    }
    
    if(lIsLazyMem)
    {
        FINALIZE_RESOURCE(dShadowMemLock, mShadowMemLock.Lock(), mShadowMemLock.Unlock());
        mShadowMemMap[(void*)lShadowMem].subscriptionInfo = lSubscriptionInfo;
        mShadowMemMap[(void*)lShadowMem].memSection = lMemSection;
    }
    
	return pmSuccess;
}

void* pmSubscriptionManager::GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem accessed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(mSubtaskMap[lPair].mShadowMem.get_ptr)() << std::endl;
    #endif

	return (void*)(mSubtaskMap[lPair].mShadowMem.get_ptr());
}

pmStatus pmSubscriptionManager::DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mSubtaskMap.find(lPair) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem destroyed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(mSubtaskMap[lPair].mShadowMem.get_ptr()) << std::endl;
    #endif

	mSubtaskMap[lPair].mShadowMem.reset(NULL);

	return pmSuccess;
}

void pmSubscriptionManager::CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBeginIter, subscription::subscriptionRecordType::const_iterator& pEndIter, ulong pShadowMemOffset)
{
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
    char* lShadowMem = NULL;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    #ifdef DUMP_SHADOW_MEM
        std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem committed for device/subtask " << pStub << "/" << pSubtaskId << " " << (void*)(mSubtaskMap[lPair].mShadowMem.get_ptr()) << std::endl;
    #endif
    
        lShadowMem = (char*)(mSubtaskMap[lPair].mShadowMem.get_ptr());
    }
    
    if(!lShadowMem)
        PMTHROW(pmFatalErrorException());
    
    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    char* lMem = (char*)(lMemSection->GetMem());
    
    subscription::subscriptionRecordType::const_iterator lIter = pBeginIter;
    for(; lIter != pEndIter; ++lIter)
        memcpy(lMem + lIter->first, lShadowMem + (lIter->first - pShadowMemOffset), lIter->second.first);
    
    DestroySubtaskShadowMem(pStub, pSubtaskId);
}
    
pmMemSection* pmSubscriptionManager::FindMemSectionContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr)
{
	FINALIZE_RESOURCE(dShadowMemLock, mShadowMemLock.Lock(), mShadowMemLock.Unlock());

    typedef std::map<void*, subscription::shadowMemDetails> mapType;
    mapType::iterator lStartIter;
    mapType::iterator* lStartIterAddr = &lStartIter;
    
    char* lAddress = static_cast<char*>(pAddr);
    FIND_FLOOR_ELEM(mapType, mShadowMemMap, lAddress, lStartIterAddr);
    
    if(lStartIterAddr)
    {
        char* lMemAddress = static_cast<char*>(lStartIter->first);
        subscription::shadowMemDetails& lShadowMemDetails = lStartIter->second;

        size_t lLength = lShadowMemDetails.subscriptionInfo.length;
        
        if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
        {
            pShadowMemOffset = lShadowMemDetails.subscriptionInfo.offset;
            pShadowMemBaseAddr = lStartIter->first;
            return lShadowMemDetails.memSection;
        }
    }
    
    return NULL;
}

pmStatus pmSubscriptionManager::FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType)
{
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
    
	if(mTask->GetMemSectionRO())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[lPair].mInputMemSubscriptions;
            if(lMap.empty())
            {
                size_t lOffset = mSubtaskMap[lPair].mConsolidatedInputMemSubscription.offset;
                lMap[lOffset].first = mSubtaskMap[lPair].mConsolidatedInputMemSubscription.length;
            }
            
            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
            FetchInputMemSubscription(pStub, pSubtaskId, pDeviceType, lSubscription, lIter->second.second);
        }
    }

	if(mTask->GetMemSectionRW())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[lPair].mOutputMemSubscriptions;
            if(lMap.empty())
            {
                size_t lOffset = mSubtaskMap[lPair].mConsolidatedOutputMemSubscription.offset;
                lMap[lOffset].first = mSubtaskMap[lPair].mConsolidatedOutputMemSubscription.length;
            }

            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
            FetchOutputMemSubscription(pStub, pSubtaskId, pDeviceType, lSubscription, lIter->second.second);
        }
    }

	return WaitForSubscriptions(pStub, pSubtaskId);
}

pmStatus pmSubscriptionManager::FetchInputMemSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, subscriptionData& pData)
{
    std::vector<pmCommunicatorCommandPtr> lReceiveVector;
	pmMemSection* lMemSection = mTask->GetMemSectionRO();
    bool lIsLazy = lMemSection->IsLazy();

    if(!lIsLazy || pDeviceType != CPU)
    {   
        size_t lOffset = pSubscriptionInfo.offset;
        size_t lLength = pSubscriptionInfo.length;

    #ifdef SUPPORT_LAZY_MEMORY
        if(lIsLazy)
            lMemSection->GetPageAlignedAddresses(lOffset, lLength);
    #endif
        
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, mTask->GetPriority(), lOffset, lLength, lReceiveVector);
    }

    pData.receiveCommandVector.insert(pData.receiveCommandVector.end(), lReceiveVector.begin(), lReceiveVector.end());
    
	return pmSuccess;
}

pmStatus pmSubscriptionManager::FetchOutputMemSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, subscriptionData& pData)
{
    std::vector<pmCommunicatorCommandPtr> lReceiveVector;
	pmMemSection* lMemSection = mTask->GetMemSectionRW();
    bool lIsWriteOnly = (lMemSection->GetMemInfo() == OUTPUT_MEM_WRITE_ONLY);
    bool lIsLazy = lMemSection->IsLazy();
    
    if(!lIsWriteOnly && (!lIsLazy || pDeviceType != CPU))
    {   
        size_t lOffset = pSubscriptionInfo.offset;
        size_t lLength = pSubscriptionInfo.length;
        
    #ifdef SUPPORT_LAZY_MEMORY
        if(lIsLazy)
            lMemSection->GetPageAlignedAddresses(lOffset, lLength);
    #endif
        
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, mTask->GetPriority(), lOffset, lLength, lReceiveVector);
    }
    
    pData.receiveCommandVector.insert(pData.receiveCommandVector.end(), lReceiveVector.begin(), lReceiveVector.end());

	return pmSuccess;
}

pmStatus pmSubscriptionManager::WaitForSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId)
{
    pmStatus lStatus;
    
    std::pair<pmExecutionStub*, ulong> lPair(pStub, pSubtaskId);
	if(mTask->GetMemSectionRO())
	{
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[lPair].mInputMemSubscriptions;
            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = lIter->second.second.receiveCommandVector;
            std::vector<pmCommunicatorCommandPtr>::iterator lInnerIter = lCommandVector.begin(), lInnerEndIter = lCommandVector.end();
            
			for(; lInnerIter != lInnerEndIter; ++lInnerIter)
			{
				if((*lInnerIter).get())
				{
                    while((lStatus = (*lInnerIter)->GetStatus()) == pmStatusUnavailable)
                    {
                        if((*lInnerIter)->WaitWithTimeOut(GetIntegralCurrentTimeInSecs() + MEMORY_TRANSFER_TIMEOUT))
                        {
                            if(pStub->RequiresPrematureExit(pSubtaskId))
                                PMTHROW_NODUMP(pmPrematureExitException());
                        }
                    }
                
                    if(lStatus != pmSuccess)
						PMTHROW(pmMemoryFetchException());
				}
			}
		}
	}

	if(mTask->GetMemSectionRW())
	{
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[lPair].mOutputMemSubscriptions;
            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = lIter->second.second.receiveCommandVector;
            std::vector<pmCommunicatorCommandPtr>::iterator lInnerIter = lCommandVector.begin(), lInnerEndIter = lCommandVector.end();
            
			for(; lInnerIter != lInnerEndIter; ++lInnerIter)
			{
				if((*lInnerIter).get())
				{
                    while((lStatus = (*lInnerIter)->GetStatus()) == pmStatusUnavailable)
                    {
                        if((*lInnerIter)->WaitWithTimeOut(GetIntegralCurrentTimeInSecs() + MEMORY_TRANSFER_TIMEOUT))
                        {
                            if(pStub->RequiresPrematureExit(pSubtaskId))
                                PMTHROW_NODUMP(pmPrematureExitException());
                        }
                    }
                
                    if(lStatus != pmSuccess)
						PMTHROW(pmMemoryFetchException());
				}
			}
		}
	}

	return pmSuccess;
}


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
		mConsolidatedOutputMemSubscription.offset = 0;
		mConsolidatedOutputMemSubscription.length = lOutputMemSection->GetLength();
	}
    
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
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(pMem);

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE(dShadowMemLock, pmSubscriptionManager::mShadowMemLock.Lock(), pmSubscriptionManager::mShadowMemLock.Unlock());

            pmSubscriptionManager::mShadowMemMap.erase(pMem);
        }

        #ifdef DUMP_SHADOW_MEM
            std::cout << "[Host " << pmGetHostId() << "]: " << "Shadow Mem deallocated " << (void*)pMem << std::endl;
        #endif
    }
}


/* class pmUserLibraryCodeAutoPtr */
pmSubtaskTerminationCheckPointAutoPtr::pmSubtaskTerminationCheckPointAutoPtr(pmExecutionStub* pStub, ulong pSubtaskId)
    : mStub(pStub)
    , mSubtaskId(pSubtaskId)
    , mExecutingLibraryCode(mStub->IsInsideLibraryCode(mIsPastCancellationStage))
{
    if(!mIsPastCancellationStage)
    {
        if(!mExecutingLibraryCode)
            mStub->MarkInsideLibraryCode(pSubtaskId);
        
        mStub->CheckForSubtaskTermination(mSubtaskId);
    }
}
    
pmSubtaskTerminationCheckPointAutoPtr::~pmSubtaskTerminationCheckPointAutoPtr()
{
    if(!mIsPastCancellationStage)
    {
        mStub->CheckForSubtaskTermination(mSubtaskId);

        if(!mExecutingLibraryCode)
            mStub->MarkInsideUserCode(mSubtaskId);
    }
}
    
}



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

namespace pm
{

using namespace subscription;

pmSubscriptionManager::pmSubscriptionManager(pmTask* pTask)
{
	mTask = pTask;
}

pmSubscriptionManager::~pmSubscriptionManager()
{
}

pmStatus pmSubscriptionManager::InitializeSubtaskDefaults(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) != mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	mSubtaskMap[pSubtaskId].Initialize(mTask);

	return pmSuccess;
}

pmStatus pmSubscriptionManager::RegisterSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if((pIsInputMem && !mTask->GetMemSectionRO()) || (!pIsInputMem && !mTask->GetMemSectionRW()))
        PMTHROW(pmFatalErrorException());

    subscriptionRecordType& lMap = pIsInputMem ? mSubtaskMap[pSubtaskId].mInputMemSubscriptions : mSubtaskMap[pSubtaskId].mOutputMemSubscriptions;
    pmSubscriptionInfo& lConsolidatedSubscription = pIsInputMem ? mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription : mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription;
    
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

pmStatus pmSubscriptionManager::SetCudaLaunchConf(ulong pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	mSubtaskMap[pSubtaskId].mCudaLaunchConf = pCudaLaunchConf;

	return pmSuccess;
}

pmCudaLaunchConf& pmSubscriptionManager::GetCudaLaunchConf(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	return mSubtaskMap[pSubtaskId].mCudaLaunchConf;
}
    
void* pmSubscriptionManager::GetScratchBuffer(ulong pSubtaskId, size_t pBufferSize)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	char* lScratchBuffer = mSubtaskMap[pSubtaskId].mScratchBuffer.get_ptr();
    if(!lScratchBuffer)
    {
        lScratchBuffer = new char[pBufferSize];
        mSubtaskMap[pSubtaskId].mScratchBuffer.reset(lScratchBuffer);
    }
    
    return lScratchBuffer;
}
    
bool pmSubscriptionManager::SubtasksHaveMatchingSubscriptions(ulong pSubtaskId1, ulong pSubtaskId2, bool pIsInputMem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(mSubtaskMap.find(pSubtaskId1) == mSubtaskMap.end() || mSubtaskMap.find(pSubtaskId2) == mSubtaskMap.end())
        PMTHROW(pmFatalErrorException());

	if((pIsInputMem && !mTask->GetMemSectionRO()) || (!pIsInputMem && !mTask->GetMemSectionRW()))
		PMTHROW(pmFatalErrorException());
    
	pmSubscriptionInfo& lConsolidatedSubscription1 = pIsInputMem ? mSubtaskMap[pSubtaskId1].mConsolidatedInputMemSubscription : mSubtaskMap[pSubtaskId1].mConsolidatedOutputMemSubscription;
	pmSubscriptionInfo& lConsolidatedSubscription2 = pIsInputMem ? mSubtaskMap[pSubtaskId2].mConsolidatedInputMemSubscription : mSubtaskMap[pSubtaskId2].mConsolidatedOutputMemSubscription;
    
    if(lConsolidatedSubscription1 != lConsolidatedSubscription2)
        return false;

    subscriptionRecordType& lSubscriptions1 = pIsInputMem ? mSubtaskMap[pSubtaskId1].mInputMemSubscriptions : mSubtaskMap[pSubtaskId1].mOutputMemSubscriptions;
    subscriptionRecordType& lSubscriptions2 = pIsInputMem ? mSubtaskMap[pSubtaskId2].mInputMemSubscriptions : mSubtaskMap[pSubtaskId2].mOutputMemSubscriptions;
    
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
    
bool pmSubscriptionManager::GetNonConsolidatedInputMemSubscriptionsForSubtask(ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	if(!mTask->GetMemSectionRO())
		return false;
    
	pBegin = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.begin();
	pEnd = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.end();
    
	return true;    
}
    
bool pmSubscriptionManager::GetNonConsolidatedOutputMemSubscriptionsForSubtask(ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());
    
	if(!mTask->GetMemSectionRW())
		return false;
    
	pBegin = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.begin();
	pEnd = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.end();
    
	return true;    
}

bool pmSubscriptionManager::GetInputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRO())
		return false;

	pSubscriptionInfo = mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription;

	return true;
}

bool pmSubscriptionManager::GetOutputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRW())
		return false;

	pSubscriptionInfo = mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription;

	return true;
}

pmStatus pmSubscriptionManager::FetchSubtaskSubscriptions(ulong pSubtaskId, pmDeviceTypes pDeviceType)
{
	if(mTask->GetMemSectionRO())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[pSubtaskId].mInputMemSubscriptions;
            if(lMap.empty())
            {
                size_t lOffset = mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.offset;
                lMap[lOffset].first = mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.length;
            }
            
            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
            FetchSubscription(pSubtaskId, true, pDeviceType, lSubscription, lIter->second.second);
        }
    }

	if(mTask->GetMemSectionRW())
    {
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions;
            if(lMap.empty())
            {
                size_t lOffset = mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.offset;
                lMap[lOffset].first = mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.length;
            }

            lIter = lMap.begin();
            lEndIter = lMap.end();
        }
        
        for(; lIter != lEndIter; ++lIter)
        {
            pmSubscriptionInfo lSubscription;
            lSubscription.offset = lIter->first;
            lSubscription.length = lIter->second.first;
            
            FetchSubscription(pSubtaskId, false, pDeviceType, lSubscription, lIter->second.second);
        }
    }

	return WaitForSubscriptions(pSubtaskId);
}

pmStatus pmSubscriptionManager::FetchSubscription(ulong pSubtaskId, bool pIsInputMem, pmDeviceTypes pDeviceType, pmSubscriptionInfo pSubscriptionInfo, subscriptionData& pData)
{
	pmMemSection* lMemSection = NULL;
    bool lIsWriteOnly = false;

	if(pIsInputMem)
    {
		lMemSection = mTask->GetMemSectionRO();
    }
	else
    {
		lMemSection = mTask->GetMemSectionRW();
        lIsWriteOnly = (((pmOutputMemSection*)lMemSection)->GetAccessType() == pmOutputMemSection::WRITE_ONLY);
    }
    
	pData.receiveCommandVector = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection->GetMem(), mTask->GetPriority(), pSubscriptionInfo.offset, pSubscriptionInfo.length, (lMemSection->IsLazy() || lIsWriteOnly));
    
#ifdef SUPPORT_LAZY_MEMORY
    if(lMemSection->IsLazy() && pDeviceType != CPU)
    {
        std::vector<pmCommunicatorCommandPtr> lReceiveVector;
        size_t lOffset = pSubscriptionInfo.offset;
        size_t lLength = pSubscriptionInfo.length;
        
        lMemSection->GetPageAlignedAddresses(lOffset, lLength);

        lReceiveVector = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection->GetMem(), mTask->GetPriority(), lOffset, lLength, false);
        
        pData.receiveCommandVector.insert(pData.receiveCommandVector.end(), lReceiveVector.begin(), lReceiveVector.end());
    }
#endif

	return pmSuccess;
}

pmStatus pmSubscriptionManager::WaitForSubscriptions(ulong pSubtaskId)
{
	if(mTask->GetMemSectionRO())
	{
        subscriptionRecordType::iterator lIter, lEndIter;
        
        // Auto lock/unlock scope
        {        
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            subscriptionRecordType& lMap = mSubtaskMap[pSubtaskId].mInputMemSubscriptions;
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
					if((*lInnerIter)->WaitForFinish() != pmSuccess)
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
            subscriptionRecordType& lMap = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions;
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
					if((*lInnerIter)->WaitForFinish() != pmSuccess)
						PMTHROW(pmMemoryFetchException());
				}
			}
		}
	}

	return pmSuccess;
}
    
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
    
    mScratchBuffer = NULL;

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

}



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

	if(pIsInputMem)
	{
		if(!mTask->GetMemSectionRO())
			PMTHROW(pmFatalErrorException());

		mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first.push_back(pSubscriptionInfo);
        if(mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first.size() == 1)
        {
            mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription = pSubscriptionInfo;
        }
        else
        {
            size_t lExistingOffset = mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.offset;
            size_t lExistingLength = mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.length;
            size_t lExistingSpan = lExistingOffset + lExistingLength;
            size_t lNewOffset = pSubscriptionInfo.offset;
            size_t lNewLength = pSubscriptionInfo.length;
            size_t lNewSpan = lNewOffset + lNewLength;
            
            mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.offset = std::min(lExistingOffset, lNewOffset);
            mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.length = (std::max(lExistingSpan, lNewSpan) - mSubtaskMap[pSubtaskId].mConsolidatedInputMemSubscription.offset);
        }
	}
	else
	{
		if(!mTask->GetMemSectionRW())
			PMTHROW(pmFatalErrorException());

		mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first.push_back(pSubscriptionInfo);
        if(mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first.size() == 1)
        {
            mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription = pSubscriptionInfo;
        }
        else
        {
            size_t lExistingOffset = mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.offset;
            size_t lExistingLength = mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.length;
            size_t lExistingSpan = lExistingOffset + lExistingLength;
            size_t lNewOffset = pSubscriptionInfo.offset;
            size_t lNewLength = pSubscriptionInfo.length;
            size_t lNewSpan = lNewOffset + lNewLength;
            
            mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.offset = std::min(lExistingOffset, lNewOffset);
            mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.length = (std::max(lExistingSpan, lNewSpan) - mSubtaskMap[pSubtaskId].mConsolidatedOutputMemSubscription.offset);
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
        size_t lCount = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first.size();
        mSubtaskMap[pSubtaskId].mInputMemSubscriptions.second.resize(lCount);
        for(size_t i=0; i<lCount; ++i)
        {
            pmSubscriptionInfo& lSubscription = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first[i];
            subscriptionData lSubscriptionData;
            FetchSubscription(pSubtaskId, true, pDeviceType, lSubscription, lSubscriptionData);

            // Auto lock/unlock scope
            {
                FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
                mSubtaskMap[pSubtaskId].mInputMemSubscriptions.second[i] = lSubscriptionData;
            }
        }
    }

	if(mTask->GetMemSectionRW())
    {
        size_t lCount = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first.size();
        mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.second.resize(lCount);
        for(size_t i=0; i<lCount; ++i)
        {
            pmSubscriptionInfo& lSubscription = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first[i];
            subscriptionData lSubscriptionData;
            FetchSubscription(pSubtaskId, false, pDeviceType, lSubscription, lSubscriptionData);

            // Auto lock/unlock scope
            {
                FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
                mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.second[i] = lSubscriptionData;
            }
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
	size_t i, j, lSize, lInnerSize;

	if(mTask->GetMemSectionRO())
	{
		std::vector<subscriptionData>* lInputMemVector = NULL;

		// Auto lock/unlock scope
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
			lInputMemVector = &(mSubtaskMap[pSubtaskId].mInputMemSubscriptions.second);
		}

		lSize = lInputMemVector->size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = ((*lInputMemVector)[i]).receiveCommandVector;

			lInnerSize = lCommandVector.size();
			for(j=0; j<lInnerSize; ++j)
			{
				pmCommunicatorCommandPtr lCommand = lCommandVector[j];
				if(lCommand)
				{
					if(lCommand->WaitForFinish() != pmSuccess)
						PMTHROW(pmMemoryFetchException());
				}
			}
		}
	}

	if(mTask->GetMemSectionRW())
	{
		std::vector<subscriptionData>* lOutputMemVector;

		// Auto lock/unlock scope
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
			lOutputMemVector = &(mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.second);
		}

		lSize = lOutputMemVector->size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = ((*lOutputMemVector)[i]).receiveCommandVector;

			lInnerSize = lCommandVector.size();
			for(j=0; j<lInnerSize; ++j)
			{
				pmCommunicatorCommandPtr lCommand = lCommandVector[j];
				if(lCommand)
				{
					if(lCommand->WaitForFinish() != pmSuccess)
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

}



#include "pmSubscriptionManager.h"
#include "pmCommunicator.h"
#include "pmTaskManager.h"
#include "pmMemoryManager.h"
#include "pmMemSection.h"
#include "pmTask.h"

namespace pm
{

pmSubscriptionManager::pmSubscriptionManager(pmTask* pTask)
{
	mTask = pTask;
}

pmSubscriptionManager::~pmSubscriptionManager()
{
}

pmStatus pmSubscriptionManager::SetDefaultSubscriptions(ulong pSubtaskId)
{
	pmSubscriptionInfo lInputMemSubscription, lOutputMemSubscription;

	pmMemSection* lInputMemSection = mTask->GetMemSectionRO();
	pmMemSection* lOutputMemSection = mTask->GetMemSectionRO();

	if(lInputMemSection)
	{
		lInputMemSubscription.offset = 0;
		lInputMemSubscription.length = lInputMemSection->GetLength();

		RegisterSubscription(pSubtaskId, true, lInputMemSubscription);
	}

	if(lOutputMemSection)
	{
		lOutputMemSubscription.offset = 0;
		lOutputMemSubscription.length = lOutputMemSection->GetLength();
		RegisterSubscription(pSubtaskId, false, lOutputMemSubscription);
	}

	return pmSuccess;
}

pmStatus pmSubscriptionManager::RegisterSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo)
{
	// Only one subscription allowed per subtask (for now)
	subscriptionData lSubscriptionData;
	if(pIsInputMem)
	{
		if(!mTask->GetMemSectionRO())
			return pmSuccess;

		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mInputMemResourceLock, Lock(), Unlock());
	/*	if(mInputMemSubscriptions.find(pSubtaskId) != mInputMemSubscriptions.end())
		{
			MAP_VALUE_DATA_TYPE lPair = mInputMemSubscriptions[pSubtaskId];
			lPair.first.push_back(pSubscriptionInfo);
			lPair.second.push_back(lSubscriptionData);
		}
		else
		{
	*/
			std::vector<pmSubscriptionInfo> lVector1;
			std::vector<subscriptionData> lVector2;
			lVector1.push_back(pSubscriptionInfo);
			lVector2.push_back(lSubscriptionData);
			mInputMemSubscriptions[pSubtaskId] = MAP_VALUE_DATA_TYPE(lVector1, lVector2);
	//	}

		//return FetchSubscription(pSubtaskId, true, pSubscriptionInfo, mInputMemSubscriptions[pSubtaskId].second[mInputMemSubscriptions[pSubtaskId].second.size()-1]);
	}
	else
	{
		if(!mTask->GetMemSectionRW())
			return pmSuccess;

		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOutputMemResourceLock, Lock(), Unlock());
	/*	if(mOutputMemSubscriptions.find(pSubtaskId) != mOutputMemSubscriptions.end())
		{
			MAP_VALUE_DATA_TYPE lPair = mOutputMemSubscriptions[pSubtaskId];
			lPair.first.push_back(pSubscriptionInfo);
			lPair.second.push_back(lSubscriptionData);
		}
		else
		{
	*/
			std::vector<pmSubscriptionInfo> lVector1;
			std::vector<subscriptionData> lVector2;
			lVector1.push_back(pSubscriptionInfo);
			lVector2.push_back(lSubscriptionData);
			mOutputMemSubscriptions[pSubtaskId] = MAP_VALUE_DATA_TYPE(lVector1, lVector2);
	//	}

		//return FetchSubscription(pSubtaskId, false, pSubscriptionInfo, mOutputMemSubscriptions[pSubtaskId].second[mInputMemSubscriptions[pSubtaskId].second.size()-1]);
	}

	return pmSuccess;
}

bool pmSubscriptionManager::GetInputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mInputMemResourceLock, Lock(), Unlock());
	if(mInputMemSubscriptions.find(pSubtaskId) != mInputMemSubscriptions.end())
	{
		pSubscriptionInfo = mInputMemSubscriptions[pSubtaskId].first[0];
		return true;
	}

	return false;
}

bool pmSubscriptionManager::GetOutputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOutputMemResourceLock, Lock(), Unlock());
	if(mOutputMemSubscriptions.find(pSubtaskId) != mOutputMemSubscriptions.end())
	{
		pSubscriptionInfo = mOutputMemSubscriptions[pSubtaskId].first[0];
		return true;
	}

	return false;
}

pmStatus pmSubscriptionManager::FetchSubtaskSubscriptions(ulong pSubtaskId)
{
	pmSubscriptionInfo lInputMemSubscription, lOutputMemSubscription;
	subscriptionData lInputMemSubscriptionData, lOutputMemSubscriptionData;

	if(GetInputMemSubscriptionForSubtask(pSubtaskId, lInputMemSubscription))
		FetchSubscription(pSubtaskId, true, lInputMemSubscription, lInputMemSubscriptionData);

	if(GetOutputMemSubscriptionForSubtask(pSubtaskId, lOutputMemSubscription))
		FetchSubscription(pSubtaskId, false, lOutputMemSubscription, lOutputMemSubscriptionData);

	return WaitForSubscriptions(pSubtaskId);
}

pmStatus pmSubscriptionManager::FetchSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo, subscriptionData& pData)
{
	pmMemSection* lMemSection = NULL;
		
	if(pIsInputMem)
		lMemSection = mTask->GetMemSectionRO();
	else
		lMemSection = mTask->GetMemSectionRW();

	pData.receiveCommandVector = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(lMemSection, mTask->GetPriority(), pSubscriptionInfo.offset, pSubscriptionInfo.length);
	lMemSection->SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(lMemSection->GetMem()), pSubscriptionInfo.offset, pSubscriptionInfo.length);

	return pmSuccess;
}

pmStatus pmSubscriptionManager::WaitForSubscriptions(ulong pSubtaskId)
{
	size_t i, j, lSize, lInnerSize;

	if(mTask->GetMemSectionRO())
	{
		mInputMemResourceLock.Lock();
		std::vector<subscriptionData> lInputMemVector = mInputMemSubscriptions[pSubtaskId].second;
		mInputMemSubscriptions.erase(pSubtaskId);
		mInputMemResourceLock.Unlock();

		lSize = lInputMemVector.size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = (lInputMemVector[i]).receiveCommandVector;
		
			lInnerSize = lCommandVector.size();
			for(j=0; j<lInnerSize; ++j)
			{
				pmCommunicatorCommandPtr lCommand = lCommandVector[i];
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
		mOutputMemResourceLock.Lock();
		std::vector<subscriptionData> lOutputMemVector = mOutputMemSubscriptions[pSubtaskId].second;
		mOutputMemSubscriptions.erase(pSubtaskId);
		mOutputMemResourceLock.Unlock();

		lSize = lOutputMemVector.size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = (lOutputMemVector[i]).receiveCommandVector;
		
			lInnerSize = lCommandVector.size();
			for(j=0; j<lInnerSize; ++j)
			{
				pmCommunicatorCommandPtr lCommand = lCommandVector[i];
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

}


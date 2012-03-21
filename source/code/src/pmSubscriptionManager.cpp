
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
    {
    std::cout << "SUBTASK ID ALREADY PRESENT " << pSubtaskId << std::endl;
		PMTHROW(pmFatalErrorException());
    }

	pmSubtask lSubtask;
	lSubtask.Initialize(mTask);
	mSubtaskMap[pSubtaskId] = lSubtask;

	return pmSuccess;
}

pmStatus pmSubscriptionManager::RegisterSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	// Only one subscription allowed per subtask (for now)
	if(pIsInputMem)
	{
		if(!mTask->GetMemSectionRO())
			PMTHROW(pmFatalErrorException());

		std::vector<pmSubscriptionInfo> lVector;
		lVector.push_back(pSubscriptionInfo);
		mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first = lVector;
	}
	else
	{
		if(!mTask->GetMemSectionRW())
			PMTHROW(pmFatalErrorException());

		std::vector<pmSubscriptionInfo> lVector;
		lVector.push_back(pSubscriptionInfo);
		mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first = lVector;
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

bool pmSubscriptionManager::GetInputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRO())
		return false;

	pSubscriptionInfo = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.first[0];

	return true;
}

bool pmSubscriptionManager::GetOutputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mSubtaskMap.find(pSubtaskId) == mSubtaskMap.end())
		PMTHROW(pmFatalErrorException());

	if(!mTask->GetMemSectionRW())
		return false;

	pSubscriptionInfo = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.first[0];

	return true;
}

pmStatus pmSubscriptionManager::FetchSubtaskSubscriptions(ulong pSubtaskId)
{
	pmSubscriptionInfo lInputMemSubscription, lOutputMemSubscription;
	subscriptionData lInputMemSubscriptionData, lOutputMemSubscriptionData;
    
	if(GetInputMemSubscriptionForSubtask(pSubtaskId, lInputMemSubscription))
    {
		FetchSubscription(pSubtaskId, true, lInputMemSubscription, lInputMemSubscriptionData);

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            mSubtaskMap[pSubtaskId].mInputMemSubscriptions.second.push_back(lInputMemSubscriptionData);
        }
    }

	if(GetOutputMemSubscriptionForSubtask(pSubtaskId, lOutputMemSubscription))
    {
		FetchSubscription(pSubtaskId, false, lOutputMemSubscription, lOutputMemSubscriptionData);

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.second.push_back(lOutputMemSubscriptionData);
        }
    }

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
            
	lMemSection->AcquireOwnershipImmediate(pSubscriptionInfo.offset, pSubscriptionInfo.length);

	return pmSuccess;
}

pmStatus pmSubscriptionManager::WaitForSubscriptions(ulong pSubtaskId)
{
	size_t i, j, lSize, lInnerSize;

	if(mTask->GetMemSectionRO())
	{
		std::vector<subscriptionData> lInputMemVector;

		// Auto lock/unlock scope
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
			lInputMemVector = mSubtaskMap[pSubtaskId].mInputMemSubscriptions.second;
		}

		lSize = lInputMemVector.size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = (lInputMemVector[i]).receiveCommandVector;

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
		std::vector<subscriptionData> lOutputMemVector;

		// Auto lock/unlock scope
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
			lOutputMemVector = mSubtaskMap[pSubtaskId].mOutputMemSubscriptions.second;
		}

		lSize = lOutputMemVector.size();
		for(i=0; i<lSize; ++i)
		{
			std::vector<pmCommunicatorCommandPtr>& lCommandVector = (lOutputMemVector[i]).receiveCommandVector;

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
	pmMemSection* lOutputMemSection = pTask->GetMemSectionRO();

	if(lInputMemSection)
	{
		pmSubscriptionInfo lInputMemSubscription;

		lInputMemSubscription.offset = 0;
		lInputMemSubscription.length = lInputMemSection->GetLength();

		subscriptionData lSubscriptionData;
		std::vector<pmSubscriptionInfo> lVector1;
		std::vector<subscriptionData> lVector2;
		lVector1.push_back(lInputMemSubscription);
		lVector2.push_back(lSubscriptionData);
		mInputMemSubscriptions = SUBSCRIPTION_DATA_TYPE(lVector1, lVector2);
	}

	if(lOutputMemSection)
	{
		pmSubscriptionInfo lOutputMemSubscription;

		lOutputMemSubscription.offset = 0;
		lOutputMemSubscription.length = lOutputMemSection->GetLength();

		subscriptionData lSubscriptionData;
		std::vector<pmSubscriptionInfo> lVector1;
		std::vector<subscriptionData> lVector2;
		lVector1.push_back(lOutputMemSubscription);
		lVector2.push_back(lSubscriptionData);
		mOutputMemSubscriptions = SUBSCRIPTION_DATA_TYPE(lVector1, lVector2);
	}

	return pmSuccess;
}

}


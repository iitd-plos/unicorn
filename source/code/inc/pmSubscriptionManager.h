
#ifndef __PM_SUBSCRIPTION_MANAGER__
#define __PM_SUBSCRIPTION_MANAGER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"

#include <map>

namespace pm
{

class pmTask;


#define SUBSCRIPTION_DATA_TYPE std::pair<std::vector<pmSubscriptionInfo>, std::vector<subscriptionData> >

namespace subscription
{
	typedef struct subscriptionData
	{
		std::vector<pmCommunicatorCommandPtr> receiveCommandVector;
	} subscriptionData;

	typedef struct pmSubtask
	{
		pmCudaLaunchConf mCudaLaunchConf;

		// Only one contiguous input mem and one output mem subscription for each subtask

		SUBSCRIPTION_DATA_TYPE mInputMemSubscriptions;
		SUBSCRIPTION_DATA_TYPE mOutputMemSubscriptions;

		pmStatus Initialize(pmTask* pTask);
	} pmSubtask;
}

class pmSubscriptionManager : public pmBase
{
	public:
		pmSubscriptionManager(pmTask* pTask);
		virtual ~pmSubscriptionManager();

		pmStatus InitializeSubtaskDefaults(ulong pSubtaskId);
		pmStatus RegisterSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo);
		pmStatus FetchSubtaskSubscriptions(ulong pSubtaskId);
		pmStatus SetCudaLaunchConf(ulong pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf);
		pmCudaLaunchConf& GetCudaLaunchConf(ulong pSubtaskId);

		bool GetInputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);
		bool GetOutputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);

	private:
		pmStatus WaitForSubscriptions(ulong pSubtaskId);
		pmStatus FetchSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo, subscription::subscriptionData& pData);

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		std::map<ulong, subscription::pmSubtask> mSubtaskMap;     // subtaskId to pmSubtask map

		pmTask* mTask;
};

} // end namespace pm

#endif


#ifndef __PM_SUBSCRIPTION_MANAGER__
#define __PM_SUBSCRIPTION_MANAGER__

#include "pmInternalDefinitions.h"
#include "pmResourceLock.h"
#include "pmTask.h"
#include "pmCommand.h"

#include <map>

namespace pm
{

typedef struct subscriptionData
{
	std::vector<pmCommunicatorCommandPtr> receiveCommandVector;

	subscriptionData();
} subscriptionData;

#define MAP_VALUE_DATA_TYPE std::pair<std::vector<pmSubscriptionInfo>, std::vector<subscriptionData> >

class pmSubscriptionManager : public pmBase
{
	public:
		pmSubscriptionManager(pmTask* pTask);
		virtual ~pmSubscriptionManager();

		pmStatus SetupNewMemRequestReceiveReception();
		pmStatus SetDefaultSubscriptions(ulong pSubtaskId);
		pmStatus RegisterSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo);
		pmStatus FetchSubtaskSubscriptions(ulong pSubtaskId);
		bool GetInputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);
		bool GetOutputMemSubscriptionForSubtask(ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);
		pmStatus WaitForSubscriptions(ulong pSubtaskId);
		
	private:
		pmStatus FetchSubscription(ulong pSubtaskId, bool pIsInputMem, pmSubscriptionInfo pSubscriptionInfo, subscriptionData& pData);

		// Only one contiguous input mem and one output mem subscriptions for each subtask
		std::map<ulong, MAP_VALUE_DATA_TYPE > mInputMemSubscriptions;	// subtaskId to input memory section subscriptions
		std::map<ulong, MAP_VALUE_DATA_TYPE > mOutputMemSubscriptions;	// subtaskId to output memory section subscriptions

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mInputMemResourceLock;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mOutputMemResourceLock;

		pmTask* mTask;
		pmCommunicatorCommandPtr mMemRequestReceiveCommand;
};

} // end namespace pm

#endif


#ifndef __PM_TASK__
#define __PM_TASK__

#include "pmInternalDefinitions.h"
#include "pmCluster.h"
#include "pmNetwork.h"

namespace pm
{

class pmMemSection;
class pmCallbackChain;

/**
 * \brief The representation of a parallel task.
 */

class pmTask : public pmBase
{
	public:
		pmTask(pmMemSection pMemRO, pmMemSection pMemRW, ulong pSubtaskCount, pmCallbackChain pChain, pmHardware pOriginatingHost = pmNetwork::GetNetwork()->GetHostId(),
			pmCluster pCluster = PM_GLOBAL_CLUSTER, ushort pPriority = MAX_PRIORITY_LEVEL);
		
		~pmTask();

		pmMemSection GetMemSectionRO() {return mMemRO;}
		pmMemSection GetMemSectionRW() {return mMemRW;}
		ulong GetSubtaskCount() {return mSubtaskCount;}
		pmCallbackChain GetCallbackChain() {return mChain;}
		pmHardware GetOriginatingHost() {return mOriginatingHost;}
		pmCluster GetCluster() {return mCluster;}
		ushort GetPriority() {return mPriority;}

	private:
		pmHardware mOriginatingHost;
		pmMemSection mMemRO;
		pmMemSection mMemRW;
		ulong mSubtaskCount;
		pmCluster mCluster;
		pmCallbackChain mChain;
		ushort mPriority;
};

} // end namespace pm

#endif

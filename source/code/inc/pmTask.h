
#ifndef __PM_TASK__
#define __PM_TASK__

#include "pmInternalDefinitions.h"

namespace pm
{

class pmMemSection;
class pmCluster;
class pmCallbackChain;

/**
 * \brief The representation of a parallel task.
 */

class pmTask
{
	public:
		pmTask();
		~pmTask();

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

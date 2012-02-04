
#ifndef __PM_SIGNAL_WAIT__
#define __PM_SIGNAL_WAIT__

#include "pmBase.h"
#include "pmResourceLock.h"

namespace pm
{

/**
 * \brief An implementation of wait and signal mechanism.
*/

class pmSignalWait : public pmBase
{
	public:
		virtual pmStatus Wait() = 0;
		virtual pmStatus Signal() = 0;

	private:
		virtual pmStatus WaitTillAllBlockedThreadsWakeup() = 0;
};

class pmPThreadSignalWait : public pmSignalWait
{
	public:
		pmPThreadSignalWait();
		virtual ~pmPThreadSignalWait();

		virtual pmStatus Wait();
		virtual pmStatus Signal();

	private:
		virtual pmStatus WaitTillAllBlockedThreadsWakeup();

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		pthread_cond_t mCondVariable;

		bool mExiting;
		bool mCondEnforcer;
		uint mWaitingThreadCount;
};

} // end namespace pm

#endif

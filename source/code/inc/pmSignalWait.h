
#ifndef __PM_SIGNAL_WAIT__
#define __PM_SIGNAL_WAIT__

#include "pmInternalDefinitions.h"

#include THREADING_IMPLEMENTATION_HEADER

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

		virtual pmStatus WaitTillAllBlockedThreadsWakeup() = 0;

	private:
};

class pmPThreadSignalWait : public pmSignalWait
{
	public:
		pmPThreadSignalWait();
		virtual ~pmPThreadSignalWait();

		virtual pmStatus Wait();
		virtual pmStatus Signal();
		virtual pmStatus WaitTillAllBlockedThreadsWakeup();

	private:
		pthread_mutex_t mMutex;
		pthread_cond_t mCondVariable;
		bool mCondEnforcer;
		uint mWaitingThreadCount;
};

} // end namespace pm

#endif

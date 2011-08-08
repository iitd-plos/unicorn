
#ifndef __PM_SIGNAL_WAIT__
#define __PM_SIGNAL_WAIT__

#include "pmInternalDefinitions.h"

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief An implementation of wait and signal mechanism.
*/

class pmSignalWait
{
	public:
		virtual pmStatus Wait() = 0;
		virtual pmStatus Signal() = 0;

	private:
};

class pmPThreadSignalWait : pmSignalWait
{
	public:
		pmPThreadSignalWait();
		virtual ~pmPThreadSignalWait();

		virtual pmStatus Wait();
		virtual pmStatus Signal();

	private:
		pthread_mutex_t mMutex;
		pthread_cond_t mCondVariable;
		bool mCondEnforcer;
};

} // end namespace pm

#endif
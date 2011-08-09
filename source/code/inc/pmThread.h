
#ifndef __PM_THREAD__
#define __PM_THREAD__

#include "pmInternalDefinitions.h"

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

class pmThreadCommand;

void* ThreadLoop(void* pThreadData);

/**
 * \brief The base thread class of PMLIB.
 * This class serves as base class to classes providing thread implementations.
 * The thread keeps on running indefinitely unless the implementing client object
 * destroys itself. The thread accepts commands, executes it and waits for next command.
 * This class is implemented based upon Abstract Factory Design Pattern so that
 * multiple thread implementations may be hooked up.
 * From their executing thread (could be the main application thread), the clients
 * call SwitchThread function which inturn calls the callback function ThreadSwitchCallback
 * (on pmThread) which all clients are required to implement. The data passed to
 * SwitchThread will be passed back to ThreadSwitchCallback and it's interpretation is client
 * specific. pmThread executes only one command at a time. The subsequent commands wait until
 * the first one returns.
*/

class pmThread
{
	public:
		virtual pmStatus SwitchThread(pmThreadCommand* pCommand) = 0;
		
		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand) = 0;
};

class pmPThread : pmThread
{
	public:
		pmPThread();
		virtual ~pmPThread();

		virtual pmStatus SwitchThread(pmThreadCommand* pCommand);

		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand) = 0;

		friend void* ThreadLoop(void* pThreadData);

	private:
		virtual pmStatus SubmitCommand(pmThreadCommand* pCommand);
		virtual pmStatus ThreadCommandLoop();

		pthread_mutex_t mMutex;
		pthread_cond_t mCondVariable;
		bool mCondEnforcer;

		pmThreadCommand* mCommand;
		pthread_t mThread;
};

} // end namespace pm

#endif

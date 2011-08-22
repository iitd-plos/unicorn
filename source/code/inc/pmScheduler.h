
#ifndef __PM_SCHEDULER__
#define __PM_SCHEDULER__

#include "pmInternalDefinitions.h"
#include "pmThread.h"

namespace pm
{

class pmThreadCommand;

/**
 * \brief This class schedules, load balances and executes all tasks on this machine.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmScheduler : public THREADING_IMPLEMENTATION_CLASS
{
	public:
		static pmScheduler* GetScheduler();
		pmStatus DestroyScheduler();

		pmStatus SubmitTask();

		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand);

	private:
		pmScheduler();

		static pmScheduler* mScheduler;
};

} // end namespace pm

#endif

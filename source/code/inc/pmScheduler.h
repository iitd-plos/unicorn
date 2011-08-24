
#ifndef __PM_SCHEDULER__
#define __PM_SCHEDULER__

#include "pmInternalDefinitions.h"
#include "pmThread.h"
#include "pmSafePriorityQueue.h"

namespace pm
{

class pmThreadCommand;
class pmTask;
class pmSignalWait;

/**
 * \brief This class schedules, load balances and executes all tasks on this machine.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmScheduler : public THREADING_IMPLEMENTATION_CLASS
{
	public:
		struct subtaskRange
		{
			pmTask task;
			ulong startSubtask;
			ulong endSubtask;
		};

		static pmScheduler* GetScheduler();
		pmStatus DestroyScheduler();

		pmStatus SubmitSubtasks(subtaskRange pRange);

		pmStatus Execute(subtaskRange pRange);

		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand);

	private:
		pmScheduler();

		pmSafePQ<subtaskRange> mPriorityQueue;
		pmSignalWait mSignalWait;

		static pmScheduler* mScheduler;
};

} // end namespace pm

#endif

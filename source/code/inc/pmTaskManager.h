
#ifndef __PM_TASK_MANAGER__
#define __PM_TASK_MANAGER__

#include "pmInternalDefinitions.h"

namespace pm
{

class pmTask;

/**
 * \brief This class manages all tasks executing on this machine. A task may originate locally or remotely.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmTaskManager
{
	public:
		static pmTaskManager* GetTaskManager();
		pmStatus DestroyTaskManager();

		pmStatus SubmitTask();

	private:
		pmTaskManager();

		static pmTaskManager* mTaskManager;
};

} // end namespace pm

#endif

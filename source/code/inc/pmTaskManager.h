
#ifndef __PM_TASK_MANAGER__
#define __PM_TASK_MANAGER__

#include "pmResourceLock.h"
#include "pmCommand.h"

#include <set>

namespace pm
{

class pmTask;
class pmLocalTask;
class pmRemoteTask;

/**
 * \brief This class manages all tasks executing on this machine. A task may originate locally or remotely.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmTaskManager : public pmBase
{
	public:
		static pmTaskManager* GetTaskManager();
		pmStatus DestroyTaskManager();

		pmStatus SubmitTask(pmLocalTask* pLocalTask);
		pmRemoteTask* CreateRemoteTask(pmCommunicatorCommand::remoteTaskAssignPacked* pRemoteTaskData);

		pmStatus DeleteTask(pmLocalTask* pLocalTask);
		pmStatus DeleteTask(pmRemoteTask* pRemoteTask);

		pmStatus CancelTask(pmLocalTask* pLocalTask);

		uint GetLocalTaskCount();
		uint GetRemoteTaskCount();

		uint FindPendingLocalTaskCount();

		pmRemoteTask* FindRemoteTask(pmMachine* pOriginatingHost, ulong pInternalTaskId);

	private:
		pmTaskManager();

		pmStatus SubmitTask(pmRemoteTask* pLocalTask);

		static std::set<pmLocalTask*> mLocalTasks;		/* Tasks that originated on this machine */
		static std::set<pmRemoteTask*> mRemoteTasks;	/* Tasks that originated on remote machines */
		static pmTaskManager* mTaskManager;

		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mLocalTaskResourceLock;
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mRemoteTaskResourceLock;
};

} // end namespace pm

#endif

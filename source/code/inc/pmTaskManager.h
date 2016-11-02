
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

#ifndef __PM_TASK_MANAGER__
#define __PM_TASK_MANAGER__

#include "pmResourceLock.h"
#include "pmSignalWait.h"
#include "pmCommunicator.h"

#include <set>
#include <map>

namespace pm
{

class pmTask;
class pmLocalTask;
class pmRemoteTask;
class pmMachine;
class pmProcessingElement;

/**
 * \brief This class manages all tasks executing on this machine. A task may originate locally or remotely.
 * Only one object of this class is created for each machine. This class is thread safe.
 */

class pmTaskManager : public pmBase
{
    typedef std::set<pmLocalTask*> localTasksSetType;
    typedef std::set<pmRemoteTask*> remoteTasksSetType;
    typedef std::set<std::pair<uint, ulong>> finishedTasksSetType;  // pair of originating host and sequence numbers
    typedef std::map<std::pair<const pmMachine*, ulong>, std::vector<std::pair<pmSubtaskRange, const pmProcessingElement*> > > enqueuedRemoteSubtasksMapType;

    public:
		static pmTaskManager* GetTaskManager();

        void SubmitTask(pmLocalTask* pLocalTask);

		void StartTask(pmLocalTask* pLocalTask);
        void StartTask(pmRemoteTask* pRemoteTask);

		pmRemoteTask* CreateRemoteTask(communicator::remoteTaskAssignPacked* pRemoteTaskData);

		void DeleteTask(pmLocalTask* pLocalTask);
		void DeleteTask(pmRemoteTask* pRemoteTask);

		void CancelTask(pmLocalTask* pLocalTask);

		uint GetLocalTaskCount();
		uint GetRemoteTaskCount();
    
        void WaitForAllTasksToFinish();

		uint FindPendingLocalTaskCount();

        bool GetRemoteTaskOrEnqueueSubtasks(pmSubtaskRange& pRange, const pmProcessingElement* pTargetDevice, const pmMachine* pOriginatingHost, ulong pSequenceNumber);
		pmTask* FindTask(const pmMachine* pOriginatingHost, ulong pSequenceNumber);
        pmTask* FindTaskNoThrow(const pmMachine* pOriginatingHost, ulong pSequenceNumber);
    
        bool DoesTaskHavePendingSubtasks(pmTask* pTask);
        bool DoesTaskHavePendingSubtasks(const pmMachine* pOriginatingHost, ulong pSequenceNumber);

        void RegisterTaskFinish(uint pOriginatingHost, ulong pSequenceNumber);
        bool IsRemoteTaskFinished(uint pOriginatingHost, ulong pSequenceNumber); // A remote task may be unknown to this machine (this information is sometimes required as the remote task may use memory which is partially owned on this host)
    
	private:
		pmTaskManager();

        void ScheduleEnqueuedRemoteSubtasksForExecution(pmRemoteTask* pRemoteTask);
        
        void SubmitTask(pmRemoteTask* pRemoteTask);

        pmLocalTask* FindLocalTask_Internal(ulong pSequenceNumber);
        pmRemoteTask* FindRemoteTask_Internal(const pmMachine* pOriginatingHost, ulong pSequenceNumber);

		static localTasksSetType& GetLocalTasks();		/* Tasks that originated on this machine */
		static remoteTasksSetType& GetRemoteTasks();	/* Tasks that originated on remote machines */
        static finishedTasksSetType& GetFinishedRemoteTasks();
        static enqueuedRemoteSubtasksMapType& GetEnqueuedRemoteSubtasksMap();
    
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetLocalTaskResourceLock();
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetRemoteTaskResourceLock();
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetFinishedRemoteTasksResourceLock();

        SIGNAL_WAIT_IMPLEMENTATION_CLASS mTaskFinishSignalWait;
};

} // end namespace pm

#endif

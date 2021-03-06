
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

#include "pmTaskManager.h"
#include "pmTask.h"
#include "pmScheduler.h"
#include "pmNetwork.h"
#include "pmHardware.h"
#include "pmCallbackUnit.h"
#include "pmDevicePool.h"
#include "pmAddressSpace.h"

namespace pm
{

STATIC_ACCESSOR(pmTaskManager::localTasksSetType, pmTaskManager, GetLocalTasks)
STATIC_ACCESSOR(pmTaskManager::remoteTasksSetType, pmTaskManager, GetRemoteTasks)
STATIC_ACCESSOR(pmTaskManager::finishedTasksSetType, pmTaskManager, GetFinishedRemoteTasks)
STATIC_ACCESSOR(pmTaskManager::enqueuedRemoteSubtasksMapType, pmTaskManager, GetEnqueuedRemoteSubtasksMap)

STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmTaskManager::mLocalTaskResourceLock"), pmTaskManager, GetLocalTaskResourceLock)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmTaskManager::mRemoteTaskResourceLock"), pmTaskManager, GetRemoteTaskResourceLock)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmTaskManager::mFinishedRemoteTasksResourceLock"), pmTaskManager, GetFinishedRemoteTasksResourceLock)


pmTaskManager::pmTaskManager()
    : mTaskFinishSignalWait(false)
{
}

pmTaskManager* pmTaskManager::GetTaskManager()
{
	static pmTaskManager lTaskManager;
    return &lTaskManager;
}

void pmTaskManager::StartTask(pmLocalTask* pLocalTask)
{
	pLocalTask->MarkTaskStart();
	pmScheduler::GetScheduler()->SubmitTaskEvent(pLocalTask);
}

void pmTaskManager::SubmitTask(pmLocalTask* pLocalTask)
{
    FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());

    GetLocalTasks().insert(pLocalTask);
}

pmRemoteTask* pmTaskManager::CreateRemoteTask(communicator::remoteTaskAssignPacked* pRemoteTaskData)
{
	const pmCallbackUnit* lCallbackUnit = pmCallbackUnit::FindCallbackUnit(pRemoteTaskData->taskStruct.callbackKey);	// throws exception if key unregistered

    std::vector<pmTaskMemory> lTaskMemVector;
    lTaskMemVector.reserve(pRemoteTaskData->taskStruct.taskMemCount);

    for(size_t memIndex = 0; memIndex < pRemoteTaskData->taskStruct.taskMemCount; ++memIndex)
    {
        communicator::taskMemoryStruct& lTaskMemStruct = pRemoteTaskData->taskMem[memIndex];
        const pmMachine* lOwnerHost = pmMachinePool::GetMachinePool()->GetMachine(lTaskMemStruct.memIdentifier.memOwnerHost);

        pmAddressSpace* lAddressSpace = NULL;
        
        EXCEPTION_ASSERT(lTaskMemStruct.addressSpaceType == ADDRESS_SPACE_LINEAR || lTaskMemStruct.addressSpaceType == ADDRESS_SPACE_2D);

        if(lTaskMemStruct.addressSpaceType == ADDRESS_SPACE_LINEAR)
            lAddressSpace = pmAddressSpace::CheckAndCreateAddressSpace(lTaskMemStruct.memLength, lOwnerHost, lTaskMemStruct.memIdentifier.generationNumber);
        else
            lAddressSpace = pmAddressSpace::CheckAndCreateAddressSpace(lTaskMemStruct.memLength, lTaskMemStruct.cols, lOwnerHost, lTaskMemStruct.memIdentifier.generationNumber);
        
        lTaskMemVector.emplace_back(lAddressSpace, (pmMemType)(lTaskMemStruct.memType), (pmSubscriptionVisibilityType)(lTaskMemStruct.subscriptionVisibility), (bool)lTaskMemStruct.flags);
    }

    std::vector<const pmProcessingElement*> lDevices;
    lDevices.reserve(pRemoteTaskData->taskStruct.assignedDeviceCount);

    for(uint i = 0; i < pRemoteTaskData->taskStruct.assignedDeviceCount; ++i)
        lDevices.emplace_back(pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(((uint*)pRemoteTaskData->devices.get_ptr())[i]));

	pmRemoteTask* lRemoteTask = new pmRemoteTask(pRemoteTaskData->taskConf, pRemoteTaskData->taskStruct.taskConfLength, pRemoteTaskData->taskStruct.taskId, std::move(lTaskMemVector), pRemoteTaskData->taskStruct.subtaskCount, lCallbackUnit, pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost), pRemoteTaskData->taskStruct.sequenceNumber, std::move(lDevices), PM_GLOBAL_CLUSTER, pRemoteTaskData->taskStruct.priority, (scheduler::schedulingModel)(pRemoteTaskData->taskStruct.schedModel), pRemoteTaskData->taskStruct.flags, (pmAffinityCriterion)(pRemoteTaskData->taskStruct.affinityCriterion));

    SubmitTask(lRemoteTask);
    lRemoteTask->LockAddressSpaces();
    
	return lRemoteTask;
}

void pmTaskManager::StartTask(pmRemoteTask* pRemoteTask)
{
    ScheduleEnqueuedRemoteSubtasksForExecution(pRemoteTask);
}
    
void pmTaskManager::SubmitTask(pmRemoteTask* pRemoteTask)
{
	FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

	GetRemoteTasks().insert(pRemoteTask);
}

// Do not dereference pLocalTask in this method. It is already deleted from memory.
void pmTaskManager::DeleteTask(pmLocalTask* pLocalTask)
{
    FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());
    localTasksSetType& lLocalTasks = GetLocalTasks();
    if(lLocalTasks.find(pLocalTask) == lLocalTasks.end())
        return;
    
    lLocalTasks.erase(pLocalTask);

    mTaskFinishSignalWait.Signal();
}

// Do not dereference pRemoteTask in this method. It is already deleted from memory.
void pmTaskManager::DeleteTask(pmRemoteTask* pRemoteTask)
{
    FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());
    remoteTasksSetType& lRemoteTasks = GetRemoteTasks();
    lRemoteTasks.erase(pRemoteTask);
    
    mTaskFinishSignalWait.Signal();
}

void pmTaskManager::CancelTask(pmLocalTask* pLocalTask)
{
	pmScheduler::GetScheduler()->CancelTask(pLocalTask);
}

uint pmTaskManager::GetLocalTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());
	return (uint)(GetLocalTasks().size());
}

uint pmTaskManager::GetRemoteTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());
	return (uint)(GetRemoteTasks().size());
}

uint pmTaskManager::FindPendingLocalTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());

	uint lPendingCount = 0;
	std::set<pmLocalTask*>::iterator lIter;

    localTasksSetType& lLocalTasks = GetLocalTasks();
	for(lIter = lLocalTasks.begin(); lIter != lLocalTasks.end(); ++lIter)
	{
		if((*lIter)->GetStatus() == pmStatusUnavailable)
			++lPendingCount;
	}

	return lPendingCount;
}
    
void pmTaskManager::WaitForAllTasksToFinish()
{
    while(GetLocalTaskCount() || GetRemoteTaskCount())
        mTaskFinishSignalWait.Wait();
}

/** If remote task already exists returns true; otherwise if remote task does not exist (due to out of order message receives),
 *  enqueues remote subtask range for later execution 
 */
bool pmTaskManager::GetRemoteTaskOrEnqueueSubtasks(pmSubtaskRange& pRange, const pmProcessingElement* pTargetDevice, const pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
	FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

    pmRemoteTask* lRemoteTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
    
    if(lRemoteTask && lRemoteTask->HasStarted())
    {
        pRange.task = lRemoteTask;
        return true;
    }
    
    std::pair<const pmMachine*, ulong> lPair(pOriginatingHost, pSequenceNumber);
    
    enqueuedRemoteSubtasksMapType& lEnqueuedRemoteSubtasksMap = GetEnqueuedRemoteSubtasksMap();
    
#ifdef _DEBUG
    if(lEnqueuedRemoteSubtasksMap.find(lPair) != lEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, const pmProcessingElement*> >::iterator lBegin = lEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, const pmProcessingElement*> >::iterator lEnd = lEnqueuedRemoteSubtasksMap[lPair].end();
        
        for(; lBegin != lEnd; ++lBegin)
        {
            if(lBegin->second == pTargetDevice)
                PMTHROW(pmFatalErrorException());
        }
    }
#endif
    
    lEnqueuedRemoteSubtasksMap[lPair].push_back(std::make_pair(pRange, pTargetDevice));
    
    return false;
}

// Must be called with mRemoteTaskResourceLock acquired
void pmTaskManager::ScheduleEnqueuedRemoteSubtasksForExecution(pmRemoteTask* pRemoteTask)
{
    enqueuedRemoteSubtasksMapType& lEnqueuedRemoteSubtasksMap = GetEnqueuedRemoteSubtasksMap();
    std::pair<const pmMachine*, ulong> lPair(pRemoteTask->GetOriginatingHost(), pRemoteTask->GetSequenceNumber());
    
    if(lEnqueuedRemoteSubtasksMap.find(lPair) != lEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, const pmProcessingElement*> >::iterator lBegin = lEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, const pmProcessingElement*> >::iterator lEnd = lEnqueuedRemoteSubtasksMap[lPair].end();
        
        for(; lBegin != lEnd; ++lBegin)
        {
            std::pair<pmSubtaskRange, const pmProcessingElement*>& lValuePair = *lBegin;
            lValuePair.first.task = pRemoteTask;
            pmScheduler::GetScheduler()->PushEvent(lValuePair.second, lValuePair.first, false);
        }
        
        lEnqueuedRemoteSubtasksMap.erase(lPair);
    }
}

pmTask* pmTaskManager::FindTask(const pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    pmTask* lTask = FindTaskNoThrow(pOriginatingHost, pSequenceNumber);

    if(!lTask)
        PMTHROW(pmFatalErrorException());
    
    return lTask;
}

pmTask* pmTaskManager::FindTaskNoThrow(const pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    pmTask* lTask = NULL;
    
    if(pOriginatingHost == PM_LOCAL_MACHINE)
    {
        FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());
        lTask = FindLocalTask_Internal(pSequenceNumber);
    }
    else
    {
        FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());
        lTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
    }
    
    return lTask;
}

/** There could be unnecessary processing requests in flight when a task finishes. So it is required to atomically check
 *  the existence of task and it's subtask completion while processing commands in the scheduler thread.
*/
bool pmTaskManager::DoesTaskHavePendingSubtasks(const pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    bool lState = false;
    
    if(pOriginatingHost == PM_LOCAL_MACHINE)
    {
        FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());

        pmTask* lTask = FindLocalTask_Internal(pSequenceNumber);
        if(lTask)
            lState = !(lTask->HasSubtaskExecutionFinished());
    }
    else
    {
        FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

        pmTask* lTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
        if(lTask)
            lState = !(lTask->HasSubtaskExecutionFinished());
    }
    
    return lState;
}
    
bool pmTaskManager::DoesTaskHavePendingSubtasks(pmTask* pTask)
{
    // Auto lock/release scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());

        localTasksSetType& lLocalTasks = GetLocalTasks();
        localTasksSetType::iterator lIter = lLocalTasks.find((pmLocalTask*)pTask);
        if(lIter != lLocalTasks.end())
            return !((*lIter)->HasSubtaskExecutionFinished());
    }

    // Auto lock/release scope    
    {
        FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

        remoteTasksSetType& lRemoteTasks = GetRemoteTasks();
        remoteTasksSetType::iterator lIter = lRemoteTasks.find((pmRemoteTask*)pTask);
        if(lIter != lRemoteTasks.end())
            return !((*lIter)->HasSubtaskExecutionFinished());
    }
    
    return false;    
}

// Must be called with mLocalTaskResourceLock acquired
pmLocalTask* pmTaskManager::FindLocalTask_Internal(ulong pSequenceNumber)
{
    localTasksSetType& lLocalTasks = GetLocalTasks();
    localTasksSetType::iterator lIter = lLocalTasks.begin(), lEndIter = lLocalTasks.end();

    for(; lIter != lEndIter; ++lIter)
    {
        pmLocalTask* lLocalTask = *lIter;
        if(lLocalTask->GetSequenceNumber() == pSequenceNumber)
            return lLocalTask;
    }
    
    return NULL;
}
    
// Must be called with mRemoteTaskResourceLock acquired
pmRemoteTask* pmTaskManager::FindRemoteTask_Internal(const pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    remoteTasksSetType& lRemoteTasks = GetRemoteTasks();
    remoteTasksSetType::iterator lIter = lRemoteTasks.begin(), lEndIter = lRemoteTasks.end();

	for(; lIter != lEndIter; ++lIter)
	{
		pmRemoteTask* lRemoteTask = *lIter;
		if(lRemoteTask->GetOriginatingHost() == pOriginatingHost && lRemoteTask->GetSequenceNumber() == pSequenceNumber)
			return lRemoteTask;
	}

    return NULL;
}
    
void pmTaskManager::RegisterTaskFinish(uint pOriginatingHost, ulong pSequenceNumber)
{
    if(pOriginatingHost == (uint)(*PM_LOCAL_MACHINE))
        return;
    
    FINALIZE_RESOURCE(dResourceLock, GetFinishedRemoteTasksResourceLock().Lock(), GetFinishedRemoteTasksResourceLock().Unlock());
    
    GetFinishedRemoteTasks().emplace(pOriginatingHost, pSequenceNumber);
}
    
bool pmTaskManager::IsRemoteTaskFinished(uint pOriginatingHost, ulong pSequenceNumber)
{
    FINALIZE_RESOURCE(dResourceLock, GetFinishedRemoteTasksResourceLock().Lock(), GetFinishedRemoteTasksResourceLock().Unlock());

    return (GetFinishedRemoteTasks().find(std::make_pair(pOriginatingHost, pSequenceNumber)) != GetFinishedRemoteTasks().end());
}

} // end namespace pm




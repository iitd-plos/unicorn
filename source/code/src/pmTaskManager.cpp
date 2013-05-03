
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#include "pmTaskManager.h"
#include "pmTask.h"
#include "pmScheduler.h"
#include "pmNetwork.h"
#include "pmHardware.h"
#include "pmCommand.h"
#include "pmCallbackUnit.h"
#include "pmDevicePool.h"
#include "pmMemSection.h"

namespace pm
{

STATIC_ACCESSOR(pmTaskManager::localTasksSetType, pmTaskManager, GetLocalTasks)
STATIC_ACCESSOR(pmTaskManager::remoteTasksSetType, pmTaskManager, GetRemoteTasks)
STATIC_ACCESSOR(pmTaskManager::enqueuedRemoteSubtasksMapType, pmTaskManager, GetEnqueuedRemoteSubtasksMap)

STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmTaskManager::mLocalTaskResourceLock"), pmTaskManager, GetLocalTaskResourceLock)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmTaskManager::mRemoteTaskResourceLock"), pmTaskManager, GetRemoteTaskResourceLock)


pmTaskManager::pmTaskManager()
{
}

pmTaskManager* pmTaskManager::GetTaskManager()
{
	static pmTaskManager lTaskManager;
    return &lTaskManager;
}

pmStatus pmTaskManager::SubmitTask(pmLocalTask* pLocalTask)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());
        GetLocalTasks().insert(pLocalTask);
    }

	pLocalTask->MarkTaskStart();
	pmScheduler::GetScheduler()->SubmitTaskEvent(pLocalTask);

	return pmSuccess;
}

pmRemoteTask* pmTaskManager::CreateRemoteTask(pmCommunicatorCommand::remoteTaskAssignPacked* pRemoteTaskData)
{
	pmCallbackUnit* lCallbackUnit = pmCallbackUnit::FindCallbackUnit(pRemoteTaskData->taskStruct.callbackKey);	// throws exception if key unregistered

	void* lTaskConf;
	if(pRemoteTaskData->taskStruct.taskConfLength == 0)
	{
		lTaskConf = NULL;
	}
	else
	{
		lTaskConf = malloc(pRemoteTaskData->taskStruct.taskConfLength);
		if(!lTaskConf)
			PMTHROW(pmOutOfMemoryException());

		memcpy(lTaskConf, pRemoteTaskData->taskConf.ptr, pRemoteTaskData->taskConf.length);
	}

	pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost);

	pmRemoteTask* lRemoteTask;
	pmMemSection* lInputMem = NULL;
	pmMemSection* lOutputMem = NULL;
	
	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		FREE_PTR_ON_EXCEPTION(lDestructionBlock, lTaskConf, lTaskConf);
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lInputMem, pmMemSection, (pRemoteTaskData->taskStruct.inputMemLength == 0) ? NULL : (pmMemSection::CheckAndCreateMemSection(pRemoteTaskData->taskStruct.inputMemLength, lOriginatingHost, pRemoteTaskData->taskStruct.inputMemGenerationNumber)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lOutputMem, pmMemSection, (pRemoteTaskData->taskStruct.outputMemLength == 0) ? NULL : (pmMemSection::CheckAndCreateMemSection(pRemoteTaskData->taskStruct.outputMemLength, lOriginatingHost, pRemoteTaskData->taskStruct.outputMemGenerationNumber)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lRemoteTask, pmRemoteTask, new pmRemoteTask(lTaskConf, pRemoteTaskData->taskStruct.taskConfLength, pRemoteTaskData->taskStruct.taskId, lInputMem, lOutputMem, (pmMemInfo)(pRemoteTaskData->taskStruct.inputMemInfo), (pmMemInfo)(pRemoteTaskData->taskStruct.outputMemInfo), pRemoteTaskData->taskStruct.subtaskCount, lCallbackUnit, pRemoteTaskData->taskStruct.assignedDeviceCount, pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost), pRemoteTaskData->taskStruct.sequenceNumber, PM_GLOBAL_CLUSTER, pRemoteTaskData->taskStruct.priority, (scheduler::schedulingModel)(pRemoteTaskData->taskStruct.schedModel), (pRemoteTaskData->taskStruct.flags & TASK_MULTI_ASSIGN_FLAG_VAL), (pRemoteTaskData->taskStruct.flags & TASK_SAME_READ_WRITE_SUBSCRIPTION_FLAG_VAL)));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

	if(pRemoteTaskData->taskStruct.schedModel == scheduler::PULL || lRemoteTask->GetCallbackUnit()->GetDataReductionCB())
	{
		for(uint i=0; i<pRemoteTaskData->taskStruct.assignedDeviceCount; ++i)
			lRemoteTask->AddAssignedDevice(pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(((uint*)pRemoteTaskData->devices.ptr)[i]));
	}

	SubmitTask(lRemoteTask);

	return lRemoteTask;
}

pmStatus pmTaskManager::SubmitTask(pmRemoteTask* pRemoteTask)
{
	FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

	GetRemoteTasks().insert(pRemoteTask);
    ScheduleEnqueuedRemoteSubtasksForExecution(pRemoteTask);

	return pmSuccess;
}

// Do not derefernce pLocalTask in this method. It is already deleted from memory.
pmStatus pmTaskManager::DeleteTask(pmLocalTask* pLocalTask)
{
    FINALIZE_RESOURCE(dResourceLock, GetLocalTaskResourceLock().Lock(), GetLocalTaskResourceLock().Unlock());
    localTasksSetType& lLocalTasks = GetLocalTasks();
    if(lLocalTasks.find(pLocalTask) == lLocalTasks.end())
        return pmSuccess;
    
    lLocalTasks.erase(pLocalTask);

    mTaskFinishSignalWait.Signal();

    return pmSuccess;
}

// Do not derefernce pRemoteTask in this method. It is already deleted from memory.
pmStatus pmTaskManager::DeleteTask(pmRemoteTask* pRemoteTask)
{
    FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());
    remoteTasksSetType& lRemoteTasks = GetRemoteTasks();
    lRemoteTasks.erase(pRemoteTask);
    
    mTaskFinishSignalWait.Signal();
    
    return pmSuccess;
}

pmStatus pmTaskManager::CancelTask(pmLocalTask* pLocalTask)
{
	return pmScheduler::GetScheduler()->CancelTask(pLocalTask);
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
    
pmStatus pmTaskManager::WaitForAllTasksToFinish()
{
    while(GetLocalTaskCount() || GetRemoteTaskCount())
        mTaskFinishSignalWait.Wait();
    
    return pmSuccess;
}

/** If remote task already exists returns true; otherwise if remote task does not exist (due to out of order message receives),
 *  enqueues remote subtask range for later execution 
 */
bool pmTaskManager::GetRemoteTaskOrEnqueueSubtasks(pmSubtaskRange& pRange, pmProcessingElement* pTargetDevice, pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
	FINALIZE_RESOURCE(dResourceLock, GetRemoteTaskResourceLock().Lock(), GetRemoteTaskResourceLock().Unlock());

    pmRemoteTask* lRemoteTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
    
    if(lRemoteTask)
    {
        pRange.task = lRemoteTask;
        return true;
    }
    
    std::pair<pmMachine*, ulong> lPair(pOriginatingHost, pSequenceNumber);
    
    enqueuedRemoteSubtasksMapType& lEnqueuedRemoteSubtasksMap = GetEnqueuedRemoteSubtasksMap();
    
#ifdef _DEBUG
    if(lEnqueuedRemoteSubtasksMap.find(lPair) != lEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lBegin = lEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lEnd = lEnqueuedRemoteSubtasksMap[lPair].end();
        
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
pmStatus pmTaskManager::ScheduleEnqueuedRemoteSubtasksForExecution(pmRemoteTask* pRemoteTask)
{
    enqueuedRemoteSubtasksMapType& lEnqueuedRemoteSubtasksMap = GetEnqueuedRemoteSubtasksMap();
    std::pair<pmMachine*, ulong> lPair(pRemoteTask->GetOriginatingHost(), pRemoteTask->GetSequenceNumber());
    
    if(lEnqueuedRemoteSubtasksMap.find(lPair) != lEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lBegin = lEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lEnd = lEnqueuedRemoteSubtasksMap[lPair].end();
        
        for(; lBegin != lEnd; ++lBegin)
        {
            std::pair<pmSubtaskRange, pmProcessingElement*>& lValuePair = *lBegin;
            lValuePair.first.task = pRemoteTask;
            pmScheduler::GetScheduler()->PushEvent(lValuePair.second, lValuePair.first);
        }
        
        lEnqueuedRemoteSubtasksMap.erase(lPair);
    }
    
    return pmSuccess;
}

pmTask* pmTaskManager::FindTask(pmMachine* pOriginatingHost, ulong pSequenceNumber)
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

    if(!lTask)
        PMTHROW(pmFatalErrorException());
    
    return lTask;
}

/** There could be unnecessary processing requests in flight when a task finishes. So it is required to atomically check
 *  the existence of task and it's subtask completion while processing commands in the scheduler thread.
*/
bool pmTaskManager::DoesTaskHavePendingSubtasks(pmMachine* pOriginatingHost, ulong pSequenceNumber)
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
pmRemoteTask* pmTaskManager::FindRemoteTask_Internal(pmMachine* pOriginatingHost, ulong pSequenceNumber)
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

} // end namespace pm




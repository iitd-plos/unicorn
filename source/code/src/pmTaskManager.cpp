
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

pmTaskManager* pmTaskManager::mTaskManager = NULL;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmTaskManager::mLocalTaskResourceLock;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmTaskManager::mRemoteTaskResourceLock;
std::set<pmLocalTask*> pmTaskManager::mLocalTasks;
std::set<pmRemoteTask*> pmTaskManager::mRemoteTasks;
std::map<std::pair<pmMachine*, ulong>, std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> > > pmTaskManager::mEnqueuedRemoteSubtasksMap;

pmTaskManager::pmTaskManager()
{
}

pmTaskManager* pmTaskManager::GetTaskManager()
{
	if(!mTaskManager)
		mTaskManager = new pmTaskManager();

	return mTaskManager;
}

pmStatus pmTaskManager::DestroyTaskManager()
{
	delete mTaskManager;
	mTaskManager = NULL;

	return pmSuccess;
}

pmStatus pmTaskManager::SubmitTask(pmLocalTask* pLocalTask)
{
	mLocalTaskResourceLock.Lock();
	mLocalTasks.insert(pLocalTask);
	mLocalTaskResourceLock.Unlock();

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
	pmInputMemSection* lInputMem = NULL;
	pmOutputMemSection* lOutputMem = NULL;
	
	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		FREE_PTR_ON_EXCEPTION(lDestructionBlock, lTaskConf, lTaskConf);
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lInputMem, pmInputMemSection, (pRemoteTaskData->taskStruct.inputMemLength == 0) ? NULL : (new pmInputMemSection(pRemoteTaskData->taskStruct.inputMemLength, lOriginatingHost, pRemoteTaskData->taskStruct.inputMemAddr)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lOutputMem, pmOutputMemSection, (pRemoteTaskData->taskStruct.outputMemLength == 0)? NULL : (new pmOutputMemSection(pRemoteTaskData->taskStruct.outputMemLength, 
			pRemoteTaskData->taskStruct.isOutputMemReadWrite?pmOutputMemSection::READ_WRITE:pmOutputMemSection::WRITE_ONLY, lOriginatingHost, pRemoteTaskData->taskStruct.outputMemAddr)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lRemoteTask, pmRemoteTask, new pmRemoteTask(lTaskConf, pRemoteTaskData->taskStruct.taskConfLength, 
			pRemoteTaskData->taskStruct.taskId, lInputMem, lOutputMem, pRemoteTaskData->taskStruct.subtaskCount, lCallbackUnit, pRemoteTaskData->taskStruct.assignedDeviceCount,
			pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost), pRemoteTaskData->taskStruct.sequenceNumber, PM_GLOBAL_CLUSTER,
			pRemoteTaskData->taskStruct.priority, (scheduler::schedulingModel)pRemoteTaskData->taskStruct.schedModel));
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
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

	mRemoteTasks.insert(pRemoteTask);
    ScheduleEnqueuedRemoteSubtasksForExecution(pRemoteTask);

	return pmSuccess;
}

pmStatus pmTaskManager::DeleteTask(pmLocalTask* pLocalTask)
{
	FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());
	mLocalTasks.erase(pLocalTask);
    
    mTaskFinishSignalWait.Signal();

	return pmSuccess;
}

pmStatus pmTaskManager::DeleteTask(pmRemoteTask* pRemoteTask)
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());
	SAFE_FREE(pRemoteTask->GetTaskConfiguration());
	mRemoteTasks.erase(pRemoteTask);

    mTaskFinishSignalWait.Signal();

	return pmSuccess;
}

pmStatus pmTaskManager::CancelTask(pmLocalTask* pLocalTask)
{
	return pmScheduler::GetScheduler()->CancelTask(pLocalTask);
}

uint pmTaskManager::GetLocalTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());
	return (uint)(mLocalTasks.size());
}

uint pmTaskManager::GetRemoteTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());
	return (uint)(mRemoteTasks.size());
}

uint pmTaskManager::FindPendingLocalTaskCount()
{
	FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());

	uint lPendingCount = 0;
	std::set<pmLocalTask*>::iterator lIter;

	for(lIter = mLocalTasks.begin(); lIter != mLocalTasks.end(); ++lIter)
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
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

    pmRemoteTask* lRemoteTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
    
    if(lRemoteTask)
    {
        pRange.task = lRemoteTask;
        return true;
    }
    
    std::pair<pmMachine*, ulong> lPair(pOriginatingHost, pSequenceNumber);
    
#ifdef _DEBUG
    if(mEnqueuedRemoteSubtasksMap.find(lPair) != mEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lBegin = mEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lEnd = mEnqueuedRemoteSubtasksMap[lPair].end();
        
        for(; lBegin != lEnd; ++lBegin)
        {
            if(lBegin->second == pTargetDevice)
                PMTHROW(pmFatalErrorException());
        }
    }
#endif
    
    mEnqueuedRemoteSubtasksMap[lPair].push_back(std::make_pair(pRange, pTargetDevice));
    
    return false;
}

// Must be called with mRemoteTaskResourceLock acquired
pmStatus pmTaskManager::ScheduleEnqueuedRemoteSubtasksForExecution(pmRemoteTask* pRemoteTask)
{
    std::pair<pmMachine*, ulong> lPair(pRemoteTask->GetOriginatingHost(), pRemoteTask->GetSequenceNumber());
    if(mEnqueuedRemoteSubtasksMap.find(lPair) != mEnqueuedRemoteSubtasksMap.end())
    {
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lBegin = mEnqueuedRemoteSubtasksMap[lPair].begin();
        std::vector<std::pair<pmSubtaskRange, pmProcessingElement*> >::iterator lEnd = mEnqueuedRemoteSubtasksMap[lPair].end();
        
        for(; lBegin != lEnd; ++lBegin)
        {
            std::pair<pmSubtaskRange, pmProcessingElement*>& lValuePair = *lBegin;
            lValuePair.first.task = pRemoteTask;
            pmScheduler::GetScheduler()->PushEvent(lValuePair.second, lValuePair.first);
        }
        
        mEnqueuedRemoteSubtasksMap.erase(lPair);
    }
    
    return pmSuccess;
}

pmTask* pmTaskManager::FindTask(pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    pmTask* lTask = NULL;
    
    if(pOriginatingHost == PM_LOCAL_MACHINE)
    {
        FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());        
        lTask = FindLocalTask_Internal(pSequenceNumber);
    }
    else
    {
        FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());
        lTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
    }

    if(!lTask)
        PMTHROW(pmFatalErrorException());
    
    return lTask;
}

/** There could be unnecessary steal requests in flight when a task finishes. Depending upon, the presence of reduce or scatter in the task, it may
 *  get deleted at different stages which are all independent of in flight steal commands. So it is required to atomically check the existence of
 *  task and it's steal completion while processing steal commands in the scheduler thread 
*/
bool pmTaskManager::IsTaskOpenToSteal(pmMachine* pOriginatingHost, ulong pSequenceNumber)
{
    bool lState = false;
    
    if(pOriginatingHost == PM_LOCAL_MACHINE)
    {
        FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());        

        pmTask* lTask = FindLocalTask_Internal(pSequenceNumber);
        if(lTask)
            lState = !(lTask->HasSubtaskExecutionFinished());
    }
    else
    {
        FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());
        pmTask* lTask = FindRemoteTask_Internal(pOriginatingHost, pSequenceNumber);
        if(lTask)
            lState = !(lTask->HasSubtaskExecutionFinished());
    }
    
    return lState;
}
    
bool pmTaskManager::IsTaskOpenToSteal(pmTask* pTask)
{
    // Auto lock/release scope
    {
        FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());        
        
        std::set<pmLocalTask*>::iterator lIter = mLocalTasks.find((pmLocalTask*)pTask);
        if(lIter != mLocalTasks.end())
            return !((*lIter)->HasSubtaskExecutionFinished());
    }

    // Auto lock/release scope    
    {
        FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

        std::set<pmRemoteTask*>::iterator lIter = mRemoteTasks.find((pmRemoteTask*)pTask);
        if(lIter != mRemoteTasks.end())
            return !((*lIter)->HasSubtaskExecutionFinished());
    }
    
    return false;    
}

// Must be called with mLocalTaskResourceLock acquired
pmLocalTask* pmTaskManager::FindLocalTask_Internal(ulong pSequenceNumber)
{
    std::set<pmLocalTask*>::iterator lIter = mLocalTasks.begin();
    std::set<pmLocalTask*>::iterator lEndIter = mLocalTasks.end();
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
	std::set<pmRemoteTask*>::iterator lIter = mRemoteTasks.begin();
	std::set<pmRemoteTask*>::iterator lEndIter = mRemoteTasks.end();
	for(; lIter != lEndIter; ++lIter)
	{
		pmRemoteTask* lRemoteTask = *lIter;
		if(lRemoteTask->GetOriginatingHost() == pOriginatingHost && lRemoteTask->GetSequenceNumber() == pSequenceNumber)
			return lRemoteTask;
	}

    return NULL;
}

} // end namespace pm





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
			pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost), pRemoteTaskData->taskStruct.internalTaskId, PM_GLOBAL_CLUSTER,
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
bool pmTaskManager::GetRemoteTaskOrEnqueueSubtasks(pmSubtaskRange& pRange, pmProcessingElement* pTargetDevice, pmMachine* pOriginatingHost, ulong pInternalTaskId)
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

    pmRemoteTask* lRemoteTask = FindRemoteTask_Internal(pOriginatingHost, pInternalTaskId);
    
    if(lRemoteTask)
    {
        pRange.task = lRemoteTask;
        return true;
    }
    
    std::pair<pmMachine*, ulong> lPair(pOriginatingHost, pInternalTaskId);
    
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
    std::pair<pmMachine*, ulong> lPair(pRemoteTask->GetOriginatingHost(), pRemoteTask->GetInternalTaskId());
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

pmRemoteTask* pmTaskManager::FindRemoteTask(pmMachine* pOriginatingHost, ulong pInternalTaskId)
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

    pmRemoteTask* lRemoteTask = NULL;
    if((lRemoteTask = FindRemoteTask_Internal(pOriginatingHost, pInternalTaskId)) == NULL)
        PMTHROW(pmFatalErrorException());

	return lRemoteTask;
}
    
// Must be called with mRemoteTaskResourceLock acquired
pmRemoteTask* pmTaskManager::FindRemoteTask_Internal(pmMachine* pOriginatingHost, ulong pInternalTaskId)
{
	std::set<pmRemoteTask*>::iterator lIter = mRemoteTasks.begin();
	std::set<pmRemoteTask*>::iterator lEndIter = mRemoteTasks.end();
	for(; lIter != lEndIter; ++lIter)
	{
		pmRemoteTask* lRemoteTask = *lIter;
		if(lRemoteTask->GetInternalTaskId() == pInternalTaskId && lRemoteTask->GetOriginatingHost() == pOriginatingHost)
			return lRemoteTask;
	}

    return NULL;
}

} // end namespace pm




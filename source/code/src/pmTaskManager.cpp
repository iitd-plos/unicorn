
#include "pmTaskManager.h"
#include "pmTask.h"
#include "pmScheduler.h"
#include "pmNetwork.h"
#include "pmHardware.h"
#include "pmCommand.h"
#include "pmCallbackUnit.h"
#include "pmDevicePool.h"

namespace pm
{

pmTaskManager* pmTaskManager::mTaskManager = NULL;

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
			throw pmOutOfMemoryException();

		memcpy(lTaskConf, pRemoteTaskData->taskConf.ptr, pRemoteTaskData->taskConf.length);
	}

	pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost);

	pmRemoteTask* lRemoteTask;
	pmMemSection* lInputMem = NULL;
	pmMemSection* lOutputMem = NULL;
	
	START_DESTROY_ON_EXCEPTION(lDestructionBlock)
		FREE_PTR_ON_EXCEPTION(lDestructionBlock, lTaskConf, lTaskConf);
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lInputMem, (pRemoteTaskData->taskStruct.inputMemLength == 0) ? NULL : (new pmInputMemSection(pRemoteTaskData->taskStruct.inputMemLength, lOriginatingHost, pRemoteTaskData->taskStruct.inputMemAddr)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lOutputMem, (pRemoteTaskData->taskStruct.outputMemLength == 0)? NULL : (new pmOutputMemSection(pRemoteTaskData->taskStruct.outputMemLength, 
			pRemoteTaskData->taskStruct.isOutputMemReadWrite?pmOutputMemSection::READ_WRITE:pmOutputMemSection::WRITE_ONLY, lOriginatingHost, pRemoteTaskData->taskStruct.outputMemAddr)));
		DESTROY_PTR_ON_EXCEPTION(lDestructionBlock, lRemoteTask, new pmRemoteTask(lTaskConf, pRemoteTaskData->taskStruct.taskConfLength, 
			pRemoteTaskData->taskStruct.taskId, lInputMem, lOutputMem, pRemoteTaskData->taskStruct.subtaskCount, lCallbackUnit, pRemoteTaskData->taskStruct.assignedDeviceCount,
			pmMachinePool::GetMachinePool()->GetMachine(pRemoteTaskData->taskStruct.originatingHost), pRemoteTaskData->taskStruct.internalTaskId, PM_GLOBAL_CLUSTER,
			pRemoteTaskData->taskStruct.priority, (pmScheduler::schedulingModel)pRemoteTaskData->taskStruct.schedModel));
	END_DESTROY_ON_EXCEPTION(lDestructionBlock)

	if(pRemoteTaskData->taskStruct.schedModel == pmScheduler::PULL || lRemoteTask->GetCallbackUnit()->GetDataReductionCB())
	{
		for(uint i=0; i<pRemoteTaskData->taskStruct.assignedDeviceCount; ++i)
			lRemoteTask->AddAssignedDevice(pmDevicePool::GetDevicePool()->GetDeviceAtGlobalIndex(((uint*)pRemoteTaskData->devices.ptr)[i]));
	}

	SubmitTask(lRemoteTask);

	return lRemoteTask;
}

pmStatus pmTaskManager::SubmitTask(pmRemoteTask* pRemoteTask)
{
	mRemoteTaskResourceLock.Lock();
	mRemoteTasks.insert(pRemoteTask);
	mRemoteTaskResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmTaskManager::DeleteTask(pmLocalTask* pLocalTask)
{
	FINALIZE_RESOURCE(dResourceLock, mLocalTaskResourceLock.Lock(), mLocalTaskResourceLock.Unlock());
	mLocalTasks.erase(pLocalTask);

	return pmSuccess;
}

pmStatus pmTaskManager::DeleteTask(pmRemoteTask* pRemoteTask)
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());
	SAFE_FREE(pRemoteTask->GetTaskConfiguration());
	mRemoteTasks.erase(pRemoteTask);

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

	uint lPendingCount;
	std::set<pmLocalTask*>::iterator lIter;

	for(lIter = mLocalTasks.begin(); lIter != mLocalTasks.end(); ++lIter)
	{
		if(lIter._Mynode()->_Myval->GetStatus() == pmStatusUnavailable)
			++lPendingCount;
	}

	return lPendingCount;
}

pmRemoteTask* pmTaskManager::FindRemoteTask(pmMachine* pOriginatingHost, ulong pInternalTaskId)
{
	FINALIZE_RESOURCE(dResourceLock, mRemoteTaskResourceLock.Lock(), mRemoteTaskResourceLock.Unlock());

	std::set<pmRemoteTask*>::iterator lIter;
	for(lIter = mRemoteTasks.begin(); lIter != mRemoteTasks.end(); ++lIter)
	{
		pmRemoteTask* lRemoteTask = lIter._Mynode()->_Myval;
		if(lRemoteTask->GetInternalTaskId() == pInternalTaskId && lRemoteTask->GetOriginatingHost() == pOriginatingHost)
			return lRemoteTask;
	}

	throw pmFatalErrorException();

	return NULL;
}

} // end namespace pm





#include "pmTask.h"
#include "pmDevicePool.h"
#include "pmCallback.h"
#include "pmCallbackUnit.h"
#include "pmCluster.h"
#include "pmCommand.h"
#include "pmHardware.h"
#include "pmTaskManager.h"
#include "pmSubtaskManager.h"
#include "pmReducer.h"
#include "pmMemSection.h"

#include <vector>
#include <algorithm>

namespace pm
{

RESOURCE_LOCK_IMPLEMENTATION_CLASS pmLocalTask::mSequenceLock;
ulong pmLocalTask::mSequenceId = 0;
    
#define SAFE_GET_DEVICE_POOL(x) { x = pmDevicePool::GetDevicePool(); if(!x) PMTHROW(pmFatalErrorException()); }

/* class pmTask */
pmTask::pmTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, uint pAssignedDeviceCount /* = 0 */, pmMachine* pOriginatingHost /* = PM_LOCAL_MACHINE */, pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = MAX_PRIORITY_LEVEL */, scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */)
	: mOriginatingHost(pOriginatingHost), mCluster(pCluster), mSubscriptionManager(this)
{
	mTaskConf = pTaskConf;
	mTaskConfLength = pTaskConfLength;
	mTaskId = pTaskId;
	mMemRO = pMemRO;
	mMemRW = pMemRW;
	mCallbackUnit = pCallbackUnit;
	mSubtaskCount = pSubtaskCount;
	mAssignedDeviceCount = pAssignedDeviceCount;
	mSchedulingModel = pSchedulingModel;

	if(pPriority < MAX_PRIORITY_LEVEL)
		mPriority = MAX_PRIORITY_LEVEL;
	else
		mPriority = pPriority;

	mTaskInfo.taskHandle = NULL;
	mSubtaskExecutionFinished = false;
    mReducer = NULL;
	mSubtasksExecuted = 0;
}

pmTask::~pmTask()
{
	if(mReducer)
		delete mReducer;
}
    
pmStatus pmTask::TaskInternallyFinished()
{
    if(mMemRO)
        mMemRO->FlushOwnerships();
    
    if(mMemRW)
        mMemRW->FlushOwnerships();

    return pmSuccess;
}

void* pmTask::GetTaskConfiguration()
{
	return mTaskConf;
}

uint pmTask::GetTaskConfigurationLength()
{
	return mTaskConfLength;
}

ulong pmTask::GetTaskId()
{
	return mTaskId;
}

pmMemSection* pmTask::GetMemSectionRO()
{
	return mMemRO;
}

pmMemSection* pmTask::GetMemSectionRW()
{
	return mMemRW;
}

pmCallbackUnit* pmTask::GetCallbackUnit()
{
	return mCallbackUnit;
}

ulong pmTask::GetSubtaskCount()
{
	return mSubtaskCount;
}

pmMachine* pmTask::GetOriginatingHost()
{
	return mOriginatingHost;
}

pmCluster* pmTask::GetCluster()
{
	return mCluster;
}

ushort pmTask::GetPriority()
{
	return mPriority;
}

uint pmTask::GetAssignedDeviceCount()
{
	return mAssignedDeviceCount;
}

scheduler::schedulingModel pmTask::GetSchedulingModel()
{
	return mSchedulingModel;
}

pmTaskExecStats& pmTask::GetTaskExecStats()
{
	return mTaskExecStats;
}

pmStatus pmTask::RandomizeDevices(std::vector<pmProcessingElement*>& pDevices)
{
	std::random_shuffle(pDevices.begin(), pDevices.end());

	return pmSuccess;
}

pmStatus pmTask::BuildTaskInfo()
{
	mTaskInfo.taskHandle = (void*)this;
	mTaskInfo.taskConf = GetTaskConfiguration();
	mTaskInfo.taskConfLength = GetTaskConfigurationLength();
	mTaskInfo.taskId = GetTaskId();
	mTaskInfo.subtaskCount = GetSubtaskCount();
	mTaskInfo.priority = GetPriority();
	mTaskInfo.originatingHost = *(GetOriginatingHost());

	return pmSuccess;
}

pmTaskInfo& pmTask::GetTaskInfo()
{
	if(!mTaskInfo.taskHandle)
		BuildTaskInfo();

	return mTaskInfo;
}

pmStatus pmTask::GetSubtaskInfo(ulong pSubtaskId, pmSubtaskInfo& pSubtaskInfo, bool& pOutputMemWriteOnly)
{
	pmSubscriptionInfo lInputMemSubscriptionInfo, lOutputMemSubscriptionInfo;
	void* lInputMem;
	void* lOutputMem;

	pSubtaskInfo.subtaskId = pSubtaskId;
	if(mMemRO && (lInputMem = mMemRO->GetMem()) && mSubscriptionManager.GetInputMemSubscriptionForSubtask(pSubtaskId, lInputMemSubscriptionInfo))
	{
		pSubtaskInfo.inputMem = ((char*)lInputMem + lInputMemSubscriptionInfo.offset);
		pSubtaskInfo.inputMemLength = lInputMemSubscriptionInfo.length;
	}
	else
	{
		pSubtaskInfo.inputMem = NULL;
		pSubtaskInfo.inputMemLength = 0;
	}
    
	if(mMemRW && (lOutputMem = mMemRW->GetMem()) && mSubscriptionManager.GetOutputMemSubscriptionForSubtask(pSubtaskId, lOutputMemSubscriptionInfo))
	{
		if(DoSubtasksNeedShadowMemory())
			pSubtaskInfo.outputMem = GetSubtaskShadowMem(pSubtaskId).addr;
		else
			pSubtaskInfo.outputMem = ((char*)lOutputMem + lOutputMemSubscriptionInfo.offset);

		pSubtaskInfo.outputMemLength = lOutputMemSubscriptionInfo.length;
        pOutputMemWriteOnly = (((pmOutputMemSection*)mMemRW)->GetAccessType() == pmOutputMemSection::WRITE_ONLY);
	}
	else
	{
		pSubtaskInfo.outputMem = NULL;
		pSubtaskInfo.outputMemLength = 0;
        pOutputMemWriteOnly = false;
	}

	return pmSuccess;
}

pmSubscriptionManager& pmTask::GetSubscriptionManager()
{
	return mSubscriptionManager;
}

pmReducer* pmTask::GetReducer()
{
	if(!mReducer)
		mReducer = new pmReducer(this);

	return mReducer;
}

pmStatus pmTask::MarkSubtaskExecutionFinished()
{
	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

		mSubtaskExecutionFinished = true;
	}

	if(mReducer)
		mReducer->CheckReductionFinish();

	return pmSuccess;
}

bool pmTask::HasSubtaskExecutionFinished()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtaskExecutionFinished;
}

pmStatus pmTask::IncrementSubtasksExecuted(ulong pSubtaskCount)
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	mSubtasksExecuted += pSubtaskCount;

	return pmSuccess;
}

ulong pmTask::GetSubtasksExecuted()
{
	FINALIZE_RESOURCE_PTR(dExecLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mExecLock, Lock(), Unlock());

	return mSubtasksExecuted;
}

pmStatus pmTask::SaveFinalReducedOutput(ulong pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	if(!mSubscriptionManager.GetOutputMemSubscriptionForSubtask(pSubtaskId, lSubscriptionInfo))
	{
		DestroySubtaskShadowMem(pSubtaskId);
		PMTHROW(pmFatalErrorException());
	}
std::cout << "Pending Implementation" << std::endl;
	subtaskShadowMem& lShadowMem = GetSubtaskShadowMem(pSubtaskId);
	(static_cast<pmOutputMemSection*>(mMemRW))->Update(lSubscriptionInfo.offset, lSubscriptionInfo.length, lShadowMem.addr);
	return DestroySubtaskShadowMem(pSubtaskId);

	//return (pmLocalTask*)this->MarkTaskEnd(GetSubtaskManager()->GetTaskExecutionStatus());
}

bool pmTask::DoSubtasksNeedShadowMemory()
{
	return (mCallbackUnit->GetDataReductionCB() != NULL);
}

pmStatus pmTask::CreateSubtaskShadowMem(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dShadowMemLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mShadowMemLock, Lock(), Unlock());

	pmSubscriptionInfo lSubscriptionInfo;

	if(mShadowMemMap.find(pSubtaskId) != mShadowMemMap.end() || !mSubscriptionManager.GetOutputMemSubscriptionForSubtask(pSubtaskId, lSubscriptionInfo))
		PMTHROW(pmFatalErrorException());

	mShadowMemMap[pSubtaskId].addr = new char[lSubscriptionInfo.length];
	mShadowMemMap[pSubtaskId].length = lSubscriptionInfo.length;

	memcpy(mShadowMemMap[pSubtaskId].addr, (void*)((char*)(mMemRW->GetMem()) + lSubscriptionInfo.offset), lSubscriptionInfo.length);

	return pmSuccess;
}

pmStatus pmTask::CreateSubtaskShadowMem(ulong pSubtaskId, char* pMem, size_t pMemLength)
{
	FINALIZE_RESOURCE_PTR(dShadowMemLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mShadowMemLock, Lock(), Unlock());

	pmSubscriptionInfo lSubscriptionInfo;

	if(mShadowMemMap.find(pSubtaskId) != mShadowMemMap.end() || !mSubscriptionManager.GetOutputMemSubscriptionForSubtask(pSubtaskId, lSubscriptionInfo))
		PMTHROW(pmFatalErrorException());

	mShadowMemMap[pSubtaskId].addr = new char[pMemLength];
	mShadowMemMap[pSubtaskId].length = pMemLength;

	memcpy(mShadowMemMap[pSubtaskId].addr, pMem, pMemLength);

	return pmSuccess;
}

pmTask::subtaskShadowMem& pmTask::GetSubtaskShadowMem(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dShadowMemLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mShadowMemLock, Lock(), Unlock());

	if(mShadowMemMap.find(pSubtaskId) == mShadowMemMap.end())
		PMTHROW(pmFatalErrorException());
	
	return mShadowMemMap[pSubtaskId];
}

pmStatus pmTask::DestroySubtaskShadowMem(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dShadowMemLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mShadowMemLock, Lock(), Unlock());

	if(mShadowMemMap.find(pSubtaskId) == mShadowMemMap.end())
		PMTHROW(pmFatalErrorException());

	delete[] mShadowMemMap[pSubtaskId].addr;

	return pmSuccess;
}

ulong pmTask::GetSequenceNumber()
{
    return mSequenceNumber;
}
    
pmStatus pmTask::SetSequenceNumber(ulong pSequenceNumber)
{
    mSequenceNumber = pSequenceNumber;
    
    return pmSuccess;
}
    

/* class pmLocalTask */
pmLocalTask::pmLocalTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit, 
	pmMachine* pOriginatingHost /* = PM_LOCAL_MACHINE */, pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = MAX_PRIORITY_LEVEL */,
	scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */)
	: pmTask(pTaskConf, pTaskConfLength, pTaskId, pMemRO, pMemRW, pSubtaskCount, pCallbackUnit, 0, pOriginatingHost, pCluster, pPriority, pSchedulingModel)
{
	mSubtaskManager = NULL;
	mTaskCommand = pmTaskCommand::CreateSharedPtr(pPriority, pmTaskCommand::BASIC_TASK);
    
	FINALIZE_RESOURCE(dSequenceLock, mSequenceLock.Lock(), mSequenceLock.Unlock());    
    SetSequenceNumber(mSequenceId);
    ++mSequenceId;
}

pmLocalTask::~pmLocalTask()
{
    pmTaskManager::GetTaskManager()->DeleteTask(this);  // could have already been erased from task manager in MarkSubtaskExecutionFinished

	if(mSubtaskManager)
		delete mSubtaskManager;
}

pmStatus pmLocalTask::MarkSubtaskExecutionFinished()
{
    pmCallbackUnit* lCallbackUnit = GetCallbackUnit();
    if(!lCallbackUnit->GetDataReductionCB() && !lCallbackUnit->GetDataScatterCB())
    {
        pmTask::MarkSubtaskExecutionFinished();
        pmTaskManager::GetTaskManager()->DeleteTask(this);  // Local task is only erased from task manager. It is actually deleted from memory by user
        MarkTaskEnd(GetSubtaskManager()->GetTaskExecutionStatus());
    }
    else
    {
        pmTask::MarkSubtaskExecutionFinished();
    }

    return pmSuccess;
}

pmStatus pmLocalTask::InitializeSubtaskManager(scheduler::schedulingModel pSchedulingModel)
{
	switch(pSchedulingModel)
	{
		case scheduler::PUSH:
			mSubtaskManager = new pmPushSchedulingManager(this);
			break;

		case scheduler::PULL:
			mSubtaskManager = new pmPullSchedulingManager(this);
			break;

		default:
			PMTHROW(pmFatalErrorException());
	}

	return pmSuccess;
}

std::vector<pmProcessingElement*>& pmLocalTask::GetAssignedDevices()
{
	return mDevices;
}

pmStatus pmLocalTask::WaitForCompletion()
{
	return mTaskCommand->WaitForFinish();
}

double pmLocalTask::GetExecutionTimeInSecs()
{
	return mTaskCommand->GetExecutionTimeInSecs();
}

pmStatus pmLocalTask::MarkTaskStart()
{
	return mTaskCommand->MarkExecutionStart();
}

pmStatus pmLocalTask::MarkTaskEnd(pmStatus pStatus)
{
	return mTaskCommand->MarkExecutionEnd(pStatus, std::tr1::static_pointer_cast<pmCommand>(mTaskCommand));
}

pmStatus pmLocalTask::GetStatus()
{
	return mTaskCommand->GetStatus();
}

pmStatus pmLocalTask::FindCandidateProcessingElements(std::set<pmProcessingElement*>& pDevices)
{
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

	mDevices.clear();
	pDevices.clear();

	pmSubtaskCB* lSubtaskCB = GetCallbackUnit()->GetSubtaskCB();
	if(lSubtaskCB)
	{
		for(uint i=0; i<MAX_DEVICE_TYPES; ++i)
		{
			if(lSubtaskCB->IsCallbackDefinedForDevice((pmDeviceTypes)i))
				lDevicePool->GetAllDevicesOfTypeInCluster((pmDeviceTypes)i, GetCluster(), pDevices);
		}
	}

	if(!pDevices.empty())
	{
		pmDeviceSelectionCB* lDeviceSelectionCB = GetCallbackUnit()->GetDeviceSelectionCB();
		if(lDeviceSelectionCB)
		{
			std::set<pmProcessingElement*> lDevices;

			std::set<pmProcessingElement*>::iterator lIter;
			for(lIter = pDevices.begin(); lIter != pDevices.end(); ++lIter)
			{
				if(lDeviceSelectionCB->Invoke(this, *lIter))
					lDevices.insert(*lIter);
			}

			pDevices = lDevices;
		}
	}

	// If the number of subtasks is less than number of devices, then discard the extra devices
	ulong lSubtaskCount = GetSubtaskCount();
	ulong lDeviceCount = (ulong)(pDevices.size());	
	ulong lFinalCount = lDeviceCount;

	if(lSubtaskCount < lDeviceCount)
		lFinalCount = lSubtaskCount;

	std::set<pmProcessingElement*>::iterator lIter = pDevices.begin();
	for(ulong i=0; i<lFinalCount; ++i)
	{
		mDevices.push_back(*lIter);
		++lIter;
	}

	mAssignedDeviceCount = (uint)(mDevices.size());
    
    pDevices.erase(lIter, pDevices.end());

	if(GetSchedulingModel() == scheduler::PULL)
		RandomizeDevices(mDevices);

	return pmSuccess;
}

pmSubtaskManager* pmLocalTask::GetSubtaskManager()
{
	return mSubtaskManager;
}


/* class pmRemoteTask */
pmRemoteTask::pmRemoteTask(void* pTaskConf, uint pTaskConfLength, ulong pTaskId, pmMemSection* pMemRO, pmMemSection* pMemRW, ulong pSubtaskCount, pmCallbackUnit* pCallbackUnit,
	uint pAssignedDeviceCount, pmMachine* pOriginatingHost, ulong pSequenceNumber, pmCluster* pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = MAX_PRIORITY_LEVEL */,
	scheduler::schedulingModel pSchedulingModel /* =  DEFAULT_SCHEDULING_MODEL */)
	: pmTask(pTaskConf, pTaskConfLength, pTaskId, pMemRO, pMemRW, pSubtaskCount, pCallbackUnit, pAssignedDeviceCount, pOriginatingHost, pCluster, pPriority, pSchedulingModel)
{
    SetSequenceNumber(pSequenceNumber);
}

pmRemoteTask::~pmRemoteTask()
{
	pmTaskManager::GetTaskManager()->DeleteTask(this);
}

pmStatus pmRemoteTask::MarkSubtaskExecutionFinished()
{
	pmCallbackUnit* lCallbackUnit = GetCallbackUnit();
	if(!lCallbackUnit->GetDataReductionCB() && !lCallbackUnit->GetDataScatterCB())
	{
		pmTask::MarkSubtaskExecutionFinished();
		delete this;
	}
	else
	{
		pmTask::MarkSubtaskExecutionFinished();
	}

	return pmSuccess;
}

pmStatus pmRemoteTask::AddAssignedDevice(pmProcessingElement* pDevice)
{
	mDevices.push_back(pDevice);

	uint lCount = GetAssignedDeviceCount();
	uint lSize = (uint)(mDevices.size());
	if(lSize > lCount)
		PMTHROW(pmFatalErrorException());

	if(lSize == lCount && GetSchedulingModel() == scheduler::PULL)
		RandomizeDevices(mDevices);

	return pmSuccess;
}

std::vector<pmProcessingElement*>& pmRemoteTask::GetAssignedDevices()
{
	return mDevices;
}

};


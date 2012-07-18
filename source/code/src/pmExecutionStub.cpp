
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

#include "pmExecutionStub.h"
#include "pmSignalWait.h"
#include "pmTask.h"
#include "pmDispatcherGPU.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmCommand.h"
#include "pmCallbackUnit.h"
#include "pmReducer.h"
#include "pmScheduler.h"

#include SYSTEM_CONFIGURATION_HEADER // for sched_setaffinity

#define INVOKE_SAFE_PROPAGATE_ON_FAILURE(objectType, object, function, ...) \
{ \
	pmStatus dStatus = pmSuccess; \
	objectType* dObject = object; \
	if(dObject) \
		dStatus = dObject->function(__VA_ARGS__); \
	if(dStatus != pmSuccess) \
		return dStatus; \
}

namespace pm
{

using namespace execStub;

/* class pmExecutionStub */
pmExecutionStub::pmExecutionStub(uint pDeviceIndexOnMachine)
{
	mDeviceIndexOnMachine = pDeviceIndexOnMachine;

	stubEvent lEvent;
	lEvent.eventId = THREAD_BIND;
	threadBind lBindDetails;
	lEvent.bindDetails = lBindDetails;
	SwitchThread(lEvent, MAX_CONTROL_PRIORITY);
}

pmExecutionStub::~pmExecutionStub()
{
	#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down stub thread");
	#endif
}

pmProcessingElement* pmExecutionStub::GetProcessingElement()
{
	return pmDevicePool::GetDevicePool()->GetDeviceAtMachineIndex(PM_LOCAL_MACHINE, mDeviceIndexOnMachine);
}

pmStatus pmExecutionStub::Push(pmSubtaskRange pRange)
{
	if(pRange.endSubtask < pRange.startSubtask)
		PMTHROW(pmFatalErrorException());

	stubEvent lEvent;
	subtaskExec lExecDetails;
	lExecDetails.range = pRange;
	lExecDetails.rangeExecutedOnce = false;
	lExecDetails.lastExecutedSubtaskId = 0;
	lEvent.eventId = SUBTASK_EXEC;
	lEvent.execDetails = lExecDetails;

	return SwitchThread(lEvent, pRange.task->GetPriority());
}

pmStatus pmExecutionStub::ReduceSubtasks(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2)
{
	stubEvent lEvent;
	subtaskReduce lReduceDetails;
	lReduceDetails.task = pTask;
	lReduceDetails.subtaskId1 = pSubtaskId1;
	lReduceDetails.subtaskId2 = pSubtaskId2;
	lEvent.eventId = SUBTASK_REDUCE;
	lEvent.reduceDetails = lReduceDetails;

	return SwitchThread(lEvent, pTask->GetPriority());
}

pmStatus pmExecutionStub::StealSubtasks(pmTask* pTask, pmProcessingElement* pRequestingDevice, double pExecutionRate)
{
	stubEvent lEvent;
	subtaskSteal lStealDetails;
	lStealDetails.requestingDevice = pRequestingDevice;
	lStealDetails.requestingDeviceExecutionRate = pExecutionRate;
	lStealDetails.task = pTask;
	lEvent.eventId = SUBTASK_STEAL;
	lEvent.stealDetails = lStealDetails;

	return SwitchThread(lEvent, pTask->GetPriority() - 1);	// Steal events are sent at one higher priority level than the task
}

pmStatus pmExecutionStub::CancelSubtasks(pmTask* pTask)
{
	stubEvent lEvent;
	subtaskCancel lCancelDetails;
	lCancelDetails.task = pTask;
	lCancelDetails.priority = pTask->GetPriority();
	lEvent.eventId = SUBTASK_CANCEL;
	lEvent.cancelDetails = lCancelDetails;

	return SwitchThread(lEvent, MAX_CONTROL_PRIORITY);
}

pmStatus pmExecutionStub::FreeGpuResources()
{
#ifdef SUPPORT_CUDA
        stubEvent lEvent;
        lEvent.eventId = FREE_GPU_RESOURCES;

        SwitchThread(lEvent, RESERVED_PRIORITY);
#endif

	return pmSuccess;
}
    
pmStatus pmExecutionStub::ClearPendingStealCommands(pmTask* pTask)
{
    DeleteMatchingCommands(pTask->GetPriority() - 1, stealEventClearMatchFunc, pTask); 	// Steal events are sent at one higher priority level than the task
    
    return WaitIfCurrentCommandMatches(stealEventClearMatchFunc, pTask);
}

pmStatus pmExecutionStub::ThreadSwitchCallback(stubEvent& pEvent)
{
	try
	{
		return ProcessEvent(pEvent);
	}
        catch(pmException e)
        {
                pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from stub thread");
        }

	return pmSuccess;
}

pmStatus pmExecutionStub::ProcessEvent(stubEvent& pEvent)
{
	switch(pEvent.eventId)
	{
		case THREAD_BIND:
		{
			BindToProcessingElement();
			break;
		}
	
		case SUBTASK_EXEC:	/* Comes from scheduler thread */
		{
			pmSubtaskRange lRange = pEvent.execDetails.range;
			ulong lCompletedCount, lLastExecutedSubtaskId;
    
			pmSubtaskRangeCommandPtr lCommand = pmSubtaskRangeCommand::CreateSharedPtr(lRange.task->GetPriority(), pmSubtaskRangeCommand::BASIC_SUBTASK_RANGE);

			pmSubtaskRange lCurrentRange;
			if(pEvent.execDetails.rangeExecutedOnce)
			{
				lCurrentRange.task = lRange.task;
				lCurrentRange.endSubtask = lRange.endSubtask;
				lCurrentRange.startSubtask = pEvent.execDetails.lastExecutedSubtaskId + 1;
			}
			else
			{
				lCurrentRange = lRange;
			}

			lCommand->MarkExecutionStart();
			pmStatus lExecStatus = Execute(lCurrentRange, lLastExecutedSubtaskId);

			lCommand->MarkExecutionEnd(lExecStatus, std::tr1::static_pointer_cast<pmCommand>(lCommand));

			if(lLastExecutedSubtaskId < lRange.startSubtask || lLastExecutedSubtaskId > lRange.endSubtask)
				PMTHROW(pmFatalErrorException());

            pEvent.execDetails.rangeExecutedOnce = true;
            lCompletedCount = lLastExecutedSubtaskId - lCurrentRange.startSubtask + 1;

			pEvent.execDetails.lastExecutedSubtaskId = lLastExecutedSubtaskId;

			lRange.task->GetTaskExecStats().RecordSubtaskExecutionStats(this, lCompletedCount, lCommand->GetExecutionTimeInSecs());

			if(lLastExecutedSubtaskId == lRange.endSubtask)
				pmScheduler::GetScheduler()->SendAcknowledment(GetProcessingElement(), lRange, lExecStatus);
            else
                SwitchThread(pEvent, lRange.task->GetPriority());

			break;
		}

		case SUBTASK_REDUCE:
		{
			DoSubtaskReduction(pEvent.reduceDetails.task, pEvent.reduceDetails.subtaskId1, pEvent.reduceDetails.subtaskId2);

			break;
		}

		case SUBTASK_CANCEL:	/* Comes from scheduler thread */
		{
			/* Do not dereference lTask as it could already have been purged */
			pmTask* lTask = pEvent.cancelDetails.task;
			ushort lPriority = pEvent.cancelDetails.priority;

			DeleteMatchingCommands(lPriority, execEventMatchFunc, lTask);

			break;
		}

		case SUBTASK_STEAL:	/* Comes from scheduler thread */
		{
			bool lStealSuccess = false;
			pmTask* lTask = pEvent.stealDetails.task;
			ushort lPriority = lTask->GetPriority();

			stubEvent lTaskEvent;
			if(DeleteAndGetFirstMatchingCommand(lPriority, execEventMatchFunc, lTask, lTaskEvent) == pmSuccess)
			{    
				double lLocalRate = lTask->GetTaskExecStats().GetStubExecutionRate(this);
				double lRemoteRate = pEvent.stealDetails.requestingDeviceExecutionRate;
				double lTotalExecRate = lLocalRate + lRemoteRate;
			
				ulong lAvailableSubtasks;
				if(lTaskEvent.execDetails.rangeExecutedOnce)
					lAvailableSubtasks = lTaskEvent.execDetails.range.endSubtask - lTaskEvent.execDetails.lastExecutedSubtaskId;
				else
					lAvailableSubtasks = lTaskEvent.execDetails.range.endSubtask - lTaskEvent.execDetails.range.startSubtask + 1;

				double lOverheadTime = 0;	// Add network and other overheads here
                ulong lStealCount = 0;
                
                if(lLocalRate == (double)0.0)
                {
                    lStealCount = lAvailableSubtasks;
                }
                else
                {
                    double lTotalExecutionTimeRequired = lAvailableSubtasks / lTotalExecRate;	// if subtasks are divided between both devices, how much time reqd
                    double lLocalExecutionTimeForAllSubtasks = lAvailableSubtasks / lLocalRate;	// if all subtasks are executed locally, how much time it will take
                    double lDividedExecutionTimeForAllSubtasks = lTotalExecutionTimeRequired + lOverheadTime;

                    if(lLocalExecutionTimeForAllSubtasks > lDividedExecutionTimeForAllSubtasks)
                    {
                        double lTimeDiff = lLocalExecutionTimeForAllSubtasks - lDividedExecutionTimeForAllSubtasks;
                        lStealCount = (ulong)(lTimeDiff * lLocalRate);
                    }
                }
                
                if(lStealCount)
                {                    
                    pmSubtaskRange lStolenRange;
                    lStolenRange.task = lTask;
                    lStolenRange.startSubtask = (lTaskEvent.execDetails.range.endSubtask - lStealCount) + 1;
                    lStolenRange.endSubtask = lTaskEvent.execDetails.range.endSubtask;
                    
                    lTaskEvent.execDetails.range.endSubtask -= lStealCount;
                    
                    if(lTaskEvent.execDetails.rangeExecutedOnce && lTaskEvent.execDetails.lastExecutedSubtaskId == lTaskEvent.execDetails.range.endSubtask)
                        pmScheduler::GetScheduler()->SendAcknowledment(GetProcessingElement(), lTaskEvent.execDetails.range, pmSuccess);
                    else
                        SwitchThread(lTaskEvent, lTask->GetPriority());
                    
                    lStealSuccess = true;
                    pmScheduler::GetScheduler()->StealSuccessEvent(pEvent.stealDetails.requestingDevice, GetProcessingElement(), lStolenRange);
                }
                else
                {
                    SwitchThread(lTaskEvent, lTask->GetPriority());                    
                }
			}

			if(!lStealSuccess)
				pmScheduler::GetScheduler()->StealFailedEvent(pEvent.stealDetails.requestingDevice, GetProcessingElement(), lTask);

			break;
		}

		case FREE_GPU_RESOURCES:
		{
	#ifdef SUPPORT_CUDA
			((pmStubGPU*)this)->FreeLastExecutionResources();
	#endif
			break;
		}
	}

	return pmSuccess;
}

bool pmExecutionStub::IsHighPriorityEventWaiting(ushort pPriority)
{
	return GetPriorityQueue().IsHighPriorityElementPresent(pPriority);
}

pmStatus pmExecutionStub::CommonPreExecuteOnCPU(pmTask* pTask, ulong pSubtaskId)
{
#ifdef ENABLE_TASK_PROFILING
    pmTaskProfiler* lTaskProfiler = pTask->GetTaskProfiler();

    lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_PARTITIONING, true);
	pTask->GetSubscriptionManager().InitializeSubtaskDefaults(pSubtaskId);
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataDistributionCB, pTask->GetCallbackUnit()->GetDataDistributionCB(), Invoke, pTask, pSubtaskId, GetType());
    lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_PARTITIONING, false);

	pTask->GetSubscriptionManager().FetchSubtaskSubscriptions(pSubtaskId, GetType());
    
	if(pTask->GetMemSectionRW() && pTask->DoSubtasksNeedShadowMemory())
    {
        lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_REDUCTION, true);
		pTask->CreateSubtaskShadowMem(pSubtaskId);
        lTaskProfiler->RecordProfileEvent(pmTaskProfiler::DATA_REDUCTION, false);
    }

    lTaskProfiler->RecordProfileEvent(pmTaskProfiler::SUBTASK_EXECUTION, true);
#else
	pTask->GetSubscriptionManager().InitializeSubtaskDefaults(pSubtaskId);
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataDistributionCB, pTask->GetCallbackUnit()->GetDataDistributionCB(), Invoke, pTask, pSubtaskId, GetType());
	pTask->GetSubscriptionManager().FetchSubtaskSubscriptions(pSubtaskId, GetType());

	if(pTask->GetMemSectionRW() && pTask->DoSubtasksNeedShadowMemory())
		pTask->CreateSubtaskShadowMem(pSubtaskId);
#endif
	
	return pmSuccess;
}

pmStatus pmExecutionStub::CommonPostExecuteOnCPU(pmTask* pTask, ulong pSubtaskId)
{
#ifdef ENABLE_TASK_PROFILING
    pTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::SUBTASK_EXECUTION, false);
#endif

	pmCallbackUnit* lCallbackUnit = pTask->GetCallbackUnit();
	pmDataReductionCB* lReduceCallback = lCallbackUnit->GetDataReductionCB();

	if(lReduceCallback)
		pTask->GetReducer()->AddSubtask(pSubtaskId);
	else
		INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmDataRedistributionCB, pTask->GetCallbackUnit()->GetDataRedistributionCB(), Invoke, pTask, pSubtaskId);

	return pmSuccess;
}

pmStatus pmExecutionStub::DoSubtaskReduction(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2)
{
	pmStatus lStatus = pTask->GetCallbackUnit()->GetDataReductionCB()->Invoke(pTask, pSubtaskId1, pSubtaskId2);

	/* Handle Transactions */
	switch(lStatus)
	{
		case pmSuccess:
		{
			pTask->DestroySubtaskShadowMem(pSubtaskId2);
			pTask->GetReducer()->AddSubtask(pSubtaskId1);

			break;
		}

		default:
		{
			pTask->DestroySubtaskShadowMem(pSubtaskId1);
			pTask->DestroySubtaskShadowMem(pSubtaskId2);
		}
	}

	return lStatus;
}


/* class pmStubCPU */
pmStubCPU::pmStubCPU(size_t pCoreId, uint pDeviceIndexOnMachine)
	: pmExecutionStub(pDeviceIndexOnMachine)
{
	mCoreId = pCoreId;
}

pmStubCPU::~pmStubCPU()
{
}

pmStatus pmStubCPU::BindToProcessingElement()
{
	 return SetProcessorAffinity((int)mCoreId);
}

size_t pmStubCPU::GetCoreId()
{
	return mCoreId;
}

std::string pmStubCPU::GetDeviceName()
{
	// Try to use processor_info or getcpuid system calls
	return std::string();
}

std::string pmStubCPU::GetDeviceDescription()
{
	// Try to use processor_info or getcpuid system calls
	return std::string();
}

pmDeviceTypes pmStubCPU::GetType()
{
	return CPU;
}

pmStatus pmStubCPU::Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId)
{
	ulong index = pRange.startSubtask;
	for(; index <= pRange.endSubtask; ++index)
	{
		Execute(pRange.task, index);

		if(IsHighPriorityEventWaiting(pRange.task->GetPriority()))
		{
			pLastExecutedSubtaskId = index;            
			return pmSuccess;
		}
	}

	pLastExecutedSubtaskId = pRange.endSubtask;

	return pmSuccess;
}

pmStatus pmStubCPU::Execute(pmTask* pTask, ulong pSubtaskId)
{
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId));
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, CPU, pTask, pSubtaskId, mCoreId);
	PROPAGATE_FAILURE_RET_STATUS(CommonPostExecuteOnCPU(pTask, pSubtaskId));
	
	return pmSuccess;
}


/* class pmStubGPU */
pmStubGPU::pmStubGPU(uint pDeviceIndexOnMachine)
	: pmExecutionStub(pDeviceIndexOnMachine)
{
}

pmStubGPU::~pmStubGPU()
{
}

/* class pmStubCUDA */
pmStubCUDA::pmStubCUDA(size_t pDeviceIndex, uint pDeviceIndexOnMachine)
	: pmStubGPU(pDeviceIndexOnMachine)
{
	mDeviceIndex = pDeviceIndex;
}

pmStubCUDA::~pmStubCUDA()
{
}

pmStatus pmStubCUDA::FreeResources()
{
#ifdef SUPPORT_CUDA
    FreeGpuResources();
#endif
    return pmSuccess;
}

pmStatus pmStubCUDA::FreeLastExecutionResources()
{
#ifdef SUPPORT_CUDA
    pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->FreeLastExecutionResources(mDeviceIndex);
#endif
    return pmSuccess;
}

pmStatus pmStubCUDA::BindToProcessingElement()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->BindToDevice(mDeviceIndex);
#else
	return pmSuccess;
#endif
}

std::string pmStubCUDA::GetDeviceName()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceName(mDeviceIndex);
#else
	return std::string();
#endif
}

std::string pmStubCUDA::GetDeviceDescription()
{
#ifdef SUPPORT_CUDA
	return pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->GetDeviceDescription(mDeviceIndex);
#else
	return std::string();
#endif
}

pmDeviceTypes pmStubCUDA::GetType()
{
#ifdef SUPPORT_CUDA
	return GPU_CUDA;
#else
	PMTHROW(pmFatalErrorException());
	return MAX_DEVICE_TYPES;
#endif
}

pmStatus pmStubCUDA::Execute(pmSubtaskRange pRange, ulong& pLastExecutedSubtaskId)
{
	ulong index = pRange.startSubtask;
	for(; index <= pRange.endSubtask; ++index)
	{
		Execute(pRange.task, index);

		if(IsHighPriorityEventWaiting(pRange.task->GetPriority()))
		{
			pLastExecutedSubtaskId = index;
			return pmSuccess;
		}
	}

	pLastExecutedSubtaskId = pRange.endSubtask;

	return pmSuccess;
}

pmStatus pmStubCUDA::Execute(pmTask* pTask, ulong pSubtaskId)
{
#ifdef SUPPORT_CUDA
	PROPAGATE_FAILURE_RET_STATUS(CommonPreExecuteOnCPU(pTask, pSubtaskId));
	INVOKE_SAFE_PROPAGATE_ON_FAILURE(pmSubtaskCB, pTask->GetCallbackUnit()->GetSubtaskCB(), Invoke, GPU_CUDA, pTask, pSubtaskId, mDeviceIndex);
	PROPAGATE_FAILURE_RET_STATUS(CommonPostExecuteOnCPU(pTask, pSubtaskId));
#endif

	return pmSuccess;
}


bool execEventMatchFunc(stubEvent& pEvent, void* pCriterion)
{
	if(pEvent.eventId == SUBTASK_EXEC && pEvent.execDetails.range.task == (pmTask*)pCriterion)
		return true;

	return false;
}
    
bool stealEventClearMatchFunc(execStub::stubEvent& pEvent, void* pCriterion)
{
	if(pEvent.eventId == SUBTASK_STEAL && pEvent.execDetails.range.task == (pmTask*)pCriterion)
		return true;
    
	return false;    
}

};


/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution,
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

#include "pmPreprocessorTask.h"
#include "pmTask.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmAddressSpace.h"
#include "pmCallbackUnit.h"
#include "pmStubManager.h"

namespace pm
{

using namespace preprocessorTask;
    
struct preprocessorTaskConf
{
    preprocessorTaskType taskType;
    ulong originalTaskSubtasks;
    uint originatingHost;   // of user task
    ulong sequenceNumber;  // of user task
};

void PostAffinityAddressSpaceFetchCallback(const pmCommandPtr& pCountDownCommand);

void PostAffinityAddressSpaceFetchCallback(const pmCommandPtr& pCountDownCommand)
{
    pmTask* lPreprocessorTask = const_cast<pmTask*>(static_cast<const pmTask*>(pCountDownCommand->GetUserIdentifier()));
    preprocessorTaskConf* lTaskConf = (preprocessorTaskConf*)lPreprocessorTask->GetTaskConfiguration();
    pmLocalTask* lUserTask = static_cast<pmLocalTask*>(pmTaskManager::GetTaskManager()->FindTask(pmMachinePool::GetMachinePool()->GetMachine(lTaskConf->originatingHost), lTaskConf->sequenceNumber));

    const std::vector<pmTaskMemory>& lPreprocessorTaskMemVector = lPreprocessorTask->GetTaskMemVector();
    pmAddressSpace* lAddressSpace = lPreprocessorTaskMemVector[lPreprocessorTaskMemVector.size() - 1].addressSpace;

    lUserTask->ComputeAffinityData(lAddressSpace);
    lUserTask->StartScheduling();
}

pmStatus preprocessorTask_dataDistributionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
    preprocessorTaskConf* lTaskConf = (preprocessorTaskConf*)pTaskInfo.taskConf;

    DEBUG_EXCEPTION_ASSERT(pmTaskManager::GetTaskManager()->FindTask(pmMachinePool::GetMachinePool()->GetMachine(lTaskConf->originatingHost), lTaskConf->sequenceNumber));
    
    switch(lTaskConf->taskType)
    {
        case AFFINITY_DEDUCER:
        {
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, pSubtaskInfo.memCount - 1, WRITE_SUBSCRIPTION, pmSubscriptionInfo(pDeviceInfo.host * lTaskConf->originalTaskSubtasks * sizeof(ulong), lTaskConf->originalTaskSubtasks * sizeof(ulong)));
    
            break;
        }

        case DEPENDENCY_EVALUATOR:
            break;
        
        case AFFINITY_AND_DEPENDENCY_TASK:
            break;
            
        default:
            EXCEPTION_ASSERT(0);
    }
    
    return pmSuccess;
}

pmStatus preprocessorTask_cpuCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
    preprocessorTaskConf* lTaskConf = (preprocessorTaskConf*)pTaskInfo.taskConf;
    pmTask* lUserTask = pmTaskManager::GetTaskManager()->FindTask(pmMachinePool::GetMachinePool()->GetMachine(lTaskConf->originatingHost), lTaskConf->sequenceNumber);

    EXCEPTION_ASSERT(lUserTask);
    
    switch(lTaskConf->taskType)
    {
        case AFFINITY_DEDUCER:
        {
            ulong* lOutputMem = (ulong*)pSubtaskInfo.memInfo[pSubtaskInfo.memCount - 1].writePtr;
            
            pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetCpuStub(pDeviceInfo.deviceIdOnHost);
            pmSubscriptionManager& lSubscriptionManager = lUserTask->GetSubscriptionManager();
            
            for(ulong i = 0; i < lTaskConf->originalTaskSubtasks; ++i)
            {
                lSubscriptionManager.FindSubtaskMemDependencies(lStub, i, NULL, true);
                lOutputMem[i] = lSubscriptionManager.FindLocalInputDataSizeForSubtask(lStub, i);
            }
            
            break;
        }

        case DEPENDENCY_EVALUATOR:
            break;
        
        case AFFINITY_AND_DEPENDENCY_TASK:
            break;
            
        default:
            EXCEPTION_ASSERT(0);
    }

    return pmSuccess;
}

pmStatus preprocessorTask_taskCompletionCallback(pmTaskInfo pTaskInfo)
{
    preprocessorTaskConf* lTaskConf = (preprocessorTaskConf*)pTaskInfo.taskConf;
    
    pmLocalTask* lPreprocessorTask = static_cast<pmLocalTask*>(pTaskInfo.taskHandle);
    pmLocalTask* lUserTask = static_cast<pmLocalTask*>(pmTaskManager::GetTaskManager()->FindTask(pmMachinePool::GetMachinePool()->GetMachine(lTaskConf->originatingHost), lTaskConf->sequenceNumber));
    const pmTaskCompletionCB* lTaskCompletionCB = lUserTask->GetCallbackUnit()->GetTaskCompletionCB();
    
    lUserTask->SetPreprocessorTask(lPreprocessorTask);
    lPreprocessorTask->SetTaskCompletionCallback(lTaskCompletionCB ? lTaskCompletionCB->GetCallback() : NULL);
    lUserTask->SetTaskCompletionCallback(userTask_auxillaryTaskCompletionCallback);

    // Consume data computed by pre-processor task and delete preprocessor task's address spaces
    const std::vector<pmTaskMemory>& lPreprocessorTaskMemVector = lPreprocessorTask->GetTaskMemVector();

    switch(lTaskConf->taskType)
    {
        case AFFINITY_DEDUCER:
        {
            pmCommandPtr lCountDownCommand = pmCountDownCommand::CreateSharedPtr(1, lUserTask->GetPriority(), 0, PostAffinityAddressSpaceFetchCallback, lPreprocessorTask);
            lCountDownCommand->MarkExecutionStart();

            pmAddressSpace* lAddressSpace = lPreprocessorTaskMemVector[lPreprocessorTaskMemVector.size() - 1].addressSpace;
            lAddressSpace->FetchAsync(lUserTask->GetPriority(), lCountDownCommand);
        
            break;
        }

        case DEPENDENCY_EVALUATOR:
            break;
        
        case AFFINITY_AND_DEPENDENCY_TASK:
            break;
            
        default:
            EXCEPTION_ASSERT(0);
    }

    return pmSuccess;
}
    
pmStatus userTask_auxillaryTaskCompletionCallback(pmTaskInfo pTaskInfo)
{
    pmLocalTask* lUserTask = static_cast<pmLocalTask*>(pTaskInfo.taskHandle);
    const pmLocalTask* lPreprocessorTask = lUserTask->GetPreprocessorTask();

    // Reset the original task callback
    lUserTask->SetTaskCompletionCallback(lPreprocessorTask->GetCallbackUnit()->GetTaskCompletionCB()->GetCallback());

    // Destroy preprocessorTask
    const std::vector<pmTaskMemory>& lPreprocessorTaskMemVector = lPreprocessorTask->GetTaskMemVector();
    pmAddressSpace* lAddressSpace = lPreprocessorTaskMemVector[lPreprocessorTaskMemVector.size() - 1].addressSpace;

    EXCEPTION_ASSERT(pmReleaseMemory(lAddressSpace->GetUserMemHandle()) == pmSuccess);
    EXCEPTION_ASSERT(pmReleaseTask((pmTaskHandle)(lPreprocessorTask)) == pmSuccess);

    // Call the original task completion callback
    return lUserTask->GetCallbackUnit()->GetTaskCompletionCB()->Invoke(lUserTask);
}
    
pmPreprocessorTask::pmPreprocessorTask()
{
    pmCallbacks lPreprocessorTaskCallbacks(preprocessorTask_dataDistributionCallback, preprocessorTask_cpuCallback, (pmSubtaskCallback_GPU_CUDA)NULL);
    lPreprocessorTaskCallbacks.taskCompletionCallback = preprocessorTask_taskCompletionCallback;

    EXCEPTION_ASSERT(pmRegisterCallbacks((char*)"PreprocessorTaskCallback", lPreprocessorTaskCallbacks, &mPreprocessorTaskCallbackHandle) == pmSuccess);
}
    
pmPreprocessorTask::~pmPreprocessorTask()
{
    // Unregister callbacks
}

pmPreprocessorTask* pmPreprocessorTask::GetPreprocessorTask()
{
	static pmPreprocessorTask lPreprocessorTask;
    return &lPreprocessorTask;
}

void pmPreprocessorTask::DeduceAffinity(pm::pmLocalTask* pLocalTask)
{
    DEBUG_EXCEPTION_ASSERT(pLocalTask->GetCallbackUnit() && pLocalTask->GetCallbackUnit()->GetDataDistributionCB() && !(static_cast<pmTask*>(*pTaskHandle))->HasReadOnlyLazyAddressSpace());

    LaunchPreprocessorTask(pLocalTask, AFFINITY_DEDUCER);
}

void pmPreprocessorTask::EvaluateDependency(pmLocalTask* pLocalTask)
{
    LaunchPreprocessorTask(pLocalTask, DEPENDENCY_EVALUATOR);
}

void pmPreprocessorTask::DeduceAffinityAndEvaluateDependency(pmLocalTask* pLocalTask)
{
    DEBUG_EXCEPTION_ASSERT(pLocalTask->GetCallbackUnit() && pLocalTask->GetCallbackUnit()->GetDataDistributionCB() && !(static_cast<pmTask*>(*pTaskHandle))->HasReadOnlyLazyAddressSpace());

    LaunchPreprocessorTask(pLocalTask, AFFINITY_AND_DEPENDENCY_TASK);
}
    
void pmPreprocessorTask::LaunchPreprocessorTask(pm::pmLocalTask* pLocalTask, preprocessorTaskType pTaskType)
{
    preprocessorTaskConf lTaskConf = {pTaskType, pLocalTask->GetSubtaskCount(), *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber()};

    std::set<const pmMachine*> lMachinesSet;
    pmProcessingElement::GetMachines(pLocalTask->GetAssignedDevices(), lMachinesSet);
    
    std::vector<pmTaskMem> lMemVector;
    
    ulong lPreprocessorSubtasks = 0;

    switch(pTaskType)
    {
        case preprocessorTask::AFFINITY_DEDUCER:
        {
            lPreprocessorSubtasks = lMachinesSet.size();

            // Create an output address space for the preprocessor task (every machine stores percentage local data for all subtasks) */
            size_t lOutputMemSize = lPreprocessorSubtasks * pLocalTask->GetSubtaskCount() * sizeof(ulong);
            pmMemHandle lMemHandle;
            pmCreateMemory(lOutputMemSize, &lMemHandle);
            
            lMemVector.emplace_back(lMemHandle, WRITE_ONLY, SUBSCRIPTION_NATURAL, true);
            
            break;
        }
            
        case preprocessorTask::DEPENDENCY_EVALUATOR:
        {
            break;
        }
            
        case preprocessorTask::AFFINITY_AND_DEPENDENCY_TASK:
        {
            break;
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
    
    pmTaskDetails lTaskDetails(&lTaskConf, sizeof(preprocessorTaskConf), &lMemVector[0], (uint)lMemVector.size(), mPreprocessorTaskCallbackHandle, lPreprocessorSubtasks);
    
	lTaskDetails.policy = EQUAL_STATIC;
	lTaskDetails.multiAssignEnabled = false;
    lTaskDetails.overlapComputeCommunication = false;
    lTaskDetails.suppressTaskLogs = true;

	pmTaskHandle lTaskHandle = NULL;

	EXCEPTION_ASSERT(pmSubmitTask(lTaskDetails, &lTaskHandle) == pmSuccess);
}
    
};



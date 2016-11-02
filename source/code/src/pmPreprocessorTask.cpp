
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

#include "pmPreprocessorTask.h"
#include "pmTask.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmAddressSpace.h"
#include "pmCallbackUnit.h"
#include "pmStubManager.h"
#include "pmController.h"
#include "pmAffinityTable.h"

#include <random>

namespace pm
{

using namespace preprocessorTask;
    
struct preprocessorTaskConf
{
    preprocessorTaskType taskType;
    ulong originalTaskSubtasks;
    uint originatingHost;   // of user task
    ulong sequenceNumber;  // of user task
    pmAffinityCriterion affinityCriterion;
    uint machinesCount;
#ifdef CENTRALIZED_AFFINITY_COMPUTATION
    uint originalTaskMachinesCount;
#endif
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

#ifdef ENABLE_TASK_PROFILING
    lUserTask->GetTaskProfiler()->RecordProfileEvent(taskProfiler::PREPROCESSOR_TASK_EXECUTION, false);
#endif

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
            size_t lSampleSize = pmPreprocessorTask::GetSampleSizeForAffinityCriterion(lTaskConf->affinityCriterion);
            
        #ifdef CENTRALIZED_AFFINITY_COMPUTATION
            lSampleSize *= lTaskConf->originalTaskMachinesCount;
        #endif

            if(lTaskConf->originalTaskSubtasks * lTaskConf->machinesCount != pTaskInfo.subtaskCount)
                lSampleSize *= 2;

            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, pSubtaskInfo.memCount - 1, WRITE_SUBSCRIPTION, pmSubscriptionInfo(pSubtaskInfo.subtaskId * lSampleSize, lSampleSize));
    
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
            void* lOutputMem = pSubtaskInfo.memInfo[pSubtaskInfo.memCount - 1].writePtr;

            pmExecutionStub* lStub = pmStubManager::GetStubManager()->GetCpuStub(pDeviceInfo.deviceIdOnHost);
            pmSubscriptionManager& lSubscriptionManager = lUserTask->GetSubscriptionManager();
            
            ulong lUserSubtaskId = (pSubtaskInfo.subtaskId % lTaskConf->originalTaskSubtasks);
            
            bool lSelectiveSubtasks = false;
            if(lTaskConf->originalTaskSubtasks * lTaskConf->machinesCount != pTaskInfo.subtaskCount)
            {
                lSelectiveSubtasks = true;
                
                ulong lSubtasksPerMachine = pTaskInfo.subtaskCount / lTaskConf->machinesCount;
                float lSubtaskSpan = (float)lTaskConf->originalTaskSubtasks / lSubtasksPerMachine;
                
                EXCEPTION_ASSERT(lSubtaskSpan > 1.0);
                
                ulong lChoiceId = (pSubtaskInfo.subtaskId % lSubtasksPerMachine);
                ulong lFirstPossibleSubtask = (ulong)((float)lChoiceId * lSubtaskSpan);
                ulong lLastPosibleSubtask = (ulong)((float)(lChoiceId + 1) * lSubtaskSpan);
                
                if(lLastPosibleSubtask > lTaskConf->originalTaskSubtasks)
                    lLastPosibleSubtask = lTaskConf->originalTaskSubtasks;
                
                if(lChoiceId == lSubtasksPerMachine - 1 && lLastPosibleSubtask != lTaskConf->originalTaskSubtasks)
                    lLastPosibleSubtask = lTaskConf->originalTaskSubtasks;
                
                ulong lPossibilities = lLastPosibleSubtask - lFirstPossibleSubtask;
                EXCEPTION_ASSERT(lPossibilities);
                
                std::random_device lRandomDevice;
                std::mt19937 lGenerator(lRandomDevice());

                lUserSubtaskId = lFirstPossibleSubtask + (lGenerator() % lPossibilities);
            }
            
            lSubscriptionManager.FindSubtaskMemDependencies(lStub, lUserSubtaskId, NULL, true);

        #ifdef CENTRALIZED_AFFINITY_COMPUTATION
            const std::vector<const pmMachine*>& lMachinesVector = ((pmLocalTask*)lUserTask)->GetAssignedMachinesInOrder();

            switch(lTaskConf->affinityCriterion)
            {
                case MAXIMIZE_LOCAL_DATA:
                    if(lSelectiveSubtasks)
                    {
                        *((ulong*)lOutputMem) = (ulong)lUserSubtaskId;
                        lOutputMem = (void*)((ulong*)lOutputMem + 1);
                    }

                    lSubscriptionManager.FindLocalInputDataSizeForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, (ulong*)lOutputMem);
                    break;

                case MINIMIZE_REMOTE_SOURCES:
                    if(lSelectiveSubtasks)
                    {
                        *((uint*)lOutputMem) = (uint)lUserSubtaskId;
                        lOutputMem = (void*)((uint*)lOutputMem + 1);
                    }

                    lSubscriptionManager.FindRemoteDataSourcesForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, (uint*)lOutputMem);
                    break;
                
                case MINIMIZE_REMOTE_TRANSFER_EVENTS:
                    if(lSelectiveSubtasks)
                    {
                        *((ulong*)lOutputMem) = (ulong)lUserSubtaskId;
                        lOutputMem = (void*)((ulong*)lOutputMem + 1);
                    }

                    lSubscriptionManager.FindRemoteTransferEventsForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, (ulong*)lOutputMem);
                    break;

                case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
                    if(lSelectiveSubtasks)
                    {
                        *((float*)lOutputMem) = (float)lUserSubtaskId;
                        lOutputMem = (void*)((float*)lOutputMem + 1);
                    }

                    lSubscriptionManager.FindRemoteTransferEstimateForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, (float*)lOutputMem);
                    break;
                    
                case DERIVED_AFFINITY:
                    if(lSelectiveSubtasks)
                    {
                        *((float*)lOutputMem) = (float)lUserSubtaskId;
                        lOutputMem = (void*)((float*)lOutputMem + 1);
                    }

                    //lSubscriptionManager.FindLocalInputDataSizeForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, &((derivedAffinityData*)lOutputMem)->localBytes, sizeof(derivedAffinityData));
                    //lSubscriptionManager.FindRemoteDataSourcesForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, &((derivedAffinityData*)lOutputMem)->remoteNodes, sizeof(derivedAffinityData));
                    lSubscriptionManager.FindRemoteTransferEventsForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, &((derivedAffinityData*)lOutputMem)->remoteEvents, sizeof(derivedAffinityData));
                    lSubscriptionManager.FindRemoteTransferEstimateForSubtaskOnMachines(lStub, lUserSubtaskId, lMachinesVector, &((derivedAffinityData*)lOutputMem)->estimatedTime, sizeof(derivedAffinityData));
                    
                    break;

                default:
                    PMTHROW(pmFatalErrorException());
            }
        #else
            switch(lTaskConf->affinityCriterion)
            {
                case MAXIMIZE_LOCAL_DATA:
                    if(lSelectiveSubtasks)
                    {
                        *((ulong*)lOutputMem) = (ulong)lUserSubtaskId;
                        lOutputMem = (void*)((ulong*)lOutputMem + 1);
                    }

                    *((ulong*)lOutputMem) = lSubscriptionManager.FindLocalInputDataSizeForSubtask(lStub, lUserSubtaskId);
                    break;

                case MINIMIZE_REMOTE_SOURCES:
                    if(lSelectiveSubtasks)
                    {
                        *((uint*)lOutputMem) = (uint)lUserSubtaskId;
                        lOutputMem = (void*)((uint*)lOutputMem + 1);
                    }

                    *((uint*)lOutputMem) = lSubscriptionManager.FindRemoteDataSourcesForSubtask(lStub, lUserSubtaskId);
                    break;
                
                case MINIMIZE_REMOTE_TRANSFER_EVENTS:
                    if(lSelectiveSubtasks)
                    {
                        *((ulong*)lOutputMem) = (ulong)lUserSubtaskId;
                        lOutputMem = (void*)((ulong*)lOutputMem + 1);
                    }

                    *((ulong*)lOutputMem) = lSubscriptionManager.FindRemoteTransferEventsForSubtask(lStub, lUserSubtaskId);
                    break;

                case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
                    if(lSelectiveSubtasks)
                    {
                        *((float*)lOutputMem) = (float)lUserSubtaskId;
                        lOutputMem = (void*)((float*)lOutputMem + 1);
                    }

                    *((float*)lOutputMem) = lSubscriptionManager.FindRemoteTransferEstimateForSubtask(lStub, lUserSubtaskId);
                    break;
                    
                case DERIVED_AFFINITY:
                    if(lSelectiveSubtasks)
                    {
                        *((float*)lOutputMem) = (float)lUserSubtaskId;
                        lOutputMem = (void*)((float*)lOutputMem + 1);
                    }

                    derivedAffinityData data;
                    //data.localBytes = lSubscriptionManager.FindLocalInputDataSizeForSubtask(lStub, lUserSubtaskId);
                    //data.remoteNodes = lSubscriptionManager.FindRemoteDataSourcesForSubtask(lStub, lUserSubtaskId);
                    data.remoteEvents = lSubscriptionManager.FindRemoteTransferEventsForSubtask(lStub, lUserSubtaskId);
                    data.estimatedTime = lSubscriptionManager.FindRemoteTransferEstimateForSubtask(lStub, lUserSubtaskId);
                    
                    *((derivedAffinityData*)lOutputMem) = data;
                    break;

                default:
                    PMTHROW(pmFatalErrorException());
            }
        #endif
            
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

    lUserTask->SetPreprocessorTask(lPreprocessorTask);

    const pmTaskCompletionCB* lUserTaskCompletionCB = lUserTask->GetCallbackUnit()->GetTaskCompletionCB();
    pmTaskCompletionCallback lOrgUserCallback = (lUserTaskCompletionCB ? lUserTaskCompletionCB->GetCallback() : NULL);
    lUserTask->SetTaskCompletionCallback(userTask_auxillaryTaskCompletionCallback);
    const_cast<pmTaskCompletionCB*>(lUserTask->GetCallbackUnit()->GetTaskCompletionCB())->SetUserData(reinterpret_cast<void*>(lOrgUserCallback));
    
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
    lUserTask->SetTaskCompletionCallback(reinterpret_cast<pmTaskCompletionCallback>(lUserTask->GetCallbackUnit()->GetTaskCompletionCB()->GetUserData()));

    // Destroy preprocessorTask
    const std::vector<pmTaskMemory>& lPreprocessorTaskMemVector = lPreprocessorTask->GetTaskMemVector();
    pmAddressSpace* lAddressSpace = lPreprocessorTaskMemVector[lPreprocessorTaskMemVector.size() - 1].addressSpace;

    EXCEPTION_ASSERT(pmReleaseMemory(lAddressSpace->GetUserMemHandle()) == pmSuccess);
    EXCEPTION_ASSERT(pmReleaseTask((pmTaskHandle)(lPreprocessorTask)) == pmSuccess);

    // Call the original user task completion callback
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

uint pmPreprocessorTask::GetPercentageSubtasksToBeEvaluatedPerHostForAffinityComputation()
{
    static const char* lVal = getenv("PMLIB_PERCENT_SUBTASKS_PER_HOST_FOR_AFFINITY_COMPUTATION");
    if(lVal)
    {
        uint lValue = (uint)atoi(lVal);

        if(lValue != 0 && lValue < 100)
            return lValue;
    }
    
    return 100;
}

void pmPreprocessorTask::DeduceAffinity(pm::pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion)
{
    DEBUG_EXCEPTION_ASSERT(pLocalTask->GetCallbackUnit() && pLocalTask->GetCallbackUnit()->GetDataDistributionCB() && !(static_cast<pmTask*>(pLocalTask))->HasReadOnlyLazyAddressSpace());

    LaunchPreprocessorTask(pLocalTask, AFFINITY_DEDUCER, pAffinityCriterion);
}

void pmPreprocessorTask::EvaluateDependency(pmLocalTask* pLocalTask)
{
    LaunchPreprocessorTask(pLocalTask, DEPENDENCY_EVALUATOR, MAX_AFFINITY_CRITERION);
}

void pmPreprocessorTask::DeduceAffinityAndEvaluateDependency(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion)
{
    DEBUG_EXCEPTION_ASSERT(pLocalTask->GetCallbackUnit() && pLocalTask->GetCallbackUnit()->GetDataDistributionCB() && !(static_cast<pmTask*>(pLocalTask))->HasReadOnlyLazyAddressSpace());

    LaunchPreprocessorTask(pLocalTask, AFFINITY_AND_DEPENDENCY_TASK, pAffinityCriterion);
}

void pmPreprocessorTask::LaunchPreprocessorTask(pm::pmLocalTask* pLocalTask, preprocessorTaskType pTaskType, pmAffinityCriterion pAffinityCriterion)
{
#ifdef ENABLE_TASK_PROFILING
    pLocalTask->GetTaskProfiler()->RecordProfileEvent(taskProfiler::PREPROCESSOR_TASK_EXECUTION, true);
#endif

    std::set<const pmMachine*> lMachinesSet;

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
    lMachinesSet.emplace(PM_LOCAL_MACHINE);
    
    const std::set<const pmMachine*>& lTargettedMachinesSet = pLocalTask->GetAssignedMachines();
#else
    lMachinesSet = pLocalTask->GetAssignedMachines();
#endif

    ulong lUserSubtasks = pLocalTask->GetSubtaskCount();

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
    preprocessorTaskConf lTaskConf = {pTaskType, lUserSubtasks, *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber(), pAffinityCriterion, (uint)lMachinesSet.size(), (uint)lTargettedMachinesSet.size()};
#else
    preprocessorTaskConf lTaskConf = {pTaskType, lUserSubtasks, *pLocalTask->GetOriginatingHost(), pLocalTask->GetSequenceNumber(), pAffinityCriterion, (uint)lMachinesSet.size()};
#endif
    
    EXCEPTION_ASSERT(lUserSubtasks);
    
    std::vector<pmTaskMem> lMemVector;
    
    ulong lPreprocessorSubtasks = 0;

    switch(pTaskType)
    {
        case preprocessorTask::AFFINITY_DEDUCER:
        {
            uint lPercentSubtasks = GetPercentageSubtasksToBeEvaluatedPerHostForAffinityComputation();
            uint lMemoryMultiple = 1;

            if(lPercentSubtasks != 100)
            {
                lUserSubtasks = std::max((ulong)1, (ulong)(((float)lPercentSubtasks / 100) * lUserSubtasks));
                lMemoryMultiple = ((lUserSubtasks < pLocalTask->GetSubtaskCount()) ? 2 : 1);    // For now, keeping subtask id in same format as affinity sample
            }
            
            lPreprocessorSubtasks = lMachinesSet.size() * lUserSubtasks;

            // Create an output address space for the preprocessor task (every machine stores local bytes for all subtasks) */
        #ifdef CENTRALIZED_AFFINITY_COMPUTATION
            size_t lOutputMemSize = lMemoryMultiple * lPreprocessorSubtasks * GetSampleSizeForAffinityCriterion(pAffinityCriterion) * lTargettedMachinesSet.size();
        #else
            size_t lOutputMemSize = lMemoryMultiple * lPreprocessorSubtasks * GetSampleSizeForAffinityCriterion(pAffinityCriterion);
        #endif

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
    
	lTaskDetails.policy = NODE_EQUAL_STATIC;
	lTaskDetails.multiAssignEnabled = false;
    lTaskDetails.overlapComputeCommunication = true;
    lTaskDetails.suppressTaskLogs = true;

	pmTaskHandle lTaskHandle = NULL;

    pmController::GetController()->SubmitTask_Public(lTaskDetails, &lTaskHandle, lMachinesSet);
}
    
size_t pmPreprocessorTask::GetSampleSizeForAffinityCriterion(pmAffinityCriterion pAffinityCriterion)
{
    switch(pAffinityCriterion)
    {
        case MAXIMIZE_LOCAL_DATA:
            return sizeof(ulong);
            break;
            
        case MINIMIZE_REMOTE_SOURCES:
            return sizeof(uint);
            break;
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
            return sizeof(ulong);
            break;
            
        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
            return sizeof(float);
            break;

        case DERIVED_AFFINITY:
            return sizeof(derivedAffinityData);
            break;

        default:
            PMTHROW(pmFatalErrorException());
    }
    
    return 0;
}
    
bool pmPreprocessorTask::IsAffinityTask(pmTask* pTask)
{
    const pmCallbackUnit* lCallbackUnit = pTask->GetCallbackUnit();
    
    if(lCallbackUnit)
    {
        const pmSubtaskCB* lSubtaskCB = lCallbackUnit->GetSubtaskCB();
        
        if(lSubtaskCB && lSubtaskCB->GetCpuCallback() == preprocessorTask_cpuCallback && ((preprocessorTaskConf*)pTask->GetTaskConfiguration())->taskType == AFFINITY_DEDUCER)
            return true;
    }
    
    return false;
}
    
};



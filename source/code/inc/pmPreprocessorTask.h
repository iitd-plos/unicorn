
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

#ifndef __PM_AFFINITY_DEDUCER__
#define __PM_AFFINITY_DEDUCER__

#include "pmBase.h"

namespace pm
{
    
class pmLocalTask;
    
namespace preprocessorTask
{

enum preprocessorTaskType
{
    AFFINITY_DEDUCER,
    DEPENDENCY_EVALUATOR,
    AFFINITY_AND_DEPENDENCY_TASK,
    MAX_TYPES
};

}

class pmPreprocessorTask : public pmBase
{
public:
    ~pmPreprocessorTask();

    static pmPreprocessorTask* GetPreprocessorTask();

    void DeduceAffinity(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion);
    void EvaluateDependency(pmLocalTask* pLocalTask);
    void DeduceAffinityAndEvaluateDependency(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion);

    static size_t GetSampleSizeForAffinityCriterion(pmAffinityCriterion pAffinityCriterion);
    static bool IsAffinityTask(pmTask* pTask);

private:
    pmPreprocessorTask();

    void LaunchPreprocessorTask(pmLocalTask* pLocalTask, preprocessorTask::preprocessorTaskType pTaskType, pmAffinityCriterion pAffinityCriterion);

    pmCallbackHandle mPreprocessorTaskCallbackHandle;
};

pmStatus preprocessorTask_dataDistributionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
pmStatus preprocessorTask_cpuCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
pmStatus preprocessorTask_taskCompletionCallback(pmTaskInfo pTaskInfo);
pmStatus userTask_auxillaryTaskCompletionCallback(pmTaskInfo pTaskInfo);
    
} // end namespace pm

#endif

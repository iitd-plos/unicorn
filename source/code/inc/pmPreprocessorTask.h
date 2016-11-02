
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
    static uint GetPercentageSubtasksToBeEvaluatedPerHostForAffinityComputation();

    pmCallbackHandle mPreprocessorTaskCallbackHandle;
};

pmStatus preprocessorTask_dataDistributionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
pmStatus preprocessorTask_cpuCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
pmStatus preprocessorTask_taskCompletionCallback(pmTaskInfo pTaskInfo);
pmStatus userTask_auxillaryTaskCompletionCallback(pmTaskInfo pTaskInfo);
    
} // end namespace pm

#endif

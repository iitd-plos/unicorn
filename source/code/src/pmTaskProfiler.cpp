
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

#include "pmTaskProfiler.h"
#include "pmStubManager.h"
#include "pmLogger.h"
#include "pmTask.h"

#include <string>
#include <sstream>

#ifdef ENABLE_TASK_PROFILING

namespace pm
{
    
using namespace taskProfiler;

static const char* profileName[] =
{
    (char*)"INPUT_MEMORY_TRANSFER",
    (char*)"OUTPUT_MEMORY_TRANSFER",
    (char*)"TOTAL_MEMORY_TRANSFER",
    (char*)"DATA_PARTITIONING",
    (char*)"SUBTASK_EXECUTION",
    (char*)"LOCAL_DATA_REDUCTION",
    (char*)"REMOTE_DATA_REDUCTION",
    (char*)"DATA_REDISTRIBUTION",
    (char*)"SHADOW_MEM_COMMIT",
    (char*)"SUBTASK_STEAL_WAIT",
    (char*)"SUBTASK_STEAL_SERVE",
    (char*)"STUB_WAIT_ON_NETWORK",
    (char*)"COPY_TO_PINNED_MEMORY",
    (char*)"COPY_FROM_PINNED_MEMORY",
    (char*)"CUDA_COMMAND_PREPARATION",
    (char*)"PREPROCESSOR_TASK_EXECUTION",
    (char*)"AFFINITY_SUBTASK_MAPPINGS",
    (char*)"AFFINITY_USE_OVERHEAD",
    (char*)"FLUSH_MEMORY_OWNERSHIPS",
    (char*)"NETWORK_DATA_COMPRESSION",
    (char*)"GPU_DATA_COMPRESSION",
    (char*)"UNIVERSAL"
};

pmTaskProfiler::pmTaskProfiler(pmTask* pTask)
    : mTask(pTask)
{
    for(int i = 0; i < MAX_PROFILE_TYPES; ++i)
    {
        mTimer[i].Start();
        mTimer[i].Pause();

        mRecursionCount[i] = 0;
        mAccumulatedTime[i] = 0;
        mActualTime[i] = 0;
    }
}

pmTaskProfiler::~pmTaskProfiler()
{
    std::stringstream lStream;
    lStream << std::endl;
    lStream << "Task Profiler [";
    lStream << "Host " << pmGetHostId() << "] ............" << std::endl;
    
    for(int i = 0; i < MAX_PROFILE_TYPES; ++i)
    {
        mTimer[i].Stop();

        lStream << profileName[i] << " => Accumulated Time: " << mAccumulatedTime[i] << "s; Actual Time = " << mActualTime[i] << "s; Overlapped Time = " << mAccumulatedTime[i] - mActualTime[i] << "s" << std::endl;
    }
    
    lStream << std::endl;
    
    if(!mTask->ShouldSuppressTaskLogs())
        pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
}

void pmTaskProfiler::AccountForElapsedTime(profileType pProfileType)
{
    double lElapsedTime = mTimer[pProfileType].GetElapsedTimeInSecs();
    
    mActualTime[pProfileType] += lElapsedTime;
    mAccumulatedTime[pProfileType] += mRecursionCount[pProfileType] * lElapsedTime;
}

void pmTaskProfiler::RecordProfileEvent(profileType pProfileType, bool pStart)
{
    if(pProfileType == UNIVERSAL || pProfileType == TOTAL_MEMORY_TRANSFER)
        PMTHROW(pmFatalErrorException());

    RecordProfileEventInternal(pProfileType, pStart);
    if(pProfileType == INPUT_MEMORY_TRANSFER || pProfileType == OUTPUT_MEMORY_TRANSFER)
        RecordProfileEventInternal(TOTAL_MEMORY_TRANSFER, pStart);
        
    RecordProfileEventInternal(UNIVERSAL, pStart);
}
    
void pmTaskProfiler::RecordProfileEventInternal(profileType pProfileType, bool pStart)
{
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = mResourceLock[pProfileType];
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

    if(pStart)
    {
        if(mRecursionCount[pProfileType] == 0)
        {
            mTimer[pProfileType].Resume();
        }
        else
        {
            mTimer[pProfileType].Pause();
            AccountForElapsedTime(pProfileType);
            mTimer[pProfileType].Reset();
        }
        
        ++mRecursionCount[pProfileType];
    }
    else
    {
        mTimer[pProfileType].Pause();
        AccountForElapsedTime(pProfileType);
        mTimer[pProfileType].Reset();
        
        --mRecursionCount[pProfileType];
        if(mRecursionCount[pProfileType] == 0)
            mTimer[pProfileType].Pause();
    }
}
    
} // end namespace pm

#endif


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

#include "pmTaskProfiler.h"
#include "pmStubManager.h"
#include "pmLogger.h"

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
    (char*)"PRE_SUBTASK_EXECUTION",
    (char*)"SUBTASK_EXECUTION",
    (char*)"DATA_REDUCTION",
    (char*)"DATA_REDISTRIBUTION",
    (char*)"SHADOW_MEM_COMMIT",
    (char*)"SUBTASK_STEAL_WAIT",
    (char*)"SUBTASK_STEAL_SERVE",
    (char*)"UNIVERSAL"
};

pmTaskProfiler::pmTaskProfiler()
{
    for(int i=0; i<MAX_PROFILE_TYPES; ++i)
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
    
    for(int i=0; i<MAX_PROFILE_TYPES; ++i)
    {
        mTimer[i].Stop();

        lStream << profileName[i] << " => Accumulated Time: " << mAccumulatedTime[i] << "s; Actual Time = " << mActualTime[i] << "s; Overlapped Time = " << mAccumulatedTime[i] - mActualTime[i] << "s" << std::endl;
    }
    
    lStream << std::endl;
    
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

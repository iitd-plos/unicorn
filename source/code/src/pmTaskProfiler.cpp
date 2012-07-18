
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
#include <string>
#include <sstream>

#ifdef ENABLE_TASK_PROFILING

namespace pm
{
    
static const char* profileName[] =
{
    (char*)"INPUT_MEMORY_TRANSFER",
    (char*)"OUTPUT_MEMORY_TRANSFER",
    (char*)"DATA_PARTITIONING",
    (char*)"SUBTASK_EXECUTION",
    (char*)"SUBTASK_REDUCTION",
    (char*)"DATA_REDISTRIBUTION",
    (char*)"UNIVERSAL",
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
    lStream << "Task Profiler (";
    lStream << "Host " << pmGetHostId() << ") ............" << std::endl;
    
    for(int i=0; i<MAX_PROFILE_TYPES; ++i)
    {
        if(mRecursionCount[i] != 0)
            PMTHROW(pmFatalErrorException());

        mTimer[i].Stop();

        lStream << profileName[i] << " => Accumulated Time: " << mAccumulatedTime[i] << "s; Actual Time = " << mActualTime[i] << "s; Overlapped Time = " << mAccumulatedTime[i] - mActualTime[i] << "s" << std::endl;
    }
    
    lStream << std::endl;
    
    std::cout << lStream.str();
}

void pmTaskProfiler::AccountForElapsedTime(profileType pProfileType)
{
    double lElapsedTime = mTimer[pProfileType].GetElapsedTimeInSecs();
    
    mActualTime[pProfileType] += lElapsedTime;
    mAccumulatedTime[pProfileType] += mRecursionCount[pProfileType] * lElapsedTime;
}

void pmTaskProfiler::RecordProfileEvent(profileType pProfileType, bool pStart)
{
    if(pProfileType == UNIVERSAL)
        PMTHROW(pmFatalErrorException());
    
    RecordProfileEventInternal(pProfileType, pStart);
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

#if 0
pmTaskProfiler::pmTaskProfiler()
    : mProfileCount(1 + pmStubManager::GetStubManager()->GetStubCount())
{
    for(int i=0; i<MAX_PROFILE_TYPES; ++i)
    {
        mTimer[i].reset(new TIMER_IMPLEMENTATION_CLASS[mProfileCount]);
        for(size_t j=0; j<mProfileCount; ++j)
        {
            (mTimer[i].get_ptr())[j].Start();
            (mTimer[i].get_ptr())[j].Pause();
        }
        
        mRecursionCount[i] = 0;
    }
}
    
pmTaskProfiler::~pmTaskProfiler()
{
    int i;
    double lAccumulatedTime[MAX_PROFILE_TYPES];
    
    for(i=0; i<MAX_PROFILE_TYPES; ++i)
    {
        lAccumulatedTime[i] = 0;
        for(size_t j=1; j<mProfileCount; ++j)
        {
            (mTimer[i].get_ptr())[j].Stop();
            lAccumulatedTime[i] += (mTimer[i].get_ptr())[j].GetElapsedTimeInSecs();
        }
    }
    
    std::stringstream lStream;
    lStream << "Task Profiler (";
    lStream << "Host " << pmGetHostId() << ") ............" << std::endl;

    for(i=0; i<MAX_PROFILE_TYPES; ++i)
    {
        (mTimer[i].get_ptr())[0].Stop();
        double lActualTime = (mTimer[i].get_ptr())[0].GetElapsedTimeInSecs();
        lStream << profileName[i] << " => Accumulated Time: " << lAccumulatedTime[i] << " s; Actual Time = " << lActualTime << " s; Overlapped Time = " << lAccumulatedTime[i] - lActualTime << " s" << std::endl;
    }
    
    lStream << std::endl;
    
    std::cout << lStream.str();
}

void pmTaskProfiler::RecordProfileEvent(profileType pProfileType, size_t pStubIndex, bool pStart)
{
    if(pStart)
    {
        (mTimer[pProfileType].get_ptr())[pStubIndex+1].Resume();
        if(mRecursionCount[pProfileType] == 0)
            (mTimer[pProfileType].get_ptr())[0].Resume();
        
        ++mRecursionCount[pProfileType];
    }
    else
    {
        (mTimer[pProfileType].get_ptr())[pStubIndex+1].Pause();

        --mRecursionCount[pProfileType];
        if(mRecursionCount[pProfileType] == 0)
            (mTimer[pProfileType].get_ptr())[0].Pause();
    }
}
#endif
    
} // end namespace pm

#endif

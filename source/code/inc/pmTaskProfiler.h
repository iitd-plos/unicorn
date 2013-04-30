
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

#ifndef __PM_TASK_PROFILER__
#define __PM_TASK_PROFILER__

#include "pmBase.h"
#include "pmTimer.h"
#include "pmResourceLock.h"

#ifdef ENABLE_TASK_PROFILING

namespace pm
{

/**
 * \brief The task profiler
 */
    
class pmTaskProfiler : public pmBase
{
public:
    pmTaskProfiler();
    ~pmTaskProfiler();

    void RecordProfileEvent(taskProfiler::profileType pProfileType, bool pStart);
    
private:
    void RecordProfileEventInternal(taskProfiler::profileType pProfileType, bool pStart);
    void AccountForElapsedTime(taskProfiler::profileType pProfileType);
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock[taskProfiler::MAX_PROFILE_TYPES];
    TIMER_IMPLEMENTATION_CLASS mTimer[taskProfiler::MAX_PROFILE_TYPES];
    uint mRecursionCount[taskProfiler::MAX_PROFILE_TYPES];
    double mAccumulatedTime[taskProfiler::MAX_PROFILE_TYPES];
    double mActualTime[taskProfiler::MAX_PROFILE_TYPES];
};
    
} // end namespace pm

#endif

#endif

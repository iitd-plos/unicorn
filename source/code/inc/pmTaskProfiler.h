
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

#ifndef __PM_TASK_PROFILER__
#define __PM_TASK_PROFILER__

#include "pmBase.h"
#include "pmTimer.h"
#include "pmResourceLock.h"

#ifdef ENABLE_TASK_PROFILING

namespace pm
{
    
class pmTask;

/**
 * \brief The task profiler
 */
    
class pmTaskProfiler : public pmBase
{
public:
    pmTaskProfiler(pmTask* pTask);
    ~pmTaskProfiler();

    void RecordProfileEvent(taskProfiler::profileType pProfileType, bool pStart);
    
private:
    void RecordProfileEventInternal(taskProfiler::profileType pProfileType, bool pStart);
    void AccountForElapsedTime(taskProfiler::profileType pProfileType);
    
    pmTask* mTask;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock[taskProfiler::MAX_PROFILE_TYPES];
    TIMER_IMPLEMENTATION_CLASS mTimer[taskProfiler::MAX_PROFILE_TYPES];
    uint mRecursionCount[taskProfiler::MAX_PROFILE_TYPES];
    double mAccumulatedTime[taskProfiler::MAX_PROFILE_TYPES];
    double mActualTime[taskProfiler::MAX_PROFILE_TYPES];
};
    
} // end namespace pm

#endif

#endif

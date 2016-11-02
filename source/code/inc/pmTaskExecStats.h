
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

#ifndef __PM_TASK_EXEC_STATS__
#define __PM_TASK_EXEC_STATS__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <map>

namespace pm
{

class pmTask;
class pmExecutionStub;
class pmProcessingElement;

class pmTaskExecStats : public pmBase
{
public:
    typedef struct stubStats
    {
    #ifdef SUPPORT_SPLIT_SUBTASKS
        double subtasksExecuted;
    #else
        ulong subtasksExecuted;
    #endif

        double executionTime;	  // in secs
        uint stealAttempts;
        uint successfulSteals;
        uint failedSteals;
        uint consecutiveFailedSteals;   // number of failed steals after the last successful one
        
        uint pipelineContinuationAcrossRanges;

        stubStats();
    } stubStats;

    pmTaskExecStats(pmTask* pTask);
    ~pmTaskExecStats();

#ifdef SUPPORT_SPLIT_SUBTASKS
    pmStatus RecordStubExecutionStats(pmExecutionStub* pStub, double pSubtasksExecuted, double pExecutionTimeInSecs);
#else
    pmStatus RecordStubExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs);
#endif

    double GetStubExecutionRate(pmExecutionStub* pStub);

    uint GetStealAttempts(pmExecutionStub* pStub);
    uint GetSuccessfulStealAttempts(pmExecutionStub* pStub);
    uint GetFailedStealAttempts(pmExecutionStub* pStub);
    uint GetFailedStealAttemptsSinceLastSuccessfulAttempt(pmExecutionStub* pStub);
    
    pmStatus RecordStealAttempt(pmExecutionStub* pStub);    
    void RecordSuccessfulStealAttempt(pmExecutionStub* pStub);
    void RecordFailedStealAttempt(pmExecutionStub* pStub);
    
    void RegisterPipelineContinuationAcrossRanges(pmExecutionStub* pStub);
    uint GetPipelineContinuationAcrossRanges(pmExecutionStub* pStub);
    
#ifdef ENABLE_MEM_PROFILING
    void RecordMemReceiveEvent(size_t pMemSize, bool pIsScattered);    // Scattered + General
    void RecordMemTransferEvent(size_t pMemSize, bool pIsScattered);    // Scattered + General
#endif

private:
    pmTask* mTask;

    #ifdef ENABLE_MEM_PROFILING
        size_t mMemReceived;    // Scattered + General
        size_t mMemTransferred;    // Scattered + General
        ulong mMemReceiveEvents;    // Scattered + General
        ulong mMemTransferEvents;    // Scattered + General
        size_t mScatteredMemReceived;
        size_t mScatteredMemTransferred;
        ulong mScatteredMemReceiveEvents;
        ulong mScatteredMemTransferEvents;
    #endif

    std::map<pmExecutionStub*, stubStats, stubSorter> mStats;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

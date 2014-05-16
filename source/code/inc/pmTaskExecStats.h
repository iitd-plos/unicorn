
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
        ulong subtasksExecuted;
        double executionTime;	  // in secs
        uint stealAttempts;
        uint successfulSteals;
        uint failedSteals;
        uint consecutiveFailedSteals;   // number of failed steals after the last successful one

        stubStats();
    } stubStats;

    pmTaskExecStats(pmTask* pTask);
    virtual ~pmTaskExecStats();

    pmStatus RecordStubExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs);
    double GetStubExecutionRate(pmExecutionStub* pStub);

    uint GetStealAttempts(pmExecutionStub* pStub);
    uint GetSuccessfulStealAttempts(pmExecutionStub* pStub);
    uint GetFailedStealAttempts(pmExecutionStub* pStub);
    uint GetFailedStealAttemptsSinceLastSuccessfulAttempt(pmExecutionStub* pStub);
    
    pmStatus RecordStealAttempt(pmExecutionStub* pStub);    
    void RecordSuccessfulStealAttempt(pmExecutionStub* pStub);
    void RecordFailedStealAttempt(pmExecutionStub* pStub);

private:
    pmTask* mTask;

    std::map<pmExecutionStub*, stubStats> mStats;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

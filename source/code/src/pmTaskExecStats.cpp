
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

#include "pmTaskExecStats.h"
#include "pmExecutionStub.h"
#include "pmHardware.h"
#include "pmLogger.h"
#include "pmTask.h"

#ifdef DUMP_TASK_EXEC_STATS
#include <sstream>
#endif

namespace pm
{

pmTaskExecStats::pmTaskExecStats(pmTask* pTask)
    : mTask(pTask)
#ifdef ENABLE_MEM_PROFILING
    , mMemReceived(0)
    , mMemTransferred(0)
    , mMemReceiveEvents(0)
    , mMemTransferEvents(0)
#endif
    , mResourceLock __LOCK_NAME__("pmTaskExecStats::mResourceLock")
{
}

pmTaskExecStats::~pmTaskExecStats()
{
#ifdef DUMP_TASK_EXEC_STATS
    if(mTask->ShouldSuppressTaskLogs())
        return;

    std::stringstream lStream;
    lStream << std::endl << "Task Exec Stats [Host " << pmGetHostId() << "] ............ " << std::endl;

#ifdef ENABLE_MEM_PROFILING
    lStream << "Memory Transfers - Received = " << mMemReceived << " bytes; Receive Events = " << mMemReceiveEvents << "; Sent = " << mMemTransferred << " bytes; Send Events = " << mMemTransferEvents << std::endl;
#endif
    
	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.begin(), lEndIter = mStats.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const pmProcessingElement* lDevice = lIter->first->GetProcessingElement();
        lStream << "Device " << lDevice->GetGlobalDeviceIndex() << " - Subtask execution rate = " << GetStubExecutionRate(lIter->first) << "; Steal attemps = " << GetStealAttempts(lIter->first) << "; Successful steals = " << GetSuccessfulStealAttempts(lIter->first) << "; Failed steals = " << GetFailedStealAttempts(lIter->first) << std::endl;
    }

    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif
}

pmStatus pmTaskExecStats::RecordStubExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mStats[pStub].subtasksExecuted += pSubtasksExecuted;
    mStats[pStub].executionTime += pExecutionTimeInSecs;

	return pmSuccess;
}

double pmTaskExecStats::GetStubExecutionRate(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
		return (double)0;

	return (double)(mStats[pStub].subtasksExecuted)/(double)(mStats[pStub].executionTime);
}

uint pmTaskExecStats::GetStealAttempts(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mStats[pStub].stealAttempts;
}

pmStatus pmTaskExecStats::RecordStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    ++(mStats[pStub].stealAttempts);

	return pmSuccess;
}

uint pmTaskExecStats::GetSuccessfulStealAttempts(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    return mStats[pStub].successfulSteals;
}

uint pmTaskExecStats::GetFailedStealAttempts(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    return mStats[pStub].failedSteals;
}
    
uint pmTaskExecStats::GetFailedStealAttemptsSinceLastSuccessfulAttempt(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    return mStats[pStub].consecutiveFailedSteals;
}

void pmTaskExecStats::RecordSuccessfulStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mStats[pStub].consecutiveFailedSteals = 0;
    ++(mStats[pStub].successfulSteals);
}
    
void pmTaskExecStats::RecordFailedStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    ++(mStats[pStub].consecutiveFailedSteals);
    ++(mStats[pStub].failedSteals);
}

#ifdef ENABLE_MEM_PROFILING
void pmTaskExecStats::RecordMemReceiveEvent(size_t pMemSize)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mMemReceived += pMemSize;
    ++(mMemReceiveEvents);
}
    
void pmTaskExecStats::RecordMemTransferEvent(size_t pMemSize)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mMemTransferred += pMemSize;
    ++(mMemTransferEvents);
}
#endif

/* struct stubStats */
pmTaskExecStats::stubStats::stubStats()
	: subtasksExecuted(0)
	, executionTime(0)
	, stealAttempts(0)
    , successfulSteals(0)
    , failedSteals(0)
    , consecutiveFailedSteals(0)
{
}

}

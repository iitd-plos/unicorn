
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
    , mScatteredMemReceived(0)
    , mScatteredMemTransferred(0)
    , mScatteredMemReceiveEvents(0)
    , mScatteredMemTransferEvents(0)
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
    lStream << "Total Memory Transfers - Received = " << mMemReceived << " bytes; Receive Events = " << mMemReceiveEvents << "; Sent = " << mMemTransferred << " bytes; Send Events = " << mMemTransferEvents << std::endl;
    lStream << "Scattered Memory Transfers - Received = " << mScatteredMemReceived << " bytes; Receive Events = " << mScatteredMemReceiveEvents << "; Sent = " << mScatteredMemTransferred << " bytes; Send Events = " << mScatteredMemTransferEvents << std::endl;
#endif
    
	auto lIter = mStats.begin(), lEndIter = mStats.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const pmProcessingElement* lDevice = lIter->first->GetProcessingElement();
        lStream << "Device " << lDevice->GetGlobalDeviceIndex() << " - Subtask execution rate = " << GetStubExecutionRate(lIter->first) << "; Steal attemps = " << GetStealAttempts(lIter->first) << "; Successful steals = " << GetSuccessfulStealAttempts(lIter->first) << "; Failed steals = " << GetFailedStealAttempts(lIter->first) << "; Pipelines across ranges = " << GetPipelineContinuationAcrossRanges(lIter->first) << std::endl;
    }

    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif
}

#ifdef SUPPORT_SPLIT_SUBTASKS
pmStatus pmTaskExecStats::RecordStubExecutionStats(pmExecutionStub* pStub, double pSubtasksExecuted, double pExecutionTimeInSecs)
#else
pmStatus pmTaskExecStats::RecordStubExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs)
#endif
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mStats[pStub].subtasksExecuted += pSubtasksExecuted;
    mStats[pStub].executionTime += pExecutionTimeInSecs;

	return pmSuccess;
}

double pmTaskExecStats::GetStubExecutionRate(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	auto lIter = mStats.find(pStub);
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
    
void pmTaskExecStats::RegisterPipelineContinuationAcrossRanges(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    ++(mStats[pStub].pipelineContinuationAcrossRanges);
}
    
uint pmTaskExecStats::GetPipelineContinuationAcrossRanges(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    return mStats[pStub].pipelineContinuationAcrossRanges;
}

#ifdef ENABLE_MEM_PROFILING
void pmTaskExecStats::RecordMemReceiveEvent(size_t pMemSize, bool pIsScattered)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mMemReceived += pMemSize;
    ++(mMemReceiveEvents);

    if(pIsScattered)
    {
        mScatteredMemReceived += pMemSize;
        ++(mScatteredMemReceiveEvents);
    }
}
    
void pmTaskExecStats::RecordMemTransferEvent(size_t pMemSize, bool pIsScattered)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mMemTransferred += pMemSize;
    ++(mMemTransferEvents);

    if(pIsScattered)
    {
        mScatteredMemTransferred += pMemSize;
        ++(mScatteredMemTransferEvents);
    }
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
    , pipelineContinuationAcrossRanges(0)
{
}

}

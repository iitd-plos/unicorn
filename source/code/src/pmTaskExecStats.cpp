
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

#include "pmTaskExecStats.h"
#include "pmExecutionStub.h"
#include "pmHardware.h"

#ifdef DUMP_TASK_EXEC_STATS
#include <sstream>
#endif

namespace pm
{

pmTaskExecStats::pmTaskExecStats()
{
}

pmTaskExecStats::~pmTaskExecStats()
{
#ifdef DUMP_TASK_EXEC_STATS
    std::stringstream lStream;
    lStream << "Task Exec Stats [Host " << pmGetHostId() << "] ............ " << std::endl;

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.begin(), lEndIter = mStats.end();
    for(; lIter != lEndIter; ++lIter)
    {
        pmProcessingElement* lDevice = lIter->first->GetProcessingElement();
        lStream << "Device " << lDevice->GetGlobalDeviceIndex() << " - Subtask execution rate = " << GetStubExecutionRate(lIter->first) << "; Steal attemps = " << GetStealAttempts(lIter->first) << "; Successful steals = " << GetSuccessfulStealAttempts(lIter->first) << "; Failed steals = " << GetFailedStealAttempts(lIter->first) << std::endl;
    }

    std::cout << lStream.str();
#endif
}

pmStatus pmTaskExecStats::RecordSubtaskExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter != mStats.end())
	{
        mStats[pStub].subtasksExecuted += pSubtasksExecuted;
        mStats[pStub].executionTime += pExecutionTimeInSecs;
	}
    else
    {
        mStats[pStub].subtasksExecuted = pSubtasksExecuted;
        mStats[pStub].executionTime = pExecutionTimeInSecs;
    }

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

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
		return 0;

	return mStats[pStub].stealAttempts;
}

pmStatus pmTaskExecStats::RecordStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
		mStats[pStub].stealAttempts = 1;
    else
        ++(mStats[pStub].stealAttempts);

	return pmSuccess;
}

pmStatus pmTaskExecStats::ClearStealAttempts(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
        mStats[pStub].stealAttempts = 0;

	return pmSuccess;
}

uint pmTaskExecStats::GetSuccessfulStealAttempts(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
    if(lIter == mStats.end())
        return 0;

    return mStats[pStub].successfulSteals;
}

uint pmTaskExecStats::GetFailedStealAttempts(pmExecutionStub* pStub)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
    if(lIter == mStats.end())
        return 0;

    return mStats[pStub].failedSteals;
}

void pmTaskExecStats::RecordSuccessfulStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
		mStats[pStub].successfulSteals = 1;
    else
        ++(mStats[pStub].successfulSteals);
}
    
void pmTaskExecStats::RecordFailedStealAttempt(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
		mStats[pStub].failedSteals = 1;
    else
        ++(mStats[pStub].failedSteals);
}


/* struct stubStats */
pmTaskExecStats::stubStats::stubStats()
	: subtasksExecuted(0)
	, executionTime(0)
	, stealAttempts(0)
    , successfulSteals(0)
    , failedSteals(0)
{
}

}

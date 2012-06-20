
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

namespace pm
{

pmTaskExecStats::pmTaskExecStats()
{
}

pmTaskExecStats::~pmTaskExecStats()
{
}

pmStatus pmTaskExecStats::RecordSubtaskExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	stubStats lStats;

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter != mStats.end())
	{
		stubStats lExistingStats = mStats[pStub];

		lStats.subtasksExecuted = lExistingStats.subtasksExecuted;
		lStats.executionTime = lExistingStats.executionTime;
	}

	lStats.subtasksExecuted += pSubtasksExecuted;
	lStats.executionTime += pExecutionTimeInSecs;

	mStats[pStub] = lStats;

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
	{
		stubStats lStats;
		mStats[pStub] = lStats;
	}

	++(mStats[pStub].stealAttempts);

	return pmSuccess;
}

pmStatus pmTaskExecStats::ClearStealAttempts(pmExecutionStub* pStub)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	std::map<pmExecutionStub*, stubStats>::iterator lIter = mStats.find(pStub);
	if(lIter == mStats.end())
	{
		stubStats lStats;
		mStats[pStub] = lStats;
	}

	mStats[pStub].stealAttempts = 0;

	return pmSuccess;
}


/* struct stubStats */
pmTaskExecStats::stubStats::stubStats()
{
	subtasksExecuted = 0;
	executionTime = (double)0;
	stealAttempts = 0;
}

}

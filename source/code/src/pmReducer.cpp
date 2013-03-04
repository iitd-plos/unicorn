
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

#include "pmReducer.h"
#include "pmCommunicator.h"
#include "pmStubManager.h"
#include "pmHardware.h"
#include "pmTask.h"
#include "pmExecutionStub.h"

#include <algorithm>

namespace pm
{

pmReducer::pmReducer(pmTask* pTask)
{
	mTask = pTask;
	mReduceState = false;
	mReductionsDone = 0;
	mExternalReductionsRequired = 0;
	mSendToMachine = NULL;

	PopulateExternalMachineList();
}

pmReducer::~pmReducer()
{
}

pmStatus pmReducer::PopulateExternalMachineList()
{
	std::set<pmMachine*> lMachines;
	if(dynamic_cast<pmLocalTask*>(mTask))
		pmProcessingElement::GetMachines(((pmLocalTask*)mTask)->GetAssignedDevices(), lMachines);
	else
		pmProcessingElement::GetMachines(((pmRemoteTask*)mTask)->GetAssignedDevices(), lMachines);
    
    if(lMachines.find(mTask->GetOriginatingHost()) == lMachines.end() || lMachines.find(PM_LOCAL_MACHINE) == lMachines.end())
		PMTHROW(pmFatalErrorException());

	std::vector<pmMachine*> lMachinesVector(lMachines.begin(), lMachines.end());
	std::vector<pmMachine*>::iterator lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), mTask->GetOriginatingHost());

	// Make originating host the first element of the vector
	std::rotate(lMachinesVector.begin(), lIter, lMachinesVector.end());

    lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), PM_LOCAL_MACHINE);
	uint lLocalMachineIndex = (uint)(lIter - lMachinesVector.begin());

	mExternalReductionsRequired = GetMaxPossibleExternalReductionReceives((uint)(lMachines.size()) - lLocalMachineIndex);

	if(lLocalMachineIndex != 0)
	{
		// Find index of first set bit while moving from LSB to MSB in mLocalMachineIndex
		// This is equivalent to how many rounds are required before a node sends. In each round odd numbered nodes send.
		// Then ranks of even numbered nodes are reduced by half.

		uint lPower = 1;
		uint lRoundCount = 0;
		uint lMachineIndex = lLocalMachineIndex;
		while((lMachineIndex & 0x1) != 0x1)
		{
			lPower <<= 1;
			++lRoundCount;
			lMachineIndex >>= 1;
		}

		uint lSendToMachineIndex = lLocalMachineIndex - lPower;
		mSendToMachine = lMachinesVector[lSendToMachineIndex];
	}

	return pmSuccess;
}

ulong pmReducer::GetMaxPossibleExternalReductionReceives(uint pFollowingMachineCountInclusive)
{
	// Find the highest set bit and the total number of set bits in pFollowingMachineCountInclusive
	int lBitCount = sizeof(pFollowingMachineCountInclusive) * 8;
	int lSetBits = 0;
	int lHighestSetBit = -1;

	int lMaxReceives = 0;

	if(pFollowingMachineCountInclusive > 1)	// If there is only one machine, then it does not receive anything
	{
		for(int i=0; pFollowingMachineCountInclusive && i<lBitCount; ++i)
		{
			if((pFollowingMachineCountInclusive & 0x1) == 0x1)
			{
				++lSetBits;
				if(i > lHighestSetBit)
					lHighestSetBit = i;
			}

			pFollowingMachineCountInclusive >>= 1; 
		}

		if(lSetBits == 1)
			lMaxReceives = lHighestSetBit;
		else
			lMaxReceives = lHighestSetBit + 1;
	}

	return lMaxReceives;
}

pmStatus pmReducer::AddSubtask(pmExecutionStub* pStub, ulong pSubtaskId)
{
#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDUCTION, true);
#endif

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mReduceState)
	{
		mReduceState = false;

		pStub->ReduceSubtasks(mTask, pSubtaskId, mLastSubtask.first, mLastSubtask.second);

		++mReductionsDone;
	}
	else
	{
		mLastSubtask = std::make_pair(pStub, pSubtaskId);
		mReduceState = true;

		CheckReductionFinishInternal();
	}

#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDUCTION, false);
#endif

	return pmSuccess;
}

pmStatus pmReducer::CheckReductionFinish()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    return CheckReductionFinishInternal();
}

/* This function must be called with mResourceLock acquired */
pmStatus pmReducer::CheckReductionFinishInternal()
{
	if(mReduceState && mTask->HasSubtaskExecutionFinished() && (mReductionsDone == (mExternalReductionsRequired + mTask->GetSubtasksExecuted() - 1)))
	{
		if(mSendToMachine)
		{
			if(mSendToMachine == PM_LOCAL_MACHINE || mLastSubtask.first == NULL)
				PMTHROW(pmFatalErrorException());

			// Send mLastSubtaskId to machine mSendToMachine for reduction
			return pmScheduler::GetScheduler()->ReduceRequestEvent(mLastSubtask.first, mTask, mSendToMachine, mLastSubtask.second);
		}
		else
		{
			(static_cast<pmLocalTask*>(mTask))->SaveFinalReducedOutput(mLastSubtask.first, mLastSubtask.second);
		}
	}

	return pmSuccess;
}

}


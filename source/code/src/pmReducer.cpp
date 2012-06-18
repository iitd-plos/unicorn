
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
	mCurrentStubId = 0;
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
	std::vector<pmMachine*> lMachines;

	if(dynamic_cast<pmLocalTask*>(mTask))
		pmProcessingElement::GetMachines(((pmLocalTask*)mTask)->GetAssignedDevices(), lMachines);
	else
		pmProcessingElement::GetMachines(((pmRemoteTask*)mTask)->GetAssignedDevices(), lMachines);

	std::vector<pmMachine*>::iterator lIter = std::find(lMachines.begin(), lMachines.end(), mTask->GetOriginatingHost());
	if(lIter == lMachines.end())
		PMTHROW(pmFatalErrorException());

	// Make originating host the first element of the vector
	std::rotate(lMachines.begin(), lIter, lMachines.end());

	lIter = std::find(lMachines.begin(), lMachines.end(), PM_LOCAL_MACHINE);
	if(lIter == lMachines.end())
		PMTHROW(pmFatalErrorException());

	uint lLocalMachineIndex = (uint)(lIter - lMachines.begin());

	mExternalReductionsRequired = GetMaxPossibleExternalReductionReceives(lMachines.size() - lLocalMachineIndex);

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
		mSendToMachine = lMachines[lSendToMachineIndex];
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

pmStatus pmReducer::AddSubtask(ulong pSubtaskId)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mReduceState)
	{
		mReduceState = false;

		pmStubManager::GetStubManager()->GetStub(mCurrentStubId)->ReduceSubtasks(mTask, mLastSubtaskId, pSubtaskId);

		++mReductionsDone;

		++mCurrentStubId;
		if(mCurrentStubId == pmStubManager::GetStubManager()->GetProcessingElementsCPU())
			mCurrentStubId = 0;
	}
	else
	{
		mLastSubtaskId = pSubtaskId;
		mReduceState = true;

		CheckReductionFinishInternal();
	}

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
			if(mSendToMachine == PM_LOCAL_MACHINE)
				PMTHROW(pmFatalErrorException());

			// Send mLastSubtaskId to machine mSendToMachine for reduction
			return pmScheduler::GetScheduler()->ReduceRequestEvent(mTask, mSendToMachine, mLastSubtaskId);
		}
		else
		{
			mTask->SaveFinalReducedOutput(mLastSubtaskId);
		}
	}

	return pmSuccess;
}

}


\
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
#include "pmMemoryManager.h"

#include <algorithm>

namespace pm
{

using namespace reducer;
    
pmReducer::pmReducer(pmTask* pTask)
	: mReductionsDone(0)
	, mExternalReductionsRequired(0)
	, mReduceState(false)
	, mSendToMachine(NULL)
    , mTask(pTask)
    , mAddedReductionFinishEvent(false)
    , mResourceLock __LOCK_NAME__("pmReducer::mResourceLock")
{
	PopulateExternalMachineList();
}

void pmReducer::PopulateExternalMachineList()
{
	std::set<const pmMachine*> lMachines;
	if(dynamic_cast<pmLocalTask*>(mTask))
		pmProcessingElement::GetMachines(((pmLocalTask*)mTask)->GetAssignedDevices(), lMachines);
	else
		pmProcessingElement::GetMachines(((pmRemoteTask*)mTask)->GetAssignedDevices(), lMachines);
    
    if(lMachines.find(mTask->GetOriginatingHost()) == lMachines.end())
        lMachines.insert(mTask->GetOriginatingHost());
    
    if(lMachines.find(PM_LOCAL_MACHINE) == lMachines.end())
		PMTHROW(pmFatalErrorException());

	std::vector<const pmMachine*> lMachinesVector(lMachines.begin(), lMachines.end());
	std::vector<const pmMachine*>::iterator lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), mTask->GetOriginatingHost());

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

void pmReducer::AddSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mReduceState)
	{
		mReduceState = false;

		pStub->ReduceSubtasks(mTask, pSubtaskId, pSplitInfo, mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());

		++mReductionsDone;
	}
	else
	{
		mLastSubtask.stub = pStub;
        mLastSubtask.subtaskId = pSubtaskId;
        mLastSubtask.splitInfo.reset(new pmSplitInfo(pSplitInfo->splitId, pSplitInfo->splitCount));

		mReduceState = true;

		CheckReductionFinishInternal();
	}
}

void pmReducer::CheckReductionFinish()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    CheckReductionFinishInternal();
}

/* This function must be called with mResourceLock acquired */
void pmReducer::CheckReductionFinishInternal()
{
	if(mReduceState && mTask->HasSubtaskExecutionFinished() && (mReductionsDone == (mExternalReductionsRequired + mTask->GetSubtasksExecuted() - 1)))
	{
		if(mSendToMachine)
		{
			if(mSendToMachine == PM_LOCAL_MACHINE || mLastSubtask.stub == NULL)
				PMTHROW(pmFatalErrorException());

			// Send mLastSubtaskId to machine mSendToMachine for reduction
			pmScheduler::GetScheduler()->ReduceRequestEvent(mLastSubtask.stub, mTask, mSendToMachine, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
		}
		else
		{
            AddReductionFinishEvent();
		}
	}
}
    
/* This function must be called with mResourceLock acquired */
void pmReducer::AddReductionFinishEvent()
{
    if(mAddedReductionFinishEvent)
        return;
    
    mAddedReductionFinishEvent = true;
    mLastSubtask.stub->ReductionFinishEvent(mTask);
}
    
void pmReducer::HandleReductionFinish()
{
    filtered_for_each(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace) {return (pAddressSpace->IsOutput() && mTask->IsReducible(pAddressSpace));},
    [&] (pmAddressSpace* pAddressSpace)
    {
        (static_cast<pmLocalTask*>(mTask))->SaveFinalReducedOutput(mLastSubtask.stub, pAddressSpace, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
    });
}

template<typename datatype>
void pmReducer::ReduceSubtasks(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    size_t lDataSize = sizeof(datatype);
    
    DEBUG_EXCEPTION_ASSERT(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize() % lDataSize == 0);

    pmSubscriptionManager& lSubscriptionManager = mTask->GetSubscriptionManager();
    
    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace) {return (pAddressSpace->IsOutput() && mTask->IsReducible(pAddressSpace));},
    [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
    {
        uint lAddressSpaceIndex = (uint)pAddressSpaceIndex;

        datatype* lShadowMem1 = (datatype*)lSubscriptionManager.GetSubtaskShadowMem(pStub1, pSubtaskId1, pSplitInfo1, lAddressSpaceIndex);
        datatype* lShadowMem2 = (datatype*)lSubscriptionManager.GetSubtaskShadowMem(pStub2, pSubtaskId2, pSplitInfo2, lAddressSpaceIndex);

    #ifdef SUPPORT_LAZY_MEMORY
        if(pAddressSpace->IsLazyWriteOnly())
        {
            size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();

            const std::map<size_t, size_t>& lMap = lSubscriptionManager.GetWriteOnlyLazyUnprotectedPageRanges(pStub2, pSubtaskId2, pSplitInfo2, lAddressSpaceIndex);
            
            std::map<size_t, size_t>::const_iterator lIter = lMap.begin(), lEndIter = lMap.end();
            
            switch(pReductionType)
            {
                case REDUCE_ADD:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] += lArray2[i];
                    }
                    
                    break;
                }
                    
                default:
                    PMTHROW(pmFatalErrorException());
            }
        }
        else
    #endif
        {
            pmSubscriptionInfo lUnifiedSubscriptionInfo1 = lSubscriptionManager.GetUnifiedReadWriteSubscription(pStub1, pSubtaskId1, pSplitInfo1, lAddressSpaceIndex);
            pmSubscriptionInfo lUnifiedSubscriptionInfo2 = lSubscriptionManager.GetUnifiedReadWriteSubscription(pStub2, pSubtaskId2, pSplitInfo2, lAddressSpaceIndex);
            
            if(lUnifiedSubscriptionInfo1.length != lUnifiedSubscriptionInfo2.length)
                PMTHROW(pmFatalErrorException());
            
            size_t lDataCount = lUnifiedSubscriptionInfo1.length / lDataSize;

            switch(pReductionType)
            {
                case REDUCE_ADD:
                {
                    for(size_t i = 0; i < lDataCount; ++i)
                        lShadowMem1[i] += lShadowMem2[i];
                    
                    break;
                }
                    
                default:
                    PMTHROW(pmFatalErrorException());
            }
        }
    });
}

void pmReducer::ReduceInts(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    ReduceSubtasks<int>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionType);
}
    
void pmReducer::ReduceUInts(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    ReduceSubtasks<uint>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionType);
}

void pmReducer::ReduceLongs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    ReduceSubtasks<long>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionType);
}

void pmReducer::ReduceULongs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    ReduceSubtasks<ulong>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionType);
}

void pmReducer::ReduceFloats(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType)
{
    ReduceSubtasks<float>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionType);
}

}


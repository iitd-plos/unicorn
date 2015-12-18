
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

#include "pmReducer.h"
#include "pmCommunicator.h"
#include "pmStubManager.h"
#include "pmHardware.h"
#include "pmTask.h"
#include "pmExecutionStub.h"
#include "pmMemoryManager.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmCallbackUnit.h"

#include <algorithm>

namespace pm
{

using namespace reducer;

struct reductionDataHolder
{
    reductionDataHolder(ulong pLength, pmExecutionStub* pStub)
    : mPtr(new char[pLength])
    , mStub(pStub)
    {}
    
    ~reductionDataHolder()
    {
        delete[] mPtr;
    }
    
    char* mPtr;
    pmExecutionStub* mStub;
};

void PostMpiReduceCommandCompletionCallback(const pmCommandPtr& pCommand)
{
    pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);

    communicator::subtaskMemoryReduceStruct* lReceiveStruct = (communicator::subtaskMemoryReduceStruct*)(lCommunicatorCommand->GetData());
    
    const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->originatingHost);
    pmTask* lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lReceiveStruct->sequenceNumber);
    
    pmScheduler::GetScheduler()->AddRegisterExternalReductionFinishEvent(lRequestingTask);
}

void PostExternalReduceCommandCompletionCallback(const pmCommandPtr& pCommand)
{
    pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);

    communicator::subtaskMemoryReduceStruct* lReceiveStruct = (communicator::subtaskMemoryReduceStruct*)(lCommunicatorCommand->GetData());
    
    const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->originatingHost);
    pmTask* lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lReceiveStruct->sequenceNumber);

    reductionDataHolder* lDataHolder = (reductionDataHolder*)(lCommunicatorCommand->GetExternalData());

    lDataHolder->mStub->ReduceExternalMemory(lRequestingTask, pCommand);
}

pmReducer::pmReducer(pmTask* pTask)
	: mReductionsDone(0)
	, mExternalReductionsRequired(0)
	, mReduceState(false)
	, mSendToMachine(NULL)
    , mTask(pTask)
    , mReductionTerminated(false)
    , mResourceLock __LOCK_NAME__("pmReducer::mResourceLock")
{
	PopulateExternalMachineList();
}
    
void pmReducer::PopulateExternalMachineList()
{
    std::set<const pmMachine*> lMachines = (dynamic_cast<pmLocalTask*>(mTask) ? ((pmLocalTask*)mTask)->GetAssignedMachines() : ((pmRemoteTask*)mTask)->GetAssignedMachines());

    if(lMachines.find(mTask->GetOriginatingHost()) == lMachines.end())
        lMachines.insert(mTask->GetOriginatingHost());
    
    EXCEPTION_ASSERT(lMachines.find(PM_LOCAL_MACHINE) != lMachines.end());

	std::vector<const pmMachine*> lMachinesVector(lMachines.begin(), lMachines.end());
	std::vector<const pmMachine*>::iterator lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), mTask->GetOriginatingHost());

	// Make originating host the first element of the vector
	std::rotate(lMachinesVector.begin(), lIter, lMachinesVector.end());

    lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), PM_LOCAL_MACHINE);
	uint lLocalMachineIndex = (uint)(lIter - lMachinesVector.begin());

	mExternalReductionsRequired = GetMaxPossibleExternalReductionReceives((uint)(lMachines.size()) - lLocalMachineIndex);

	if(lLocalMachineIndex != 0)
	{
		// Find index of first set bit while moving from LSB to MSB in lLocalMachineIndex
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

        mExternalReductionsRequired = std::min<ulong>(mExternalReductionsRequired, (ulong)lRoundCount);

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
		for(int i = 0; pFollowingMachineCountInclusive && i < lBitCount; ++i)
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

void pmReducer::PrepareForExternalReceive(communicator::subtaskMemoryReduceStruct& pStruct)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mSubtaskMemoryReduceStructVector.emplace_back(pStruct);
    
    CheckReductionFinishInternal();
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
        
        if(pSplitInfo)
            mLastSubtask.splitInfo.reset(new pmSplitInfo(pSplitInfo->splitId, pSplitInfo->splitCount));
        else
            mLastSubtask.splitInfo.reset(NULL);

		mReduceState = true;

		CheckReductionFinishInternal();
	}
}
    
void pmReducer::RegisterExternalReductionFinish()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    ++mReductionsDone;

    CheckReductionFinishInternal();
}

void pmReducer::CheckReductionFinish()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    CheckReductionFinishInternal();
}

void pmReducer::PerformDirectExternalReductions()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    EXCEPTION_ASSERT(mExternalReductionsRequired);

    if(mSubtaskMemoryReduceStructVector.empty())
        return;

    DEBUG_EXCEPTION_ASSERT(mTask->HasSubtaskExecutionFinished());

    communicator::subtaskMemoryReduceStruct& lSubtaskMemoryReduceStruct = mSubtaskMemoryReduceStructVector.back();
    
    pmSubscriptionManager& lSubscriptionManager = mTask->GetSubscriptionManager();
    bool lHasScratchBuffers = lSubscriptionManager.HasScratchBuffers(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
    
    const pmAddressSpace* lAddressSpace = 0;
    size_t lReducibleAddressSpaces = 0;
    size_t lAddressSpaceIndex = 0;
    for_each_with_index(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pIndex)
    {
        if(mTask->IsWritable(pAddressSpace) && mTask->IsReducible(pAddressSpace))
        {
            lAddressSpace = pAddressSpace;
            lAddressSpaceIndex = pIndex;
            ++lReducibleAddressSpaces;
        }
    });
    
    EXCEPTION_ASSERT(!lHasScratchBuffers && lReducibleAddressSpaces == 1);

    communicator::communicatorCommandTags lTag = (communicator::communicatorCommandTags)lSubtaskMemoryReduceStruct.mpiTag;
    const pmMachine* lSendingMachine = pmMachinePool::GetMachinePool()->GetMachine(lSubtaskMemoryReduceStruct.senderHost);

    finalize_ptr<communicator::subtaskMemoryReduceStruct> lData(new communicator::subtaskMemoryReduceStruct(lSubtaskMemoryReduceStruct));

#ifdef USE_MPI_REDUCE
    void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);
    ulong lOffset = 0;
    
    if(mTask->GetAddressSpaceSubscriptionVisibility(lAddressSpace, mLastSubtask.stub) == SUBSCRIPTION_NATURAL)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
        
        EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

        lOffset =  lBeginIter->first - lUnifiedSubscriptionInfo.offset;
    }
    else    // SUBSCRIPTION_COMPACT
    {
        const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
        
        auto lCompactWriteIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
        
        EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

        lOffset = *lCompactWriteIter;
    }

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::subtaskMemoryReduceStruct>::CreateSharedPtr(mTask->GetPriority(), communicator::RECEIVE, lTag, lSendingMachine, communicator::BYTE, lData, 1, PostMpiReduceCommandCompletionCallback, (static_cast<char*>(lShadowMem) + lOffset));
#else
    std::shared_ptr<reductionDataHolder> lDataPtr(new reductionDataHolder(lData->length, mLastSubtask.stub));
    
    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::subtaskMemoryReduceStruct>::CreateSharedPtr(mTask->GetPriority(), communicator::RECEIVE, lTag, lSendingMachine, communicator::BYTE, lData, 1, PostExternalReduceCommandCompletionCallback, lDataPtr->mPtr);
    
    lCommand->HoldExternalDataForLifetimeOfCommand(lDataPtr);
#endif

    mSubtaskMemoryReduceStructVector.resize(mSubtaskMemoryReduceStructVector.size() - 1);

    pmCommunicator::GetCommunicator()->ReceiveReduce(lCommand);
}

/* This function must be called with mResourceLock acquired */
void pmReducer::CheckReductionFinishInternal()
{
    ulong lSubtasksSplitted = 0;
    ulong lSplitCount = mTask->GetTotalSplitCount(lSubtasksSplitted);
    ulong lExtraReductionsForSplits = lSplitCount - lSubtasksSplitted;
    ulong lInternalReductionsCount = lExtraReductionsForSplits + mTask->GetSubtasksExecuted() - 1;

    if(mReduceState && mTask->HasSubtaskExecutionFinished())
    {
        if(mReductionsDone == (mExternalReductionsRequired + lInternalReductionsCount))
        {
            if(!mReductionTerminated)
            {
                mReductionTerminated = true;
        
                if(mSendToMachine)
                {
                    EXCEPTION_ASSERT(mSendToMachine != PM_LOCAL_MACHINE && mLastSubtask.stub != NULL);

                    // Send mLastSubtaskId to machine mSendToMachine for reduction
                    pmScheduler::GetScheduler()->ReduceRequestEvent(mLastSubtask.stub, mTask, mSendToMachine, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
                }
                else
                {
                    AddReductionFinishEvent();
                }
            }
        }
        else if(mReductionsDone >= lInternalReductionsCount)
        {
            mLastSubtask.stub->AddPlaceHolderEventForDirectExternalReduction(mTask);
        }
    }
}
    
/* This function must be called with mResourceLock acquired */
void pmReducer::AddReductionFinishEvent()
{
    mLastSubtask.stub->ReductionFinishEvent(mTask);
}

void pmReducer::HandleReductionFinish()
{
    filtered_for_each(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace) {return (mTask->IsWritable(pAddressSpace) && mTask->IsReducible(pAddressSpace));},
    [&] (pmAddressSpace* pAddressSpace)
    {
        (static_cast<pmLocalTask*>(mTask))->SaveFinalReducedOutput(mLastSubtask.stub, pAddressSpace, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
    });
 
    pmScheduler::GetScheduler()->AllReductionsDoneEvent(static_cast<pmLocalTask*>(mTask), mLastSubtask.stub, mLastSubtask.subtaskId, pmSplitData(mLastSubtask.splitInfo.get_ptr()));
}
    
void pmReducer::SignalSendToMachineAboutNoLocalReduction()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    SignalSendToMachineAboutNoLocalReductionInternal();
}

/* This function must be called with mResourceLock acquired */
void pmReducer::SignalSendToMachineAboutNoLocalReductionInternal()
{
    if(mSendToMachine && !mExternalReductionsRequired)
    {
        if(!mReductionTerminated)
        {
            mReductionTerminated = true;
            
            // No subtask has been executed on this machine
            EXCEPTION_ASSERT(mSendToMachine != PM_LOCAL_MACHINE && mLastSubtask.stub == NULL);
            
            if(mSendToMachine)
                pmScheduler::GetScheduler()->NoReductionRequiredEvent(mTask, mSendToMachine);
            else
                AddReductionFinishEvent();
        }
    }
}
    
void pmReducer::RegisterNoReductionReqdResponse()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    EXCEPTION_ASSERT(mExternalReductionsRequired);

    --mExternalReductionsRequired;

    if(mTask->GetSubtasksExecuted())
        CheckReductionFinishInternal();
    else
        SignalSendToMachineAboutNoLocalReductionInternal();
}

template<typename datatype>
struct getBitwiseOperatableType
{
    typedef datatype type;
};
    
template<>
struct getBitwiseOperatableType<float>
{
    typedef uint type;
};
    
template<>
struct getBitwiseOperatableType<double>
{
    typedef ulong type;
};

void pmReducer::ReduceExternalMemory(pmExecutionStub* pStub, const pmCommandPtr& pCommand)
{
    pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);
    reductionDataHolder* lDataHolder = (reductionDataHolder*)(lCommunicatorCommand->GetExternalData());

    communicator::subtaskMemoryReduceStruct* lReceiveStruct = (communicator::subtaskMemoryReduceStruct*)(lCommunicatorCommand->GetData());
    bool lCompressed = lReceiveStruct->compressed;
    
    EXCEPTION_ASSERT(pStub == lDataHolder->mStub);
    
#ifdef USE_MPI_REDUCE
    PMTHROW(pmFatalErrorException());
#endif

    pmSubscriptionManager& lSubscriptionManager = mTask->GetSubscriptionManager();
    bool lHasScratchBuffers = lSubscriptionManager.HasScratchBuffers(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr());
    
    const pmAddressSpace* lAddressSpace = 0;
    size_t lReducibleAddressSpaces = 0;
    size_t lAddressSpaceIndex = 0;
    for_each_with_index(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pIndex)
    {
        if(mTask->IsWritable(pAddressSpace) && mTask->IsReducible(pAddressSpace))
        {
            lAddressSpace = pAddressSpace;
            lAddressSpaceIndex = pIndex;
            ++lReducibleAddressSpaces;
        }
    });
    
    EXCEPTION_ASSERT(!lHasScratchBuffers && lReducibleAddressSpaces == 1);

    void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);
    ulong lOffset = 0, lLength = 0;
    
    if(mTask->GetAddressSpaceSubscriptionVisibility(lAddressSpace, mLastSubtask.stub) == SUBSCRIPTION_NATURAL)
    {
        pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
        
        EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

        lOffset =  lBeginIter->first - lUnifiedSubscriptionInfo.offset;
        lLength = (uint)lBeginIter->second.first;
    }
    else    // SUBSCRIPTION_COMPACT
    {
        const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex);

        subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
        lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(mLastSubtask.stub, mLastSubtask.subtaskId, mLastSubtask.splitInfo.get_ptr(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
        
        auto lCompactWriteIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
        
        EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

        lOffset = *lCompactWriteIter;
        lLength = (uint)lBeginIter->second.first;
    }

#if 0
    if(lCompressed)
        std::cout << "Received " << lReceiveStruct->length << " bytes instead of " << lLength << " (" << (double)(lReceiveStruct->length * 100) / lLength << "%)" << std::endl;
#endif
    
    void* lMem = (static_cast<char*>(lShadowMem) + lOffset);

    pmReductionOpType lOpType;
    pmReductionDataType lDataType;
    
    findReductionOpAndDataType(mTask->GetCallbackUnit()->GetDataReductionCB()->GetCallback(), lOpType, lDataType);
    
    if(!lCompressed)
    {
        switch(lDataType)
        {
            case REDUCE_INTS:
            {
                ReduceMemories<int>((int*)lMem, (int*)lDataHolder->mPtr, lLength / sizeof(int), lOpType);
                break;
            }
                
            case REDUCE_UNSIGNED_INTS:
            {
                ReduceMemories<uint>((uint*)lMem, (uint*)lDataHolder->mPtr, lLength / sizeof(uint), lOpType);
                break;
            }
                
            case REDUCE_LONGS:
            {
                ReduceMemories<long>((long*)lMem, (long*)lDataHolder->mPtr, lLength / sizeof(long), lOpType);
                break;
            }
                
            case REDUCE_UNSIGNED_LONGS:
            {
                ReduceMemories<ulong>((ulong*)lMem, (ulong*)lDataHolder->mPtr, lLength / sizeof(ulong), lOpType);
                break;
            }
                
            case REDUCE_FLOATS:
            {
                ReduceMemories<float>((float*)lMem, (float*)lDataHolder->mPtr, lLength / sizeof(float), lOpType);
                break;
            }
                
            case REDUCE_DOUBLES:
            {
                ReduceMemories<double>((double*)lMem, (double*)lDataHolder->mPtr, lLength / sizeof(double), lOpType);
                break;
            }
                
            default:
                PMTHROW(pmFatalErrorException());
        }
    }
    else
    {
        switch(lDataType)
        {
            case REDUCE_INTS:
            {
                ReduceMemoriesCompressed<int>((int*)lMem, (int*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(int), lOpType, 0);
                break;
            }
                
            case REDUCE_UNSIGNED_INTS:
            {
                ReduceMemoriesCompressed<uint>((uint*)lMem, (uint*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(uint), lOpType, 0);
                break;
            }
                
            case REDUCE_LONGS:
            {
                ReduceMemoriesCompressed<long>((long*)lMem, (long*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(long), lOpType, 0);
                break;
            }
                
            case REDUCE_UNSIGNED_LONGS:
            {
                ReduceMemoriesCompressed<ulong>((ulong*)lMem, (ulong*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(ulong), lOpType, 0);
                break;
            }
                
            case REDUCE_FLOATS:
            {
                ReduceMemoriesCompressed<float>((float*)lMem, (float*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(float), lOpType, 0);
                break;
            }
                
            case REDUCE_DOUBLES:
            {
                ReduceMemoriesCompressed<double>((double*)lMem, (double*)lDataHolder->mPtr, lReceiveStruct->length / sizeof(double), lOpType, 0);
                break;
            }
                
            default:
                PMTHROW(pmFatalErrorException());
        }
    }

    RegisterExternalReductionFinish();
}

template<typename datatype>
void pmReducer::ReduceMemories(datatype* pShadowMem1, datatype* pShadowMem2, size_t pDataCount, pmReductionOpType pReductionType)
{
    switch(pReductionType)
    {
        case REDUCE_ADD:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] += pShadowMem2[i];
            
            break;
        }
            
        case REDUCE_MIN:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = std::min(pShadowMem1[i], pShadowMem2[i]);
            
            break;
        }
        
        case REDUCE_MAX:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = std::max(pShadowMem1[i], pShadowMem2[i]);
            
            break;
        }
        
        case REDUCE_PRODUCT:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] *= pShadowMem2[i];
            
            break;
        }
        
        case REDUCE_LOGICAL_AND:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (pShadowMem1[i] && pShadowMem2[i]);
            
            break;
        }
        
        case REDUCE_BITWISE_AND:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[i]) & (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            
            break;
        }
        
        case REDUCE_LOGICAL_OR:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (pShadowMem1[i] || pShadowMem2[i]);
            
            break;
        }
        
        case REDUCE_BITWISE_OR:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[i]) | (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            
            break;
        }
        
        case REDUCE_LOGICAL_XOR:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (pShadowMem1[i] != pShadowMem2[i]);
            
            break;
        }

        case REDUCE_BITWISE_XOR:
        {
            for(size_t i = 0; i < pDataCount; ++i)
                pShadowMem1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[i]) ^ (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            
            break;
        }

        default:
            PMTHROW(pmFatalErrorException());
    }
}

template<typename datatype>
void pmReducer::ReduceMemoriesCompressed(datatype* pShadowMem1, datatype* pShadowMem2, size_t pDataCount, pmReductionOpType pReductionType, datatype pSentinel)
{
    size_t index = 0;

    switch(pReductionType)
    {
        case REDUCE_ADD:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] += pShadowMem2[i];
            }
            
            break;
        }
            
        case REDUCE_MIN:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = std::min(pShadowMem1[index], pShadowMem2[i]);
            }
            
            break;
        }
        
        case REDUCE_MAX:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = std::max(pShadowMem1[index], pShadowMem2[i]);
            }
            
            break;
        }
        
        case REDUCE_PRODUCT:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] *= pShadowMem2[i];
            }
            
            break;
        }
        
        case REDUCE_LOGICAL_AND:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (pShadowMem1[index] && pShadowMem2[i]);
            }
            
            break;
        }
        
        case REDUCE_BITWISE_AND:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[index]) & (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            }
            
            break;
        }
        
        case REDUCE_LOGICAL_OR:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (pShadowMem1[index] || pShadowMem2[i]);
            }
            
            break;
        }
        
        case REDUCE_BITWISE_OR:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[index]) | (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            }
            
            break;
        }
        
        case REDUCE_LOGICAL_XOR:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (pShadowMem1[index] != pShadowMem2[i]);
            }
            
            break;
        }

        case REDUCE_BITWISE_XOR:
        {
            for(size_t i = 0; i < pDataCount; ++i, ++index)
            {
                if(pShadowMem2[i] == pSentinel)
                {
                    ++i;
                    index = (size_t)pShadowMem2[i++];
                }

                pShadowMem1[index] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(pShadowMem1[index]) ^ (typename getBitwiseOperatableType<datatype>::type)(pShadowMem2[i]));
            }
            
            break;
        }

        default:
            PMTHROW(pmFatalErrorException());
    }
}
    
template<typename datatype>
void pmReducer::ReduceSubtasks(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionOpType pReductionType)
{
    size_t lDataSize = sizeof(datatype);
    
    DEBUG_EXCEPTION_ASSERT(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize() % lDataSize == 0);

    pmSubscriptionManager& lSubscriptionManager = mTask->GetSubscriptionManager();
    
    filtered_for_each_with_index(mTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace) {return (mTask->IsWritable(pAddressSpace) && mTask->IsReducible(pAddressSpace));},
    [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
    {
        uint lAddressSpaceIndex = (uint)pAddressSpaceIndex;

        datatype* lShadowMem1 = (datatype*)lSubscriptionManager.GetSubtaskShadowMem(pStub1, pSubtaskId1, pSplitInfo1, lAddressSpaceIndex);
        datatype* lShadowMem2 = (datatype*)lSubscriptionManager.GetSubtaskShadowMem(pStub2, pSubtaskId2, pSplitInfo2, lAddressSpaceIndex);

    #ifdef SUPPORT_LAZY_MEMORY
        if(mTask->IsLazyWriteOnly(pAddressSpace))
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
                    
                case REDUCE_MIN:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = std::min(lArray1[i], lArray2[i]);
                    }
                    
                    break;
                }

                case REDUCE_MAX:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = std::max(lArray1[i], lArray2[i]);
                    }
                    
                    break;
                }

                case REDUCE_PRODUCT:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] *= lArray2[i];
                    }
                    
                    break;
                }

                case REDUCE_LOGICAL_AND:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (lArray1[i] && lArray2[i]);
                    }
                    
                    break;
                }

                case REDUCE_BITWISE_AND:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(lArray1[i]) & (typename getBitwiseOperatableType<datatype>::type)(lArray2[i]));
                    }
                    
                    break;
                }

                case REDUCE_LOGICAL_OR:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (lArray1[i] || lArray2[i]);
                    }
                    
                    break;
                }

                case REDUCE_BOTWISE_OR:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(lArray1[i]) | (typename getBitwiseOperatableType<datatype>::type)(lArray2[i]));
                    }
                    
                    break;
                }

                case REDUCE_LOGICAL_XOR:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (lArray1[i] != lArray2[i]);
                    }
                    
                    break;
                }

                case REDUCE_BITWISE_XOR:
                {
                    for(; lIter != lEndIter; ++lIter)
                    {
                        size_t lStartPage = lIter->first;
                        size_t lPageCount = lIter->second;
                        
                        size_t lDataCount = (lPageCount * lPageSize) / lDataSize;
                        datatype* lArray1 = lShadowMem1 + ((lStartPage * lPageSize) / lDataSize);
                        datatype* lArray2 = lShadowMem2 + ((lStartPage * lPageSize) / lDataSize);
                        
                        for(size_t i = 0; i < lDataCount; ++i)
                            lArray1[i] = (datatype)((typename getBitwiseOperatableType<datatype>::type)(lArray1[i]) ^ (typename getBitwiseOperatableType<datatype>::type)(lArray2[i]));
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
            
            EXCEPTION_ASSERT(lUnifiedSubscriptionInfo1.length == lUnifiedSubscriptionInfo2.length);
            
            size_t lDataCount = lUnifiedSubscriptionInfo1.length / lDataSize;

            ReduceMemories(lShadowMem1, lShadowMem2, lDataCount, pReductionType);
        }
    });
}

void pmReducer::ReduceSubtasks(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionOpType pReductionOperation, pmReductionDataType pReductionDataType)
{
    switch(pReductionDataType)
    {
        case REDUCE_INTS:
            return ReduceSubtasks<int>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        case REDUCE_UNSIGNED_INTS:
            return ReduceSubtasks<uint>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        case REDUCE_LONGS:
            return ReduceSubtasks<long>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        case REDUCE_UNSIGNED_LONGS:
            return ReduceSubtasks<ulong>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        case REDUCE_FLOATS:
            return ReduceSubtasks<float>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        case REDUCE_DOUBLES:
            return ReduceSubtasks<double>(pStub1, pSubtaskId1, pSplitInfo1, pStub2, pSubtaskId2, pSplitInfo2, pReductionOperation);
            
        default:
            PMTHROW(pmFatalErrorException());
    }
}

}


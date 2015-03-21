
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

#include "pmAddressSpace.h"
#include "pmMemoryManager.h"
#include "pmHardware.h"
#include "pmScheduler.h"
#include "pmTask.h"
#include "pmCallbackUnit.h"
#include "pmHeavyOperations.h"
#include "pmDevicePool.h"
#include "pmStubManager.h"
#include "pmUtility.h"
#include "pmHardware.h"

#ifdef ENABLE_MEM_PROFILING
#include "pmLogger.h"
#endif

#include <string.h>
#include <sstream>
#include <cmath>
#include <iterator>

namespace pm
{

STATIC_ACCESSOR_INIT(ulong, pmAddressSpace, GetGenerationId, 0)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmAddressSpace::mGenerationLock"), pmAddressSpace, GetGenerationLock)

STATIC_ACCESSOR(pmAddressSpace::addressSpaceMapType, pmAddressSpace, GetAddressSpaceMap)
STATIC_ACCESSOR(pmAddressSpace::augmentaryAddressSpaceMapType, pmAddressSpace, GetAugmentaryAddressSpaceMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmAddressSpace::mResourceLock"), pmAddressSpace, GetResourceLock)


void FetchCallback(const pmCommandPtr& pCommand)
{
    pmAddressSpace* lAddressSpace = const_cast<pmAddressSpace*>(static_cast<const pmAddressSpace*>(pCommand->GetUserIdentifier()));
    
    lAddressSpace->FetchCompletionCallback(pCommand);
}


/* class pmAddressSpace */
pmAddressSpace::pmAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner)
    : mOwner(pOwner?pOwner:PM_LOCAL_MACHINE)
    , mGenerationNumberOnOwner(pGenerationNumberOnOwner)
    , mUserMemHandle(NULL)
    , mRequestedLength(pLength)
    , mAllocatedLength(pLength)
    , mVMPageCount(0)
    , mLazy(false)
    , mMem(NULL)
    , mReadOnlyLazyMapping(NULL)
    , mWaitingTasksLock __LOCK_NAME__("pmAddressSpace::mWaitingTasksLock")
    , mOwnershipLock __LOCK_NAME__("pmAddressSpace::mOwnershipLock")
    , mWaitingForOwnershipChange(false)
    , mOwnershipTransferLock __LOCK_NAME__("pmAddressSpace::mOwnershipTransferLock")
    , mLockingTask(NULL)
    , mTaskLock __LOCK_NAME__("pmAddressSpace::mTaskLock")
    , mUserDelete(false)
    , mDeleteLock __LOCK_NAME__("pmAddressSpace::mDeleteLock")
#ifdef ENABLE_MEM_PROFILING
    , mMemReceived(0)
    , mMemTransferred(0)
    , mMemReceiveEvents(0)
    , mMemTransferEvents(0)
    , mMemProfileLock __LOCK_NAME__("pmAddressSpace::mMemProfileLock")
#endif
{
    EXCEPTION_ASSERT(mGenerationNumberOnOwner != 0);

	mMem = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->AllocateMemory(this, mAllocatedLength, mVMPageCount);
    Init(mOwner);

    // Auto lock/unlock scope
	{
		FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::pair<const pmMachine*, ulong> lPair(mOwner, mGenerationNumberOnOwner);

        addressSpaceMapType& lAddressSpaceMap = GetAddressSpaceMap();
        EXCEPTION_ASSERT(lAddressSpaceMap.find(lPair) == lAddressSpaceMap.end());

		lAddressSpaceMap[lPair] = this;
	}
}

pmAddressSpace::~pmAddressSpace()
{
#ifdef ENABLE_MEM_PROFILING
    std::stringstream lStream;

    lStream << "Address Space [" << (uint)(*mOwner) << ", " << mGenerationNumberOnOwner << "] memory transfer statistics on [Host " << pmGetHostId() << "] ..." << std::endl;
    lStream << mMemReceived << " bytes memory received in " << mMemReceiveEvents << " events" << std::endl;
    lStream << mMemTransferred << " bytes memory transferred in " << mMemTransferEvents << " events" << std::endl;

    pmLogger::GetLogger()->LogDeferred(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif    

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        DEBUG_EXCEPTION_ASSERT(!mLockingTask);

    #ifdef SUPPORT_LAZY_MEMORY
        if(mReadOnlyLazyMapping)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeleteReadOnlyMemoryMapping(mReadOnlyLazyMapping, mAllocatedLength);

            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
            lAugmentaryAddressSpaceMap.erase(mReadOnlyLazyMapping);

            mReadOnlyLazyMapping = NULL;
        }
    #endif
    }
    
    DisposeMemory();
}
    
std::vector<uint> pmAddressSpace::GetMachinesForDistribution(bool pRandomize)
{
    std::set<const pmMachine*> lMachinesSet;
    std::map<uint, const pmMachine*> lMachinesMap;
    std::vector<uint> lMachinesVector;

    if(dynamic_cast<pmLocalTask*>(mLockingTask))
        pmProcessingElement::GetMachines(((pmLocalTask*)mLockingTask)->GetAssignedDevices(), lMachinesSet);
    else
        pmProcessingElement::GetMachines(((pmRemoteTask*)mLockingTask)->GetAssignedDevices(), lMachinesSet);
    
    lMachinesSet.emplace(mOwner);
    
    for_each(lMachinesSet, [&lMachinesMap] (const pmMachine* pMachine)
    {
        lMachinesMap.emplace((uint)(*pMachine), pMachine);
    });

    lMachinesVector.reserve(lMachinesMap.size());

    for_each(lMachinesMap, [&lMachinesVector] (typename decltype(lMachinesMap)::value_type& pPair)
    {
        lMachinesVector.emplace_back(pPair.first);
    });
    
    if(pRandomize)
    {
        // Using same seed on all machines, so that they produce same randomization.
        // This may not be portable.
        std::srand((uint)mGenerationNumberOnOwner);
        std::random_shuffle(lMachinesVector.begin(), lMachinesVector.end());

        std::cout << "Randomized list on machine " << (uint)(*PM_LOCAL_MACHINE);
        std::copy(lMachinesVector.begin(), lMachinesVector.end(), std::ostream_iterator<uint>(std::cout, " "));
        std::cout << std::endl;
    }
    
    return lMachinesVector;
}
    
pmAddressSpace* pmAddressSpace::CreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner /* = GetNextGenerationNumber() */)
{
    return new pmAddressSpace(pLength, pOwner, pGenerationNumberOnOwner);
}

pmAddressSpace* pmAddressSpace::CheckAndCreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner)
{
    pmAddressSpace* lAddressSpace = FindAddressSpace(pOwner, pGenerationNumberOnOwner);
    if(!lAddressSpace)
        lAddressSpace = CreateAddressSpace(pLength, pOwner, pGenerationNumberOnOwner);

    return lAddressSpace;
}
    
const char* pmAddressSpace::GetName()
{
    if(mName.empty())
    {
        std::stringstream lStream;
        lStream << "/pm_" << ::getpid() << "_" << (uint)(*(GetMemOwnerHost())) << "_" << GetGenerationNumber();

        mName = lStream.str();
    }
    
    return mName.c_str();
}

const pmMachine* pmAddressSpace::GetMemOwnerHost() const
{
    return mOwner;
}
    
ulong pmAddressSpace::GetGenerationNumber() const
{
    return mGenerationNumberOnOwner;
}
    
ulong pmAddressSpace::GetNextGenerationNumber()
{
    FINALIZE_RESOURCE(dGenerationLock, GetGenerationLock().Lock(), GetGenerationLock().Unlock());
    return (++GetGenerationId());   // Generation number 0 is reserved
}

void pmAddressSpace::Update(size_t pOffset, size_t pLength, void* pSrcAddr)
{
	void* lDestAddr = (void*)((char*)GetMem() + pOffset);
	memcpy(lDestAddr, pSrcAddr, pLength);
}

void pmAddressSpace::UserDelete()
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDeleteLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDeleteLock, Lock(), Unlock());
        mUserDelete = true;
    }
    
    pmTask* lTask = NULL;    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());
        lTask = mLockingTask;
    }

    if(!lTask)
        delete this;
}
    
void pmAddressSpace::DisposeMemory()
{
    if(mMem)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            addressSpaceMapType& lAddressSpaceMap = GetAddressSpaceMap();
            lAddressSpaceMap.erase(std::make_pair(mOwner, mGenerationNumberOnOwner));
        }
    
        pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->CancelMemoryTransferEvents(this);

        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(this);
        mMem = NULL;
    }
}

void pmAddressSpace::SetUserMemHandle(pmUserMemHandle* pUserMemHandle)
{
    mUserMemHandle = pUserMemHandle;
}

pmUserMemHandle* pmAddressSpace::GetUserMemHandle()
{
    return mUserMemHandle;
}

void pmAddressSpace::Init(const pmMachine* pOwner)
{
	mOwnershipMap.emplace(std::piecewise_construct, std::forward_as_tuple(0), std::forward_as_tuple(std::piecewise_construct, std::forward_as_tuple(mRequestedLength), std::forward_as_tuple(pOwner, 0, communicator::memoryIdentifierStruct(*(mOwner), mGenerationNumberOnOwner))));
}

void* pmAddressSpace::GetMem() const
{
	return mMem;
}

size_t pmAddressSpace::GetAllocatedLength() const
{
	return mAllocatedLength;
}

size_t pmAddressSpace::GetLength() const
{
	return mRequestedLength;
}

pmAddressSpace* pmAddressSpace::FindAddressSpace(const pmMachine* pOwner, ulong pGenerationNumber)
{
	FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

    addressSpaceMapType& lAddressSpaceMap = GetAddressSpaceMap();
	std::map<std::pair<const pmMachine*, ulong>, pmAddressSpace*>::iterator lIter = lAddressSpaceMap.find(std::make_pair(pOwner, pGenerationNumber));
	if(lIter != lAddressSpaceMap.end())
		return lIter->second;

	return NULL;
}
    
pmAddressSpace* pmAddressSpace::FindAddressSpaceContainingLazyAddress(void* pPtr)
{
	FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

    typedef std::map<void*, pmAddressSpace*> mapType;
    mapType::iterator lStartIter;
    mapType::iterator* lStartIterAddr = &lStartIter;
    
    char* lAddress = static_cast<char*>(pPtr);
    augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
    FIND_FLOOR_ELEM(mapType, lAugmentaryAddressSpaceMap, lAddress, lStartIterAddr);
    
    if(lStartIterAddr)
    {
        char* lMemAddress = static_cast<char*>(lStartIter->first);
        pmAddressSpace* lAddressSpace = lStartIter->second;

        size_t lLength = lAddressSpace->GetLength();
        
        if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
            return lAddressSpace;
    }
    
    return NULL;
}
    
void pmAddressSpace::SetWaitingForOwnershipChange()
{
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    EXCEPTION_ASSERT(!mWaitingForOwnershipChange);
    
    mWaitingForOwnershipChange = true;
}

bool pmAddressSpace::IsWaitingForOwnershipChange()
{
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    return mWaitingForOwnershipChange;
}

void pmAddressSpace::ChangeOwnership(std::shared_ptr<std::vector<communicator::ownershipChangeStruct>>& pOwnershipData)
{
    EXCEPTION_ASSERT(!GetLockingTask());

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

        EXCEPTION_ASSERT(mWaitingForOwnershipChange);

        for_each(*pOwnershipData.get(), [this] (communicator::ownershipChangeStruct& pStruct)
        {
            TransferOwnershipImmediate(pStruct.offset, pStruct.length, pmMachinePool::GetMachinePool()->GetMachine(pStruct.newOwnerHost));
        });

        mWaitingForOwnershipChange = false;
    }

    ScanLockQueue();
}
    
void pmAddressSpace::ScanLockQueue()
{
    FINALIZE_RESOURCE_PTR(dWaitingTasksLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mWaitingTasksLock, Lock(), Unlock());

    if(mTasksWaitingForLock.empty() || IsWaitingForOwnershipChange() || GetLockingTask())
        return;

    auto lValue = mTasksWaitingForLock.front();  // For now, only one task can acquire lock

    Lock(lValue.first.first, lValue.first.second);
    lValue.second->MarkExecutionEnd(pmSuccess, lValue.second);
    
    mTasksWaitingForLock.pop_front();
}

void pmAddressSpace::EnqueueForLock(pm::pmTask* pTask, pmMemType pMemType, pmCommandPtr& pCountDownCommand)
{
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    if(mWaitingForOwnershipChange || GetLockingTask())
    {
        FINALIZE_RESOURCE_PTR(dWaitingTasksLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mWaitingTasksLock, Lock(), Unlock());

        mTasksWaitingForLock.emplace_back(std::make_pair(pTask, pMemType), pCountDownCommand);
    }
    else
    {
        Lock(pTask, pMemType);
        pCountDownCommand->MarkExecutionEnd(pmSuccess, pCountDownCommand);
    }
}
    
void pmAddressSpace::Lock(pmTask* pTask, pmMemType pMemType)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        EXCEPTION_ASSERT(!mLockingTask && pMemType != MAX_MEM_TYPE);

        mLockingTask = pTask;
        
    #ifdef SUPPORT_LAZY_MEMORY
        if((pmUtility::IsWritable(pMemType) || !pmUtility::IsLazy(pMemType)) && mReadOnlyLazyMapping)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeleteReadOnlyMemoryMapping(mReadOnlyLazyMapping, mAllocatedLength);
 
            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
            lAugmentaryAddressSpaceMap.erase(mReadOnlyLazyMapping);

            mReadOnlyLazyMapping = NULL;
        }
    
        if(pmUtility::IsReadOnly(pMemType) && pmUtility::IsLazy(pMemType) && !mReadOnlyLazyMapping)
        {
            mReadOnlyLazyMapping = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateReadOnlyMemoryMapping(this);

            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
            lAugmentaryAddressSpaceMap[mReadOnlyLazyMapping] = this;
        }
    #endif
    }

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

        // FlushOwnerships is called for writable address spaces which causes their mOriginalOwnershipMap to be cleared
        bool lAddressSpaceReadOnlyLastTime = !mOriginalOwnershipMap.empty();

        // Nothing has to be done if address space was read only last time and is read only even now. In this case, the
        // already kept mOriginalOwnership map, must be kept as it is (for future restorations when address space becomes writable)
        if(!(lAddressSpaceReadOnlyLastTime && pmUtility::IsReadOnly(pMemType)))
        {
            // If the address space was read only last time but writable now, then its original ownership map must be restored
            if(lAddressSpaceReadOnlyLastTime && pmUtility::IsWritable(pMemType))
                mOwnershipMap = mOriginalOwnershipMap;
            else
                mOriginalOwnershipMap = mOwnershipMap;
        }
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(pmUtility::IsReadOnly(pMemType) && pmUtility::IsLazy(pMemType))
    {
        if(IsRegionLocallyOwned(0, GetLength()))
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(mReadOnlyLazyMapping, mAllocatedLength, true, true);
    }
#endif

#ifdef SUPPORT_CUDA
    if(pmUtility::IsWritable(pMemType))
        pmStubManager::GetStubManager()->PurgeAddressSpaceEntriesFromGpuCaches(this);
#endif
}
    
void pmAddressSpace::Unlock(pmTask* pTask)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        
        EXCEPTION_ASSERT(mOwnershipTransferVector.empty());
    }
#endif
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        EXCEPTION_ASSERT(mLockingTask == pTask);
        
        mLockingTask = NULL;
    }

    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CancelUnreferencedRequests(this);
    
    bool lUserDelete = false;
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDeleteLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDeleteLock, Lock(), Unlock());
        lUserDelete = mUserDelete;
    }
    
    if(lUserDelete)
    {
        delete this;
    }
    else
    {
        bool lOwnershipTransferRequired = (pTask->IsWritable(this) && (mOwner != PM_LOCAL_MACHINE) && (!pTask->GetCallbackUnit()->GetDataReductionCB()) && (!pTask->GetCallbackUnit()->GetDataRedistributionCB()));

        if(lOwnershipTransferRequired)
            SetWaitingForOwnershipChange();

        ScanLockQueue();
    }
}
    
pmTask* pmAddressSpace::GetLockingTask()
{
	FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

    return mLockingTask;
}

void pmAddressSpace::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = mOwnershipLock;
    pmMemOwnership& lMap = mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    SetRangeOwnerInternal(pRangeOwner, pOffset, pLength, lMap);
}

// This method must be called with mOwnershipLock acquired
void pmAddressSpace::SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength, pmMemOwnership& pMap)
{
    pmMemOwnership& lMap = pMap;
    
#ifdef _DEBUG
#if 0
    PrintOwnerships();
    if(pRangeOwner.memIdentifier.memOwnerHost == *(mOwner) && pRangeOwner.memIdentifier.generationNumber == mGenerationNumberOnOwner)
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner, Owner Offset): (" << pOffset << ", " << pLength << ", " << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
    else
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner address space (Host, Generation Number), Owner, Owner Offset): (" << pOffset << ", " << pLength << ", (" << pRangeOwner.memIdentifier.memOwnerHost << ", " << pRangeOwner.memIdentifier.generationNumber << ")," << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
#endif
#endif
    
	// Remove present ownership
	size_t lLastAddr = pOffset + pLength - 1;
	size_t lOwnerLastAddr = pRangeOwner.hostOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, lMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, lMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		PMTHROW(pmFatalErrorException());
    
	assert(lStartIter->first <= pOffset);
	assert(lEndIter->first <= lLastAddr);
	assert(lStartIter->first + lStartIter->second.first > pOffset);
	assert(lEndIter->first + lEndIter->second.first > lLastAddr);

	size_t lStartOffset = lStartIter->first;
	//size_t lStartLength = lStartIter->second.first;
	vmRangeOwner lStartOwner = lStartIter->second.second;

	size_t lEndOffset = lEndIter->first;
	size_t lEndLength = lEndIter->second.first;
	vmRangeOwner lEndOwner = lEndIter->second.second;

	lMap.erase(lStartIter, lEndIter);
    lMap.erase(lEndIter);

	if(lStartOffset < pOffset)
	{
		if(lStartOwner.host == pRangeOwner.host && lStartOwner.memIdentifier == pRangeOwner.memIdentifier && lStartOwner.hostOffset == (pRangeOwner.hostOffset - (pOffset - lStartOffset)))
        {
            pRangeOwner.hostOffset -= (pOffset - lStartOffset);
			pOffset = lStartOffset;		// Combine with previous range
        }
		else
        {
			lMap.insert(std::make_pair(lStartOffset, std::make_pair(pOffset-lStartOffset, lStartOwner)));
        }
	}
    else
    {
        if(lStartOffset != pOffset)
            PMTHROW(pmFatalErrorException());
        
        pmMemOwnership::iterator lPrevIter;
        pmMemOwnership::iterator* lPrevIterAddr = &lPrevIter;

        if(pOffset)
        {
            size_t lPrevAddr = pOffset - 1;
            FIND_FLOOR_ELEM(pmMemOwnership, lMap, lPrevAddr, lPrevIterAddr);
            if(lPrevIterAddr)
            {
                size_t lPrevOffset = lPrevIter->first;
                size_t lPrevLength = lPrevIter->second.first;
                vmRangeOwner lPrevOwner = lPrevIter->second.second;
                
                if(lPrevOwner.host == pRangeOwner.host && lPrevOwner.memIdentifier == pRangeOwner.memIdentifier && lPrevOwner.hostOffset + lPrevLength == pRangeOwner.hostOffset)
                {
                    pRangeOwner.hostOffset -= (lStartOffset - lPrevOffset);
                    pOffset = lPrevOffset;		// Combine with previous range                

                    lMap.erase(lPrevIter);
                }
            }
        }
    }

	if(lEndOffset + lEndLength - 1 > lLastAddr)
	{
		if(lEndOwner.host == pRangeOwner.host && lEndOwner.memIdentifier == pRangeOwner.memIdentifier && (lEndOwner.hostOffset + (lLastAddr - lEndOffset)) == lOwnerLastAddr)
        {
			lLastAddr = lEndOffset + lEndLength - 1;	// Combine with following range
        }
		else
        {
            vmRangeOwner lEndRangeOwner = lEndOwner;
            lEndRangeOwner.hostOffset += (lLastAddr - lEndOffset + 1);
			lMap.insert(std::make_pair(lLastAddr + 1, std::make_pair(lEndOffset + lEndLength - 1 - lLastAddr, lEndRangeOwner)));
        }
	}
    else
    {
        if(lEndOffset + lEndLength - 1 != lLastAddr)
            PMTHROW(pmFatalErrorException());

        pmMemOwnership::iterator lNextIter;
        pmMemOwnership::iterator* lNextIterAddr = &lNextIter;
    
        if(lLastAddr + 1 < GetLength())
        {
            size_t lNextAddr = lLastAddr + 1;
            FIND_FLOOR_ELEM(pmMemOwnership, lMap, lNextAddr, lNextIterAddr);
            if(lNextIterAddr)
            {
                size_t lNextOffset = lNextIter->first;
                size_t lNextLength = lNextIter->second.first;
                vmRangeOwner lNextOwner = lNextIter->second.second;
                
                if(lNextOwner.host == pRangeOwner.host && lNextOwner.memIdentifier == pRangeOwner.memIdentifier && lNextOwner.hostOffset == lOwnerLastAddr + 1)
                {
                    lLastAddr = lNextOffset + lNextLength - 1;	// Combine with following range

                    lMap.erase(lNextIter);
                }
            }
        }
    }

	lMap.insert(std::make_pair(pOffset, std::make_pair(lLastAddr - pOffset + 1, pRangeOwner)));

#ifdef _DEBUG
    SanitizeOwnerships();
#endif
}

#ifdef _DEBUG
void pmAddressSpace::CheckMergability(pmMemOwnership::iterator& pRange1, pmMemOwnership::iterator& pRange2)
{
    size_t lOffset1 = pRange1->first;
    size_t lOffset2 = pRange2->first;
    size_t lLength1 = pRange1->second.first;
    size_t lLength2 = pRange2->second.first;
    vmRangeOwner& lRangeOwner1 = pRange1->second.second;
    vmRangeOwner& lRangeOwner2 = pRange2->second.second;
    
    if(lOffset1 + lLength1 != lOffset2)
        std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Range end points don't match. Range 1: Offset = " << lOffset1 << " Length = " << lLength1 << " Range 2: Offset = " << lOffset2 << std::endl;
    
    if(lRangeOwner1.host == lRangeOwner2.host && lRangeOwner1.memIdentifier == lRangeOwner2.memIdentifier && lRangeOwner1.hostOffset + lLength1 == lRangeOwner2.hostOffset)
        std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Mergable Ranges Found (" << lOffset1 << ", " << lLength1 << ") - (" << lOffset2 << ", " << lLength2 << ") map to (" << lRangeOwner1.hostOffset << ", " << lLength1 << ") - (" << lRangeOwner2.hostOffset << ", " << lLength2 << ") on host " << (uint)(*lRangeOwner1.host) << std::endl;
}
    
void pmAddressSpace::SanitizeOwnerships()
{
    if(mOwnershipMap.size() == 1)
        return;
    
    pmMemOwnership::iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end(), lPenultimate = lEnd;
    --lPenultimate;
    
    for(lIter = lBegin; lIter != lPenultimate; ++lIter)
    {
        pmMemOwnership::iterator lNext = lIter;
        ++lNext;
        
        CheckMergability(lIter, lNext);
    }
}

void pmAddressSpace::PrintOwnerships()
{
    std::cout << "Host " << pmGetHostId() << " Ownership Dump " << std::endl;
    pmMemOwnership::iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end();
    for(lIter = lBegin; lIter != lEnd; ++lIter)
        std::cout << "Range (" << lIter->first << " , " << lIter->second.first << ") is owned by host " << (uint)(*(lIter->second.second.host)) << " (" << lIter->second.second.hostOffset << ", " << lIter->second.first << ")" << std::endl;
        
    std::cout << std::endl;
}
#endif

void pmAddressSpace::AcquireOwnershipImmediate(ulong pOffset, ulong pLength)
{
    SetRangeOwner(vmRangeOwner(PM_LOCAL_MACHINE, pOffset, communicator::memoryIdentifierStruct(*(mOwner), mGenerationNumberOnOwner)), pOffset, pLength);
}
    
void pmAddressSpace::TransferOwnershipImmediate(ulong pOffset, ulong pLength, const pmMachine* pNewOwnerHost)
{
    DEBUG_EXCEPTION_ASSERT(pNewOwnerHost != PM_LOCAL_MACHINE);

    SetRangeOwner(vmRangeOwner(pNewOwnerHost, pOffset, communicator::memoryIdentifierStruct(*(mOwner), mGenerationNumberOnOwner)), pOffset, pLength);
}


#ifdef SUPPORT_LAZY_MEMORY
void* pmAddressSpace::GetReadOnlyLazyMemoryMapping()
{
    FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

    if(!mReadOnlyLazyMapping)
        PMTHROW(pmFatalErrorException());

    return mReadOnlyLazyMapping;
}

uint pmAddressSpace::GetLazyForwardPrefetchPageCount()
{
    return LAZY_FORWARD_PREFETCH_PAGE_COUNT;
}
#endif

void pmAddressSpace::GetPageAlignedAddresses(size_t& pOffset, size_t& pLength)
{
    size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();

    size_t lStartAddress = reinterpret_cast<size_t>(GetMem()) + pOffset;
    size_t lEndPageAddress = lStartAddress + pLength - 1;

    pOffset = GET_VM_PAGE_START_ADDRESS(lStartAddress, lPageSize);
    lEndPageAddress = GET_VM_PAGE_START_ADDRESS(lEndPageAddress, lPageSize);

    size_t lEndAddress = lEndPageAddress + lPageSize - 1;
    size_t lMaxAddress = reinterpret_cast<size_t>(GetMem()) + GetLength() - 1;
    if(lEndAddress > lMaxAddress)
        lEndAddress = lMaxAddress;

    pLength = lEndAddress - pOffset + 1;
    pOffset -= reinterpret_cast<size_t>(GetMem());
}
    
void pmAddressSpace::TransferOwnershipPostTaskCompletion(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    DEBUG_EXCEPTION_ASSERT(!GetLockingTask()->IsReadOnly(this));

	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

    mOwnershipTransferVector.push_back(pmMemTransferData(pRangeOwner, pOffset, pLength));
}

void pmAddressSpace::FlushOwnerships()
{
    pmTask* lTask = GetLockingTask();

    DEBUG_EXCEPTION_ASSERT(lTask && lTask->IsWritable(this));
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());    
 
    mOwnershipMap = mOriginalOwnershipMap;
    mOriginalOwnershipMap.clear();
    
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

    bool lOwnershipTransferRequired = ((mOwner == PM_LOCAL_MACHINE) && (!lTask->GetCallbackUnit()->GetDataReductionCB()) && (!lTask->GetCallbackUnit()->GetDataRedistributionCB()));
    pmOwnershipTransferMap lOwnershipTransferMap;
    
    if(lOwnershipTransferRequired)
    {
        std::set<const pmMachine*> lMachines;
        if(dynamic_cast<pmLocalTask*>(lTask))
            pmProcessingElement::GetMachines(((pmLocalTask*)lTask)->GetAssignedDevices(), lMachines);
        else
            pmProcessingElement::GetMachines(((pmRemoteTask*)lTask)->GetAssignedDevices(), lMachines);
        
        // All machines that have locked task memory expect an ownership change message
        for_each(lMachines, [&] (const pmMachine* pMachine)
        {
            if(pMachine != PM_LOCAL_MACHINE)
                lOwnershipTransferMap.emplace(std::piecewise_construct, std::forward_as_tuple(pMachine), std::forward_as_tuple(new std::vector<communicator::ownershipChangeStruct>()));
        });
    }
    
    if(lOwnershipTransferRequired)
    {
        for_each(mOwnershipTransferVector, [&] (const pmMemTransferData& pTransferData)
        {
            pmMemOwnership lOwnerships;
            GetOwnersInternal(mOwnershipMap, pTransferData.offset, pTransferData.length, lOwnerships);

            for_each(lOwnerships, [&] (const decltype(lOwnerships)::value_type& pPair)
            {
                const pmAddressSpace::vmRangeOwner& lRangeOwner = pPair.second.second;

                if(lRangeOwner.host != PM_LOCAL_MACHINE && lRangeOwner.host != pTransferData.rangeOwner.host)
                    lOwnershipTransferMap.find(lRangeOwner.host)->second->emplace_back(pPair.first, pPair.second.first, *(pTransferData.rangeOwner.host));
            });

            SetRangeOwnerInternal(pTransferData.rangeOwner, pTransferData.offset, pTransferData.length, mOwnershipMap);
        });
    }
    else
    {
        for_each(mOwnershipTransferVector, [&] (const pmMemTransferData& pTransferData)
        {
            SetRangeOwnerInternal(pTransferData.rangeOwner, pTransferData.offset, pTransferData.length, mOwnershipMap);
        });
    }
    
    mOwnershipTransferVector.clear();

    if(lOwnershipTransferRequired)
        SendRemoteOwnershipChangeMessages(lOwnershipTransferMap);
}
    
void pmAddressSpace::Fetch(ushort pPriority)
{
    FetchRange(pPriority, 0, GetLength());
}
    
void pmAddressSpace::FetchCompletionCallback(const pmCommandPtr& pCommand)
{
    FINALIZE_RESOURCE_PTR(dWaitingFetchLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mWaitingFetchLock, Lock(), Unlock());
    
    auto lIter = mCommandsWaitingForFetch.find(pCommand);
    EXCEPTION_ASSERT(lIter != mCommandsWaitingForFetch.end());
    
    lIter->second->MarkExecutionEnd(pmSuccess, lIter->second);
    mCommandsWaitingForFetch.erase(lIter);
}
    
void pmAddressSpace::FetchAsync(ushort pPriority, pmCommandPtr pCommand)
{
    std::vector<pmCommandPtr> lVector;
    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(this, pPriority, 0, GetLength(), lVector);

    if(lVector.size())
    {
        FINALIZE_RESOURCE_PTR(dWaitingFetchLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mWaitingFetchLock, Lock(), Unlock());

        pmCommandPtr lAccumulatorCommand = pmAccumulatorCommand::CreateSharedPtr(lVector, FetchCallback, this);
        mCommandsWaitingForFetch.emplace(lAccumulatorCommand, pCommand);
    }
    else
    {
        pCommand->MarkExecutionEnd(pmSuccess, pCommand);
    }
}

void pmAddressSpace::FetchRange(ushort pPriority, ulong pOffset, ulong pLength)
{
#ifdef ENABLE_MEM_PROFILING
    TIMER_IMPLEMENTATION_CLASS lTimer;
    lTimer.Start();
#endif

    std::vector<pmCommandPtr> lVector;
    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(this, pPriority, pOffset, pLength, lVector);
    
    std::vector<pmCommandPtr>::const_iterator lIter = lVector.begin(), lEndIter = lVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForFinish();

#ifdef ENABLE_MEM_PROFILING
    lTimer.Stop();
    
    std::stringstream lStream;
    lStream << std::endl << "Address Space [" << (uint)(*mOwner) << ", " << mGenerationNumberOnOwner << "] explicit fetch time on [Host " << pmGetHostId() << "] is " << lTimer.GetElapsedTimeInSecs() << "s" << std::endl;

    pmLogger::GetLogger()->LogDeferred(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str(), false);
#endif
}

/* This method does not acquire mOwnershipLock (directly calls GetOwnersInternal instead of GetOwners).
 It is only meant to be called by preprocessor task as it does not actually fetch data on user task's
 address spaces, but just determines its current ownership.
 */
ulong pmAddressSpace::FindLocalDataSizeUnprotected(ulong pOffset, ulong pLength)
{
    pmAddressSpace::pmMemOwnership lOwners;
    GetOwnersInternal(mOwnershipMap, pOffset, pLength, lOwners);
    
    ulong lLocalDataSize = 0;
    for_each(lOwners, [&] (const typename decltype(lOwners)::value_type& pPair)
    {
        if(pPair.second.second.host == PM_LOCAL_MACHINE)
            lLocalDataSize += pPair.second.first;
    });
    
    return lLocalDataSize;
}

/* This method does not acquire mOwnershipLock (directly calls GetOwnersInternal instead of GetOwners).
 It is only meant to be called by preprocessor task as it does not actually fetch data on user task's
 address spaces, but just determines its current ownership.
 */
std::set<const pmMachine*> pmAddressSpace::FindRemoteDataSourcesUnprotected(ulong pOffset, ulong pLength)
{
    pmAddressSpace::pmMemOwnership lOwners;
    GetOwnersInternal(mOwnershipMap, pOffset, pLength, lOwners);
    
    std::set<const pmMachine*> lSet;
    for_each(lOwners, [&] (const typename decltype(lOwners)::value_type& pPair)
    {
        if(pPair.second.second.host != PM_LOCAL_MACHINE)
            lSet.emplace(pPair.second.second.host);
    });
    
    return lSet;
}

bool pmAddressSpace::IsRegionLocallyOwned(ulong pOffset, ulong pLength)
{
    pmAddressSpace::pmMemOwnership lOwners;
    GetOwners(pOffset, pLength, lOwners);
    
    return ((lOwners.size() == 1) && (lOwners.begin()->second.second.host == PM_LOCAL_MACHINE));
}

void pmAddressSpace::GetOwners(ulong pOffset, ulong pLength, pmAddressSpace::pmMemOwnership& pOwnerships)
{
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = mOwnershipLock;
    pmMemOwnership& lMap = mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
    GetOwnersInternal(lMap, pOffset, pLength, pOwnerships);
}

/* This method does not acquire mOwnershipLock (directly calls GetOwnersInternal instead of GetOwners).
 It is only meant to be called by preprocessor task as it does not actually fetch data on user task's
 address spaces, but just determines its current ownership.
 */
void pmAddressSpace::GetOwnersUnprotected(ulong pOffset, ulong pLength, pmAddressSpace::pmMemOwnership& pOwnerships)
{
    GetOwnersInternal(mOwnershipMap, pOffset, pLength, pOwnerships);
}

/* This method must be called with mOwnershipLock acquired */
void pmAddressSpace::GetOwnersInternal(pmMemOwnership& pMap, ulong pOffset, ulong pLength, pmAddressSpace::pmMemOwnership& pOwnerships)
{
    pmMemOwnership& lMap = pMap;
    
	ulong lLastAddr = pOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, lMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, lMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		PMTHROW(pmFatalErrorException());

    size_t lSpan = lStartIter->first + lStartIter->second.first - 1;
    if(lLastAddr < lSpan)
    {
        lSpan = lLastAddr;
        if(lStartIter != lEndIter)
            PMTHROW(pmFatalErrorException());
    }
    
    vmRangeOwner lRangeOwner = lStartIter->second.second;
    lRangeOwner.hostOffset += (pOffset - lStartIter->first);
	pOwnerships.insert(std::make_pair(pOffset, std::make_pair(lSpan - pOffset + 1, lRangeOwner)));
	
	pmMemOwnership::iterator lIter = lStartIter;
	++lIter;

	if(lStartIter != lEndIter)
	{
		for(; lIter != lEndIter; ++lIter)
        {
            lSpan = lIter->first + lIter->second.first - 1;
            if(lLastAddr < lSpan)
            {
                lSpan = lLastAddr;
                if(lIter != lEndIter)
                    PMTHROW(pmFatalErrorException());
            }
            
			pOwnerships.insert(std::make_pair(lIter->first, std::make_pair(lSpan - lIter->first + 1, lIter->second.second)));
        }
        
        lSpan = lEndIter->first + lEndIter->second.first - 1;
        if(lLastAddr < lSpan)
            lSpan = lLastAddr;
        
        pOwnerships.insert(std::make_pair(lEndIter->first, std::make_pair(lSpan - lEndIter->first + 1, lEndIter->second.second)));
	}
}

void pmAddressSpace::SendRemoteOwnershipChangeMessages(pmOwnershipTransferMap& pOwnershipTransferMap)
{
    pmOwnershipTransferMap::iterator lIter = pOwnershipTransferMap.begin(), lEndIter = pOwnershipTransferMap.end();

    for(; lIter != lEndIter; ++lIter)
        pmScheduler::GetScheduler()->SendPostTaskOwnershipTransfer(this, lIter->first, lIter->second);
}
    
void pmAddressSpace::DeleteAllLocalAddressSpaces()
{
    std::vector<pmAddressSpace*> lAddressSpaces;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        for_each(GetAddressSpaceMap(), [&] (const addressSpaceMapType::value_type& pPair) {lAddressSpaces.push_back(pPair.second);});
    }

    for_each(lAddressSpaces, [] (pmAddressSpace* pAddressSpace) {delete pAddressSpace;});
}

#ifdef ENABLE_MEM_PROFILING
void pmAddressSpace::RecordMemReceive(size_t pReceiveSize)
{
    FINALIZE_RESOURCE_PTR(dMemProfileLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemProfileLock, Lock(), Unlock());

    mMemReceived += pReceiveSize;
    ++mMemReceiveEvents;
}

void pmAddressSpace::RecordMemTransfer(size_t pTransferSize)
{
    FINALIZE_RESOURCE_PTR(dMemProfileLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemProfileLock, Lock(), Unlock());

    mMemTransferred += pTransferSize;
    ++mMemTransferEvents;
}
#endif

/* class pmUserMemHandle */
pmUserMemHandle::pmUserMemHandle(pmAddressSpace* pAddressSpace)
    : mAddressSpace(pAddressSpace)
{
    pAddressSpace->SetUserMemHandle(this);
}
    
pmUserMemHandle::~pmUserMemHandle()
{
}

void pmUserMemHandle::Reset(pmAddressSpace* pAddressSpace)
{
    mAddressSpace->SetUserMemHandle(NULL);
    mAddressSpace = pAddressSpace;
    mAddressSpace->SetUserMemHandle(this);
}

pmAddressSpace* pmUserMemHandle::GetAddressSpace()
{
    return mAddressSpace;
}
    
    
/* class pmScatteredSubscriptionFilter */
pmScatteredSubscriptionFilter::pmScatteredSubscriptionFilter(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
    : mScatteredSubscriptionInfo(pScatteredSubscriptionInfo)
{}
    
const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, pmAddressSpace::vmRangeOwner>>>& pmScatteredSubscriptionFilter::FilterBlocks(const std::function<void (size_t)>& pRowFunctor)
{
    for(size_t i = 0; i < mScatteredSubscriptionInfo.count; ++i)
        pRowFunctor(i);
    
    return GetLeftoverBlocks();
}

// AddNextSubRow must be called in increasing y first and then increasing x values (i.e. first one or more times with increasing x for row one, then similarly for row two and so on)
void pmScatteredSubscriptionFilter::AddNextSubRow(ulong pOffset, ulong pLength, pmAddressSpace::vmRangeOwner& pRangeOwner)
{
    ulong lStartCol = (pOffset - (ulong)mScatteredSubscriptionInfo.offset) % (ulong)mScatteredSubscriptionInfo.step;
    
    bool lRangeCombined = false;

    auto lIter = mCurrentBlocks.begin(), lEndIter = mCurrentBlocks.end();
    while(lIter != lEndIter)
    {
        blockData& lData = (*lIter);

        ulong lEndCol1 = lData.startCol + lData.colCount - 1;
        ulong lEndCol2 = lStartCol + pLength - 1;
        
        bool lRemoveCurrentRange = false;
        
        if(!(lEndCol2 < lData.startCol || lStartCol > lEndCol1))    // If the ranges overlap
        {
            // Total overlap and to be fetched from same host and there is no gap between rows
            if(lData.startCol == lStartCol && lData.colCount == pLength && lData.rangeOwner.host == pRangeOwner.host
               && lData.rangeOwner.memIdentifier == pRangeOwner.memIdentifier
               && (lData.rangeOwner.hostOffset + lData.subscriptionInfo.count * lData.subscriptionInfo.step == pRangeOwner.hostOffset)
               && (lData.subscriptionInfo.offset + lData.subscriptionInfo.count * lData.subscriptionInfo.step == pOffset))
            {
                ++lData.subscriptionInfo.count; // Combine with previous range
                lRangeCombined = true;
            }
            else
            {
                lRemoveCurrentRange = true;
            }
        }
        
        if(lRemoveCurrentRange)
        {
            mBlocksToBeFetched[lData.rangeOwner.host].emplace_back(lData.subscriptionInfo, lData.rangeOwner);
            mCurrentBlocks.erase(lIter++);
        }
        else
        {
            ++lIter;
        }
    }
    
    if(!lRangeCombined)
        mCurrentBlocks.emplace_back(lStartCol, pLength, pmScatteredSubscriptionInfo(pOffset, pLength, mScatteredSubscriptionInfo.step, 1), pRangeOwner);
}

const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, pmAddressSpace::vmRangeOwner>>>& pmScatteredSubscriptionFilter::GetLeftoverBlocks()
{
    PromoteCurrentBlocks();
    
    return mBlocksToBeFetched;
}

void pmScatteredSubscriptionFilter::PromoteCurrentBlocks()
{
    for_each(mCurrentBlocks, [&] (const blockData& pData)
    {
        mBlocksToBeFetched[pData.rangeOwner.host].emplace_back(pData.subscriptionInfo, pData.rangeOwner);
    });
    
    mCurrentBlocks.clear();
}
    
};





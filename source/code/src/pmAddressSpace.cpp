
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

#include "pmAddressSpace.h"
#include "pmMemoryManager.h"
#include "pmHardware.h"
#include "pmScheduler.h"
#include "pmTask.h"
#include "pmCallbackUnit.h"
#include "pmHeavyOperations.h"

#include <string.h>
#include <sstream>

namespace pm
{

STATIC_ACCESSOR_INIT(ulong, pmAddressSpace, GetGenerationId, 0)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmAddressSpace::mGenerationLock"), pmAddressSpace, GetGenerationLock)

STATIC_ACCESSOR(pmAddressSpace::addressSpaceMapType, pmAddressSpace, GetAddressSpaceMap)
STATIC_ACCESSOR(pmAddressSpace::augmentaryAddressSpaceMapType, pmAddressSpace, GetAugmentaryAddressSpaceMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmAddressSpace::mResourceLock"), pmAddressSpace, GetResourceLock)

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
    , mOwnershipLock __LOCK_NAME__("pmAddressSpace::mOwnershipLock")
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
    if(mGenerationNumberOnOwner == 0)
        PMTHROW(pmFatalErrorException());

	mMem = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->AllocateMemory(this, mAllocatedLength, mVMPageCount);
    Init(mOwner);

    // Auto lock/unlock scope
	{
		FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::pair<const pmMachine*, ulong> lPair(mOwner, mGenerationNumberOnOwner);

        addressSpaceMapType& lAddressSpaceMap = GetAddressSpaceMap();
        if(lAddressSpaceMap.find(lPair) != lAddressSpaceMap.end())
            PMTHROW(pmFatalErrorException());

		lAddressSpaceMap[lPair] = this;
	}
}

pmAddressSpace::~pmAddressSpace()
{
#ifdef ENABLE_MEM_PROFILING
    std::stringstream lStream;
    if(IsReadOnly())
        lStream << mMemReceived << " bytes input memory received in " << mMemReceiveEvents << " events";
    else
        lStream << mMemReceived << " bytes output memory received in " << mMemReceiveEvents << " events";
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
    
    lStream.str(std::string()); // clear stream
    if(IsReadOnly())
        lStream << mMemTransferred << " bytes input memory transferred in " << mMemTransferEvents << " events";
    else
        lStream << mMemTransferred << " bytes output memory transferred in " << mMemTransferEvents << " events";
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif    

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

    #ifdef _DEBUG
        if(mLockingTask)
            PMTHROW(pmFatalErrorException());
    #endif

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
    DEBUG_EXCEPTION_ASSERT(!GetLockingTask()->IsReadOnly(this));
    
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
	mOwnershipMap.insert(std::make_pair(0, std::make_pair(mRequestedLength, vmRangeOwner(pOwner, 0, communicator::memoryIdentifierStruct(*(mOwner), mGenerationNumberOnOwner)))));
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
    
void pmAddressSpace::Lock(pmTask* pTask, pmMemType pMemType)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        if(mLockingTask || pMemType == MAX_MEM_TYPE)
            PMTHROW(pmFatalErrorException());
        
        mLockingTask = pTask;

    #ifdef SUPPORT_LAZY_MEMORY
        if((pTask->IsWritable(this) || !pTask->IsLazy(this)) && mReadOnlyLazyMapping)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeleteReadOnlyMemoryMapping(mReadOnlyLazyMapping, mAllocatedLength);
 
            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
            lAugmentaryAddressSpaceMap.erase(mReadOnlyLazyMapping);

            mReadOnlyLazyMapping = NULL;
        }
    
        if(pTask->IsReadOnly(this) && pTask->IsLazy(this) && !mReadOnlyLazyMapping)
        {
            mReadOnlyLazyMapping = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateReadOnlyMemoryMapping(this);

            FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

            augmentaryAddressSpaceMapType& lAugmentaryAddressSpaceMap = GetAugmentaryAddressSpaceMap();
            lAugmentaryAddressSpaceMap[mReadOnlyLazyMapping] = this;
        }
    #endif
    }
    
    if(pTask->IsWritable(this))
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        mOriginalOwnershipMap = mOwnershipMap;
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(pTask->IsReadOnly(this) && pTask->IsLazy(this))
    {
        if(IsRegionLocallyOwned(0, GetLength()))
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(mReadOnlyLazyMapping, mAllocatedLength, true, true);
    }
#endif
}
    
void pmAddressSpace::Unlock(pmTask* pTask)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        
        if(!mOwnershipTransferVector.empty())
        {
            std::cout << mLockingTask->IsReadOnly(this) << " " << (uint)(*GetMemOwnerHost()) << " " << GetGenerationNumber() << std::endl;
            PMTHROW(pmFatalErrorException());
        }
    }
#endif
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        if(mLockingTask != pTask)
            PMTHROW(pmFatalErrorException());
        
        mLockingTask = NULL;
    }

    bool lUserDelete = false;
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dDeleteLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDeleteLock, Lock(), Unlock());
        lUserDelete = mUserDelete;
    }
    
    if(lUserDelete)
        delete this;
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
    if(GetLockingTask())
        PMTHROW(pmFatalErrorException());

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
    DEBUG_EXCEPTION_ASSERT(GetLockingTask() && GetLockingTask()->IsWritable(this));
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());    
 
    mOwnershipMap = mOriginalOwnershipMap;
    mOriginalOwnershipMap.clear();
    
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

#if 0
    bool lOwnershipTransferRequired = (mOwner != PM_LOCAL_MACHINE) && (!lTask->GetCallbackUnit()->GetDataReductionCB()) && (!lTask->GetCallbackUnit()->GetDataRedistributionCB());
#else
    bool lOwnershipTransferRequired = false;
#endif

    pmOwnershipTransferMap lOwnershipTransferMap;
    pmMemOwnership lOwnerships;
    
    std::vector<pmMemTransferData>::iterator lIter = mOwnershipTransferVector.begin(), lEndIter = mOwnershipTransferVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        pmMemTransferData& lTransferData = *lIter;

        if(lOwnershipTransferRequired)
        {
            GetOwnersInternal(mOwnershipMap, lTransferData.offset, lTransferData.length, lOwnerships);
            pmAddressSpace::pmMemOwnership::iterator lIter = lOwnerships.begin(), lEndIter = lOwnerships.end();
            for(; lIter != lEndIter; ++lIter)
            {
                ulong lInternalOffset = lIter->first;
                ulong lInternalLength = lIter->second.first;
                pmAddressSpace::vmRangeOwner& lRangeOwner = lIter->second.second;

                if(mOwner == PM_LOCAL_MACHINE && lRangeOwner.host != PM_LOCAL_MACHINE && lRangeOwner.host != lTransferData.rangeOwner.host)
                {
                    if(lOwnershipTransferMap.find(lRangeOwner.host) == lOwnershipTransferMap.end())
                        lOwnershipTransferMap[lRangeOwner.host].reset(new std::vector<communicator::ownershipChangeStruct>());

                    lOwnershipTransferMap[lRangeOwner.host]->push_back(communicator::ownershipChangeStruct(lInternalOffset, lInternalLength, *(lTransferData.rangeOwner.host)));
                }
            }
        }

        SetRangeOwnerInternal(lTransferData.rangeOwner, lTransferData.offset, lTransferData.length, mOwnershipMap);
    }
    
    mOwnershipTransferVector.clear();

    if(lOwnershipTransferRequired)
        SendRemoteOwnershipChangeMessages(lOwnershipTransferMap);
}
    
void pmAddressSpace::Fetch(ushort pPriority)
{
    FetchRange(pPriority, 0, GetLength());
}
    
void pmAddressSpace::FetchRange(ushort pPriority, ulong pOffset, ulong pLength)
{
#ifdef ENABLE_MEM_PROFILING
    TIMER_IMPLEMENTATION_CLASS lTimer;
    lTimer.Start();
#endif

    std::vector<pmCommunicatorCommandPtr> lVector;
    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(this, pPriority, pOffset, pLength, lVector);
    
    std::vector<pmCommunicatorCommandPtr>::const_iterator lIter = lVector.begin(), lEndIter = lVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForFinish();

#ifdef ENABLE_MEM_PROFILING
    lTimer.Stop();
    
    char lStr[512];
    sprintf(lStr, "%s memory Fetch Time = %lfs", (IsReadOnly()?(char*)"Input":(char*)"Output"), lTimer.GetElapsedTimeInSecs());
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr, true);
#endif
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
    
};





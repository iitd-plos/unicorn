
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

#include "pmMemSection.h"
#include "pmMemoryManager.h"
#include "pmHardware.h"
#include "pmScheduler.h"
#include "pmTask.h"
#include "pmCallbackUnit.h"

#include <string.h>
#include <sstream>

namespace pm
{

RESOURCE_LOCK_IMPLEMENTATION_CLASS pmMemSection::mGenerationLock;
ulong pmMemSection::mGenerationId = 0;

std::map<std::pair<pmMachine*, ulong>, pmMemSection*> pmMemSection::mMemSectionMap;
std::map<void*, pmMemSection*> pmMemSection::mAugmentaryMemSectionMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmMemSection::mResourceLock;

/* class pmMemSection */
pmMemSection::pmMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner)
    : mOwner(pOwner?pOwner:PM_LOCAL_MACHINE)
    , mGenerationNumberOnOwner(pGenerationNumberOnOwner)
    , mUserMemHandle(NULL)
    , mRequestedLength(pLength)
    , mAllocatedLength(pLength)
    , mVMPageCount(0)
    , mLazy(false)
    , mMem(NULL)
    , mReadOnlyLazyMapping(NULL)
    , mLockingTask(NULL)
    , mUserDelete(false)
    , mMemInfo(MAX_MEM_INFO)
#ifdef ENABLE_MEM_PROFILING
    , mMemReceived(0)
    , mMemTransferred(0)
    , mMemReceiveEvents(0)
    , mMemTransferEvents(0)
#endif
{
    if(mGenerationNumberOnOwner == 0)
        PMTHROW(pmFatalErrorException());

	mMem = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->AllocateMemory(this, mAllocatedLength, mVMPageCount);
    Init(mOwner);

    // Auto lock/unlock scope
	{
		FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

        std::pair<pmMachine*, ulong> lPair(mOwner, mGenerationNumberOnOwner);
    
        if(mMemSectionMap.find(lPair) != mMemSectionMap.end())
            PMTHROW(pmFatalErrorException());

		mMemSectionMap[lPair] = this;
	}
}

pmMemSection::~pmMemSection()
{
#ifdef ENABLE_MEM_PROFILING
    std::stringstream lStream;
    if(IsInput())
        lStream << mMemReceived << " bytes input memory received in " << mMemReceiveEvents << " events";
    else
        lStream << mMemReceived << " bytes output memory received in " << mMemReceiveEvents << " events";
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str());
    
    lStream.str(std::string()); // clear stream
    if(IsInput())
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

        if(mReadOnlyLazyMapping)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeleteReadOnlyMemoryMapping(mReadOnlyLazyMapping, mAllocatedLength);

            mAugmentaryMemSectionMap.erase(mReadOnlyLazyMapping);
            mReadOnlyLazyMapping = NULL;
        }
    }
    
    DisposeMemory();
}
    
pmMemSection* pmMemSection::CreateMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner /* = GetNextGenerationNumber() */)
{
    return new pmMemSection(pLength, pOwner, pGenerationNumberOnOwner);
}

pmMemSection* pmMemSection::CheckAndCreateMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner)
{
    pmMemSection* lMemSection = FindMemSection(pOwner, pGenerationNumberOnOwner);
    if(!lMemSection)
        lMemSection = CreateMemSection(pLength, pOwner, pGenerationNumberOnOwner);

    return lMemSection;
}
    
const char* pmMemSection::GetName()
{
    if(mName.empty())
    {
        std::stringstream lStream;
        lStream << "/pm_" << ::getpid() << "_" << (uint)(*(GetMemOwnerHost())) << "_" << GetGenerationNumber();

        mName = lStream.str();
    }
    
    return mName.c_str();
}

pmMachine* pmMemSection::GetMemOwnerHost()
{
    return mOwner;
}
    
ulong pmMemSection::GetGenerationNumber()
{
    return mGenerationNumberOnOwner;
}
    
ulong pmMemSection::GetNextGenerationNumber()
{
    FINALIZE_RESOURCE(dGenerationLock, mGenerationLock.Lock(), mGenerationLock.Unlock());
    return (++mGenerationId);   // Generation number 0 is reserved
}

pmStatus pmMemSection::Update(size_t pOffset, size_t pLength, void* pSrcAddr)
{
    if(IsInput())
        PMTHROW(pmFatalErrorException());
    
	void* lDestAddr = (void*)((char*)GetMem() + pOffset);
	memcpy(lDestAddr, pSrcAddr, pLength);

	return pmSuccess;
}

pmMemInfo pmMemSection::GetMemInfo()
{
	return mMemInfo;
}
    
bool pmMemSection::IsInput() const
{
    return (mMemInfo == INPUT_MEM_READ_ONLY || mMemInfo == INPUT_MEM_READ_ONLY_LAZY);
}

bool pmMemSection::IsOutput() const
{
    return (mMemInfo == OUTPUT_MEM_WRITE_ONLY || mMemInfo == OUTPUT_MEM_READ_WRITE || mMemInfo == OUTPUT_MEM_READ_WRITE_LAZY);
}

void pmMemSection::UserDelete()
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
    
void pmMemSection::DisposeMemory()
{
    if(mMem)
    {
        FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
        mMemSectionMap.erase(std::make_pair(mOwner, mGenerationNumberOnOwner));
    
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(this);
        mMem = NULL;
    }
}

void pmMemSection::SetUserMemHandle(pmUserMemHandle* pUserMemHandle)
{
    mUserMemHandle = pUserMemHandle;
}

pmUserMemHandle* pmMemSection::GetUserMemHandle()
{
    return mUserMemHandle;
}

void pmMemSection::Init(pmMachine* pOwner)
{
	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
    lRangeOwner.hostOffset = 0;
    lRangeOwner.memIdentifier.memOwnerHost = *(mOwner);
    lRangeOwner.memIdentifier.generationNumber = mGenerationNumberOnOwner;
    
	mOwnershipMap[0] = std::pair<size_t, vmRangeOwner>(mRequestedLength, lRangeOwner);
}

void* pmMemSection::GetMem()
{
	return mMem;
}

size_t pmMemSection::GetAllocatedLength()
{
	return mAllocatedLength;
}

size_t pmMemSection::GetLength()
{
	return mRequestedLength;
}

pmMemSection* pmMemSection::FindMemSection(pmMachine* pOwner, ulong pGenerationNumber)
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	std::map<std::pair<pmMachine*, ulong>, pmMemSection*>::iterator lIter = mMemSectionMap.find(std::make_pair(pOwner, pGenerationNumber));
	if(lIter != mMemSectionMap.end())
		return lIter->second;

	return NULL;
}
    
pmMemSection* pmMemSection::FindMemSectionContainingLazyAddress(void* pPtr)
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
    
    typedef std::map<void*, pmMemSection*> mapType;
    mapType::iterator lStartIter;
    mapType::iterator* lStartIterAddr = &lStartIter;
    
    char* lAddress = static_cast<char*>(pPtr);
    FIND_FLOOR_ELEM(mapType, mAugmentaryMemSectionMap, lAddress, lStartIterAddr);
    
    if(lStartIterAddr)
    {
        char* lMemAddress = static_cast<char*>(lStartIter->first);
        pmMemSection* lMemSection = lStartIter->second;

        size_t lLength = lMemSection->GetLength();
        
        if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
            return lMemSection;
    }
    
    return NULL;
}
    
void pmMemSection::Lock(pmTask* pTask, pmMemInfo pMemInfo)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

        if(mLockingTask || pMemInfo == MAX_MEM_INFO)
            PMTHROW(pmFatalErrorException());
        
        mLockingTask = pTask;
        mMemInfo = pMemInfo;

    #ifdef SUPPORT_LAZY_MEMORY
        if((IsOutput() || !IsLazy()) && mReadOnlyLazyMapping)
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeleteReadOnlyMemoryMapping(mReadOnlyLazyMapping, mAllocatedLength);
 
            mAugmentaryMemSectionMap.erase(mReadOnlyLazyMapping);
            mReadOnlyLazyMapping = NULL;
        }
    
        if(IsInput() && IsLazy() && !mReadOnlyLazyMapping)
        {
            mReadOnlyLazyMapping = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateReadOnlyMemoryMapping(this);
            mAugmentaryMemSectionMap[mReadOnlyLazyMapping] = this;
        }
    #endif
    }
    
    if(IsOutput())
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        mOriginalOwnershipMap = mOwnershipMap;
    }
    
#ifdef SUPPORT_LAZY_MEMORY
    if(IsInput() && IsLazy())
    {
        if(IsRegionLocallyOwned(0, GetLength()))
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->SetLazyProtection(mReadOnlyLazyMapping, mAllocatedLength, true, true);
    }
#endif
}
    
void pmMemSection::Unlock(pmTask* pTask)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        
        if(!mOwnershipTransferVector.empty())
        {
            std::cout << IsInput() << " " << (uint)(*GetMemOwnerHost()) << " " << GetGenerationNumber() << std::endl;
            abort();
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
        mMemInfo = MAX_MEM_INFO;
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
    
pmTask* pmMemSection::GetLockingTask()
{
	FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

    return mLockingTask;
}

pmStatus pmMemSection::SetRangeOwner(vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = mOwnershipLock;
    pmMemOwnership& lMap = mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    return SetRangeOwnerInternal(pRangeOwner, pOffset, pLength, lMap);
}

// This method must be called with mOwnershipLock acquired
pmStatus pmMemSection::SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength, pmMemOwnership& pMap)
{
    pmMemOwnership& lMap = pMap;
    
#if _DEBUG
#if 0
    PrintOwnerships();
    if(pRangeOwner.memIdentifier.memOwnerHost == *(mOwner) && pRangeOwner.memIdentifier.generationNumber == mGenerationNumberOnOwner)
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner, Owner Offset): (" << pOffset << ", " << pLength << ", " << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
    else
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner Mem Section (Host, Generation Number), Owner, Owner Offset): (" << pOffset << ", " << pLength << ", (" << pRangeOwner.memIdentifier.memOwnerHost << ", " << pRangeOwner.memIdentifier.generationNumber << ")," << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
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
			lMap[lStartOffset] = std::pair<size_t, vmRangeOwner>(pOffset-lStartOffset, lStartOwner);
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
			lMap[lLastAddr + 1] = std::pair<size_t, vmRangeOwner>(lEndOffset + lEndLength - 1 - lLastAddr, lEndRangeOwner);
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

	lMap[pOffset] = std::pair<size_t, vmRangeOwner>(lLastAddr - pOffset + 1, pRangeOwner);

#ifdef _DEBUG
    SanitizeOwnerships();
#endif
    
	return pmSuccess;
}

#ifdef _DEBUG
void pmMemSection::CheckMergability(pmMemOwnership::iterator& pRange1, pmMemOwnership::iterator& pRange2)
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
    
void pmMemSection::SanitizeOwnerships()
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

void pmMemSection::PrintOwnerships()
{
    std::cout << "Host " << pmGetHostId() << " Ownership Dump " << std::endl;
    pmMemOwnership::iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end();
    for(lIter = lBegin; lIter != lEnd; ++lIter)
        std::cout << "Range (" << lIter->first << " , " << lIter->second.first << ") is owned by host " << (uint)(*(lIter->second.second.host)) << " (" << lIter->second.second.hostOffset << ", " << lIter->second.first << ")" << std::endl;
        
    std::cout << std::endl;
}
#endif

pmStatus pmMemSection::AcquireOwnershipImmediate(ulong pOffset, ulong pLength)
{
    vmRangeOwner lRangeOwner;
    lRangeOwner.host = PM_LOCAL_MACHINE;
    lRangeOwner.hostOffset = pOffset;
    lRangeOwner.memIdentifier.memOwnerHost = *(mOwner);
    lRangeOwner.memIdentifier.generationNumber = mGenerationNumberOnOwner;
    return SetRangeOwner(lRangeOwner, pOffset, pLength);
}
    
pmStatus pmMemSection::TransferOwnershipImmediate(ulong pOffset, ulong pLength, pmMachine* pNewOwnerHost)
{
    if(GetLockingTask())
        PMTHROW(pmFatalErrorException());
    
    if(pNewOwnerHost == PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());

    vmRangeOwner lRangeOwner;
    lRangeOwner.host = pNewOwnerHost;
    lRangeOwner.hostOffset = pOffset;
    lRangeOwner.memIdentifier.memOwnerHost = *(mOwner);
    lRangeOwner.memIdentifier.generationNumber = mGenerationNumberOnOwner;

    //std::cout << "Host " << pmGetHostId() << " Transferring Ownership " << pOffset << " " << pLength << " " << *(pNewOwnerHost) << std::endl;
    return SetRangeOwner(lRangeOwner, pOffset, pLength);
}


#ifdef SUPPORT_LAZY_MEMORY
void* pmMemSection::GetReadOnlyLazyMemoryMapping()
{
    FINALIZE_RESOURCE_PTR(dTaskLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTaskLock, Lock(), Unlock());

    if(!mReadOnlyLazyMapping)
        PMTHROW(pmFatalErrorException());

    return mReadOnlyLazyMapping;
}

uint pmMemSection::GetLazyForwardPrefetchPageCount()
{
    return LAZY_FORWARD_PREFETCH_PAGE_COUNT;
}

void pmMemSection::GetPageAlignedAddresses(size_t& pOffset, size_t& pLength)
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
#endif
    
pmStatus pmMemSection::TransferOwnershipPostTaskCompletion(vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
#ifdef _DEBUG
    if(IsInput())
        PMTHROW(pmFatalErrorException());
#endif
    
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    pmMemTransferData lTransferData;
    lTransferData.rangeOwner = pRangeOwner;
    lTransferData.offset = pOffset;
    lTransferData.length = pLength;

    mOwnershipTransferVector.push_back(lTransferData);

    return pmSuccess;
}

pmStatus pmMemSection::FlushOwnerships()
{
#ifdef _DEBUG
    if(!GetLockingTask() || !IsOutput())
       PMTHROW(pmFatalErrorException());
#endif
    
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
    pmCommunicatorCommand::ownershipChangeStruct lChangeStruct;

    std::vector<pmMemTransferData>::iterator lIter = mOwnershipTransferVector.begin(), lEndIter = mOwnershipTransferVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        pmMemTransferData& lTransferData = *lIter;

        if(lOwnershipTransferRequired)
        {
            GetOwnersInternal(mOwnershipMap, lTransferData.offset, lTransferData.length, lOwnerships);
            pmMemSection::pmMemOwnership::iterator lIter = lOwnerships.begin(), lEndIter = lOwnerships.end();
            for(; lIter != lEndIter; ++lIter)
            {
                ulong lInternalOffset = lIter->first;
                ulong lInternalLength = lIter->second.first;
                pmMemSection::vmRangeOwner& lRangeOwner = lIter->second.second;

                if(mOwner == PM_LOCAL_MACHINE && lRangeOwner.host != PM_LOCAL_MACHINE && lRangeOwner.host != lTransferData.rangeOwner.host)
                {
                    lChangeStruct.offset = lInternalOffset;
                    lChangeStruct.length = lInternalLength;
                    lChangeStruct.newOwnerHost = *(lTransferData.rangeOwner.host);
                
                    if(lOwnershipTransferMap.find(lRangeOwner.host) == lOwnershipTransferMap.end())
                        lOwnershipTransferMap[lRangeOwner.host].reset(new std::vector<pmCommunicatorCommand::ownershipChangeStruct>());

                    lOwnershipTransferMap[lRangeOwner.host]->push_back(lChangeStruct);
                }
            }
        }

        SetRangeOwnerInternal(lTransferData.rangeOwner, lTransferData.offset, lTransferData.length, mOwnershipMap);
    }
    
    mOwnershipTransferVector.clear();

    if(lOwnershipTransferRequired)
        SendRemoteOwnershipChangeMessages(lOwnershipTransferMap);

	return pmSuccess;
}
    
pmStatus pmMemSection::Fetch(ushort pPriority)
{
#ifdef ENABLE_MEM_PROFILING
    TIMER_IMPLEMENTATION_CLASS lTimer;
    lTimer.Start();
#endif

    std::vector<pmCommunicatorCommandPtr> lVector;
    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(this, pPriority, 0, GetLength(), lVector);
    
    std::vector<pmCommunicatorCommandPtr>::const_iterator lIter = lVector.begin(), lEndIter = lVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForFinish();

#ifdef ENABLE_MEM_PROFILING
    lTimer.Stop();
    
    char lStr[512];
    sprintf(lStr, "%s memory Fetch Time = %lfs", (IsInput()?(char*)"Input":(char*)"Output"), lTimer.GetElapsedTimeInSecs());
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr, true);
#endif
    
    return pmSuccess;
}

bool pmMemSection::IsRegionLocallyOwned(ulong pOffset, ulong pLength)
{
    pmMemSection::pmMemOwnership lOwners;
    GetOwners(pOffset, pLength, lOwners);
    
    return ((lOwners.size() == 1) && (lOwners.begin()->second.second.host == PM_LOCAL_MACHINE));
}

pmStatus pmMemSection::GetOwners(ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships)
{
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = mOwnershipLock;
    pmMemOwnership& lMap = mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
    return GetOwnersInternal(lMap, pOffset, pLength, pOwnerships);
}

/* This method must be called with mOwnershipLock acquired */
pmStatus pmMemSection::GetOwnersInternal(pmMemOwnership& pMap, ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships)
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
	pOwnerships[pOffset] = std::make_pair(lSpan - pOffset + 1, lRangeOwner);
	
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
            
			pOwnerships[lIter->first] = std::make_pair(lSpan - lIter->first + 1, lIter->second.second);
        }
        
        lSpan = lEndIter->first + lEndIter->second.first - 1;
        if(lLastAddr < lSpan)
            lSpan = lLastAddr;
        
        pOwnerships[lEndIter->first] = std::make_pair(lSpan - lEndIter->first + 1, lEndIter->second.second);        
	}

	return pmSuccess;
}

void pmMemSection::SendRemoteOwnershipChangeMessages(pmOwnershipTransferMap& pOwnershipTransferMap)
{
    pmOwnershipTransferMap::iterator lIter = pOwnershipTransferMap.begin(), lEndIter = pOwnershipTransferMap.end();
    
    for(; lIter != lEndIter; ++lIter)
        pmScheduler::GetScheduler()->SendPostTaskOwnershipTransfer(this, lIter->first, lIter->second);
}
    
void pmMemSection::DeleteAllLocalMemSections()
{
    std::vector<pmMemSection*> lMemSections;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

        std::map<std::pair<pmMachine*, ulong>, pmMemSection*>::iterator lIter = mMemSectionMap.begin(), lEndIter = mMemSectionMap.end();
        for(; lIter != lEndIter; ++lIter)
            lMemSections.push_back(lIter->second);
    }

    std::vector<pmMemSection*>::iterator lExternalIter = lMemSections.begin(), lExternalEndIter = lMemSections.end();
    for(; lExternalIter != lExternalEndIter; ++lExternalIter)
        delete *lExternalIter;
}
    
bool pmMemSection::IsLazy()
{
    return ((mMemInfo == INPUT_MEM_READ_ONLY_LAZY) || (mMemInfo == OUTPUT_MEM_READ_WRITE_LAZY));
}

#ifdef ENABLE_MEM_PROFILING
void pmMemSection::RecordMemReceive(size_t pReceiveSize)
{
    FINALIZE_RESOURCE_PTR(dMemProfileLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemProfileLock, Lock(), Unlock());

    mMemReceived += pReceiveSize;
    ++mMemReceiveEvents;
}

void pmMemSection::RecordMemTransfer(size_t pTransferSize)
{
    FINALIZE_RESOURCE_PTR(dMemProfileLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemProfileLock, Lock(), Unlock());

    mMemTransferred += pTransferSize;
    ++mMemTransferEvents;
}
#endif

/* class pmUserMemHandle */
pmUserMemHandle::pmUserMemHandle(pmMemSection* pMemSection)
    : mMemSection(pMemSection)
{
    pMemSection->SetUserMemHandle(this);
}
    
pmUserMemHandle::~pmUserMemHandle()
{
}

void pmUserMemHandle::Reset(pmMemSection* pMemSection)
{
    mMemSection->SetUserMemHandle(NULL);
    mMemSection = pMemSection;
    mMemSection->SetUserMemHandle(this);
}

pmMemSection* pmUserMemHandle::GetMemSection()
{
    return mMemSection;
}
    
};





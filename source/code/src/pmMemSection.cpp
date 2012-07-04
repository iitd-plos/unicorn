
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

#include <string.h>

namespace pm
{

std::map<void*, pmMemSection*> pmMemSection::mMemSectionMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmMemSection::mResourceLock;

/* class pmMemSection */
pmMemSection::pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerBaseMemAddr, bool pIsLazy)
    : mOwner(pOwner?pOwner:PM_LOCAL_MACHINE),
    mUserMemHandle(NULL),
    mRequestedLength(pLength)
{
    pmMemoryManager* lMemoryManager = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager();

#ifdef SUPPORT_LAZY_MEMORY
    mLazy = pIsLazy;
    
    if(mLazy)
        mMem = lMemoryManager->AllocateLazyMemory(pLength, mVMPageCount);
    else
        mMem = lMemoryManager->AllocateMemory(pLength, mVMPageCount);
#else
    mLazy = false;
	mMem = lMemoryManager->AllocateMemory(pLength, mVMPageCount);
#endif

	mAllocatedLength = pLength;

    ResetOwnerships(mOwner, ((mOwner == PM_LOCAL_MACHINE) ? (ulong)mMem : pOwnerBaseMemAddr));
    
	if(mMem)
	{
		FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
		mMemSectionMap[mMem] = this;
	}
}
    
pmMemSection::pmMemSection(const pmMemSection& pMemSection)
{
    mOwner = pMemSection.mOwner;
    mUserMemHandle = NULL;
    mRequestedLength = pMemSection.mRequestedLength;
    mAllocatedLength = pMemSection.mAllocatedLength;
    mVMPageCount = pMemSection.mVMPageCount;
    mLazy = pMemSection.mLazy;
    mMem = NULL;
}
		
pmMemSection::~pmMemSection()
{    
    DisposeMemory();
    
    if(mOwner == PM_LOCAL_MACHINE)
        DeleteAssociations();
}
    
void pmMemSection::DisposeMemory()
{
    if(mMem)
    {
        FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
        mMemSectionMap.erase(mMem);
        
        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(mMem);
        mMem = NULL;
    }
}

void pmMemSection::DeleteAssociations()
{
    DeleteLocalAssociations();
    DeleteRemoteAssociations();
}
    
void pmMemSection::DeleteLocalAssociations()
{
    std::vector<pmMemSection*>::iterator lStart = mLocalAssociations.begin(), lEnd = mLocalAssociations.end();
    for(; lStart != lEnd; ++lStart)
        delete *lStart;
}

void pmMemSection::DeleteRemoteAssociations()
{
}
    
void pmMemSection::CreateLocalAssociation(pmMemSection* pMemSection)
{
    mLocalAssociations.push_back(pMemSection);
}

void pmMemSection::SetUserMemHandle(pmUserMemHandle* pUserMemHandle)
{
    mUserMemHandle = pUserMemHandle;
}

pmUserMemHandle* pmMemSection::GetUserMemHandle()
{
    return mUserMemHandle;
}

void pmMemSection::SwapMemoryAndOwnerships(pmMemSection* pMemSection1, pmMemSection* pMemSection2)
{
    std::swap(pMemSection1->mMem, pMemSection2->mMem);
    std::swap(pMemSection1->mOwnershipMap, pMemSection2->mOwnershipMap);
    
    pMemSection1->mLazyOwnershipMap = pMemSection1->mOwnershipMap;
    pMemSection2->mLazyOwnershipMap = pMemSection2->mOwnershipMap;
    
    FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

    if(pMemSection1->mMem)
        mMemSectionMap[pMemSection1->mMem] = pMemSection1;

    if(pMemSection2->mMem)
        mMemSectionMap[pMemSection2->mMem] = pMemSection2;
}

pmInputMemSection* pmMemSection::ConvertOutputMemSectionToInputMemSection(pmOutputMemSection* pOutputMemSection)
{
    pmInputMemSection* lInputMemSection = new pmInputMemSection(*pOutputMemSection);
    
    SwapMemoryAndOwnerships(lInputMemSection, pOutputMemSection);
    
    pmUserMemHandle* lUserMemHandle = pOutputMemSection->GetUserMemHandle();
    if(lUserMemHandle)
        lUserMemHandle->Reset(lInputMemSection);
    
    lInputMemSection->CreateLocalAssociation(pOutputMemSection);
    
    return lInputMemSection;
}

pmOutputMemSection* pmMemSection::ConvertInputMemSectionToOutputMemSection(pmInputMemSection* pInputMemSection)
{
    pmOutputMemSection* lOutputMemSection = new pmOutputMemSection(*pInputMemSection);
    
    SwapMemoryAndOwnerships(lOutputMemSection, pInputMemSection);
    
    pmUserMemHandle* lUserMemHandle = pInputMemSection->GetUserMemHandle();
    if(lUserMemHandle)
        lUserMemHandle->Reset(lOutputMemSection);
    
    lOutputMemSection->CreateLocalAssociation(pInputMemSection);
    
    return lOutputMemSection;
}

void pmMemSection::ResetOwnerships(pmMachine* pOwner, ulong pBaseAddr)
{
    mOwnershipMap.clear();
    
	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
	lRangeOwner.hostBaseAddr = pBaseAddr;
    lRangeOwner.hostOffset = 0;
    
	mOwnershipMap[0] = std::pair<size_t, vmRangeOwner>(mRequestedLength, lRangeOwner);
    
    if(mLazy)
        mLazyOwnershipMap = mOwnershipMap;    

#ifdef SUPPORT_LAZY_MEMORY
    if(mLazy && (!pOwner || pOwner == PM_LOCAL_MACHINE))
    {
        pmMemoryManager* lMemoryManager = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager();
        lMemoryManager->RemoveLazyProtection(mMem, mRequestedLength);
    }
#endif
}

void* pmMemSection::GetMem()
{
	return mMem;
}

size_t pmMemSection::GetLength()
{
	return mRequestedLength;
}

pmMemSection* pmMemSection::FindMemSection(void* pMem)
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	std::map<void*, pmMemSection*>::iterator lIter = mMemSectionMap.find(pMem);
	if(lIter != mMemSectionMap.end())
		return lIter->second;

	PMTHROW(pmUnrecognizedMemoryException());

	return NULL;
}
    
pmMemSection* pmMemSection::FindMemSectionContainingAddress(void* pPtr)
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
    
    typedef std::map<void*, pmMemSection*> mapType;
    mapType::iterator lStartIter;
    mapType::iterator *lStartIterAddr = &lStartIter;
    
    char* lAddress = static_cast<char*>(pPtr);
    FIND_FLOOR_ELEM(mapType, mMemSectionMap, lAddress, lStartIterAddr);
    
    if(lStartIterAddr)
    {
        char* lMemAddress = static_cast<char*>((void*)(lStartIter->first));
        pmMemSection* lMemSection = static_cast<pmMemSection*>(lStartIter->second);

        size_t lLength = lMemSection->GetLength();
        
        if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
            return lMemSection;
    }
    
    return NULL;
}

pmStatus pmMemSection::SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, ulong pOffset, ulong pLength, bool pIsLazyAcquisition)
{
    if(pIsLazyAcquisition && !IsLazy())
        PMTHROW(pmFatalErrorException());
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = pIsLazyAcquisition ? mLazyOwnershipLock : mOwnershipLock;
    pmMemOwnership& lMap = pIsLazyAcquisition ? mLazyOwnershipMap : mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

#if _DEBUG
#if 0
    if(!pIsLazyAcquisition)
    {
        PrintOwnerships();
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner, Owner Offset, OwnerBaseAddr): (" << pOffset << ", " << pLength << ", " << (uint)(*pOwner) << ", " << pOwnerOffset << ", " << (void*)pOwnerBaseMemAddr << ")" << std::endl;
    }
#endif
#endif
    
	// Remove present ownership
	size_t lLastAddr = pOffset + pLength - 1;
	size_t lOwnerLastAddr = pOwnerOffset + pLength - 1;

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
		if(lStartOwner.host == pOwner && lStartOwner.hostBaseAddr == pOwnerBaseMemAddr && lStartOwner.hostOffset == (pOwnerOffset - (pOffset - lStartOffset)))
        {
            pOwnerOffset -= (pOffset - lStartOffset);
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
                
                if(lPrevOwner.host == pOwner && lPrevOwner.hostBaseAddr == pOwnerBaseMemAddr && lPrevOwner.hostOffset + lPrevLength == pOwnerOffset)
                {
                    pOwnerOffset -= (lStartOffset - lPrevOffset);
                    pOffset = lPrevOffset;		// Combine with previous range                

                    lMap.erase(lPrevIter);
                }
            }
        }
    }

	if(lEndOffset + lEndLength - 1 > lLastAddr)
	{
		if(lEndOwner.host == pOwner && lEndOwner.hostBaseAddr == pOwnerBaseMemAddr && (lEndOwner.hostOffset + (lLastAddr - lEndOffset)) == lOwnerLastAddr)
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
                
                if(lNextOwner.host == pOwner && lNextOwner.hostBaseAddr == pOwnerBaseMemAddr && lNextOwner.hostOffset == lOwnerLastAddr + 1)
                {
                    lLastAddr = lNextOffset + lNextLength - 1;	// Combine with following range

                    lMap.erase(lNextIter);
                }
            }
        }
    }

	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
	lRangeOwner.hostBaseAddr = pOwnerBaseMemAddr;
    lRangeOwner.hostOffset = pOwnerOffset;
	lMap[pOffset] = std::pair<size_t, vmRangeOwner>(lLastAddr - pOffset + 1, lRangeOwner);

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
    
    if(lRangeOwner1.host == lRangeOwner2.host && lRangeOwner1.hostBaseAddr == lRangeOwner2.hostBaseAddr && lRangeOwner1.hostOffset + lLength1 == lRangeOwner2.hostOffset)
        std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Mergable Ranges Found (" << lOffset1 << ", " << lLength1 << ") - (" << lOffset2 << ", " << lLength2 << ") map to (" << lRangeOwner1.hostOffset << ", " << lLength1 << ") - (" << lRangeOwner2.hostOffset << ", " << lLength2 << ") on host " << (uint)(*lRangeOwner1.host) << " Base addr " << mMem << " maps to " << (void*)lRangeOwner1.hostBaseAddr << std::endl;
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
    return SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(GetMem()), pOffset, pOffset, pLength, false);
}

#ifdef SUPPORT_LAZY_MEMORY
uint pmMemSection::GetLazyForwardPrefetchPageCount()
{
    return LAZY_FORWARD_PREFETCH_PAGE_COUNT;
}
    
pmStatus pmMemSection::AcquireOwnershipLazy(ulong pOffset, ulong pLength)
{
    return SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(GetMem()), pOffset, pOffset, pLength, true);
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
    
pmStatus pmMemSection::TransferOwnershipPostTaskCompletion(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    vmRangeOwner lRangeOwner;
    lRangeOwner.host = pOwner;
    lRangeOwner.hostBaseAddr = pOwnerBaseMemAddr;
    lRangeOwner.hostOffset = pOwnerOffset;
    
    pmMemTransferData lTransferData;
    lTransferData.rangeOwner = lRangeOwner;
    lTransferData.offset = pOffset;
    lTransferData.length = pLength;

    mOwnershipTransferVector.push_back(lTransferData);

    return pmSuccess;
}

void pmMemSection::ClearOwnerships()
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());    
        mOwnershipTransferVector.clear();
    }
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        mOwnershipMap.clear();
    }

    if(IsLazy())
    {
        FINALIZE_RESOURCE_PTR(dLazyOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLazyOwnershipLock, Lock(), Unlock());
        mLazyOwnershipMap.clear();
    }
}
    
pmStatus pmMemSection::FlushOwnerships()
{
    if(IsLazy())
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        FINALIZE_RESOURCE_PTR(dLazyOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLazyOwnershipLock, Lock(), Unlock());
        
        mOwnershipMap = mLazyOwnershipMap;
        mLazyOwnershipMap.clear();
    }
        
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    std::vector<pmMemTransferData>::iterator lIter = mOwnershipTransferVector.begin();
    std::vector<pmMemTransferData>::iterator lEnd = mOwnershipTransferVector.end();
    
    for(; lIter != lEnd; ++lIter)
    {
        pmMemTransferData& lTransferData = *lIter;

        if(lTransferData.rangeOwner.host == PM_LOCAL_MACHINE)
            PMTHROW(pmFatalErrorException());

        SetRangeOwner(lTransferData.rangeOwner.host, lTransferData.rangeOwner.hostBaseAddr, lTransferData.rangeOwner.hostOffset, lTransferData.offset, lTransferData.length, false);

#ifdef SUPPORT_LAZY_MEMORY
        if(IsLazy())
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->ApplyLazyProtection((void*)((char*)mMem + lTransferData.offset), lTransferData.length);
#endif
    }
    
    mOwnershipTransferVector.clear();    

	return pmSuccess;
}
    
pmStatus pmMemSection::Fetch(ushort pPriority)
{
    if(IsLazy())
        return pmSuccess;
    
    const std::vector<pmCommunicatorCommandPtr>& lVector = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(mMem, pPriority, 0, GetLength(), false);
    
    std::vector<pmCommunicatorCommandPtr>::const_iterator lIter = lVector.begin();
    std::vector<pmCommunicatorCommandPtr>::const_iterator lEndIter = lVector.end();

    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForFinish();

    return pmSuccess;
}
    
pmStatus pmMemSection::GetOwners(ulong pOffset, ulong pLength, bool pIsLazyRegisteration, pmMemSection::pmMemOwnership& pOwnerships)
{
    if(pIsLazyRegisteration && !IsLazy())
        PMTHROW(pmFatalErrorException());

    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = pIsLazyRegisteration ? mLazyOwnershipLock : mOwnershipLock;
    pmMemOwnership& lMap = pIsLazyRegisteration ? mLazyOwnershipMap : mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

	ulong lLastAddr = pOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, lMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, lMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
    {
        std::cout << pOffset << " " << pLength << " " << lStartIterAddr << " " << lEndIterAddr << " " << lMap.size() << " " << lMap.begin()->first << " " << lMap.begin()->second.first << std::endl;
		PMTHROW(pmFatalErrorException());
    }

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

bool pmMemSection::IsLazy()
{
    return mLazy;
}


/* class pmInputMemSection */
pmInputMemSection::pmInputMemSection(size_t pLength, bool pIsLazy, pmMachine* pOwner /* = NULL */, ulong pOwnerMemSectionAddr /* = 0 */)
	: pmMemSection(pLength, pOwner, pOwnerMemSectionAddr, pIsLazy)
{
}
    
pmInputMemSection::pmInputMemSection(const pmOutputMemSection& pOutputMemSection)
    : pmMemSection(pOutputMemSection)
{
}

pmInputMemSection::~pmInputMemSection()
{
}


/* class pmOutputMemSection */
pmOutputMemSection::pmOutputMemSection(size_t pLength, accessType pAccess, bool pIsLazy, pmMachine* pOwner /* = NULL */, ulong pOwnerMemSectionAddr /* = 0 */)
	: pmMemSection(pLength, pOwner, pOwnerMemSectionAddr, pIsLazy)
{
	mAccess = pAccess;
}
    
pmOutputMemSection::pmOutputMemSection(const pmInputMemSection& pInputMemSection)
    : pmMemSection(pInputMemSection)
{
    mAccess = READ_WRITE;
}

pmOutputMemSection::~pmOutputMemSection()
{
}

pmStatus pmOutputMemSection::Update(size_t pOffset, size_t pLength, void* pSrcAddr)
{
	void* lDestAddr = (void*)((char*)GetMem() + pOffset);
	memcpy(lDestAddr, pSrcAddr, pLength);

	return pmSuccess;
}

pmOutputMemSection::accessType pmOutputMemSection::GetAccessType()
{
	return mAccess;
}
    

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





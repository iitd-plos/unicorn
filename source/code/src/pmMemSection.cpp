
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
{
	mRequestedLength = pLength;
    
    pmMemoryManager* lMemoryManager = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager();

#ifdef USE_LAZY_MEMORY
    mLazy = pIsLazy;
    
    if(mLazy)
    {
        mMem = lMemoryManager->AllocateLazyMemory(pLength, mVMPageCount);
        
        if(!pOwner || pOwner == PM_LOCAL_MACHINE)
            lMemoryManager->RemoveLazyProtection(mMem, pLength);
    }
    else
    {
        mMem = lMemoryManager->AllocateMemory(pLength, mVMPageCount);
    }
#else
    mLazy = false;
	mMem = lMemoryManager->AllocateMemory(pLength, mVMPageCount);
#endif

	mAllocatedLength = pLength;

	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner?pOwner:PM_LOCAL_MACHINE;
	lRangeOwner.hostBaseAddr = pOwner?pOwnerBaseMemAddr:(ulong)mMem;

	mOwnershipMap[0] = std::pair<size_t, vmRangeOwner>(pLength, lRangeOwner);
	if(mMem)
	{
		FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
		mMemSectionMap[mMem] = this;
	}

    if(mLazy)
        mLazyOwnershipMap = mOwnershipMap;
}
		
pmMemSection::~pmMemSection()
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
	mMemSectionMap.erase(mMem);

	MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(mMem);
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

pmStatus pmMemSection::SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength, bool pIsLazyAcquisition)
{
    if(pIsLazyAcquisition && !IsLazy())
        PMTHROW(pmFatalErrorException());
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = pIsLazyAcquisition ? mLazyOwnershipLock : mOwnershipLock;
    pmMemOwnership& lMap = pIsLazyAcquisition ? mLazyOwnershipMap : mOwnershipMap;
    
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
	// Remove present ownership
	size_t lLastAddr = pOffset + pLength - 1;

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
		if(lStartOwner.host == pOwner && lStartOwner.hostBaseAddr == pOwnerBaseMemAddr)
			pOffset = lStartOffset;		// Combine with previous range
		else
			lMap[lStartOffset] = std::pair<size_t, vmRangeOwner>(pOffset-lStartOffset, lStartOwner);
	}

	if(lEndOffset + lEndLength - 1 > lLastAddr)
	{
		if(lEndOwner.host == pOwner && lEndOwner.hostBaseAddr == pOwnerBaseMemAddr)
			lLastAddr = lEndOffset + lEndLength - 1;	// Combine with following range
		else
			lMap[lLastAddr + 1] = std::pair<size_t, vmRangeOwner>(lEndOffset + lEndLength - 1 - lLastAddr, lEndOwner);
	}

	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
	lRangeOwner.hostBaseAddr = pOwnerBaseMemAddr;
	lMap[pOffset] = std::pair<size_t, vmRangeOwner>(lLastAddr - pOffset + 1, lRangeOwner);

	return pmSuccess;
}

pmStatus pmMemSection::AcquireOwnershipImmediate(ulong pOffset, ulong pLength)
{
    return SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(GetMem()), pOffset, pLength, false);
}

#ifdef USE_LAZY_MEMORY
pmStatus pmMemSection::AcquireOwnershipLazy(ulong pOffset, ulong pLength)
{
    return SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(GetMem()), pOffset, pLength, true);
}
#endif
    
pmStatus pmMemSection::TransferOwnershipPostTaskCompletion(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

    vmRangeOwner lRangeOwner;
    lRangeOwner.host = pOwner;
    lRangeOwner.hostBaseAddr = pOwnerBaseMemAddr;
    
    pmMemTransferData lTransferData;
    lTransferData.rangeOwner = lRangeOwner;
    lTransferData.offset = pOffset;
    lTransferData.length = pLength;

    mOwnershipTransferVector.push_back(lTransferData);

    return pmSuccess;
}

pmStatus pmMemSection::FlushOwnerships()
{
    if(IsLazy())
    {
        FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
        FINALIZE_RESOURCE_PTR(dLazyOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLazyOwnershipLock, Lock(), Unlock());
        
        mOwnershipMap = mLazyOwnershipMap;
    }

	FINALIZE_RESOURCE_PTR(dTransferLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());
    
    std::vector<pmMemTransferData>::iterator lIter = mOwnershipTransferVector.begin();
    std::vector<pmMemTransferData>::iterator lEnd = mOwnershipTransferVector.end();
    
    for(; lIter != lEnd; ++lIter)
    {
        pmMemTransferData& lTransferData = *lIter;

        if(lTransferData.rangeOwner.host == PM_LOCAL_MACHINE)
            PMTHROW(pmFatalErrorException());

        SetRangeOwner(lTransferData.rangeOwner.host, lTransferData.rangeOwner.hostBaseAddr, lTransferData.offset, lTransferData.length, false);

#ifdef USE_LAZY_MEMORY
        if(IsLazy())
        {
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->ApplyLazyProtection((void*)((char*)mMem + lTransferData.offset), lTransferData.length);
        }
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
		PMTHROW(pmFatalErrorException());

    size_t lSpan = lStartIter->first + lStartIter->second.first - 1;
    if(lLastAddr < lSpan)
    {
        lSpan = lLastAddr;
        assert(lStartIter == lEndIter);
    }
    
	pOwnerships[pOffset] = std::make_pair(lSpan - pOffset + 1, lStartIter->second.second);
	
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
                assert(lIter == lEndIter);
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

pmInputMemSection::~pmInputMemSection()
{
}


/* class pmOutputMemSection */
pmOutputMemSection::pmOutputMemSection(size_t pLength, accessType pAccess, bool pIsLazy, pmMachine* pOwner /* = NULL */, ulong pOwnerMemSectionAddr /* = 0 */)
	: pmMemSection(pLength, pOwner, pOwnerMemSectionAddr, pIsLazy)
{
	mAccess = pAccess;
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

};


#include "pmMemSection.h"
#include "pmMemoryManager.h"
#include "pmHardware.h"

#include <string.h>

namespace pm
{

std::map<void*, pmMemSection*> pmMemSection::mMemSectionMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmMemSection::mResourceLock;

/* class pmMemSection */
pmMemSection::pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerBaseMemAddr)
{
	mRequestedLength = pLength;

#ifdef USE_LAZY_MEMORY
	mMem = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->AllocateLazyMemory(pLength, mVMPageCount);
#else
	mMem = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->AllocateMemory(pLength, mVMPageCount);
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

pmStatus pmMemSection::SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    pmMemOwnership& lMap = mOwnershipMap;
    
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
    return SetRangeOwner(PM_LOCAL_MACHINE, (ulong)(GetMem()), pOffset, pLength);
}
    
pmStatus pmMemSection::TransferOwnershipPostTaskCompletion(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

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
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipTransferLock, Lock(), Unlock());

    std::vector<pmMemTransferData>::iterator lIter = mOwnershipTransferVector.begin();
    std::vector<pmMemTransferData>::iterator lEnd = mOwnershipTransferVector.end();
    
    for(; lIter != lEnd; ++lIter)
    {
        pmMemTransferData& lTransferData = *lIter;

        SetRangeOwner(lTransferData.rangeOwner.host, lTransferData.rangeOwner.hostBaseAddr, lTransferData.offset, lTransferData.length);
    }
    
    mOwnershipTransferVector.clear();

	return pmSuccess;
}

pmStatus pmMemSection::Fetch(ushort pPriority)
{
    const std::vector<pmCommunicatorCommandPtr>& lVector = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FetchMemoryRegion(this, pPriority, 0, GetLength());
    
    std::vector<pmCommunicatorCommandPtr>::const_iterator lIter = lVector.begin();
    std::vector<pmCommunicatorCommandPtr>::const_iterator lEndIter = lVector.end();

    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForFinish();

    return pmSuccess;
}

pmStatus pmMemSection::GetOwners(ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    pmMemOwnership& lMap = mOwnershipMap;

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

/* class pmInputMemSection */
pmInputMemSection::pmInputMemSection(size_t pLength, pmMachine* pOwner /* = NULL */, ulong pOwnerMemSectionAddr /* = 0 */)
	: pmMemSection(pLength, pOwner, pOwnerMemSectionAddr)
{
}

pmInputMemSection::~pmInputMemSection()
{
}


/* class pmOutputMemSection */
pmOutputMemSection::pmOutputMemSection(size_t pLength, accessType pAccess, pmMachine* pOwner /* = NULL */, ulong pOwnerMemSectionAddr /* = 0 */)
	: pmMemSection(pLength, pOwner, pOwnerMemSectionAddr)
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

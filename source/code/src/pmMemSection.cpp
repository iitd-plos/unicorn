
#include "pmMemSection.h"
#include "pmMemoryManager.h"

#include <string.h>

namespace pm
{

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
		mResourceLock.Lock();
		mMemSectionMap[mMem] = this;
		mResourceLock.Unlock();
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

	throw pmUnrecognizedMemoryException();

	return NULL;
}

pmStatus pmMemSection::SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

	if(mShadowOwnershipMap.empty())
		mShadowOwnershipMap = mOwnershipMap;

	// Remove present ownership
	size_t lLastAddr = pOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, mShadowOwnershipMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, mShadowOwnershipMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		throw pmFatalErrorException();

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

	mShadowOwnershipMap.erase(lStartIter, lEndIter);

	if(lStartOffset < pOffset)
	{
		if(lStartOwner.host == pOwner && lStartOwner.hostBaseAddr == pOwnerBaseMemAddr)
			pOffset = lStartOffset;		// Combine with previous range
		else
			mShadowOwnershipMap[lStartOffset] = std::pair<size_t, vmRangeOwner>(pOffset-lStartOffset, lStartOwner);
	}

	if(lEndOffset + lEndLength - 1 > lLastAddr)
	{
		if(lEndOwner.host == pOwner && lEndOwner.hostBaseAddr == pOwnerBaseMemAddr)
			lLastAddr = lEndOffset + lEndLength - 1;	// Combine with following range
		else
			mShadowOwnershipMap[lLastAddr + 1] = std::pair<size_t, vmRangeOwner>(lEndOffset + lEndLength - 1 - lLastAddr, lEndOwner);
	}

	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
	lRangeOwner.hostBaseAddr = pOwnerBaseMemAddr;
	mShadowOwnershipMap[pOffset] = std::pair<size_t, vmRangeOwner>(lLastAddr - pOffset + 1, lRangeOwner);

	return pmSuccess;
}

pmStatus pmMemSection::FlushOwnerships()
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

	if(!mShadowOwnershipMap.empty())
	{
		mOwnershipMap = mShadowOwnershipMap;
		mShadowOwnershipMap.clear();
	}

	return pmSuccess;
}

pmStatus pmMemSection::GetOwners(ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

	pmMemOwnership* lTargetMap = NULL;
	if(mShadowOwnershipMap.empty())
		lTargetMap = &mOwnershipMap;
	else
		lTargetMap = &mShadowOwnershipMap;

	pmMemOwnership lMap = *lTargetMap;

	ulong lLastAddr = pOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, lMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, lMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		throw pmFatalErrorException();

	pOwnerships[pOffset] = lStartIter->second;
	
	pmMemOwnership::iterator lIter = lStartIter;
	++lIter;

	if(lStartIter != lEndIter)
	{
		for(; lIter != lEndIter; ++lIter)
			pOwnerships[lIter->first] = lIter->second;
	}

	vmRangeOwner lLastOwner = lEndIter->second.second;
	pOwnerships[lEndIter->first] = std::pair<size_t, vmRangeOwner>(lLastAddr - lEndIter->first + 1, lLastOwner);

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

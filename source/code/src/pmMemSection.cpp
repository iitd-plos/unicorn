
#include "pmMemSection.h"
#include "pmMemoryManager.h"

namespace pm
{

/* class pmMemSection */
pmMemSection::pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerMemSectionAddr)
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
	lRangeOwner.hostMemSection = pOwner?pOwnerMemSectionAddr:(ulong)this;
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

pmStatus pmMemSection::SetRangeOwner(pmMachine* pOwner, ulong pOwnerMemSectionAddr, ulong pOffset, ulong pLength)
{
	FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());



	vmRangeOwner lRangeOwner;
	lRangeOwner.host = pOwner;
	lRangeOwner.hostMemSection = pOwnerMemSectionAddr;
	mOwnershipMap[pOffset] = std::pair<size_t, vmRangeOwner>(pLength, lRangeOwner);

	return pmSuccess;
}

pmStatus pmMemSection::GetOwner(pmMachine*& pHost, ulong& pAddr)
{
	FINALIZE_RESOURCE_PTR(dSectionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mSectionLock, Lock(), Unlock());
	pHost = mMemOrganizationMap[0].host;
	pAddr = mMemOrganizationMap[0].addr;

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


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

#include "pmPoolAllocator.h"
#include "pmMemoryManager.h"

namespace pm
{

pmPoolAllocator::pmPoolAllocator(size_t pIndividualAllocationSize, size_t pMaxAllocations, bool pPageAlignedAllocations)
    : mIndividualAllocationSize(pIndividualAllocationSize)
    , mMaxAllocations(pMaxAllocations)
    , mPageAlignedAllocations(pPageAlignedAllocations)
    , mMasterAllocation(NULL)
    , mResourceLock __LOCK_NAME__("pmPoolAllocator::mResourceLock")
{
    if(!mIndividualAllocationSize || !mMaxAllocations)
        PMTHROW(pmFatalErrorException());
    
    if(mPageAlignedAllocations)
    {
        size_t lPageCount = 0;
        mIndividualAllocationSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->FindAllocationSize(mIndividualAllocationSize, lPageCount);
    }
}
    
pmPoolAllocator::~pmPoolAllocator()
{
    if(mMasterAllocation)
    {
        if(mPageAlignedAllocations)
            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->DeallocateMemory(mMasterAllocation);
        else
            free(mMasterAllocation);
    }
}
    
void* pmPoolAllocator::Allocate(size_t pSize)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    if(mMasterAllocation)
    {
        if(pSize > mIndividualAllocationSize)
            return NULL;
    }
    else
    {
        DEBUG_EXCEPTION_ASSERT(mUnallocatedPool.empty());
        
        if(mPageAlignedAllocations)
            mMasterAllocation = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CreateCheckOutMemory(mMaxAllocations * mIndividualAllocationSize);
        else
            mMasterAllocation = malloc(mMaxAllocations * mIndividualAllocationSize);

        mUnallocatedPool.reserve(mMaxAllocations);
        for(size_t i = 0, lAddr = reinterpret_cast<size_t>(mMasterAllocation); i < mMaxAllocations; ++i, lAddr += mIndividualAllocationSize)
            mUnallocatedPool.push_back(reinterpret_cast<void*>(lAddr));
    }
    
    if(!mUnallocatedPool.empty())
    {
        void* lMem = mUnallocatedPool.back();
        mUnallocatedPool.pop_back();
        
        return lMem;
    }
    
    return NULL;
}

void pmPoolAllocator::Deallocate(void* pMem)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    DEBUG_EXCEPTION_ASSERT(std::find(mUnallocatedPool.begin(), mUnallocatedPool.end(), pMem) == mUnallocatedPool.end());
    DEBUG_EXCEPTION_ASSERT(reinterpret_cast<size_t>(pMem) >= reinterpret_cast<size_t>(mMasterAllocation));
    DEBUG_EXCEPTION_ASSERT(reinterpret_cast<size_t>(pMem) < reinterpret_cast<size_t>(mMasterAllocation) + (mMaxAllocations * mIndividualAllocationSize))
    DEBUG_EXCEPTION_ASSERT(((reinterpret_cast<size_t>(pMem) - reinterpret_cast<size_t>(mMasterAllocation)) % mIndividualAllocationSize) == 0);
    
    mUnallocatedPool.push_back(pMem);
}
    
bool pmPoolAllocator::HasNoAllocations()
{
    return (mUnallocatedPool.size() == mMaxAllocations);
}
    
}

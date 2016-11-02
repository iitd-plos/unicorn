
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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
            pmBase::DeallocateMemory(mMasterAllocation);
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
            mMasterAllocation = pmBase::AllocateMemory(mMaxAllocations * mIndividualAllocationSize);

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

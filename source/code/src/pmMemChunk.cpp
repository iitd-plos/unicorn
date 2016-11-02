
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

#include "pmMemChunk.h"

namespace pm
{
 
pmMemChunk::pmMemChunk(void* pChunk, size_t pSize)
    : mChunk(pChunk)
    , mSize(pSize)
    , mResourceLock __LOCK_NAME__("pmMemChunk::mResourceLock")
{
    mFree.insert(std::make_pair(pSize, 0));
    mFreeBlocks[pChunk] = pSize;
}
    
const void* pmMemChunk::GetChunk() const
{
    return mChunk;
}

void* pmMemChunk::Allocate(size_t pSize, size_t pAlignment)
{
    if(!pSize)
        return NULL;

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    std::multimap<size_t, size_t>::reverse_iterator lFreeIter = mFree.rbegin(), lFreeEndIter = mFree.rend();
    for(; lFreeIter != lFreeEndIter; ++lFreeIter)
    {
        if(lFreeIter->first < pSize)
            return NULL;
        
        size_t lAddr = reinterpret_cast<size_t>(mChunk) + lFreeIter->second;
        size_t lHighestVal = pAlignment - 1;
        size_t lPossibleAllocation = ((lAddr + lHighestVal) & ~lHighestVal);
        
        size_t lLastReqdAddr = lPossibleAllocation + pSize - 1;
        size_t lLastAvailableAddr = lAddr + lFreeIter->first - 1;
        if(lLastReqdAddr <= lLastAvailableAddr)
        {
            size_t lSecond = lFreeIter->second;
            
            mFreeBlocks.erase(reinterpret_cast<void*>(lAddr));
            mFree.erase(--lFreeIter.base());
            
            if(lPossibleAllocation != lAddr)
            {
                mFreeBlocks[reinterpret_cast<void*>(lAddr)] = lPossibleAllocation - lAddr;
                mFree.insert(std::make_pair(lPossibleAllocation - lAddr, lSecond));
            }
            
            if(lLastReqdAddr != lLastAvailableAddr)
            {
                mFreeBlocks[reinterpret_cast<void*>(lLastReqdAddr + 1)] = lLastAvailableAddr - lLastReqdAddr;
                mFree.insert(std::make_pair(lLastAvailableAddr - lLastReqdAddr, lLastReqdAddr + 1 - reinterpret_cast<size_t>(mChunk)));
            }
            
            void* lAllocation = reinterpret_cast<void*>(lPossibleAllocation);
            mAllocations[lAllocation] = pSize;

            return lAllocation;
        }
    }
    
    return NULL;
}
    
void pmMemChunk::Deallocate(void* pPtr)
{
    if(!pPtr)
        return;
    
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::map<void*, size_t>::iterator lIter = mAllocations.find(pPtr);
    EXCEPTION_ASSERT(lIter != mAllocations.end());
    
    size_t lSize = lIter->second;
    mAllocations.erase(lIter);
    
    size_t lAddr = reinterpret_cast<size_t>(pPtr);
    size_t lEndAddr = lAddr + lSize;
    
    std::map<void*, size_t>::iterator lFreeBlockIter = mFreeBlocks.find(reinterpret_cast<void*>(lEndAddr));
    if(lFreeBlockIter != mFreeBlocks.end())
    {
        size_t lNextSize = lFreeBlockIter->second;
        mFreeBlocks.erase(lFreeBlockIter);

        std::multimap<size_t, size_t>::iterator lFreeIter = mFree.find(lNextSize), lFreeEndIter = mFree.end();
        for(; lFreeIter != lFreeEndIter; ++lFreeIter)
        {
            DEBUG_EXCEPTION_ASSERT(lFreeIter->first == lNextSize);
            
            if(lFreeIter->second == (lEndAddr - reinterpret_cast<size_t>(mChunk)))
            {
                mFree.erase(lFreeIter);
                break;
            }
        }
        
        lSize += lNextSize;
        lEndAddr += lNextSize;
    }
    
    if(lAddr != reinterpret_cast<size_t>(mChunk))
    {
        size_t lFormerAddr = lAddr - 1;
        std::map<void*, size_t>::iterator lUpperIter = mFreeBlocks.upper_bound(reinterpret_cast<void*>(lFormerAddr));
        if(lUpperIter != mFreeBlocks.begin())
        {
            --lUpperIter;
            
            if(reinterpret_cast<size_t>(lUpperIter->first) + lUpperIter->second == lAddr)
            {
                size_t lPrevAddr = reinterpret_cast<size_t>(lUpperIter->first);
                size_t lPrevSize = lUpperIter->second;
                mFreeBlocks.erase(lUpperIter);
                
                std::multimap<size_t, size_t>::iterator lFreeIter = mFree.find(lPrevSize), lFreeEndIter = mFree.end();
                for(; lFreeIter != lFreeEndIter; ++lFreeIter)
                {
                    DEBUG_EXCEPTION_ASSERT(lFreeIter->first == lPrevSize);
                    
                    if(lFreeIter->second == (lPrevAddr - reinterpret_cast<size_t>(mChunk)))
                    {
                        mFree.erase(lFreeIter);
                        break;
                    }
                }
                
                lSize += lPrevSize;
                lAddr = lPrevAddr;
            }
        }
    }
    
    mFreeBlocks[reinterpret_cast<void*>(lAddr)] = lSize;
    mFree.insert(std::make_pair(lSize, lAddr - reinterpret_cast<size_t>(mChunk)));
}
    
bool pmMemChunk::HasNoAllocations()
{
    return (GetBiggestAvaialbleContiguousAllocation() == GetSize());
}
    
size_t pmMemChunk::GetBiggestAvaialbleContiguousAllocation()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(mFree.empty())
        return 0;
    
    return mFree.rbegin()->first;
}
    
size_t pmMemChunk::GetSize() const
{
    return mSize;
}

}




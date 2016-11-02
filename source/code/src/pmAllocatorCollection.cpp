
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

namespace pm
{

template<typename __allocator, bool>
struct CallAllocate
{};
    
template<typename __allocator>
struct CallAllocate<__allocator, true>
{
    void* operator()(std::shared_ptr<__allocator>& pAllocator, size_t pSize, size_t pAlignment)
    {
        return pAllocator->Allocate(pSize, pAlignment);
    }
};

template<typename __allocator>
struct CallAllocate<__allocator, false>
{
    void* operator()(std::shared_ptr<__allocator>& pAllocator, size_t pSize, size_t pAlignment)
    {
        return pAllocator->Allocate(pSize);
    }
};

template<typename __allocator_traits>
inline void pmAllocatorCollection<__allocator_traits>::SetChunkSizeMultiplier(size_t pMultiplier)
{
    DEBUG_EXCEPTION_ASSERT(pMultiplier && !mChunkSizeMultiplier);

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mChunkSizeMultiplier = pMultiplier;
}

template<typename __allocator_traits>
void* pmAllocatorCollection<__allocator_traits>::Allocate(size_t pSize, size_t pAlignment)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    void* lPtr = NULL;
    
    typename decltype(mMemChunksList)::iterator lIter = mMemChunksList.begin(), lEndIter = mMemChunksList.end();
    for(; lIter != lEndIter ; ++lIter)
    {
        lPtr = CallAllocate<typename __allocator_traits::allocator, __allocator_traits::alignedAllocations>()(*lIter, pSize, pAlignment);
        
        if(lPtr)
        {
            mAllocatedPtrs[lPtr] = lIter;
            break;
        }
    }
    
    if(!lPtr)
    {
        DEBUG_EXCEPTION_ASSERT(mChunkSizeMultiplier);

        size_t lTempSize = pSize + ((pAlignment > 1) ? pAlignment : 0);
        
        size_t lChunkSize = (((lTempSize / mChunkSizeMultiplier) + ((lTempSize % mChunkSizeMultiplier) ? 1 : 0)) * mChunkSizeMultiplier);
        std::shared_ptr<typename __allocator_traits::allocator> lNewChunk = typename __allocator_traits::creator()(lChunkSize);
        
        if(!lNewChunk.get())
            PMTHROW_NODUMP(pmOutOfMemoryException());

        mMemChunksList.emplace_back(std::move(lNewChunk));
        typename decltype(mMemChunksList)::iterator lChunkIter = --mMemChunksList.end();
        
        lPtr = CallAllocate<typename __allocator_traits::allocator, __allocator_traits::alignedAllocations>()(*lChunkIter, pSize, pAlignment);
        
        EXCEPTION_ASSERT(lPtr);
        
        mAllocatedPtrs[lPtr] = lChunkIter;
    }

    return lPtr;
}

template<typename __allocator_traits>
void* pmAllocatorCollection<__allocator_traits>::Allocate(size_t pSize)
{
    return Allocate(pSize, 1);
}
    
template<typename __allocator_traits>
void* pmAllocatorCollection<__allocator_traits>::AllocateNoThrow(size_t pSize, size_t pAlignment)
{
    try
    {
        return Allocate(pSize, pAlignment);
    }
    catch(pmOutOfMemoryException&)
    {}
    
    return NULL;
}

template<typename __allocator_traits>
void* pmAllocatorCollection<__allocator_traits>::AllocateNoThrow(size_t pSize)
{
    try
    {
        return Allocate(pSize);
    }
    catch(pmOutOfMemoryException&)
    {}
    
    return NULL;
}


template<typename __allocator_traits>
inline void pmAllocatorCollection<__allocator_traits>::Deallocate(void* pPtr)
{
    EXCEPTION_ASSERT(pPtr);

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    typename decltype(mAllocatedPtrs)::iterator lIter = mAllocatedPtrs.find(pPtr);
    EXCEPTION_ASSERT(lIter != mAllocatedPtrs.end());

    typename decltype(mMemChunksList)::iterator lChunkIter = lIter->second;
    mAllocatedPtrs.erase(lIter);

    (*lChunkIter)->Deallocate(pPtr);

    if((*lChunkIter)->HasNoAllocations())
    {
        typename __allocator_traits::destructor()((*lChunkIter));
        mMemChunksList.erase(lChunkIter);
    }
}
    
template<typename __allocator_traits>
inline void pmAllocatorCollection<__allocator_traits>::Reset()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    for_each(mMemChunksList, [] (std::shared_ptr<typename __allocator_traits::allocator>& pChunkPtr)
    {
        typename __allocator_traits::destructor()(pChunkPtr);
    });
    
    mAllocatedPtrs.clear();
    mMemChunksList.clear();
}
    
};


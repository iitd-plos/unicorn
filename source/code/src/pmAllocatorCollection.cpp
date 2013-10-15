
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
    DEBUG_EXCEPTION_ASSERT(pMultiplier);

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
            PMTHROW(pmOutOfMemoryException());

        mMemChunksList.emplace_back(std::move(lNewChunk));
        typename decltype(mMemChunksList)::iterator lChunkIter = --mMemChunksList.end();
        
        lPtr = CallAllocate<typename __allocator_traits::allocator, __allocator_traits::alignedAllocations>()(*lChunkIter, pSize, pAlignment);
        
        if(!lPtr)
            PMTHROW_NODUMP(pmOutOfMemoryException());
        
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
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename decltype(mAllocatedPtrs)::iterator lIter = mAllocatedPtrs.find(pPtr);
    if(lIter == mAllocatedPtrs.end())
        PMTHROW(pmFatalErrorException());

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


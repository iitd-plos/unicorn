
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

#ifndef __PM_ALLOCATOR_COLLECTION__
#define __PM_ALLOCATOR_COLLECTION__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <map>
#include <list>
#include <memory>

namespace pm
{

/* __allocator_traits must define
(a) Two functors
 1. creator - To create a new chunk. It should define std::shared_ptr<__allocator_traits::allocator> operator()(size_t size)
 2. destructor - To destroy an existing chunk. It should define void operator()(const std::shared_ptr<__allocator_traits::allocator>&)
 
(b) an allocator that defines the following functions
 1. void* Allocate(size_t size, size_t alignment) or void* Allocate(size_t size)
 2. void Deallocate(void*)
 3. bool HasNoAllocations()

(c) static const bool alignedAllocations
*/
    
template<typename __allocator_traits>
class pmAllocatorCollection : public pmBase
{
public:
    pmAllocatorCollection(size_t pChunkSizeMultiplier = 0)
    : mChunkSizeMultiplier(pChunkSizeMultiplier)
    {}
    
    ~pmAllocatorCollection()
    {
        Reset();
    }

    void SetChunkSizeMultiplier(size_t pMultiplier);
    
    void* Allocate(size_t pSize, size_t pAlignment);
    void* Allocate(size_t pSize);

    void* AllocateNoThrow(size_t pSize, size_t pAlignment);
    void* AllocateNoThrow(size_t pSize);

    void Deallocate(void* pPtr);
    
    void Reset();
    
private:
    size_t mChunkSizeMultiplier;
    
    std::list<std::shared_ptr<typename __allocator_traits::allocator>> mMemChunksList;
    std::map<void*, typename std::list<std::shared_ptr<typename __allocator_traits::allocator>>::iterator> mAllocatedPtrs;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#include "../src/pmAllocatorCollection.cpp"

#endif

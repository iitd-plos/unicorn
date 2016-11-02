
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

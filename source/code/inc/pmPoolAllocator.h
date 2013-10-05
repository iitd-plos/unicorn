
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

#ifndef __PM_POOL_ALLOCATOR__
#define __PM_POOL_ALLOCATOR__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <vector>

namespace pm
{
    
class pmPoolAllocator : public pmBase
{
public:
    pmPoolAllocator(size_t pIndividualAllocationSize, size_t pMaxAllocations, bool pPageAlignedAllocations);
    ~pmPoolAllocator();
    
    void* Allocate(size_t pSize);
    void Deallocate(void* pMem);
    
private:
    size_t mIndividualAllocationSize;
    size_t mMaxAllocations;
    bool mPageAlignedAllocations;
    
    void* mMasterAllocation;
    std::vector<void*> mUnallocatedPool;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};
    
} // end namespace pm

#endif

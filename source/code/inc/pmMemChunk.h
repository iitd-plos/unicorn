
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

#ifndef __PM_MEM_CHUNK__
#define __PM_MEM_CHUNK__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <map>

namespace pm
{

class pmMemChunk : public pmBase
{
public:
    pmMemChunk(void* pChunk, size_t pSize);
    ~pmMemChunk();

    void* GetChunk();
    
    void* Allocate(size_t pSize, size_t pAlignment);
    void Deallocate(void* pPtr);
    
    size_t GetBiggestAvaialbleContiguousAllocation();
    
private:
    void* mChunk;
    size_t mSize;
    
    std::map<void*, size_t> mAllocations;
    std::map<void*, size_t> mFreeBlocks;
    std::multimap<size_t, size_t> mFree; // free size versus offset
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif


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
    
    void* Allocate(size_t pSize, size_t pAlignment);
    void Deallocate(void* pPtr);
    
    const void* GetChunk() const;
    bool HasNoAllocations();
    
private:
    size_t GetBiggestAvaialbleContiguousAllocation();
    size_t GetSize() const;

    const void* mChunk;
    const size_t mSize;
    
    std::map<void*, size_t> mAllocations;
    std::map<void*, size_t> mFreeBlocks;
    std::multimap<size_t, size_t> mFree; // free size versus offset
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

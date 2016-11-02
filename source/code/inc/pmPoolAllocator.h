
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
    
    bool HasNoAllocations();
    
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

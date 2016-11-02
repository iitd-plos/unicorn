
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

#ifndef __PM_MEMORY_MANAGER__
#define __PM_MEMORY_MANAGER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"
#include "pmMemoryDirectory.h"

#include <map>
#include <stdlib.h>

#include VM_IMPLEMENTATION_HEADER1

#ifdef VM_IMPLEMENTATION_HEADER2
#include VM_IMPLEMENTATION_HEADER2
#endif

#ifdef VM_IMPLEMENTATION_HEADER3
#include VM_IMPLEMENTATION_HEADER3
#endif

namespace pm
{

class pmMachine;
extern pmMachine* PM_LOCAL_MACHINE;

#ifdef SUPPORT_LAZY_MEMORY
	void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);
#endif
    
/**
 * \brief Memory Management Routines and Virtual Memory Optimizations
 * This class provides an implementation of lazy arrays which fill their VM pages on access.
*/
class pmMemoryManager : public pmBase
{
	public:
		virtual ~pmMemoryManager() {}
    
		virtual void* AllocateMemory(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount) = 0;
        virtual void DeallocateMemory(pmAddressSpace* pAddressSpace) = 0;
        virtual void DeallocateMemory(void* pMem) = 0;

        virtual uint GetMemoryFetchEvents(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength) = 0;
        virtual ulong GetMemoryFetchPages(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength) = 0;
        virtual void FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector) = 0;

        virtual uint GetScatteredMemoryFetchEvents(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo) = 0;
        virtual ulong GetScatteredMemoryFetchPages(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo) = 0;
    
    #ifdef CENTRALIZED_AFFINITY_COMPUTATION
        virtual uint GetScatteredMemoryFetchEventsForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine) = 0;
        virtual ulong GetScatteredMemoryFetchPagesForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine) = 0;
    #endif

        virtual void FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector) = 0;
    
        virtual size_t GetVirtualMemoryPageSize() const = 0;
        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount) = 0;

        virtual void* CreateCheckOutMemory(size_t pLength) = 0;
    
#ifdef SUPPORT_LAZY_MEMORY
        virtual void* CreateReadOnlyMemoryMapping(pmAddressSpace* pAddressSpace) = 0;
        virtual void DeleteReadOnlyMemoryMapping(void* pReadOnlyMemoryMapping, size_t pLength) = 0;
        virtual void SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed) = 0;
#endif

	protected:
		size_t mPageSize;
};


namespace linuxMemManager
{
    typedef struct addressSpaceSpecifics
    {
        addressSpaceSpecifics();

        int mSharedMemDescriptor;
    } addressSpaceSpecifics;
}
    
class pmLinuxMemoryManager : public pmMemoryManager
{
    private:
        friend void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);

        class sharedMemAutoPtr
        {
            public:
                sharedMemAutoPtr(const char* pSharedMemName);
                ~sharedMemAutoPtr();
        
            private:
                const char* mSharedMemName;
        };
    
	public:
        static pmMemoryManager* GetMemoryManager();

		virtual void* AllocateMemory(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount);
        virtual void DeallocateMemory(pmAddressSpace* pAddressSpace);
        virtual void DeallocateMemory(void* pMem);

        virtual uint GetMemoryFetchEvents(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength);
        virtual ulong GetMemoryFetchPages(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength);
        virtual void FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector);

        virtual uint GetScatteredMemoryFetchEvents(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);
        virtual ulong GetScatteredMemoryFetchPages(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);
    
    #ifdef CENTRALIZED_AFFINITY_COMPUTATION
        virtual uint GetScatteredMemoryFetchEventsForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine);
        virtual ulong GetScatteredMemoryFetchPagesForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine);
    #endif

        virtual void FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector);

        virtual size_t GetVirtualMemoryPageSize() const;

        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount);	// Allocation size must be a multiple of page size
    
        void InstallSegFaultHandler();
		void UninstallSegFaultHandler();
    
    private:
		pmLinuxMemoryManager();
		virtual ~pmLinuxMemoryManager();

        void CreateAddressSpaceSpecifics(pmAddressSpace* pAddressSpace, int pSharedMemDescriptor);
        linuxMemManager::addressSpaceSpecifics& GetAddressSpaceSpecifics(pmAddressSpace* pAddressSpace);
    
        void* AllocatePageAlignedMemoryInternal(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount, int& pSharedMemDescriptor);

        void FetchNonOverlappingMemoryRegion(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, communicator::memoryTransferType pTransferType, size_t pOffset, size_t pLength, size_t pStep, size_t pCount, const vmRangeOwner& pRangeOwner, const pmCommandPtr& pCommand);

        void FetchNonOverlappingScatteredMemoryRegions(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, std::vector<std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>>& pVector, std::vector<pmCommandPtr>& pCommandVector);

        virtual void* CreateCheckOutMemory(size_t pLength);

#ifdef SUPPORT_LAZY_MEMORY
    public:
        virtual void* CreateReadOnlyMemoryMapping(pmAddressSpace* pAddressSpace);
        virtual void DeleteReadOnlyMemoryMapping(void* pReadOnlyMemoryMapping, size_t pLength);
        virtual void SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed);

    private:
        void LoadLazyMemoryPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pLazyMemAddr);
        void LoadLazyMemoryPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pLazyMemAddr, uint pForwardPrefetchPageCount);
        void CopyLazyInputMemPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pFaultAddr);
        void CopyShadowMemPage(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmAddressSpace* pAddressSpace, pmTask* pTask, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr);

        void* CreateMemoryMapping(int pSharedMemDescriptor, size_t pLength, bool pReadAllowed, bool pWriteAllowed);
        void DeleteMemoryMapping(void* pMem, size_t pLength);
#endif

    private:
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mAddressSpaceSpecificsMapLock;
        std::map<pmAddressSpace*, linuxMemManager::addressSpaceSpecifics> mAddressSpaceSpecificsMap;  // Singleton class (one instance of this map exists)
 
#ifdef TRACK_MEMORY_ALLOCATIONS
		ulong mTotalAllocatedMemory;	// Lazy + Non-Lazy
		ulong mTotalAllocations;
		ulong mTotalDeallocations;
		ulong mTotalLazySegFaults;
        double mTotalAllocationTime;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mTrackLock;
#endif
};

} // end namespace pm

#endif

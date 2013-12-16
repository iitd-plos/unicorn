
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

#ifndef __PM_MEMORY_MANAGER__
#define __PM_MEMORY_MANAGER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"
#include "pmAddressSpace.h"

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

        virtual void FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector) = 0;
        virtual void FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector) = 0;
        virtual void CopyReceivedMemory(pmAddressSpace* pAddressSpace, ulong pOffset, ulong pLength, void* pSrcMem, pmTask* pRequestingTask) = 0;
    
        virtual size_t GetVirtualMemoryPageSize() const = 0;
        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount) = 0;

        virtual void* CreateCheckOutMemory(size_t pLength) = 0;
    
        virtual void CancelUnreferencedRequests(pmAddressSpace* pAddressSpace) = 0;
    
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
    typedef struct regionFetchData
    {
        pmCommandPtr receiveCommand;
        
        std::map<size_t, size_t> partialReceiveRecordMap;
        size_t accumulatedPartialReceivesLength;
        
        regionFetchData()
        : accumulatedPartialReceivesLength(0)
        {}
        
        regionFetchData(pmCommandPtr& pCommand)
        : receiveCommand(pCommand)
        , accumulatedPartialReceivesLength(0)
        {}
    } regionFetchData;
        
    typedef std::map<void*, std::pair<size_t, regionFetchData> > pmInFlightRegions;
        
    typedef struct addressSpaceSpecifics
    {
        addressSpaceSpecifics();

        int mSharedMemDescriptor;
        pmInFlightRegions mInFlightMemoryMap;	// Map for regions being fetched; pair is length of region and regionFetchData
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mInFlightLock;
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

        virtual void FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector);
        virtual void FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector);
        virtual void CopyReceivedMemory(pmAddressSpace* pAddressSpace, ulong pOffset, ulong pLength, void* pSrcMem, pmTask* pRequestingTask);

        virtual size_t GetVirtualMemoryPageSize() const;

        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount);	// Allocation size must be a multiple of page size
    
        void InstallSegFaultHandler();
		void UninstallSegFaultHandler();
    
        virtual void CancelUnreferencedRequests(pmAddressSpace* pAddressSpace);

    private:
		pmLinuxMemoryManager();
		virtual ~pmLinuxMemoryManager();

        void CreateAddressSpaceSpecifics(pmAddressSpace* pAddressSpace, int pSharedMemDescriptor);
        linuxMemManager::addressSpaceSpecifics& GetAddressSpaceSpecifics(pmAddressSpace* pAddressSpace);
    
        void* AllocatePageAlignedMemoryInternal(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount, int& pSharedMemDescriptor);

        void FetchNonOverlappingMemoryRegion(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, communicator::memoryTransferType pTransferType, size_t pOffset, size_t pLength, size_t pStep, size_t pCount, pmAddressSpace::vmRangeOwner& pRangeOwner, linuxMemManager::pmInFlightRegions& pInFlightMap, pmCommandPtr& pCommand);

        void FindRegionsNotInFlight(linuxMemManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommandPtr>& pCommandVector);

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

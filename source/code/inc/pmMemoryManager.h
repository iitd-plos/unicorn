
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
#include "pmMemSection.h"

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
    
		virtual void* AllocateMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount) = 0;
        virtual pmStatus DeallocateMemory(pmMemSection* pMemSection) = 0;
        virtual pmStatus DeallocateMemory(void* pMem) = 0;

        virtual void FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommunicatorCommandPtr>& pCommandVector) = 0;
        virtual pmStatus CopyReceivedMemory(pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem, pmTask* pRequestingTask) = 0;
    
        virtual size_t GetVirtualMemoryPageSize() const = 0;
        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount) = 0;

        virtual void* CreateCheckOutMemory(size_t pLength) = 0;
    
#ifdef SUPPORT_LAZY_MEMORY
        virtual void* CreateReadOnlyMemoryMapping(pmMemSection* pMemSection) = 0;
        virtual void DeleteReadOnlyMemoryMapping(void* pReadOnlyMemoryMapping, size_t pLength) = 0;
        virtual pmStatus SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed) = 0;
#endif

	protected:
		size_t mPageSize;
};


namespace linuxMemManager
{
    typedef struct regionFetchData
    {
        pmCommunicatorCommandPtr receiveCommand;
        
        std::map<size_t, size_t> partialReceiveRecordMap;
        size_t accumulatedPartialReceivesLength;
        
        regionFetchData()
        : accumulatedPartialReceivesLength(0)
        {}
    } regionFetchData;
        
    typedef std::map<void*, std::pair<size_t, regionFetchData> > pmInFlightRegions;
        
    typedef struct memSectionSpecifics
    {
        memSectionSpecifics();

        int mSharedMemDescriptor;
        pmInFlightRegions mInFlightMemoryMap;	// Map for regions being fetched; pair is length of region and regionFetchData
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mInFlightLock;
    } memSectionSpecifics;
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

		virtual void* AllocateMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount);
        virtual pmStatus DeallocateMemory(pmMemSection* pMemSection);
        virtual pmStatus DeallocateMemory(void* pMem);

        virtual void FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommunicatorCommandPtr>& pCommandVector);
        virtual pmStatus CopyReceivedMemory(pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem, pmTask* pRequestingTask);

        virtual size_t GetVirtualMemoryPageSize() const;

        virtual size_t FindAllocationSize(size_t pLength, size_t& pPageCount);	// Allocation size must be a multiple of page size
    
        pmStatus InstallSegFaultHandler();
		pmStatus UninstallSegFaultHandler();

    private:
		pmLinuxMemoryManager();
		virtual ~pmLinuxMemoryManager();

        void CreateMemSectionSpecifics(pmMemSection* pMemSection, int pSharedMemDescriptor);
        linuxMemManager::memSectionSpecifics& GetMemSectionSpecifics(pmMemSection* pMemSection);
    
        void* AllocatePageAlignedMemoryInternal(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount, int& pSharedMemDescriptor);

        void FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMemSection::vmRangeOwner& pRangeOwner, linuxMemManager::pmInFlightRegions& pInFlightMap, pmCommunicatorCommandPtr& pCommand);

        void FindRegionsNotInFlight(linuxMemManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommunicatorCommandPtr>& pCommandVector);

        virtual void* CreateCheckOutMemory(size_t pLength);

#ifdef SUPPORT_LAZY_MEMORY
    public:
        virtual void* CreateReadOnlyMemoryMapping(pmMemSection* pMemSection);
        virtual void DeleteReadOnlyMemoryMapping(void* pReadOnlyMemoryMapping, size_t pLength);
        virtual pmStatus SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed);

    private:
        pmStatus LoadLazyMemoryPage(pmExecutionStub* pStub, pmMemSection* pMemSection, void* pLazyMemAddr);
        pmStatus LoadLazyMemoryPage(pmExecutionStub* pStub, pmMemSection* pMemSection, void* pLazyMemAddr, uint pForwardPrefetchPageCount);
        pmStatus CopyLazyInputMemPage(pmExecutionStub* pStub, pmMemSection* pMemSection, void* pFaultAddr);
        pmStatus CopyShadowMemPage(pmExecutionStub* pStub, pmMemSection* pMemSection, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr);

        void* CreateMemoryMapping(int pSharedMemDescriptor, size_t pLength, bool pReadAllowed, bool pWriteAllowed);
        void DeleteMemoryMapping(void* pMem, size_t pLength);
#endif

    private:
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mMemSectionSpecificsMapLock;
        std::map<pmMemSection*, linuxMemManager::memSectionSpecifics> mMemSectionSpecificsMap;  // Singleton class (one instance of this map exists)
 
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

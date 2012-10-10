
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

#ifndef __PM_VIRTUAL_MEMORY__
#define __PM_VIRTUAL_MEMORY__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"

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

class pmMemSection;

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

        virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength) = 0;
        virtual pmStatus CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem) = 0;
    
        virtual size_t GetVirtualMemoryPageSize() = 0;
    
#ifdef SUPPORT_LAZY_MEMORY
		virtual void* AllocateLazyMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount) = 0;
        virtual pmStatus ApplyLazyProtection(void* pAddr, size_t pLength) = 0;
        virtual pmStatus RemoveLazyProtection(void* pAddr, size_t pLength) = 0;
        virtual pmStatus RemoveLazyWriteProtection(void* pAddr, size_t pLength) = 0;
#endif

	protected:
        static pmMemoryManager* mMemoryManager;
		size_t mPageSize;
};


namespace linuxMemManager
{
    typedef struct regionFetchData
    {
        pmCommunicatorCommandPtr sendCommand;
        pmCommunicatorCommandPtr receiveCommand;
        
        std::map<size_t, size_t> partialReceiveRecordMap;
    #ifdef SUPPORT_LAZY_MEMORY
        std::map<void*, std::vector<char> > partialPageReceiveBufferMap;
    #endif
        size_t accumulatedPartialReceivesLength;
        
        regionFetchData();
    } regionFetchData;
        
    typedef std::map<void*, std::pair<size_t, regionFetchData> > pmInFlightRegions;
        
    typedef struct memSectionSpecifics
    {
        pmInFlightRegions mInFlightMemoryMap;	// Map for regions being fetched; pair is length of region and regionFetchData
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mInFlightLock;
    } memSectionSpecifics;
}
    
class pmLinuxMemoryManager : public pmMemoryManager
{
    friend class pmController;
    
	public:
        static pmMemoryManager* GetMemoryManager();

		virtual void* AllocateMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount);
        virtual pmStatus DeallocateMemory(pmMemSection* pMemSection);
        virtual pmStatus DeallocateMemory(void* pMem);

        virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength);
        virtual pmStatus CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem);

        virtual size_t GetVirtualMemoryPageSize();
    
    private:
		pmLinuxMemoryManager();
		virtual ~pmLinuxMemoryManager();

        void CreateMemSectionSpecifics(pmMemSection* pMemSection);
        linuxMemManager::memSectionSpecifics& GetMemSectionSpecifics(pmMemSection* pMemSection);
    
        size_t FindAllocationSize(size_t pLength, size_t& pPageCount);	// Allocation size must be a multiple of page size
        void* AllocatePageAlignedMemoryInternal(size_t& pLength, size_t& pPageCount);

        pmCommunicatorCommandPtr FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMachine* pOwnerMachine, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, linuxMemManager::pmInFlightRegions& pInFlightMap);

        void FindRegionsNotInFlight(linuxMemManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommunicatorCommandPtr>& pCommandVector);
        
#ifdef SUPPORT_LAZY_MEMORY
    public:
        virtual void* AllocateLazyMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount);
        virtual pmStatus ApplyLazyProtection(void* pAddr, size_t pLength);
        virtual pmStatus RemoveLazyProtection(void* pAddr, size_t pLength);
        virtual pmStatus RemoveLazyWriteProtection(void* pAddr, size_t pLength);

    private:
        pmStatus LoadLazyMemoryPage(pmMemSection* pMemSection, void* pLazyMemAddr);
        pmStatus LoadLazyMemoryPage(pmMemSection* pMemSection, void* pLazyMemAddr, uint pForwardPrefetchPageCount);
        pmStatus CopyShadowMemPage(pmMemSection* pMemSection, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr);

		pmStatus InstallSegFaultHandler();
		pmStatus UninstallSegFaultHandler();

        friend void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);
#endif

    private:
        std::map<pmMemSection*, linuxMemManager::memSectionSpecifics> mMemSectionSpecificsMap;  // Singleton class (one instance of this map exists)

#ifdef TRACK_MEMORY_ALLOCATIONS
		ulong mTotalAllocatedMemory;	// Lazy + Non-Lazy
		ulong mTotalLazyMemory;
		ulong mTotalAllocations;
		ulong mTotalDeallocations;
		ulong mTotalLazySegFaults;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mTrackLock;
#endif
};

} // end namespace pm

#endif

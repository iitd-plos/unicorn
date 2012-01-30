
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

#ifdef USE_LAZY_MEMORY
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

		virtual pmStatus DestroyMemoryManager() = 0;

		virtual void* AllocateMemory(size_t& pLength, size_t& pPageCount) = 0;

#ifdef USE_LAZY_MEMORY
		virtual void* AllocateLazyMemory(size_t& pLength, size_t& pPageCount) = 0;
#endif

		virtual pmStatus DeallocateMemory(void* pMem) = 0;

		virtual bool IsLazyMemory(void* pPtr) = 0;
		virtual uint GetVirtualMemoryPageSize() = 0;

		virtual ulong GetLowerPageSizeMultiple(ulong pNum) = 0;
		virtual ulong GetHigherPageSizeMultiple(ulong pNum) = 0;

		virtual pmStatus LoadLazyMemoryPage(void* pLazyMemAddr) = 0;
		virtual pmStatus CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem) = 0;

		virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(void* pMem, ushort pPriority, size_t pOffset, size_t pLength) = 0;
		virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength) = 0;

	private:

	protected:
		uint mPageSize;
};

class pmLinuxMemoryManager : public pmMemoryManager
{
	public:
		typedef struct regionFetchData
		{
			pmCommunicatorCommandPtr sendCommand;
			pmCommunicatorCommandPtr receiveCommand;

			regionFetchData();
		} regionFetchData;

		static pmMemoryManager* GetMemoryManager();
		virtual pmStatus DestroyMemoryManager();

		virtual void* AllocateMemory(size_t& pLength, size_t& pPageCount);

#ifdef USE_LAZY_MEMORY
		virtual void* AllocateLazyMemory(size_t& pLength, size_t& pPageCount);
		virtual pmStatus LoadLazyMemoryPage(void* pLazyMemAddr);

		friend void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);
#endif

		virtual pmStatus DeallocateMemory(void* pMem);

		virtual bool IsLazyMemory(void* pPtr);
	
		virtual uint GetVirtualMemoryPageSize();

		virtual ulong GetLowerPageSizeMultiple(ulong pNum);
		virtual ulong GetHigherPageSizeMultiple(ulong pNum);

		virtual pmStatus CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem);

		virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(void* pMem, ushort pPriority, size_t pOffset, size_t pLength);
		virtual std::vector<pmCommunicatorCommandPtr> FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength);

	private:
		pmLinuxMemoryManager();
		virtual ~pmLinuxMemoryManager();

		virtual pmCommunicatorCommandPtr FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMachine* pOwnerMachine, ulong pOwnerBaseMemAddr);

		size_t FindAllocationSize(size_t pLength, size_t& pPageCount);	// Allocation size must be a multiple of page size

#ifdef USE_LAZY_MEMORY
		pmStatus InstallSegFaultHandler();
		pmStatus UninstallSegFaultHandler();
#endif

		void* GetLazyMemoryStartAddr(void* pPtr, size_t& pLength);

		static std::map<void*, size_t> mLazyMemoryMap;	// Map from allocated region's address to length
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;

		static std::map<void*, std::pair<size_t, regionFetchData> > mInFlightMemoryMap;	// Map for lazy regions/pages being fetched; pair is length of region and regionFetchData
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mInFlightLock;

		static pmMemoryManager* mMemoryManager;

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

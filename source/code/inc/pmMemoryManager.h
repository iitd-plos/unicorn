
#ifndef __PM_VIRTUAL_MEMORY__
#define __PM_VIRTUAL_MEMORY__

#include "pmInternalDefinitions.h"

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

	void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);

/**
 * \brief Memory Management Routines and Virtual Memory Optimizations
 * This class provides an implementation of lazy arrays which fill their VM pages on access.
*/
class pmMemoryManager
{
	public:
		virtual pmStatus DestroyMemoryManager() = 0;

		virtual void* AllocateMemory(size_t pLength) = 0;
		virtual void* AllocateLazyMemory(size_t pLength) = 0;

		virtual pmStatus DeallocateMemory(void* pMem) = 0;

		virtual bool IsLazyMemory(void* pPtr) = 0;
		virtual uint GetVirtualMemoryPageSize() = 0;

		virtual pmStatus LoadLazyMemory(void* pLazyMem, ulong pLoadOffset, ulong pLoadLength) = 0;

	private:
};

class pmLinuxMemoryManager : public pmMemoryManager
{
	public:
		static pmMemoryManager* GetMemoryManager();
		virtual pmStatus DestroyMemoryManager();

		virtual void* AllocateMemory(size_t pLength);
		virtual void* AllocateLazyMemory(size_t pLength);

		virtual pmStatus DeallocateMemory(void* pMem);

		virtual bool IsLazyMemory(void* pPtr);
	
		virtual uint GetVirtualMemoryPageSize();

		virtual pmStatus LoadLazyMemoryPage(void* pLazyMemAddr);

		friend void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);

	private:
		pmLinuxMemoryManager();
		~pmLinuxMemoryManager();

		pmStatus InstallSegFaultHandler();
		pmStatus UninstallSegFaultHandler();

		void* GetLazyMemoryStartAddr(void* pPtr, size_t& pLength);

		bool mSegFaultHandlerInstalled;
		std::map<void*, size_t> mLazyMemoryMap;	// Map from allocated region's address to length

		static pmMemoryManager* mMemoryManager;

#ifdef TRACK_MEMORY_ALLOCATIONS
		ulong mTotalAllocatedMemory;	// Lazy + Non-Lazy
		ulong mTotalLazyMemory;
		ulong mTotalAllocations;
		ulong mTotalDeallocations;
		ulong mTotalLazySegFaults;
#endif
};

} // end namespace pm

#endif

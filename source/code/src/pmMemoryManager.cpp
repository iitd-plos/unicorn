
#include "pmMemoryManager.h"
#include "pmController.h"

namespace pm
{

/* class pmLinuxMemoryManager */
pmMemoryManager* pmLinuxMemoryManager::mMemoryManager = NULL;

pmLinuxMemoryManager::pmLinuxMemoryManager()
{
	InstallSegFaultHandler();
}

pmLinuxMemoryManager::~pmLinuxMemoryManager()
{
	UninstallSegFaultHandler();
}

pmMemoryManager* pmLinuxMemoryManager::GetMemoryManager()
{
	if(!mMemoryManager)
		mMemoryManager = new pmLinuxMemoryManager();

	return mMemoryManager;
}

pmStatus pmLinuxMemoryManager::DestroyMemoryManager()
{
	delete mMemoryManager;
	mMemoryManager = NULL;

	return pmSuccess;
}

void* pmLinuxMemoryManager::AllocateMemory(size_t pLength)
{
	void* lPtr = ::malloc(pLength);

	if(!lPtr)
		throw pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED);

	return lPtr;
}

void* pmLinuxMemoryManager::AllocateLazyMemory(size_t pLength)
{
	uint lPageSize = GetVirtualMemoryPageSize();

	void* lPtr = NULL;
	void** lRef = (void**)(&lPtr);
	if(::posix_memalign(lRef, lPageSize, pLength) != 0)
		throw pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED);

	if(::mprotect(lPtr, pLength, PROT_NONE) != 0)
	{
		::free(lPtr);
		throw pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_NONE_FAILED);
	}

	mLazyMemoryMap[lPtr] = pLength;
}

pmStatus pmLinuxMemoryManager::DeallocateMemory(void* pMem)
{
	std::map<void*, size_t>::iterator lIter = mLazyMemoryMap.find(pMem);
	if(lIter != mLazyMemoryMap.end())
		mLazyMemoryMap.erase(lIter);

	::free(pMem);

	return pmSuccess;
}

bool pmLinuxMemoryManager::IsLazyMemory(void* pPtr)
{
	char* lAddress = static_cast<char*>(pPtr);
	std::map<void*, size_t>::iterator lIter = mLazyMemoryMap.begin();

	for(lIter; lIter != mLazyMemoryMap.end(); ++lIter)
	{
		char* lMemAddress = static_cast<char*>((void*)(lIter->first));
		size_t lLength = static_cast<size_t>(lIter->second);

		if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
			return true;
	}

	return false;
}

void* pmLinuxMemoryManager::GetLazyMemoryStartAddr(void* pPtr, size_t& pLength)
{
	char* lAddress = static_cast<char*>(pPtr);
	std::map<void*, size_t>::iterator lIter = mLazyMemoryMap.begin();

	for(lIter; lIter != mLazyMemoryMap.end(); ++lIter)
	{
		char* lMemAddress = static_cast<char*>((void*)(lIter->first));
		size_t lLength = static_cast<size_t>(lIter->second);

		if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
		{
			pLength = lLength;
			return static_cast<void*>(lMemAddress);
		}
	}

	return NULL;
}

uint pmLinuxMemoryManager::GetVirtualMemoryPageSize()
{
	return ::getPageSize();
}

pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(void* pLazyMemAddr)
{
	// Do not throw from this function as it is called by seg fault handler

	size_t lLength = 0;
	char* lStartAddr = static_cast<char*>(GetLazyMemoryStartAddr(pLazyMemAddr, lLength));

	if(!lStartAddr)
		return pmFatalError;

	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	char* lLastAddr = lStartAddr + lLength;

	char* lMemAddr = static_cast<char*>(pLazyMemAddr);
	char* lPageAddr = lMemAddr - (reinterpret_cast<size_t>(lMemAddr) % lPageSize);

	size_t lOffset = lPageAddr - lStartAddr;
	size_t lLeftoverLength = lLastAddr - lPageAddr;

	if(lLeftoverLength > lPageSize)
		lLeftoverLength = lPageSize;

	pmController* lController = pmController::GetController();
	if(!lController)
		return pmFatalError;

	lController->FetchMemoryRegion(lStartAddr, lOffset, lLeftoverLength);

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::InstallSegFaultHandler()
{    
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_sigaction = SegFaultHandler;

	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		throw pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_INSTALL_FAILED);

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::UninstallSegFaultHandler()
{
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_sigaction = SIG_DFL;

	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		throw pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_UNINSTALL_FAILED);

	return pmSuccess;
}

void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext)
{
	pmLinuxMemoryManager* lMemoryManager = dynamic_cast<pmLinuxMemoryManager*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager());

	if(!lMemoryManager->IsLazyMemory((void*)(pSigInfo->si_addr)))
		lMemoryManager->UninstallSegFaultHandler();

	if(::mprotect((void*)(pSigInfo->si_addr), 1, PROT_READ | PROT_WRITE) != 0)
		lMemoryManager->UninstallSegFaultHandler();

	if(lMemoryManager->LoadLazyMemoryPage((void*)(pSigInfo->si_addr)) != pmSuccess)
		exit(EXIT_FAILURE);
}

}

#include "pmMemoryManager.h"
#include "pmController.h"
#include "pmCommunicator.h"
#include "pmMemSection.h"
#include "pmHardware.h"

#include <string.h>

namespace pm
{
        
#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_req(const pmMemSection* memSection, const void* addr, size_t offset, size_t length, uint host)
{
    char lStr[512];
   
    if(dynamic_cast<const pmInputMemSection*>(memSection))
        sprintf(lStr, "Requesting input memory %p (Mem Section %p) at offset %ld for length %ld from host %d", addr, memSection, offset, length, host);
    else
        sprintf(lStr, "Requesting output memory %p (Mem Section %p) at offset %ld for length %ld from host %d", addr, memSection, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_REQ_DUMP(memSection, addr, offset, length, host) __dump_mem_req(memSection, addr, offset, length, host);
#else
#define MEM_REQ_DUMP(memSection, addr, offset, length, host)    
#endif

RESOURCE_LOCK_IMPLEMENTATION_CLASS pmLinuxMemoryManager::mInFlightLock;
std::map<void*, size_t> pmLinuxMemoryManager::mLazyMemoryMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmLinuxMemoryManager::mResourceLock;
std::map<void*, std::pair<size_t, pmLinuxMemoryManager::regionFetchData> > pmLinuxMemoryManager::mInFlightMemoryMap;

/*
pmStatus MemoryManagerCommandCompletionCallback(pmCommandPtr pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::tr1::dynamic_pointer_cast<pmCommunicatorCommand>(pCommand);
	if(!lCommunicatorCommand)
		PMTHROW(pmFatalErrorException());

	switch(lCommunicatorCommand->GetType())
	{
		case pmCommunicatorCommand::SEND:
		{
			if(lCommunicatorCommand->GetTag() == pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG)
				delete lCommunicatorCommand->GetData();

			break;
		}

		case pmCommunicatorCommand::RECEIVE:
		{
			pmCommunicatorCommand::memorySubscriptionRequest* lData = (pmCommunicatorCommand::memorySubscriptionRequest*)(pCommand->GetData());
			ulong lAddr = lData->addr;
			ulong lOffset = lData->offset;
			ulong lLength = lData->length;
			ulong lTransferId = lData->transferId;
			pmMachine* lDestHost = pmMachinePool::GetMachinePool()->GetMachine(lData->destHost);

			pmCommunicatorCommandPtr lSendCommand = pmCommunicatorCommand::CreateSharedPtr(pCommand->GetPriority(), pmCommunicatorCommand::SEND, (pmCommunicatorCommand::communicatorCommandTags)(lData->transferId), lDestHost, pmCommunicatorCommand::BYTE, 
				((char*)lAddr + lOffset),  lLength, NULL, 0, gCommandCompletionCallback);

			pmCommunicator::GetCommunicator()->Send(lSendCommand);

			pmSubscriptionManager* lManager = (pmSubscriptionManager*)(lCommunicatorCommand->GetSecondaryData());
			lManager->SetupNewMemRequestReceiveReception();
		}

		default:
			PMTHROW(pmFatalErrorException());
	}

	return pmSuccess;
}

static pmCommandCompletionCallback gCommandCompletionCallback = MemoryManagerCommandCompletionCallback;
*/

/* class pmLinuxMemoryManager */
pmMemoryManager* pmLinuxMemoryManager::mMemoryManager = NULL;

pmLinuxMemoryManager::pmLinuxMemoryManager()
{
#ifdef USE_LAZY_MEMORY
	InstallSegFaultHandler();
#endif

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());

	mTotalAllocatedMemory = 0;
	mTotalLazyMemory = 0;
	mTotalAllocations = 0;
	mTotalDeallocations = 0;
	mTotalLazySegFaults = 0;
#endif

	mPageSize = ::getpagesize();
}

pmLinuxMemoryManager::~pmLinuxMemoryManager()
{
#ifdef USE_LAZY_MEMORY
	UninstallSegFaultHandler();
#endif
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

void* pmLinuxMemoryManager::AllocateMemory(size_t& pLength, size_t& pPageCount)
{
	if(pLength == 0)
		return NULL;

	pLength = FindAllocationSize(pLength, pPageCount);

	uint lPageSize = GetVirtualMemoryPageSize();

	void* lPtr = NULL;
	void** lRef = (void**)(&lPtr);
	if(::posix_memalign(lRef, lPageSize, pLength) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED));

	if(!lPtr)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED));

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	mTotalAllocatedMemory += pLength;
	++mTotalAllocations;
#endif

	return lPtr;
}

#ifdef USE_LAZY_MEMORY
void* pmLinuxMemoryManager::AllocateLazyMemory(size_t& pLength, size_t& pPageCount)
{
	if(pLength == 0)
		return NULL;

	pLength = FindAllocationSize(pLength, pPageCount);

	uint lPageSize = GetVirtualMemoryPageSize();

	void* lPtr = NULL;
	void** lRef = (void**)(&lPtr);
	if(::posix_memalign(lRef, lPageSize, pLength) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED));

	if(!lPtr)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED));

	if(::mprotect(lPtr, pLength, PROT_NONE) != 0)
	{
		::free(lPtr);
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_NONE_FAILED));
	}

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	mTotalLazyMemory += pLength;
#endif

	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
	mLazyMemoryMap[lPtr] = pLength;

	return lPtr;
}
#endif

pmStatus pmLinuxMemoryManager::DeallocateMemory(void* pMem)
{
#ifdef USE_LAZY_MEMORY
	mResourceLock.Lock();
	std::map<void*, size_t>::iterator lIter = mLazyMemoryMap.find(pMem);
	if(lIter != mLazyMemoryMap.end())
		mLazyMemoryMap.erase(lIter);
	mResourceLock.Unlock();
#endif

	::free(pMem);

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	++mTotalDeallocations;
#endif

	return pmSuccess;
}

size_t pmLinuxMemoryManager::FindAllocationSize(size_t pLength, size_t& pPageCount)
{
	uint lPageSize = GetVirtualMemoryPageSize();
	pPageCount = ((pLength/lPageSize) + ((pLength%lPageSize != 0)?1:0));

	return (pPageCount*lPageSize);
}

bool pmLinuxMemoryManager::IsLazyMemory(void* pPtr)
{
#ifdef USE_LAZY_MEMORY
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	typedef std::map<void*, size_t> mapType;
	mapType::iterator lStartIter;
	mapType::iterator *lStartIterAddr = &lStartIter;

	char* lAddress = static_cast<char*>(pPtr);
	FIND_FLOOR_ELEM(mapType, mLazyMemoryMap, lAddress, lStartIterAddr);

	if(lStartIterAddr)
	{
		char* lMemAddress = static_cast<char*>((void*)(lStartIter->first));
		size_t lLength = static_cast<size_t>(lStartIter->second);

		if(lMemAddress <= lAddress && lAddress < lMemAddress + lLength)
			return true;
	}
#endif

	return false;
}

void* pmLinuxMemoryManager::GetLazyMemoryStartAddr(void* pPtr, size_t& pLength)
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	typedef std::map<void*, size_t> mapType;
	mapType::iterator lStartIter;
	mapType::iterator *lStartIterAddr = &lStartIter;

	char* lAddress = static_cast<char*>(pPtr);
	FIND_FLOOR_ELEM(mapType, mLazyMemoryMap, lAddress, lStartIterAddr);

	if(lStartIterAddr)
	{
		char* lMemAddress = static_cast<char*>((void*)(lStartIter->first));
		size_t lLength = static_cast<size_t>(lStartIter->second);

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
	return mPageSize;
}

ulong pmLinuxMemoryManager::GetLowerPageSizeMultiple(ulong pNum)
{
	uint lPageSize = GetVirtualMemoryPageSize();

	return ((pNum/lPageSize) * lPageSize);
}

ulong pmLinuxMemoryManager::GetHigherPageSizeMultiple(ulong pNum)
{
	uint lPageSize = GetVirtualMemoryPageSize();

	return (((pNum/lPageSize) + 1) * lPageSize);
}

pmStatus pmLinuxMemoryManager::CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem)
{
	FINALIZE_RESOURCE(dInFlightLock, mInFlightLock.Lock(), mInFlightLock.Unlock());

	//if(::mprotect((void*)(pSigInfo->si_addr), 1, PROT_READ | PROT_WRITE) != 0)
	//	lMemoryManager->UninstallSegFaultHandler();
	// Receive this after removing memory protection from page under mInFlightLock

	char* lAddr = (char*)pDestMem;
	lAddr += pOffset;

	memcpy((void*)lAddr, pSrcMem, pLength);

	std::map<void*, std::pair<size_t, regionFetchData> >::iterator lIter;
	if(mInFlightMemoryMap.find(lAddr) == mInFlightMemoryMap.end())
		PMTHROW(pmFatalErrorException());

	regionFetchData& lData = mInFlightMemoryMap[lAddr].second;    
	delete (pmCommunicatorCommand::memorySubscriptionRequest*)(lData.sendCommand->GetData());
	lData.receiveCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(lData.receiveCommand));
    mInFlightMemoryMap.erase(lAddr);

	return pmSuccess;
}

std::vector<pmCommunicatorCommandPtr> pmLinuxMemoryManager::FetchMemoryRegion(void* pMem, ushort pPriority, size_t pOffset, size_t pLength)
{
	std::vector<pmCommunicatorCommandPtr> lCommandVector;

	pmMemSection* lMemSection = pmMemSection::FindMemSection(pMem);
	if(!lMemSection)
		PMTHROW(pmFatalErrorException());

	//pOffset = GetLowerPageSizeMultiple(pOffset);
	//pLength = GetHigherPageSizeMultiple(pLength);

	char* lFetchAddress = (char*)pMem + pOffset;
	char* lLastFetchAddress = lFetchAddress + pLength - 1;

	typedef std::map<void*, std::pair<size_t, regionFetchData> > mapType;
	mapType::iterator lStartIter, lEndIter;
	mapType::iterator* lStartIterAddr = &lStartIter;
	mapType::iterator* lEndIterAddr = &lEndIter;

	std::vector<std::pair<ulong, ulong> > lRegionsToBeFetched;	// Each pair is start address and last address of sub ranges to be fetched

	FINALIZE_RESOURCE(dInFlightLock, mInFlightLock.Lock(), mInFlightLock.Unlock());

	FIND_FLOOR_ELEM(mapType, mInFlightMemoryMap, lFetchAddress, lStartIterAddr);	// Find mem fetch range in flight just previous to the start of new range
	FIND_FLOOR_ELEM(mapType, mInFlightMemoryMap, lLastFetchAddress, lEndIterAddr);	// Find mem fetch range in flight just previous to the end of new range

	// Both start and end of new range fall prior to all ranges in flight or there is no range in flight
	if(!lStartIterAddr && !lEndIterAddr)
	{
		lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, (ulong)lLastFetchAddress));
	}
	else
	{
		// If start of new range falls prior to all ranges in flight but end of new range does not
		if(!lStartIterAddr)
		{
			char* lFirstAddr = (char*)(mInFlightMemoryMap.begin()->first);
			lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, ((ulong)lFirstAddr)-1));
			lFetchAddress = lFirstAddr;
			lStartIter = mInFlightMemoryMap.begin();
		}
	
		// Both start and end of new range have atleast one in flight range prior to them
		
		// Check if start and end of new range fall within their just prior ranges or outside
		bool lStartInside = ((lFetchAddress >= (char*)(lStartIter->first)) && (lFetchAddress < ((char*)(lStartIter->first) + lStartIter->second.first)));
		bool lEndInside = ((lLastFetchAddress >= (char*)(lEndIter->first)) && (lLastFetchAddress < ((char*)(lEndIter->first) + lEndIter->second.first)));

		// If both start and end of new range have the same in flight range just prior to them
		if(lStartIter == lEndIter)
		{
			// If both start and end lie within the same in flight range, then the new range is already being fetched
			if(lStartInside && lEndInside)
			{
				lCommandVector.push_back(lStartIter->second.second.receiveCommand);
				return lCommandVector;
			}
			else if(lStartInside && !lEndInside)
            {
                // If start of new range is within an in flight range and that range is just prior to the end of new range
				lCommandVector.push_back(lStartIter->second.second.receiveCommand);
                
                lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lStartIter->first) + lStartIter->second.first), (ulong)lLastFetchAddress));
            }
            else
            {
                // If both start and end of new range have the same in flight range just prior to them and they don't fall within that range
                lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, (ulong)lLastFetchAddress));
			}
		}
		else
		{
			// If start and end of new range have different in flight ranges prior to them

			// If start of new range does not fall within the in flight range
			if(!lStartInside)
			{
				++lStartIter;
				lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, ((ulong)(lStartIter->first))-1));
			}

			// If end of new range does not fall within the in flight range
			if(!lEndInside)
            {
				lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lEndIter->first) + lEndIter->second.first), (ulong)lLastFetchAddress));
            }

            lCommandVector.push_back(lEndIter->second.second.receiveCommand);

            // Fetch all non in flight data between in flight ranges
			if(lStartIter != lEndIter)
			{
				for(mapType::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
				{
                    lCommandVector.push_back(lTempIter->second.second.receiveCommand);
                    
					mapType::iterator lNextIter = lTempIter;
					++lNextIter;
					lRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lTempIter->first) + lTempIter->second.first), ((ulong)(lNextIter->first))-1));
				}
			}
		}
	}

	size_t lRegionCount = lRegionsToBeFetched.size();
	for(size_t i=0; i<lRegionCount; ++i)
	{
		ulong lOffset = lRegionsToBeFetched[i].first-(ulong)pMem;
		ulong lLength = lRegionsToBeFetched[i].second - lRegionsToBeFetched[i].first+ 1;

		pmMemSection::pmMemOwnership lOwnerships;
		lMemSection->GetOwners(lOffset, lLength, lOwnerships);

		pmMemSection::pmMemOwnership::iterator lStartIter, lEndIter, lIter;
		lStartIter = lOwnerships.begin();
		lEndIter = lOwnerships.end();

		for(lIter = lStartIter; lIter != lEndIter; ++lIter)
		{
			ulong lInternalOffset = lIter->first;
			ulong lInternalLength = lIter->second.first;
			pmMemSection::vmRangeOwner& lRangeOwner = lIter->second.second;

			if(lRangeOwner.host != PM_LOCAL_MACHINE)
			{
				pmCommunicatorCommandPtr lCommand = FetchNonOverlappingMemoryRegion(pPriority, lMemSection, pMem, lInternalOffset, lInternalLength, lRangeOwner.host, lRangeOwner.hostBaseAddr);
				if(lCommand)
					lCommandVector.push_back(lCommand);
			}
		}
	}

	return lCommandVector;
}

std::vector<pmCommunicatorCommandPtr> pmLinuxMemoryManager::FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength)
{
	void* lMem = NULL;
	if(dynamic_cast<pmInputMemSection*>(pMemSection))
		lMem = ((pmInputMemSection*)pMemSection)->GetMem();
	else
		lMem = ((pmOutputMemSection*)pMemSection)->GetMem();

	return FetchMemoryRegion(lMem, pPriority, pOffset, pLength);
}

pmCommunicatorCommandPtr pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMachine* pOwnerMachine, ulong pOwnerBaseMemAddr)
{	
	regionFetchData lFetchData;
	lFetchData.receiveCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::RECEIVE, pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG,
		pOwnerMachine, pmCommunicatorCommand::BYTE,	NULL, 0, NULL, 0);	// Dummy command just to allow threads to wait on it

	pmCommunicatorCommand::memorySubscriptionRequest* lData = new pmCommunicatorCommand::memorySubscriptionRequest();
	lData->ownerBaseAddr = pOwnerBaseMemAddr;	// page aligned
	lData->receiverBaseAddr = (ulong)pMem;		// page aligned
	lData->offset = pOffset;
	lData->length = pLength;
	lData->destHost = *PM_LOCAL_MACHINE;

	lFetchData.sendCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG, pOwnerMachine,	pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT, (void*)lData, 1, NULL, 0);

	char* lAddr = (char*)pMem;
	lAddr += lData->offset;

	pmCommunicator::GetCommunicator()->Send(lFetchData.sendCommand);

    MEM_REQ_DUMP(pMemSection, pMem, pOffset, pLength, (uint)(*pOwnerMachine));
    
	std::pair<size_t, regionFetchData> lPair(lData->length, lFetchData);
	mInFlightMemoryMap[lAddr] = lPair;

	lFetchData.receiveCommand->MarkExecutionStart();
	return lFetchData.receiveCommand;
}

pmLinuxMemoryManager::regionFetchData::regionFetchData()
{
}

#ifdef USE_LAZY_MEMORY
pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(void* pLazyMemAddr)
{
	// Do not throw from this function as it is called by seg fault handler

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	++mTotalLazySegFaults;
#endif

	size_t lLength = 0;
	char* lStartAddr = static_cast<char*>(GetLazyMemoryStartAddr(pLazyMemAddr, lLength));

	if(!lStartAddr)
		return pmFatalError;

	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	char* lLastAddr = lStartAddr + lLength;

	char* lMemAddr = static_cast<char*>(pLazyMemAddr);
	char* lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);

	//size_t lOffset = lPageAddr - lStartAddr;
	size_t lLeftoverLength = lLastAddr - lPageAddr;

	if(lLeftoverLength > lPageSize)
		lLeftoverLength = lPageSize;

	pmController* lController = pmController::GetController();
	if(!lController)
		return pmFatalError;

	//FetchMemoryRegion(lStartAddr, lOffset, lLeftoverLength);

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::InstallSegFaultHandler()
{    
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_sigaction = SegFaultHandler;

	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_INSTALL_FAILED));

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::UninstallSegFaultHandler()
{
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_handler = SIG_DFL;

	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_UNINSTALL_FAILED));

	return pmSuccess;
}

void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext)
{
	pmLinuxMemoryManager* lMemoryManager = dynamic_cast<pmLinuxMemoryManager*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager());

	if(!lMemoryManager->IsLazyMemory((void*)(pSigInfo->si_addr)))
		lMemoryManager->UninstallSegFaultHandler();

	//if(::mprotect((void*)(pSigInfo->si_addr), 1, PROT_READ | PROT_WRITE) != 0)
	//	lMemoryManager->UninstallSegFaultHandler();

	if(lMemoryManager->LoadLazyMemoryPage((void*)(pSigInfo->si_addr)) != pmSuccess)
		exit(EXIT_FAILURE);
}
#endif

}



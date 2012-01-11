
#include "pmMemoryManager.h"
#include "pmController.h"
#include "pmCommunicator.h"

namespace pm
{

pmStatus MemoryManagerCommandCompletionCallback(pmCommandPtr pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::tr1::dynamic_pointer_cast<pmCommunicatorCommand>(pCommand);
	if(!lCommunicatorCommand)
		throw pmFatalErrorException();

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
			throw pmFatalErrorException();
	}

	return pmSuccess;
}

static pmCommandCompletionCallback gCommandCompletionCallback = MemoryManagerCommandCompletionCallback;

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

	mPageSize = ::getPageSize();
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
		throw pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED);

	if(!lPtr)
		throw pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED);

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
		throw pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED);

	if(!lPtr)
		throw pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED);

	if(::mprotect(lPtr, pLength, PROT_NONE) != 0)
	{
		::free(lPtr);
		throw pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_NONE_FAILED);
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

	size_t lOffset = lPageAddr - lStartAddr;
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
		throw pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_INSTALL_FAILED);

	return pSuccess;
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

std::vector<pmCommunicatorCommandPtr> pmLinuxMemoryManager::FetchMemoryRegion(void* pMem, ushort pPriority, size_t pOffset, size_t pLength)
{
	std::vector<pmCommunicatorCommandPtr> lCommandVector;

	pmMemSection* lMemSection = pmMemSection::FindMemSection(pMem);
	if(!lMemSection)
		throw pmFatalErrorException();

	pOffset = GetLowerPageSizeMultiple(pOffset);
	pLength = GetHigherPageSizeMultiple(pLength);

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

	// Both start and end of new range fall prior to all ranges in flight
	if(!lStartIterAddr && !lEndIterAddr)
	{
		lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(lFetchAddress, lLastFetchAddress));
	}
	else
	{
		// If start of new range falls prior to all ranges in flight but end of new range does not
		if(!lStartIterAddr)
		{
			char* lFirstAddr = (char*)(mInFlightMemoryMap.begin()->first);
			lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(lFetchAddress, ((ulong)lFirstAddr)-1));
			lFetchAddress = lFirstAddr;
			lStartIter = mInFlightMemoryMap.begin();
		}
	
		// Both start and end of new range have atleast have in flight range prior to them
		
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
			else
			{
				// If start of new range is within an in flight range and that range is just prior to the end of new range
				if(lStartInside && !lEndInside)
					lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(((char*)(lStartIter->first) + lStartIter->second.first), lLastFetchAddress));
				else	// If both start and end of new range have the same in flight range just prior to them and they don't fall within that range
					lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(lFetchAddress, lLastFetchAddress));
			}
		}
		else
		{
			// If start and end of new range have different in flight ranges prior to them

			// If start of new range does not fall within the in flight range
			if(!lStartInside)
			{
				++lStartIter;
				lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(lFetchAddress, ((ulong)(lStartIter->first))-1));
			}

			// If end of new range does not fall within the in flight range
			if(!lEndInside)
				lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(((char*)(lEndIter->first) + lEndIter->second.first), lLastFetchAddress));

			// Fetch all non in flight data between in flight ranges
			if(lStartIter != lEndIter)
			{
				for(mapType::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
				{
					mapType::iterator lNextIter = lTempIter;
					++lNextIter;
					lRegionsToBeFetched.push_back(std::pair<ulong, ulong>(((char*)(lTempIter->first) + lTempIter->second.first), ((ulong)(lNextIter->first))-1)));			
				}
			}
		}
	}

	size_t lRegionCount = lRegionsToBeFetched.size();
	for(size_t i=0; i<lRegionCount; ++i)
		lCommandVector.push_back(FetchNonOverlappingMemoryRegion(pPriority, lMemSection, pMem, lRegionsToBeFetched[i].first-(ulong)pMem, lRegionsToBeFetched[i].second - lRegionsToBeFetched[i].first+ 1));

	mInFlightLock.Unlock();

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

pmCommunicatorCommandPtr pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength)
{
	pmMachine* lOwner = NULL;
	ulong lOwnerAddr;
	pMemSection->GetOwner(lOwner, lOwnerAddr);

	if(lOwner == PM_LOCAL_MACHINE)
		return NULL;	// success; memory already available

	pmCommunicatorCommand::memorySubscriptionRequest* lData = new pmCommunicatorCommand::memorySubscriptionRequest();
	
	lData->addr = lOwnerAddr;	// already page aligned
	lData->offset = pOffset;
	lData->length = pLength;
	lData->transferId = pmCommunicatorCommand::GetNextDynamicTag();
	lData->destHost = *PM_LOCAL_MACHINE;

	regionFetchData lFetchData;

	lFetchData.sendCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG, lOwner,
		pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT, (void*)lData, sizeof(pmCommunicatorCommand::memorySubscriptionRequest), NULL, 0, gCommandCompletionCallback);

	char* lAddr = (char*)pMem;
	lAddr += lData->offset;

	//if(::mprotect((void*)(pSigInfo->si_addr), 1, PROT_READ | PROT_WRITE) != 0)
	//	lMemoryManager->UninstallSegFaultHandler();
	// Receive this after removing memory protection from page under mInFlightLock
	lFetchData.receiveCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::RECEIVE, (pmCommunicatorCommand::communicatorCommandTags)(lData->transferId), lOwner, pmCommunicatorCommand::BYTE,
		lAddr, lData->length, NULL, 0, gCommandCompletionCallback);

	pmCommunicator::GetCommunicator()->Send(lFetchData.sendCommand);
	pmCommunicator::GetCommunicator()->Receive(lFetchData.receiveCommand);

	std::pair<size_t, regionFetchData> lPair(lData->length, lFetchData);
	mInFlightMemoryMap[lAddr] = lPair;

	return lFetchData.receiveCommand;
}

pmLinuxMemoryManager::regionFetchData::regionFetchData()
{
	sendCommand = NULL;
	receiveCommand = NULL;
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
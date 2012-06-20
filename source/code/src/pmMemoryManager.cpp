
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

#include "pmMemoryManager.h"
#include "pmController.h"
#include "pmCommunicator.h"
#include "pmMemSection.h"
#include "pmHardware.h"

#include <string.h>

namespace pm
{
        
#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_req(const pmMemSection* memSection, const void* addr, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
   
    if(dynamic_cast<const pmInputMemSection*>(memSection))
        sprintf(lStr, "Requesting input memory %p (Mem Section %p) at offset %ld (Remote Offset %ld) for length %ld from host %d", addr, memSection, receiverOffset, offset, length, host);
    else
        sprintf(lStr, "Requesting output memory %p (Mem Section %p) at offset %ld (Remote Offset %ld) for length %ld from host %d", addr, memSection, receiverOffset, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_REQ_DUMP(memSection, addr, receiverOffset, offset, length, host) __dump_mem_req(memSection, addr, receiverOffset, offset, length, host);
#else
#define MEM_REQ_DUMP(memSection, addr, receiverOffset, offset, length, host)    
#endif
    
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmLinuxMemoryManager::mInFlightLock;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmLinuxMemoryManager::mInFlightLazyRegisterationLock;
pmLinuxMemoryManager::pmInFlightRegions pmLinuxMemoryManager::mInFlightMemoryMap;
pmLinuxMemoryManager::pmInFlightRegions pmLinuxMemoryManager::mInFlightLazyRegisterations;


/* class pmLinuxMemoryManager */
pmMemoryManager* pmMemoryManager::mMemoryManager = NULL;

pmLinuxMemoryManager::pmLinuxMemoryManager()
{
    if(mMemoryManager)
        PMTHROW(pmFatalErrorException());
    
    mMemoryManager = this;

#ifdef SUPPORT_LAZY_MEMORY
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
#ifdef SUPPORT_LAZY_MEMORY
	UninstallSegFaultHandler();
#endif
}

pmMemoryManager* pmLinuxMemoryManager::GetMemoryManager()
{
	return mMemoryManager;
}

void* pmLinuxMemoryManager::AllocatePageAlignedMemoryInternal(size_t& pLength, size_t& pPageCount)
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
    
    return lPtr;
}

void* pmLinuxMemoryManager::AllocateMemory(size_t& pLength, size_t& pPageCount)
{
    void* lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);
    
#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	mTotalAllocatedMemory += pLength;
	++mTotalAllocations;
#endif

	return lPtr;
}

#ifdef SUPPORT_LAZY_MEMORY
void* pmLinuxMemoryManager::AllocateLazyMemory(size_t& pLength, size_t& pPageCount)
{   
    void* lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);

#ifdef MACOS
    if(pPageCount < 31)
    {
    #ifdef _DEBUG
        std::cout << "WARNING: Less than 31 pages of lazy memory allocated on Mac OS X " << std::endl << std::flush;
    #endif
        
        ::free(lPtr);
        pLength = 31 * GetVirtualMemoryPageSize();
        
        lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);
    }
#endif

    ApplyLazyProtection(lPtr, pLength);

#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	mTotalLazyMemory += pLength;
#endif

	return lPtr;
}
#endif

pmStatus pmLinuxMemoryManager::DeallocateMemory(void* pMem)
{
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

uint pmLinuxMemoryManager::GetVirtualMemoryPageSize()
{
	return mPageSize;
}

pmStatus pmLinuxMemoryManager::CopyReceivedMemory(void* pDestMem, pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem)
{
    bool lIsLazyRegisteration = (pMemSection->IsLazy() && !pLength);
    pmInFlightRegions& lMap = lIsLazyRegisteration ? mInFlightLazyRegisterations : mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lIsLazyRegisteration ? mInFlightLazyRegisterationLock : mInFlightLock;
    
	FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

	char* lAddr = (char*)pDestMem + pOffset;

    if(lMap.find(lAddr) == lMap.end())
        PMTHROW(pmFatalErrorException());
        
    std::pair<size_t, regionFetchData> lPair = lMap[lAddr];
    lMap.erase(lAddr);
    
    // Zero length is sent as an acknowledgement when registerOnly request is sent
    if(pLength)
    {
        assert(lPair.first == pLength);
        
#ifdef SUPPORT_LAZY_MEMORY
        if(pMemSection->IsLazy())
        {
            if(pLength > GetVirtualMemoryPageSize())
                PMTHROW(pmFatalErrorException());
            
            RemoveLazyProtection((void*)lAddr, (size_t)pLength);
        }
#endif
        
        memcpy((void*)lAddr, pSrcMem, pLength);
    }
    
    regionFetchData& lData = lPair.second;        

#ifdef SUPPORT_LAZY_MEMORY
    if(lIsLazyRegisteration)
        pMemSection->AcquireOwnershipLazy(pOffset, lPair.first);
    else
#endif
        pMemSection->AcquireOwnershipImmediate(pOffset, lPair.first);
    
	delete (pmCommunicatorCommand::memorySubscriptionRequest*)(lData.sendCommand->GetData());
	lData.receiveCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(lData.receiveCommand));

	return pmSuccess;
}

// This function must be called after acquiring lock on pInFlightMap
void pmLinuxMemoryManager::FindRegionsNotInFlight(pmLinuxMemoryManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommunicatorCommandPtr>& pCommandVector)
{
	pmInFlightRegions::iterator lStartIter, lEndIter;
	pmInFlightRegions::iterator* lStartIterAddr = &lStartIter;
	pmInFlightRegions::iterator* lEndIterAddr = &lEndIter;

	char* lFetchAddress = (char*)pMem + pOffset;
	char* lLastFetchAddress = lFetchAddress + pLength - 1;

    FIND_FLOOR_ELEM(pmInFlightRegions, pInFlightMap, lFetchAddress, lStartIterAddr);	// Find range in flight just previous to the start of new range
    FIND_FLOOR_ELEM(pmInFlightRegions, pInFlightMap, lLastFetchAddress, lEndIterAddr);	// Find range in flight just previous to the end of new range
    
    // Both start and end of new range fall prior to all ranges in flight or there is no range in flight
    if(!lStartIterAddr && !lEndIterAddr)
    {
        pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, (ulong)lLastFetchAddress));
    }
    else
    {
        // If start of new range falls prior to all ranges in flight but end of new range does not
        if(!lStartIterAddr)
        {
            char* lFirstAddr = (char*)(pInFlightMap.begin()->first);
            pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, ((ulong)lFirstAddr)-1));
            lFetchAddress = lFirstAddr;
            lStartIter = pInFlightMap.begin();
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
                pCommandVector.push_back(lStartIter->second.second.receiveCommand);
                return;
            }
            else if(lStartInside && !lEndInside)
            {
                // If start of new range is within an in flight range and that range is just prior to the end of new range
                pCommandVector.push_back(lStartIter->second.second.receiveCommand);
                
                pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lStartIter->first) + lStartIter->second.first), (ulong)lLastFetchAddress));
            }
            else
            {
                // If both start and end of new range have the same in flight range just prior to them and they don't fall within that range
                pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, (ulong)lLastFetchAddress));
            }
        }
        else
        {
            // If start and end of new range have different in flight ranges prior to them
            
            // If start of new range does not fall within the in flight range
            if(!lStartInside)
            {
                ++lStartIter;
                pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)lFetchAddress, ((ulong)(lStartIter->first))-1));
            }
            
            // If end of new range does not fall within the in flight range
            if(!lEndInside)
            {
                pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lEndIter->first) + lEndIter->second.first), (ulong)lLastFetchAddress));
            }
            
            pCommandVector.push_back(lEndIter->second.second.receiveCommand);
            
            // Fetch all non in flight data between in flight ranges
            if(lStartIter != lEndIter)
            {
                for(pmInFlightRegions::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
                {
                    pCommandVector.push_back(lTempIter->second.second.receiveCommand);
                    
                    pmInFlightRegions::iterator lNextIter = lTempIter;
                    ++lNextIter;
                    pRegionsToBeFetched.push_back(std::pair<ulong, ulong>((ulong)((char*)(lTempIter->first) + lTempIter->second.first), ((ulong)(lNextIter->first))-1));
                }
            }
        }
    }
}

std::vector<pmCommunicatorCommandPtr> pmLinuxMemoryManager::FetchMemoryRegion(void* pMem, ushort pPriority, size_t pOffset, size_t pLength, bool pRegisterOnly)
{
	pmMemSection* lMemSection = pmMemSection::FindMemSection(pMem);
	if(!lMemSection)
		PMTHROW(pmFatalErrorException());

	std::vector<std::pair<ulong, ulong> > lRegionsToBeFetched;	// Each pair is start address and last address of sub ranges to be fetched
	std::vector<pmCommunicatorCommandPtr> lCommandVector;
    bool lIsLazyRegisteration = (lMemSection->IsLazy() && pRegisterOnly);
    pmInFlightRegions& lMap = lIsLazyRegisteration ? mInFlightLazyRegisterations : mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lIsLazyRegisteration ? mInFlightLazyRegisterationLock : mInFlightLock;

	FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

    FindRegionsNotInFlight(lMap, pMem, pOffset, pLength, lRegionsToBeFetched, lCommandVector);

	size_t lRegionCount = lRegionsToBeFetched.size();

	for(size_t i=0; i<lRegionCount; ++i)
	{
		ulong lOffset = lRegionsToBeFetched[i].first - (ulong)pMem;
		ulong lLength = lRegionsToBeFetched[i].second - lRegionsToBeFetched[i].first + 1;
        
		pmMemSection::pmMemOwnership lOwnerships;
        lMemSection->GetOwners(lOffset, lLength, lIsLazyRegisteration, lOwnerships);

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
				pmCommunicatorCommandPtr lCommand = FetchNonOverlappingMemoryRegion(pPriority, lMemSection, pMem, lInternalOffset, lInternalLength, lRangeOwner.host, lRangeOwner.hostBaseAddr, lRangeOwner.hostOffset, pRegisterOnly, lMap);

				if(lCommand.get())
					lCommandVector.push_back(lCommand);
			}
		}
	}

	return lCommandVector;
}

pmCommunicatorCommandPtr pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMachine* pOwnerMachine, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, bool pRegisterOnly, pmInFlightRegions& pInFlightMap)
{	
	regionFetchData lFetchData;

	pmCommunicatorCommand::memorySubscriptionRequest* lData = new pmCommunicatorCommand::memorySubscriptionRequest();
	lData->ownerBaseAddr = pOwnerBaseMemAddr;	// page aligned
	lData->receiverBaseAddr = (ulong)pMem;		// page aligned
    lData->receiverOffset = pOffset;
	lData->offset = pOwnerOffset;
	lData->length = pLength;
	lData->destHost = *PM_LOCAL_MACHINE;
    lData->registerOnly = pRegisterOnly?1:0;
    
	lFetchData.sendCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG, pOwnerMachine,	pmCommunicatorCommand::MEMORY_SUBSCRIPTION_STRUCT, (void*)lData, 1, NULL, 0);

	pmCommunicator::GetCommunicator()->Send(lFetchData.sendCommand);

    MEM_REQ_DUMP(pMemSection, pMem, pOffset, pOwnerOffset, pLength, (uint)(*pOwnerMachine));
        
    // For write only memory, a zero length buffer will be received back as an acknowledgement of subscription registration
	lFetchData.receiveCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::RECEIVE, pmCommunicatorCommand::MEMORY_SUBSCRIPTION_TAG, pOwnerMachine, pmCommunicatorCommand::BYTE, NULL, 0, NULL, 0);	// Dummy command just to allow threads to wait on it
    
	char* lAddr = (char*)pMem + lData->receiverOffset;
    pInFlightMap[lAddr] = std::make_pair(lData->length, lFetchData);

	lFetchData.receiveCommand->MarkExecutionStart();
	return lFetchData.receiveCommand;
}
    
pmLinuxMemoryManager::regionFetchData::regionFetchData()
{
}

#ifdef SUPPORT_LAZY_MEMORY
pmStatus pmLinuxMemoryManager::ApplyLazyProtection(void* pAddr, size_t pLength)
{
	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	char* lPageAddr = GET_VM_PAGE_START_ADDRESS(static_cast<char*>(pAddr), lPageSize);

	if(::mprotect(lPageAddr, pLength + reinterpret_cast<size_t>(pAddr) - reinterpret_cast<size_t>(lPageAddr), PROT_NONE) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_NONE_FAILED));

    return pmSuccess;
}

pmStatus pmLinuxMemoryManager::RemoveLazyProtection(void* pAddr, size_t pLength)
{
	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	char* lPageAddr = GET_VM_PAGE_START_ADDRESS(static_cast<char*>(pAddr), lPageSize);
    
	if(::mprotect(lPageAddr, pLength + reinterpret_cast<size_t>(pAddr) - reinterpret_cast<size_t>(lPageAddr), PROT_READ | PROT_WRITE) != 0)
        PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_RW_FAILED));
    
    return pmSuccess;
}

pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(pmMemSection* pMemSection, void* pLazyMemAddr)
{
#ifdef TRACK_MEMORY_ALLOCATIONS
	FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
	++mTotalLazySegFaults;
#endif

	char* lStartAddr = static_cast<char*>(pMemSection->GetMem());
	size_t lLength = pMemSection->GetLength();

	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	char* lLastAddr = lStartAddr + lLength;

	char* lMemAddr = static_cast<char*>(pLazyMemAddr);
	char* lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);

	size_t lOffset = lPageAddr - lStartAddr;
	size_t lLeftoverLength = (size_t)(lLastAddr - lPageAddr);

	if(lLeftoverLength > lPageSize)
		lLeftoverLength = lPageSize;

	const std::vector<pmCommunicatorCommandPtr>& lCommandVector = FetchMemoryRegion(lStartAddr, MAX_CONTROL_PRIORITY, lOffset, lLeftoverLength, false);
    
    std::vector<pmCommunicatorCommandPtr>::const_iterator lStartIter = lCommandVector.begin();
    std::vector<pmCommunicatorCommandPtr>::const_iterator lEndIter = lCommandVector.end();

    for(; lStartIter != lEndIter; ++lStartIter)
    {
        pmCommunicatorCommandPtr lCommand = *lStartIter;
        if(lCommand)
        {
            if(lCommand->WaitForFinish() != pmSuccess)
                PMTHROW(pmMemoryFetchException());
        }
    }
    
	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::InstallSegFaultHandler()
{    
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_sigaction = SegFaultHandler;

#ifdef MACOS
	if(sigaction(SIGBUS, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_INSTALL_FAILED));
#else
	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_INSTALL_FAILED));
#endif
    
	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::UninstallSegFaultHandler()
{
	struct sigaction lSigAction;

	lSigAction.sa_flags = SA_SIGINFO;
	sigemptyset(&lSigAction.sa_mask);
	lSigAction.sa_handler = SIG_DFL;

#ifdef MACOS
	if(sigaction(SIGBUS, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_UNINSTALL_FAILED));
#else
	if(sigaction(SIGSEGV, &lSigAction, NULL) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SEGFAULT_HANDLER_UNINSTALL_FAILED));
#endif
    
	return pmSuccess;
}

void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext)
{
    pmMemSection* lMemSection = pmMemSection::FindMemSectionContainingAddress((void*)(pSigInfo->si_addr));
    if(!lMemSection || !lMemSection->IsLazy())
        abort();
    
    pmLinuxMemoryManager* lMemoryManager = dynamic_cast<pmLinuxMemoryManager*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager());

	if(lMemoryManager->LoadLazyMemoryPage(lMemSection, (void*)(pSigInfo->si_addr)) != pmSuccess)
		abort();
}
#endif

}



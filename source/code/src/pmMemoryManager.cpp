
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
#include "pmHardware.h"
#include "pmExecutionStub.h"
#include "pmTaskProfiler.h"
#include "pmTask.h"
#include "pmTls.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>

#include <string.h>
#include <sstream>
#include <limits>

namespace pm
{
        
#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_req(const pmAddressSpace* addressSpace, const void* addr, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_req(const pmAddressSpace* addressSpace, const void* addr, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
   
    sprintf(lStr, "Requesting memory %p (address space %p) at offset %ld (Remote Offset %ld) for length %ld from host %d", addr, addressSpace, receiverOffset, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_REQ_DUMP(addressSpace, addr, receiverOffset, offset, length, host) __dump_mem_req(addressSpace, addr, receiverOffset, offset, length, host);
#else
#define MEM_REQ_DUMP(addressSpace, addr, receiverOffset, offset, length, host)    
#endif
    

/* class pmLinuxMemoryManager */
pmLinuxMemoryManager::pmLinuxMemoryManager()
    : mAddressSpaceSpecificsMapLock __LOCK_NAME__("pmLinuxMemoryManager::mAddressSpaceSpecificsMapLock")
#ifdef TRACK_MEMORY_ALLOCATIONS
	, mTotalAllocatedMemory(0)
	, mTotalAllocations(0)
	, mTotalDeallocations(0)
	, mTotalLazySegFaults(0)
    , mTotalAllocationTime(0)
    , mTrackLock __LOCK_NAME__("pmLinuxMemoryManager::mTrackLock")
#endif
{
	InstallSegFaultHandler();

	mPageSize = ::getpagesize();
}

pmLinuxMemoryManager::~pmLinuxMemoryManager()
{
	UninstallSegFaultHandler();

#ifdef TRACK_MEMORY_ALLOCATIONS
    std::stringstream lStream;
    lStream << "Memory Allocation Tracking ... " << std::endl;
    lStream << "Total Allocated Memory = " << mTotalAllocatedMemory << std::endl;
    lStream << "Total Allocations = " << mTotalAllocations << std::endl;
    lStream << "Total Deallocations = " << mTotalDeallocations << std::endl;
    lStream << "Total Lazy Traps = " << mTotalLazySegFaults << std::endl;
    lStream << "Total Allocation Time = " << mTotalAllocationTime << std::endl;
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStream.str().c_str(), true);

#endif
}

pmMemoryManager* pmLinuxMemoryManager::GetMemoryManager()
{
	static pmLinuxMemoryManager lMemoryManager;
    return &lMemoryManager;
}

#ifdef SUPPORT_LAZY_MEMORY
void* pmLinuxMemoryManager::CreateMemoryMapping(int pSharedMemDescriptor, size_t pLength, bool pReadAllowed, bool pWriteAllowed)
{
    if(pSharedMemDescriptor == -1)
        PMTHROW(pmFatalErrorException());

    int lFlags = PROT_NONE;
    if(pReadAllowed)
        lFlags |= PROT_READ;
    if(pWriteAllowed)
        lFlags |= PROT_WRITE;

	void* lMem = mmap(NULL, pLength, lFlags, MAP_SHARED, pSharedMemDescriptor, 0);
	if(lMem == MAP_FAILED || !lMem)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MMAP_FAILED));

	return lMem;
}

void pmLinuxMemoryManager::DeleteMemoryMapping(void* pMem, size_t pLength)
{
	if(munmap(pMem, pLength) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MUNMAP_FAILED));
}
    
void* pmLinuxMemoryManager::CreateReadOnlyMemoryMapping(pmAddressSpace* pAddressSpace)
{
#ifdef TRACK_MEMORY_ALLOCATIONS
    double lTrackTime = GetCurrentTimeInSecs();
#endif

    linuxMemManager::addressSpaceSpecifics& lSpecifics = GetAddressSpaceSpecifics(pAddressSpace);
    
    void* lPtr = CreateMemoryMapping(lSpecifics.mSharedMemDescriptor, pAddressSpace->GetAllocatedLength(), false, false);
    
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalAllocationTime += (GetCurrentTimeInSecs() - lTrackTime);
    }
#endif
    
    return lPtr;
}
    
void pmLinuxMemoryManager::DeleteReadOnlyMemoryMapping(void* pReadOnlyMemoryMapping, size_t pLength)
{
    DeleteMemoryMapping(pReadOnlyMemoryMapping, pLength);
}
#endif

void* pmLinuxMemoryManager::AllocatePageAlignedMemoryInternal(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount, int& pSharedMemDescriptor)
{
#ifdef TRACK_MEMORY_ALLOCATIONS
    double lTrackTime = GetCurrentTimeInSecs();
#endif
    
	if(pLength == 0)
		return NULL;

	pLength = FindAllocationSize(pLength, pPageCount);
    pSharedMemDescriptor = -1;
    
	void* lPtr = NULL;

#ifdef SUPPORT_LAZY_MEMORY
    if(pAddressSpace)
    {
        const char* lSharedMemName = pAddressSpace->GetName();

        int lSharedMemDescriptor = shm_open(lSharedMemName, O_RDWR | O_CREAT | O_EXCL, 0600);
        if(lSharedMemDescriptor == -1)
        {
            shm_unlink(lSharedMemName);
            lSharedMemDescriptor = shm_open(lSharedMemName, O_RDWR | O_CREAT | O_EXCL, 0600);
        }

        if(lSharedMemDescriptor == -1)
            PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SHM_OPEN_FAILED));

        sharedMemAutoPtr lSharedMemAutoPtr(lSharedMemName);
        if(ftruncate(lSharedMemDescriptor, pLength) != 0)
            PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::FTRUNCATE_FAILED));

        lPtr = CreateMemoryMapping(lSharedMemDescriptor, pLength, true, true);
        pSharedMemDescriptor = lSharedMemDescriptor;
    }
    else
#endif
    {
        size_t lPageSize = GetVirtualMemoryPageSize();
        void** lRef = (void**)(&lPtr);

        if(::posix_memalign(lRef, lPageSize, pLength) != 0)
            PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED));
    }
    
	if(!lPtr)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED));
    
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalAllocationTime += (GetCurrentTimeInSecs() - lTrackTime);
    }
#endif
    
    return lPtr;
}
    
void pmLinuxMemoryManager::CreateAddressSpaceSpecifics(pmAddressSpace* pAddressSpace, int pSharedMemDescriptor)
{
    FINALIZE_RESOURCE_PTR(dAddressSpaceSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAddressSpaceSpecificsMapLock, Lock(), Unlock());

    if(mAddressSpaceSpecificsMap.find(pAddressSpace) != mAddressSpaceSpecificsMap.end())
        PMTHROW(pmFatalErrorException());
    
    mAddressSpaceSpecificsMap[pAddressSpace].mSharedMemDescriptor = pSharedMemDescriptor;
}

linuxMemManager::addressSpaceSpecifics& pmLinuxMemoryManager::GetAddressSpaceSpecifics(pmAddressSpace* pAddressSpace)
{
    FINALIZE_RESOURCE_PTR(dAddressSpaceSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAddressSpaceSpecificsMapLock, Lock(), Unlock());

    if(mAddressSpaceSpecificsMap.find(pAddressSpace) == mAddressSpaceSpecificsMap.end())
        PMTHROW(pmFatalErrorException());
    
    return mAddressSpaceSpecificsMap[pAddressSpace];
}

void* pmLinuxMemoryManager::AllocateMemory(pmAddressSpace* pAddressSpace, size_t& pLength, size_t& pPageCount)
{
    int lSharedMemDescriptor = -1;

    void* lPtr = AllocatePageAlignedMemoryInternal(pAddressSpace, pLength, pPageCount, lSharedMemDescriptor);
    
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalAllocatedMemory += pLength;
        ++mTotalAllocations;
    }
#endif

    if(pAddressSpace)
        CreateAddressSpaceSpecifics(pAddressSpace, lSharedMemDescriptor);

    return lPtr;
}

void* pmLinuxMemoryManager::CreateCheckOutMemory(size_t pLength)
{
    size_t lPageCount = 0;
    FindAllocationSize(pLength, lPageCount);

#if 0
    if(pIsLazy && lPageCount < 31)
    {
    #ifdef _DEBUG
        std::cout << "WARNING: Less than 31 pages of lazy memory allocated on Mac OS X " << std::endl << std::flush;
    #endif
        
        pLength = 31 * GetVirtualMemoryPageSize();
    }
#endif

    return AllocateMemory(NULL, pLength, lPageCount);
}

void pmLinuxMemoryManager::DeallocateMemory(pmAddressSpace* pAddressSpace)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dAddressSpaceSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAddressSpaceSpecificsMapLock, Lock(), Unlock());
    
        if(mAddressSpaceSpecificsMap.find(pAddressSpace) == mAddressSpaceSpecificsMap.end())
            PMTHROW(pmFatalErrorException());
    
        mAddressSpaceSpecificsMap.erase(pAddressSpace);
    }

    #ifdef SUPPORT_LAZY_MEMORY
        DeleteMemoryMapping(pAddressSpace->GetMem(), pAddressSpace->GetAllocatedLength());
    #else
        ::free(pAddressSpace->GetMem());
    #endif

#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalDeallocations;
    }
#endif
}

void pmLinuxMemoryManager::DeallocateMemory(void* pMem)
{
	::free(pMem);

#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalDeallocations;
    }
#endif
}
    
size_t pmLinuxMemoryManager::FindAllocationSize(size_t pLength, size_t& pPageCount)
{
	size_t lPageSize = GetVirtualMemoryPageSize();
	pPageCount = ((pLength / lPageSize) + ((pLength % lPageSize != 0) ? 1 : 0));

	return (pPageCount * lPageSize);
}

size_t pmLinuxMemoryManager::GetVirtualMemoryPageSize() const
{
	return mPageSize;
}

void pmLinuxMemoryManager::CancelUnreferencedRequests(pmAddressSpace* pAddressSpace)
{
    using namespace linuxMemManager;

    addressSpaceSpecifics& lSpecifics = GetAddressSpaceSpecifics(pAddressSpace);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

    FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
    auto lIter = lMap.begin(), lEnd = lMap.end();
    while(lIter != lEnd)
    {
        if(lIter->second.second.receiveCommand.unique())
            lMap.erase(lIter++);
        else
            ++lIter;
    }
}
    
// This function must be called after acquiring lock on pInFlightMap
void pmLinuxMemoryManager::FindRegionsNotInFlight(linuxMemManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommandPtr>& pCommandVector)
{
    using namespace linuxMemManager;
    
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
        pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, (ulong)lLastFetchAddress);
    }
    else
    {
        // If start of new range falls prior to all ranges in flight but end of new range does not
        if(!lStartIterAddr)
        {
            char* lFirstAddr = (char*)(pInFlightMap.begin()->first);
            pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, ((ulong)lFirstAddr)-1);
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
                pCommandVector.emplace_back(lStartIter->second.second.receiveCommand);
                return;
            }
            else if(lStartInside && !lEndInside)
            {
                // If start of new range is within an in flight range and that range is just prior to the end of new range
                pCommandVector.emplace_back(lStartIter->second.second.receiveCommand);
                
                pRegionsToBeFetched.emplace_back((ulong)((char*)(lStartIter->first) + lStartIter->second.first), (ulong)lLastFetchAddress);
            }
            else
            {
                // If both start and end of new range have the same in flight range just prior to them and they don't fall within that range
                pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, (ulong)lLastFetchAddress);
            }
        }
        else
        {
            // If start and end of new range have different in flight ranges prior to them
            
            // If start of new range does not fall within the in flight range
            if(!lStartInside)
            {
                ++lStartIter;
                pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, ((ulong)(lStartIter->first))-1);
            }
            
            // If end of new range does not fall within the in flight range
            if(!lEndInside)
            {
                pRegionsToBeFetched.emplace_back((ulong)((char*)(lEndIter->first) + lEndIter->second.first), (ulong)lLastFetchAddress);
            }
            
            pCommandVector.emplace_back(lEndIter->second.second.receiveCommand);
            
            // Fetch all non in flight data between in flight ranges
            if(lStartIter != lEndIter)
            {
                for(pmInFlightRegions::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
                {
                    pCommandVector.emplace_back(lTempIter->second.second.receiveCommand);
                    
                    pmInFlightRegions::iterator lNextIter = lTempIter;
                    ++lNextIter;
                    pRegionsToBeFetched.emplace_back((ulong)((char*)(lTempIter->first) + lTempIter->second.first), ((ulong)(lNextIter->first))-1);
                }
            }
        }
    }
}

void pmLinuxMemoryManager::FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);

    // Check if entire scattered range is in flight or entire scattered range is on one machine (local or remote). Otherwise, split scattered fetch into general fetch.
    
    using namespace linuxMemManager;
    void* lMem = pAddressSpace->GetMem();
    
	std::vector<std::pair<ulong, ulong>> lRegionsToBeFetched;	// Start address and last address of sub ranges to be fetched
    std::vector<pmCommandPtr> lTempCommandVector;
    addressSpaceSpecifics& lSpecifics = GetAddressSpaceSpecifics(pAddressSpace);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

    bool lPartiallyInFlight = false;
    bool lRangeLiesOnOneMachine = true;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

        for(size_t i = 0; i < pScatteredSubscriptionInfo.count; ++i)
        {
            FindRegionsNotInFlight(lMap, lMem, pScatteredSubscriptionInfo.offset + i * pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.size, lRegionsToBeFetched, lTempCommandVector);
            lPartiallyInFlight |= (lTempCommandVector.size() > 1);
        }

        if(!lPartiallyInFlight)
        {
            if(lTempCommandVector.size() == 0)  // Nothing is in flight
            {
                pmAddressSpace::pmMemOwnership lOwnerships;
                for(size_t i = 0; i < pScatteredSubscriptionInfo.count; ++i)
                    pAddressSpace->GetOwners(pScatteredSubscriptionInfo.offset + i * pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.size, lOwnerships);

                if(lOwnerships.size() == pScatteredSubscriptionInfo.count)
                {
                    const pmMachine* lServingHost = lOwnerships.begin()->second.second.host;

                    for_each(lOwnerships, [&] (pmAddressSpace::pmMemOwnership::value_type& pPair)
                    {
                        DEBUG_EXCEPTION_ASSERT(pPair.second.first == pLength);
                        
                        DEBUG_EXCEPTION_ASSERT((pPair.second.second.hostOffset - pOffset) % pStep == 0);
                        
                        lRangeLiesOnOneMachine &= (lServingHost == pPair.second.second.host);
                    });
                    
                    if(lRangeLiesOnOneMachine && lServingHost != PM_LOCAL_MACHINE)
                    {
                        pmCommandPtr lCommand;
                        FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, lMem, communicator::TRANSFER_SCATTERED, pScatteredSubscriptionInfo.offset, pScatteredSubscriptionInfo.size, pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.count, lOwnerships.begin()->second.second, lMap, lCommand);

                        if(lCommand.get())
                            pCommandVector.emplace_back(std::move(lCommand));
                    }
                }
                else
                {
                    lRangeLiesOnOneMachine = false;
                }
            }
            else if(lTempCommandVector.size() == pScatteredSubscriptionInfo.count)  // Everything is in flight
            {
                pCommandVector.insert(pCommandVector.end(), lTempCommandVector.begin(), lTempCommandVector.end());
            }
            else    // Partially in flight
            {
                lPartiallyInFlight = true;
            }
        }
    }
    
    if(lPartiallyInFlight || !lRangeLiesOnOneMachine)
    {
        for(size_t i = 0; i < pScatteredSubscriptionInfo.count; ++i)
            FetchMemoryRegion(pAddressSpace, pPriority, pScatteredSubscriptionInfo.offset + i * pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.size, pCommandVector);
    }
}

void pmLinuxMemoryManager::FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector)
{
    EXCEPTION_ASSERT(pLength);

    using namespace linuxMemManager;
    void* lMem = pAddressSpace->GetMem();
    
	std::vector<std::pair<ulong, ulong>> lRegionsToBeFetched;	// Start address and last address of sub ranges to be fetched
    addressSpaceSpecifics& lSpecifics = GetAddressSpaceSpecifics(pAddressSpace);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

	FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

    FindRegionsNotInFlight(lMap, lMem, pOffset, pLength, lRegionsToBeFetched, pCommandVector);

	size_t lRegionCount = lRegionsToBeFetched.size();

	for(size_t i = 0; i < lRegionCount; ++i)
	{
		ulong lOffset = lRegionsToBeFetched[i].first - (ulong)lMem;
		ulong lLength = lRegionsToBeFetched[i].second - lRegionsToBeFetched[i].first + 1;
        
        if(lLength)
        {
            pmAddressSpace::pmMemOwnership lOwnerships;
            pAddressSpace->GetOwners(lOffset, lLength, lOwnerships);

            for_each(lOwnerships, [&] (pmAddressSpace::pmMemOwnership::value_type& pPair)
            {
                pmAddressSpace::vmRangeOwner& lRangeOwner = pPair.second.second;

                if(lRangeOwner.host != PM_LOCAL_MACHINE)
                {
                    pmCommandPtr lCommand;
                    FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, lMem, communicator::TRANSFER_GENERAL, pPair.first,  pPair.second.first, 0, 0, lRangeOwner, lMap, lCommand);

                    if(lCommand.get())
                        pCommandVector.emplace_back(std::move(lCommand));
                }
            });
        }
	}
}

void pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, communicator::memoryTransferType pTransferType, size_t pOffset, size_t pLength, size_t pStep, size_t pCount, pmAddressSpace::vmRangeOwner& pRangeOwner, linuxMemManager::pmInFlightRegions& pInFlightMap, pmCommandPtr& pCommand)
{
    using namespace linuxMemManager;
    
    pmTask* lLockingTask = pAddressSpace->GetLockingTask();
    
    uint lOriginatingHost = lLockingTask ? (uint)(*(lLockingTask->GetOriginatingHost())) : std::numeric_limits<uint>::max();
    ulong lSequenceNumber = lLockingTask ? lLockingTask->GetSequenceNumber() : std::numeric_limits<ulong>::max();

	finalize_ptr<communicator::memoryTransferRequest> lData(new communicator::memoryTransferRequest(communicator::memoryIdentifierStruct(pRangeOwner.memIdentifier.memOwnerHost, pRangeOwner.memIdentifier.generationNumber), communicator::memoryIdentifierStruct(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber()), pTransferType, pOffset, pRangeOwner.hostOffset, pLength, pStep, pCount, *PM_LOCAL_MACHINE, 0, (ushort)(lLockingTask != NULL), lOriginatingHost, lSequenceNumber, pPriority));
    
	pmCommunicatorCommandPtr lSendCommand = pmCommunicatorCommand<communicator::memoryTransferRequest>::CreateSharedPtr(pPriority, communicator::SEND, communicator::MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host, communicator::MEMORY_TRANSFER_REQUEST_STRUCT, lData, 1, pmScheduler::GetScheduler()->GetSchedulerCommandCompletionCallback());

    communicator::memoryTransferRequest* lRequestData = (communicator::memoryTransferRequest*)(lSendCommand->GetData());

    if(pTransferType == communicator::TRANSFER_GENERAL)
    {
        pCommand = pmCommand::CreateSharedPtr(pPriority, communicator::RECEIVE, 0);	// Dummy command just to allow threads to wait on it

        char* lAddr = (char*)pMem + lRequestData->receiverOffset;
        pInFlightMap.emplace(std::piecewise_construct, std::forward_as_tuple(lAddr), std::forward_as_tuple(lRequestData->length, regionFetchData(pCommand)));
    
    #ifdef ENABLE_TASK_PROFILING
        if(lLockingTask)
            pAddressSpace->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, true);
    #endif
    }
    else
    {
        DEBUG_EXCEPTION_ASSERT(pTransferType == communicator::TRANSFER_SCATTERED);
        
        pCommand = pmCountDownCommand::CreateSharedPtr(pCount, pPriority, communicator::RECEIVE, 0);	// Dummy command just to allow threads to wait on it

        for(size_t i = 0; i < pCount; ++i)
        {
            char* lAddr = (char*)pMem + lRequestData->receiverOffset + i * pStep;
            pInFlightMap.emplace(std::piecewise_construct, std::forward_as_tuple(lAddr), std::forward_as_tuple(pLength, regionFetchData(pCommand)));

        #ifdef ENABLE_TASK_PROFILING
            if(lLockingTask)
                pAddressSpace->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, true);
        #endif
        }
    }
    
	pCommand->MarkExecutionStart();

    MEM_REQ_DUMP(pAddressSpace, pMem, pOffset, pRangeOwner.hostOffset, pLength, (uint)(*pRangeOwner.host));

	pmCommunicator::GetCommunicator()->Send(lSendCommand);
}
    
void pmLinuxMemoryManager::CopyReceivedMemory(pmAddressSpace* pAddressSpace, ulong pOffset, ulong pLength, void* pSrcMem, pmTask* pRequestingTask)
{
    using namespace linuxMemManager;

    if(!pLength)
        PMTHROW(pmFatalErrorException());
    
    pmTask* lLockingTask = pAddressSpace->GetLockingTask();
    if(lLockingTask != pRequestingTask)
        return;

    addressSpaceSpecifics& lSpecifics = GetAddressSpaceSpecifics(pAddressSpace);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

    FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
    void* lDestMem = pAddressSpace->GetMem();
    char* lAddr = (char*)lDestMem + pOffset;
    
    pmInFlightRegions::iterator lIter = lMap.find(lAddr);
    if((lIter != lMap.end()) && (lIter->second.first == pLength))
    {
        std::pair<size_t, regionFetchData>& lPair = lIter->second;
        memcpy((void*)lAddr, pSrcMem, pLength);

        regionFetchData& lData = lPair.second;
        pAddressSpace->AcquireOwnershipImmediate(pOffset, lPair.first);

    #ifdef ENABLE_TASK_PROFILING
        if(lLockingTask)
            lLockingTask->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
    #endif

    #ifdef ENABLE_MEM_PROFILING
        pAddressSpace->RecordMemReceive(pLength);
    #endif

        pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
        lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

        lMap.erase(lIter);
    }
    else
    {
        pmInFlightRegions::iterator lBaseIter;
        pmInFlightRegions::iterator* lBaseIterAddr = &lBaseIter;
        FIND_FLOOR_ELEM(pmInFlightRegions, lSpecifics.mInFlightMemoryMap, lAddr, lBaseIterAddr);
        
        if(!lBaseIterAddr)
            PMTHROW(pmFatalErrorException());
        
        size_t lStartAddr = reinterpret_cast<size_t>(lBaseIter->first);
        std::pair<size_t, regionFetchData>& lPair = lBaseIter->second;
        
        size_t lRecvAddr = reinterpret_cast<size_t>(lAddr);
        if((lRecvAddr < lStartAddr) || (lRecvAddr + pLength > lStartAddr + lPair.first))
            PMTHROW(pmFatalErrorException());
        
        typedef std::map<size_t, size_t> partialReceiveRecordType;
        regionFetchData& lData = lPair.second;
        partialReceiveRecordType& lPartialReceiveRecordMap = lData.partialReceiveRecordMap;
                
        partialReceiveRecordType::iterator lPartialIter;
        partialReceiveRecordType::iterator* lPartialIterAddr = &lPartialIter;
        FIND_FLOOR_ELEM(partialReceiveRecordType, lPartialReceiveRecordMap, lRecvAddr, lPartialIterAddr);

        if(lPartialIterAddr && lPartialIter->first + lPartialIter->second - 1 >= lRecvAddr)
            PMTHROW(pmFatalErrorException());   // Multiple overlapping partial receives

        lData.accumulatedPartialReceivesLength += pLength;
        if(lData.accumulatedPartialReceivesLength > lPair.first)
            PMTHROW(pmFatalErrorException());

        bool lTransferComplete = (lData.accumulatedPartialReceivesLength == lPair.first);

        if(lTransferComplete)
        {
            memcpy((void*)lAddr, pSrcMem, pLength);

            size_t lOffset = lStartAddr - reinterpret_cast<size_t>(lDestMem);
            pAddressSpace->AcquireOwnershipImmediate(lOffset, lPair.first);
            
        #ifdef ENABLE_TASK_PROFILING
            if(lLockingTask)
                pAddressSpace->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
        #endif

        #ifdef ENABLE_MEM_PROFILING
            pAddressSpace->RecordMemReceive(pLength);
        #endif

            pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
            lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

            lSpecifics.mInFlightMemoryMap.erase(lBaseIter);
        }
        else
        {            
            // Make partial receive entry
            lPartialReceiveRecordMap[lRecvAddr] = pLength;
            
            memcpy((void*)lAddr, pSrcMem, pLength);

            size_t lOffset = lRecvAddr - reinterpret_cast<size_t>(lDestMem);
            pAddressSpace->AcquireOwnershipImmediate(lOffset, pLength);
        }
    }
}
    
#ifdef SUPPORT_LAZY_MEMORY
void pmLinuxMemoryManager::SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed)
{
    ACCUMULATION_TIMER(Timer_ACC, "SetLazyProtection");
    
	size_t lPageSize = static_cast<size_t>(GetVirtualMemoryPageSize());
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(reinterpret_cast<size_t>(pAddr), lPageSize);

    int lFlags = PROT_NONE;
    if(pReadAllowed)
        lFlags |= PROT_READ;
    if(pWriteAllowed)
        lFlags |= PROT_WRITE;
    
	if(::mprotect(reinterpret_cast<void*>(lPageAddr), pLength + reinterpret_cast<size_t>(pAddr) - lPageAddr, lFlags) != 0)
        PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_PROT_RW_FAILED));
}

void pmLinuxMemoryManager::LoadLazyMemoryPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pLazyMemAddr, uint pForwardPrefetchPageCount)
{
	size_t lPageSize = GetVirtualMemoryPageSize();
    size_t lBytesToBeFetched = (1 + pForwardPrefetchPageCount) * lPageSize;

	size_t lStartAddr = reinterpret_cast<size_t>(pAddressSpace->GetMem());
	size_t lLength = pAddressSpace->GetLength();
	size_t lLastAddr = lStartAddr + lLength;

	size_t lMemAddr = reinterpret_cast<size_t>(pLazyMemAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);

	size_t lOffset = lPageAddr - lStartAddr;
	size_t lLeftoverLength = lLastAddr - lPageAddr;

	if(lLeftoverLength > lBytesToBeFetched)
		lLeftoverLength = lBytesToBeFetched;

    pmTask* lLockingTask = pAddressSpace->GetLockingTask();
    ushort lPriority = (lLockingTask ? lLockingTask->GetPriority() : MAX_CONTROL_PRIORITY);
    
    if(!pAddressSpace->IsRegionLocallyOwned(lOffset, ((lLeftoverLength > lPageSize) ? lPageSize : lLeftoverLength)))
    {
        // We want to fetch lazy memory page and prefetch pages collectively. But we do want this thread to resume execution as soon as
        // lazy memory page is fetched without waiting for prefetch pages to come. To do this, we make two FetchMemory requests - first with
        // lazy memory page and prefetch pages and second with lazy memory page only. The in-flight memory system will piggy back second
        // request onto the first one. The current thread only waits on commands returned by second FetchMemory statement.
        std::vector<pmCommandPtr> lCommandVector;
        if(pForwardPrefetchPageCount)
            FetchMemoryRegion(pAddressSpace, lPriority, lOffset, lLeftoverLength, lCommandVector);

        lCommandVector.clear();
        FetchMemoryRegion(pAddressSpace, lPriority, lOffset, ((lLeftoverLength > lPageSize) ? lPageSize : lLeftoverLength), lCommandVector);

        pStub->WaitForNetworkFetch(lCommandVector);
    }
}

void pmLinuxMemoryManager::LoadLazyMemoryPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pLazyMemAddr)
{
    LoadLazyMemoryPage(pStub, pAddressSpace, pLazyMemAddr, pAddressSpace->GetLazyForwardPrefetchPageCount());
}

void pmLinuxMemoryManager::CopyLazyInputMemPage(pmExecutionStub* pStub, pmAddressSpace* pAddressSpace, void* pFaultAddr)
{
    DEBUG_EXCEPTION_ASSERT(!pAddressSpace->GetLockingTask()->IsWritable(pAddressSpace) && pAddressSpace->GetLockingTask()->IsLazy(pAddressSpace));

	size_t lPageSize = GetVirtualMemoryPageSize();
	size_t lMemAddr = reinterpret_cast<size_t>(pFaultAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
    size_t lOffset = (lPageAddr - reinterpret_cast<size_t>(pAddressSpace->GetReadOnlyLazyMemoryMapping()));
    
    void* lDestAddr = reinterpret_cast<void*>(lPageAddr);
    void* lSrcAddr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pAddressSpace->GetMem()) + lOffset);

    LoadLazyMemoryPage(pStub, pAddressSpace, lSrcAddr);
    SetLazyProtection(lDestAddr, lPageSize, true, true);    // we may actually not allow writes here at all and abort if a write access is done to RO memory
}

void pmLinuxMemoryManager::CopyShadowMemPage(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmAddressSpace* pAddressSpace, pmTask* pTask, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr)
{
    DEBUG_EXCEPTION_ASSERT(!pAddressSpace->GetLockingTask()->IsReadOnly(pAddressSpace) && pAddressSpace->GetLockingTask()->IsLazyReadWrite(pAddressSpace));

	size_t lPageSize = GetVirtualMemoryPageSize();
	size_t lMemAddr = reinterpret_cast<size_t>(pFaultAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
    size_t lSrcMemBaseAddr = reinterpret_cast<size_t>(pAddressSpace->GetMem());
    
    void* lDestAddr = reinterpret_cast<void*>(lPageAddr);

    if(pTask->GetAddressSpaceSubscriptionVisibility(pAddressSpace, pStub) == SUBSCRIPTION_NATURAL)
    {
        size_t lOffset = pShadowMemOffset + (lPageAddr - reinterpret_cast<size_t>(pShadowMemBaseAddr));

        void* lSrcAddr = reinterpret_cast<void*>(lSrcMemBaseAddr + lOffset);

        LoadLazyMemoryPage(pStub, pAddressSpace, lSrcAddr);
        SetLazyProtection(lDestAddr, lPageSize, true, true);

        size_t lMaxSrcAddr = lSrcMemBaseAddr + pAddressSpace->GetLength();
        size_t lMaxCopyAddr = reinterpret_cast<size_t>(lSrcAddr) + lPageSize;

        if(lMaxCopyAddr > lMaxSrcAddr)
            ::memcpy(lDestAddr, lSrcAddr, lMaxSrcAddr - reinterpret_cast<size_t>(lSrcAddr));
        else
            ::memcpy(lDestAddr, lSrcAddr, lPageSize);
    }
    else    // SUBSCRIPTION_COMPACT
    {
        std::vector<subscription::pmCompactPageInfo> lCompactedPages = pTask->GetSubscriptionManager().GetReadSubscriptionPagesForCompactViewPage(pStub, pSubtaskId, pSplitInfo, pTask->GetAddressSpaceIndex(pAddressSpace), lPageAddr, lPageSize);

        for_each(lCompactedPages, [&] (const subscription::pmCompactPageInfo& pInfo)
        {
            void* lSrcAddr = reinterpret_cast<void*>(lSrcMemBaseAddr + pInfo.addressSpaceOffset);

            LoadLazyMemoryPage(pStub, pAddressSpace, lSrcAddr);
        });

        SetLazyProtection(lDestAddr, lPageSize, true, true);
        
        for_each(lCompactedPages, [&] (const subscription::pmCompactPageInfo& pInfo)
        {
            void* lSrcAddr = reinterpret_cast<void*>(lSrcMemBaseAddr + pInfo.addressSpaceOffset);
            void* lCopyAddr = reinterpret_cast<void*>(lPageAddr + pInfo.compactViewOffset);

            size_t lMaxSrcAddr = lSrcMemBaseAddr + pAddressSpace->GetLength();
            size_t lMaxCopyAddr = reinterpret_cast<size_t>(lSrcAddr) + std::min(lPageSize, pInfo.length);

            if(lMaxCopyAddr > lMaxSrcAddr)
                ::memcpy(lCopyAddr, lSrcAddr, lMaxSrcAddr - reinterpret_cast<size_t>(lSrcAddr));
            else
                ::memcpy(lCopyAddr, lSrcAddr, std::min(lPageSize, pInfo.length));
        });
    }
}
#endif
    
void pmLinuxMemoryManager::InstallSegFaultHandler()
{
    void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext);

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
}

void pmLinuxMemoryManager::UninstallSegFaultHandler()
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
}

void SegFaultHandler(int pSignalNum, siginfo_t* pSigInfo, void* pContext)
{
    ACCUMULATION_TIMER(Timer_ACC, "SegFaultHandler");

    const std::pair<void*, void*>& lPair = TLS_IMPLEMENTATION_CLASS::GetTls()->GetThreadLocalStoragePair(TLS_EXEC_STUB, TLS_CURRENT_SUBTASK_ID);
    pmExecutionStub* lStub = static_cast<pmExecutionStub*>(lPair.first);
    void* lSubtaskPtr = lPair.second;
    if(!lStub || !lSubtaskPtr)
        abort();

    pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(lStub);

#ifdef SUPPORT_LAZY_MEMORY
    try
    {
        ulong lSubtaskId = *(static_cast<ulong*>(lSubtaskPtr));

        size_t lShadowMemOffset = 0;
        void* lShadowMemBaseAddr = NULL;

        pmLinuxMemoryManager* lMemoryManager = dynamic_cast<pmLinuxMemoryManager*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager());

    #ifdef TRACK_MEMORY_ALLOCATIONS
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lMemoryManager->mTrackLock, Lock(), Unlock());
            ++lMemoryManager->mTotalLazySegFaults;
        }
    #endif
    
        /* Check if the address belongs to a lazy read only memory */
        pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpaceContainingLazyAddress((void*)(pSigInfo->si_addr));
        if(lAddressSpace)
        {
            lMemoryManager->CopyLazyInputMemPage(lStub, lAddressSpace, (void*)(pSigInfo->si_addr));
        }
        else    /* Check if the address belongs to a lazy read write/write only memory */
        {
            pmTask* lTask = NULL;

            lAddressSpace = pmSubscriptionManager::FindAddressSpaceContainingShadowAddr((void*)(pSigInfo->si_addr), lShadowMemOffset, lShadowMemBaseAddr, lTask);
            if(lAddressSpace && lShadowMemBaseAddr && lTask)
            {
                const std::pair<void*, void*>& lPair = TLS_IMPLEMENTATION_CLASS::GetTls()->GetThreadLocalStoragePair(TLS_SPLIT_ID, TLS_SPLIT_COUNT);

                pmSplitInfo* lSplitInfoPtr = NULL;
                pmSplitInfo lSplitInfo;

                if(lPair.first && lPair.second)
                {
                    lSplitInfoPtr = &lSplitInfo;
                    lSplitInfo.splitId = *((uint*)lPair.first);
                    lSplitInfo.splitCount = *((uint*)lPair.second);
                }

                if(lTask->IsLazyReadWrite(lAddressSpace))
                {
                    lMemoryManager->CopyShadowMemPage(lStub, lSubtaskId, lSplitInfoPtr, lAddressSpace, lTask, lShadowMemOffset, lShadowMemBaseAddr, (void*)(pSigInfo->si_addr));
                }
                else    // Write only address space
                {
                	size_t lPageSize = lMemoryManager->GetVirtualMemoryPageSize();
                    size_t lMemAddr = reinterpret_cast<size_t>((void*)(pSigInfo->si_addr));
                    size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
                    size_t lMemOffset = (lPageAddr - reinterpret_cast<size_t>(lShadowMemBaseAddr));
                    size_t lOffset = lShadowMemOffset + lMemOffset;
                    
                    uint lAddressSpaceIndex = lTask->GetAddressSpaceIndex(lAddressSpace);

                    lMemoryManager->SetLazyProtection(reinterpret_cast<void*>(lPageAddr), lPageSize, true, true);
                    lTask->GetSubscriptionManager().AddWriteOnlyLazyUnprotection(lStub, lSubtaskId, lSplitInfoPtr, lAddressSpaceIndex, lMemOffset / lPageSize);
                    lTask->GetSubscriptionManager().InitializeWriteOnlyLazyMemory(lStub, lSubtaskId, lSplitInfoPtr, lAddressSpaceIndex, lTask, lAddressSpace, lOffset, reinterpret_cast<void*>(lPageAddr), lPageSize);
                }
            }
            else
            {
                abort();
            }
        }
    }
    catch(pmPrematureExitException&)
    {
    }
    catch(...)
    {
        abort();
    }
#endif
}

linuxMemManager::addressSpaceSpecifics::addressSpaceSpecifics()
    : mInFlightLock __LOCK_NAME__("linuxMemManager::addressSpaceSpecifics::mInFlightLock")
{
}
    
/* pmLinuxMemoryManager::sharedMemAutoPtr */
pmLinuxMemoryManager::sharedMemAutoPtr::sharedMemAutoPtr(const char* pSharedMemName)
    : mSharedMemName(pSharedMemName)
{
}
    
pmLinuxMemoryManager::sharedMemAutoPtr::~sharedMemAutoPtr()
{
    if(shm_unlink(mSharedMemName) == -1)
        PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::SHM_UNLINK_FAILED));
}


}

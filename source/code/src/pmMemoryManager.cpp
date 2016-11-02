
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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
void __dump_mem_req(const pmAddressSpace* addressSpace, const void* addr, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host);
    
void __dump_mem_req(const pmAddressSpace* addressSpace, const void* addr, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host)
{
    char lStr[512];
   
    sprintf(lStr, "Requesting memory %p (address space %p) at offset %ld (Remote Offset %ld) for length %ld (step %ld, count %ld) from host %d", addr, addressSpace, receiverOffset, offset, length, step, count, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_REQ_DUMP(addressSpace, addr, receiverOffset, offset, length, step, count, host) __dump_mem_req(addressSpace, addr, receiverOffset, offset, length, step, count, host);
#else
#define MEM_REQ_DUMP(addressSpace, addr, receiverOffset, offset, length, step, count, host)
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

// This method assumes nothing is in flight
uint pmLinuxMemoryManager::GetScatteredMemoryFetchEvents(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);

    auto lBlocks = pAddressSpace->GetRemoteRegionsInfo(pScatteredSubscriptionInfo);

    size_t lCount = 0;
    for_each(lBlocks, [&] (const typename decltype(lBlocks)::value_type& pMapKeyValue)
    {
        lCount += pMapKeyValue.second.size();
    });
    
    return (uint)lCount;
}

// This method assumes nothing is in flight
ulong pmLinuxMemoryManager::GetScatteredMemoryFetchPages(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);
    
    size_t lPageSize = GetVirtualMemoryPageSize();
    auto lBlocks = pAddressSpace->GetRemoteRegionsInfo(pScatteredSubscriptionInfo);

    ulong lPages = 0;
    for_each(lBlocks, [&] (const typename decltype(lBlocks)::value_type& pMapKeyValue)
    {
        for_each(pMapKeyValue.second, [&] (const std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>& pPair)
        {
            DEBUG_EXCEPTION_ASSERT(pPair.first.size && pPair.first.step && pPair.first.count);
            
            lPages += ((pPair.first.size * pPair.first.count) + lPageSize - 1) / lPageSize;
        });
    });

    return lPages;
}

#ifdef CENTRALIZED_AFFINITY_COMPUTATION
// This method assumes nothing is in flight
uint pmLinuxMemoryManager::GetScatteredMemoryFetchEventsForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);

    auto lBlocks = pAddressSpace->GetRemoteRegionsInfo(pScatteredSubscriptionInfo);

    size_t lCount = 0;
    for_each(lBlocks, [&] (const typename decltype(lBlocks)::value_type& pMapKeyValue)
    {
        lCount += pMapKeyValue.second.size();
    });
    
    return (uint)lCount;
}

// This method assumes nothing is in flight
ulong pmLinuxMemoryManager::GetScatteredMemoryFetchPagesForMachine(pmAddressSpace* pAddressSpace, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, const pmMachine* pMachine)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);
    
    size_t lPageSize = GetVirtualMemoryPageSize();
    auto lBlocks = pAddressSpace->GetRemoteRegionsInfo(pScatteredSubscriptionInfo);

    ulong lPages = 0;
    for_each(lBlocks, [&] (const typename decltype(lBlocks)::value_type& pMapKeyValue)
    {
        for_each(pMapKeyValue.second, [&] (const std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>& pPair)
        {
            DEBUG_EXCEPTION_ASSERT(pPair.first.size && pPair.first.step && pPair.first.count);
            
            lPages += ((pPair.first.size * pPair.first.count) + lPageSize - 1) / lPageSize;
        });
    });

    return lPages;
}
#endif
    
void pmLinuxMemoryManager::FetchScatteredMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, std::vector<pmCommandPtr>& pCommandVector)
{
    EXCEPTION_ASSERT(pScatteredSubscriptionInfo.size && pScatteredSubscriptionInfo.step && pScatteredSubscriptionInfo.count);

    using namespace linuxMemManager;
    void* lMem = pAddressSpace->GetMem();
    
    std::set<pmCommandPtr> lCommandsAlreadyIssuedSet;
    pmScatteredTransferMapType lMachineVersusTupleVectorMap = pAddressSpace->SetupRemoteRegionsForFetching(pScatteredSubscriptionInfo, pPriority, lCommandsAlreadyIssuedSet);
    
#ifdef GROUP_SCATTERED_REQUESTS
    for_each(lMachineVersusTupleVectorMap, [&] (pmScatteredTransferMapType::value_type& pMapKeyValue)
    {
        if(pMapKeyValue.second.size() == 1)
        {
            auto& lTuple = *pMapKeyValue.second.begin();

            const pmScatteredSubscriptionInfo& lInfo = std::get<0>(lTuple);
            const vmRangeOwner& lRangeOwner = std::get<1>(lTuple);
            pmCommandPtr& lCommandPtr = std::get<2>(lTuple);

            FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, lMem, communicator::TRANSFER_SCATTERED, lInfo.offset, lInfo.size, lInfo.step, lInfo.count, lRangeOwner, lCommandPtr);
            
            pCommandVector.emplace_back(std::move(lCommandPtr));
        }
        else
        {
            std::vector<pmCommandPtr> lCommandVector;
            FetchNonOverlappingScatteredMemoryRegions(pPriority, pAddressSpace, lMem, pMapKeyValue.second, lCommandVector);

            if(!lCommandVector.empty())
                std::move(lCommandVector.begin(), lCommandVector.end(), std::back_inserter(pCommandVector));
        }
    });
#else
    for_each(lMachineVersusTupleVectorMap, [&] (pmScatteredTransferMapType::value_type& pMapKeyValue)
    {
        for_each(pMapKeyValue.second, [&] (std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>& pTuple)
        {
            const pmScatteredSubscriptionInfo& lInfo = std::get<0>(pTuple);
            const vmRangeOwner& lRangeOwner = std::get<1>(pTuple);
            pmCommandPtr& lCommandPtr = std::get<2>(pTuple);

            FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, lMem, communicator::TRANSFER_SCATTERED, lInfo.offset, lInfo.size, lInfo.step, lInfo.count, lRangeOwner, lCommandPtr);

            pCommandVector.emplace_back(std::move(lCommandPtr));
        });
    });
#endif
    
    pCommandVector.reserve(pCommandVector.size() + lCommandsAlreadyIssuedSet.size());
    pCommandVector.insert(pCommandVector.end(), lCommandsAlreadyIssuedSet.begin(), lCommandsAlreadyIssuedSet.end());
}

// This method assumes nothing is in flight
uint pmLinuxMemoryManager::GetMemoryFetchEvents(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength)
{
    EXCEPTION_ASSERT(pLength);
    
    pmMemOwnership lOwnerships;
    pAddressSpace->GetOwnersUnprotected(pOffset, pLength, lOwnerships);
    
    uint lEvents = 0;

    for_each(lOwnerships, [&] (pmMemOwnership::value_type& pPair)
    {
        vmRangeOwner& lRangeOwner = pPair.second.second;

        if(lRangeOwner.host != PM_LOCAL_MACHINE)
            ++lEvents;
    });
    
    return lEvents;
}

// This method assumes nothing is in flight
ulong pmLinuxMemoryManager::GetMemoryFetchPages(pmAddressSpace* pAddressSpace, size_t pOffset, size_t pLength)
{
    EXCEPTION_ASSERT(pLength);
    
    size_t lPageSize = GetVirtualMemoryPageSize();
    
    pmMemOwnership lOwnerships;
    pAddressSpace->GetOwnersUnprotected(pOffset, pLength, lOwnerships);
    
    ulong lPages = 0;

    for_each(lOwnerships, [&] (pmMemOwnership::value_type& pPair)
    {
        vmRangeOwner& lRangeOwner = pPair.second.second;

        if(lRangeOwner.host != PM_LOCAL_MACHINE)
            lPages += ((pPair.second.first + lPageSize - 1) / lPageSize);
    });
    
    return lPages;
}
    
void pmLinuxMemoryManager::FetchMemoryRegion(pmAddressSpace* pAddressSpace, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommandPtr>& pCommandVector)
{
    EXCEPTION_ASSERT(pLength);

    if(pAddressSpace->GetAddressSpaceType() == ADDRESS_SPACE_2D)
    {
        ulong lAddressSpaceCols = pAddressSpace->GetCols();
        size_t lOffset = ((ulong)(pOffset / lAddressSpaceCols)) * lAddressSpaceCols;    // Floor offset to a multiple of lAddressSpaceCols
        size_t lLength = ((ulong)((pLength + lAddressSpaceCols - 1) / lAddressSpaceCols)) * lAddressSpaceCols;    // Ceil length to a multiple of lAddressSpaceCols
        
        EXCEPTION_ASSERT(lLength <= lAddressSpaceCols * pAddressSpace->GetRows());

        std::set<pmCommandPtr> lCommandsAlreadyIssuedSet;
        pmScatteredTransferMapType lMachineVersusTupleVectorMap = pAddressSpace->SetupRemoteRegionsForFetching(pmScatteredSubscriptionInfo(lOffset, lAddressSpaceCols, lAddressSpaceCols, lLength / lAddressSpaceCols), pPriority, lCommandsAlreadyIssuedSet);
        
        for_each(lMachineVersusTupleVectorMap, [&] (const std::pair<const pmMachine*, std::vector<std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>>>& pMapKeyValue)
        {
            for_each(pMapKeyValue.second, [&] (const std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>& pTuple)
            {
                const pmScatteredSubscriptionInfo& lInfo = std::get<0>(pTuple);
                const vmRangeOwner& lRangeOwner = std::get<1>(pTuple);
                const pmCommandPtr& lCommandPtr = std::get<2>(pTuple);

                FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, pAddressSpace->GetMem(), communicator::TRANSFER_SCATTERED, lInfo.offset, lInfo.size, lInfo.step, lInfo.count, lRangeOwner, lCommandPtr);

                pCommandVector.emplace_back(std::move(lCommandPtr));
            });
        });
    }
    else
    {
        pmLinearTransferVectorType lTupleVector = pAddressSpace->SetupRemoteRegionsForFetching(pmSubscriptionInfo(pOffset, pLength), pPriority, pCommandVector);

        for_each(lTupleVector, [&] (std::tuple<pmSubscriptionInfo, vmRangeOwner, pmCommandPtr>& pTuple)
        {
            const pmSubscriptionInfo& lInfo = std::get<0>(pTuple);
            const vmRangeOwner& lRangeOwner = std::get<1>(pTuple);
            pmCommandPtr& lCommandPtr = std::get<2>(pTuple);

            FetchNonOverlappingMemoryRegion(pPriority, pAddressSpace, pAddressSpace->GetMem(), communicator::TRANSFER_GENERAL, lInfo.offset, lInfo.length, 0, 0, lRangeOwner, lCommandPtr);

            pCommandVector.emplace_back(std::move(lCommandPtr));
        });
    }
}

// This method must be called with mInFlightLock on mInFlightMap of the address space acquired
void pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, communicator::memoryTransferType pTransferType, size_t pOffset, size_t pLength, size_t pStep, size_t pCount, const vmRangeOwner& pRangeOwner, const pmCommandPtr& pCommand)
{
    using namespace linuxMemManager;

    pmTask* lLockingTask = pAddressSpace->GetLockingTask();
    
    uint lOriginatingHost = lLockingTask ? (uint)(*(lLockingTask->GetOriginatingHost())) : std::numeric_limits<uint>::max();
    ulong lSequenceNumber = lLockingTask ? lLockingTask->GetSequenceNumber() : std::numeric_limits<ulong>::max();

	finalize_ptr<communicator::memoryTransferRequest> lData(new communicator::memoryTransferRequest(pRangeOwner.memIdentifier, communicator::memoryIdentifierStruct(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber()), pTransferType, pOffset, pRangeOwner.hostOffset, pLength, pStep, pCount, *PM_LOCAL_MACHINE, 0, (ushort)(lLockingTask != NULL), lOriginatingHost, lSequenceNumber, pPriority));
    
	pmCommunicatorCommandPtr lSendCommand = pmCommunicatorCommand<communicator::memoryTransferRequest>::CreateSharedPtr(pPriority, communicator::SEND, communicator::MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host, communicator::MEMORY_TRANSFER_REQUEST_STRUCT, lData, 1);

#ifdef ENABLE_TASK_PROFILING
    if(lLockingTask)
        pAddressSpace->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, true);
#endif
    
    MEM_REQ_DUMP(pAddressSpace, pMem, pOffset, pRangeOwner.hostOffset, pLength, pStep, pCount, (uint)(*pRangeOwner.host));

	pmCommunicator::GetCommunicator()->Send(lSendCommand);
}

// This method must be called with mInFlightLock on mInFlightMap of the address space acquired
void pmLinuxMemoryManager::FetchNonOverlappingScatteredMemoryRegions(ushort pPriority, pmAddressSpace* pAddressSpace, void* pMem, std::vector<std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>>& pVector, std::vector<pmCommandPtr>& pCommandVector)
{
    pmTask* lLockingTask = pAddressSpace->GetLockingTask();
    
    uint lOriginatingHost = lLockingTask ? (uint)(*(lLockingTask->GetOriginatingHost())) : std::numeric_limits<uint>::max();
    ulong lSequenceNumber = lLockingTask ? lLockingTask->GetSequenceNumber() : std::numeric_limits<ulong>::max();
    
    finalize_ptr<std::vector<communicator::scatteredMemoryTransferRequestCombinedStruct>> lAutoPtr(new std::vector<communicator::scatteredMemoryTransferRequestCombinedStruct>());
    std::vector<communicator::scatteredMemoryTransferRequestCombinedStruct>* lVector = lAutoPtr.get_ptr();

    lVector->reserve(pVector.size());
    pCommandVector.reserve(pVector.size());
    
    for_each(pVector, [&] (const std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>& pTuple)
    {
        const pmScatteredSubscriptionInfo& lInfo = std::get<0>(pTuple);
        const vmRangeOwner& lRangeOwner = std::get<1>(pTuple);
        const pmCommandPtr& lCommandPtr = std::get<2>(pTuple);

        lVector->emplace_back(lInfo.offset, lRangeOwner.hostOffset, lInfo.size, lInfo.step, lInfo.count);
    
    #ifdef ENABLE_TASK_PROFILING
        if(lLockingTask)
            pAddressSpace->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(lLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, true);
    #endif
     
        pCommandVector.emplace_back(lCommandPtr);
    });

    communicator::memoryIdentifierStruct lDestStruct(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber());

    finalize_ptr<communicator::scatteredMemoryTransferRequestCombinedPacked> lPackedData(new communicator::scatteredMemoryTransferRequestCombinedPacked(std::get<1>(pVector[0]).memIdentifier, lDestStruct, *PM_LOCAL_MACHINE, (ushort)(lLockingTask != NULL), lOriginatingHost, lSequenceNumber, pPriority, lAutoPtr));
    
    pmCommunicatorCommandPtr lSendCommand = pmCommunicatorCommand<communicator::scatteredMemoryTransferRequestCombinedPacked>::CreateSharedPtr(pPriority, communicator::SEND, communicator::SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_TAG, std::get<1>(pVector[0]).host, communicator::SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_PACKED, lPackedData, 1);

//    MEM_REQ_DUMP(pAddressSpace, pMem, pOffset, pRangeOwner.hostOffset, pLength, pStep, pCount, (uint)(*pRangeOwner.host));

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lSendCommand);
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
            PMLIB_MEMCPY(lDestAddr, lSrcAddr, lMaxSrcAddr - reinterpret_cast<size_t>(lSrcAddr), std::string("pmLinuxMemoryManager::CopyShadowMemPage1"));
        else
            PMLIB_MEMCPY(lDestAddr, lSrcAddr, lPageSize, std::string("pmLinuxMemoryManager::CopyShadowMemPage2"));
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
                PMLIB_MEMCPY(lCopyAddr, lSrcAddr, lMaxSrcAddr - reinterpret_cast<size_t>(lSrcAddr), std::string("pmLinuxMemoryManager::CopyShadowMemPage3"));
            else
                PMLIB_MEMCPY(lCopyAddr, lSrcAddr, std::min(lPageSize, pInfo.length), std::string("pmLinuxMemoryManager::CopyShadowMemPage4"));
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
    : mSharedMemDescriptor(-1)
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

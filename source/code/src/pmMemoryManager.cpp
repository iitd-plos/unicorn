
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

namespace pm
{
        
#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_req(const pmMemSection* memSection, const void* addr, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_req(const pmMemSection* memSection, const void* addr, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
   
    if(memSection->IsInput())
        sprintf(lStr, "Requesting input memory %p (Mem Section %p) at offset %ld (Remote Offset %ld) for length %ld from host %d", addr, memSection, receiverOffset, offset, length, host);
    else
        sprintf(lStr, "Requesting output memory %p (Mem Section %p) at offset %ld (Remote Offset %ld) for length %ld from host %d", addr, memSection, receiverOffset, offset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_REQ_DUMP(memSection, addr, receiverOffset, offset, length, host) __dump_mem_req(memSection, addr, receiverOffset, offset, length, host);
#else
#define MEM_REQ_DUMP(memSection, addr, receiverOffset, offset, length, host)    
#endif
    

/* class pmLinuxMemoryManager */
pmLinuxMemoryManager::pmLinuxMemoryManager()
    : mMemSectionSpecificsMapLock __LOCK_NAME__("pmLinuxMemoryManager::mMemSectionSpecificsMapLock")
#ifdef TRACK_MEMORY_ALLOCATIONS
	, mTotalAllocatedMemory(0)
	, mTotalAllocations(0)
	, mTotalDeallocations(0)
	, mTotalLazySegFaults(0)
    , mTotalAllocationTime(0)
    , mTrackLock __LOCK_NAME__("pmLinuxMemoryManager::mTrackLock")
#endif
{
#ifdef SUPPORT_LAZY_MEMORY
	InstallSegFaultHandler();
#endif

	mPageSize = ::getpagesize();
}

pmLinuxMemoryManager::~pmLinuxMemoryManager()
{
#ifdef SUPPORT_LAZY_MEMORY
	UninstallSegFaultHandler();
#endif

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
    
void* pmLinuxMemoryManager::CreateReadOnlyMemoryMapping(pmMemSection* pMemSection)
{
#ifdef TRACK_MEMORY_ALLOCATIONS
    double lTrackTime = GetCurrentTimeInSecs();
#endif

    linuxMemManager::memSectionSpecifics& lSpecifics = GetMemSectionSpecifics(pMemSection);
    
    void* lPtr = CreateMemoryMapping(lSpecifics.mSharedMemDescriptor, pMemSection->GetAllocatedLength(), false, false);
    
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

void* pmLinuxMemoryManager::AllocatePageAlignedMemoryInternal(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount, int& pSharedMemDescriptor)
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
    if(pMemSection)
    {
        const char* lSharedMemName = pMemSection->GetName();

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
    
void pmLinuxMemoryManager::CreateMemSectionSpecifics(pmMemSection* pMemSection, int pSharedMemDescriptor)
{
    FINALIZE_RESOURCE_PTR(dMemSectionSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemSectionSpecificsMapLock, Lock(), Unlock());

    if(mMemSectionSpecificsMap.find(pMemSection) != mMemSectionSpecificsMap.end())
        PMTHROW(pmFatalErrorException());
    
    mMemSectionSpecificsMap[pMemSection].mSharedMemDescriptor = pSharedMemDescriptor;
}

linuxMemManager::memSectionSpecifics& pmLinuxMemoryManager::GetMemSectionSpecifics(pmMemSection* pMemSection)
{
    FINALIZE_RESOURCE_PTR(dMemSectionSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemSectionSpecificsMapLock, Lock(), Unlock());

    if(mMemSectionSpecificsMap.find(pMemSection) == mMemSectionSpecificsMap.end())
        PMTHROW(pmFatalErrorException());
    
    return mMemSectionSpecificsMap[pMemSection];
}

void* pmLinuxMemoryManager::AllocateMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount)
{
    int lSharedMemDescriptor = -1;

    void* lPtr = AllocatePageAlignedMemoryInternal(pMemSection, pLength, pPageCount, lSharedMemDescriptor);
    
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalAllocatedMemory += pLength;
        ++mTotalAllocations;
    }
#endif

    if(pMemSection)
        CreateMemSectionSpecifics(pMemSection, lSharedMemDescriptor);

    return lPtr;
}

#ifdef SUPPORT_LAZY_MEMORY
void* pmLinuxMemoryManager::CreateCheckOutMemory(size_t pLength, bool pIsLazy)
{
    size_t lPageCount = 0;
    FindAllocationSize(pLength, lPageCount);

#ifdef MACOS
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
#endif

pmStatus pmLinuxMemoryManager::DeallocateMemory(pmMemSection* pMemSection)
{
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dMemSectionSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemSectionSpecificsMapLock, Lock(), Unlock());
    
        if(mMemSectionSpecificsMap.find(pMemSection) == mMemSectionSpecificsMap.end())
            PMTHROW(pmFatalErrorException());
    
        mMemSectionSpecificsMap.erase(pMemSection);
    }

    #ifdef SUPPORT_LAZY_MEMORY
        DeleteMemoryMapping(pMemSection->GetMem(), pMemSection->GetAllocatedLength());
    #else
        ::free(pMemSection->GetMem());
    #endif

#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalDeallocations;
    }
#endif

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::DeallocateMemory(void* pMem)
{
	::free(pMem);

#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalDeallocations;
    }
#endif

	return pmSuccess;
}
    
size_t pmLinuxMemoryManager::FindAllocationSize(size_t pLength, size_t& pPageCount)
{
	size_t lPageSize = GetVirtualMemoryPageSize();
	pPageCount = ((pLength/lPageSize) + ((pLength%lPageSize != 0)?1:0));

	return (pPageCount*lPageSize);
}

size_t pmLinuxMemoryManager::GetVirtualMemoryPageSize()
{
	return mPageSize;
}

// This function must be called after acquiring lock on pInFlightMap
void pmLinuxMemoryManager::FindRegionsNotInFlight(linuxMemManager::pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, std::vector<std::pair<ulong, ulong> >& pRegionsToBeFetched, std::vector<pmCommunicatorCommandPtr>& pCommandVector)
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

void pmLinuxMemoryManager::FetchMemoryRegion(pmMemSection* pMemSection, ushort pPriority, size_t pOffset, size_t pLength, std::vector<pmCommunicatorCommandPtr>& pCommandVector)
{
    if(!pLength)
        PMTHROW(pmFatalErrorException());

    using namespace linuxMemManager;
    void* lMem = pMemSection->GetMem();
    
	std::vector<std::pair<ulong, ulong> > lRegionsToBeFetched;	// Start address and last address of sub ranges to be fetched
    memSectionSpecifics& lSpecifics = GetMemSectionSpecifics(pMemSection);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

	FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());

    FindRegionsNotInFlight(lMap, lMem, pOffset, pLength, lRegionsToBeFetched, pCommandVector);

	size_t lRegionCount = lRegionsToBeFetched.size();

	for(size_t i=0; i<lRegionCount; ++i)
	{
		ulong lOffset = lRegionsToBeFetched[i].first - (ulong)lMem;
		ulong lLength = lRegionsToBeFetched[i].second - lRegionsToBeFetched[i].first + 1;
        
        if(lLength)
        {
            pmMemSection::pmMemOwnership lOwnerships;
            pMemSection->GetOwners(lOffset, lLength, lOwnerships);

            pmMemSection::pmMemOwnership::iterator lIter = lOwnerships.begin(), lEndIter = lOwnerships.end();            
            for(; lIter != lEndIter; ++lIter)
            {
                ulong lInternalOffset = lIter->first;
                ulong lInternalLength = lIter->second.first;
                pmMemSection::vmRangeOwner& lRangeOwner = lIter->second.second;

                if(lRangeOwner.host != PM_LOCAL_MACHINE)
                {
                    pmCommunicatorCommandPtr lCommand;
                    FetchNonOverlappingMemoryRegion(pPriority, pMemSection, lMem, lInternalOffset, lInternalLength, lRangeOwner, lMap, lCommand);

                    if(lCommand.get())
                        pCommandVector.push_back(lCommand);
                }
            }
        }
	}
}

void pmLinuxMemoryManager::FetchNonOverlappingMemoryRegion(ushort pPriority, pmMemSection* pMemSection, void* pMem, size_t pOffset, size_t pLength, pmMemSection::vmRangeOwner& pRangeOwner, linuxMemManager::pmInFlightRegions& pInFlightMap, pmCommunicatorCommandPtr& pCommand)
{	
    using namespace linuxMemManager;
    
	regionFetchData lFetchData;

	pmCommunicatorCommand::memoryTransferRequest* lData = new pmCommunicatorCommand::memoryTransferRequest();
	lData->sourceMemIdentifier.memOwnerHost = pRangeOwner.memIdentifier.memOwnerHost;
	lData->sourceMemIdentifier.generationNumber = pRangeOwner.memIdentifier.generationNumber;
	lData->destMemIdentifier.memOwnerHost = *(pMemSection->GetMemOwnerHost());
	lData->destMemIdentifier.generationNumber = pMemSection->GetGenerationNumber();
    lData->receiverOffset = pOffset;
	lData->offset = pRangeOwner.hostOffset;
	lData->length = pLength;
	lData->destHost = *PM_LOCAL_MACHINE;
    lData->isForwarded = 0;
    
	lFetchData.sendCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host,	pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT, (void*)lData, 1, NULL, 0);

	pmCommunicator::GetCommunicator()->Send(lFetchData.sendCommand);

    MEM_REQ_DUMP(pMemSection, pMem, pOffset, pRangeOwner.hostOffset, pLength, (uint)(*pRangeOwner.host));
        
	lFetchData.receiveCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::RECEIVE, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host, pmCommunicatorCommand::BYTE, NULL, 0, NULL, 0);	// Dummy command just to allow threads to wait on it
    
	char* lAddr = (char*)pMem + lData->receiverOffset;
    pInFlightMap[lAddr] = std::make_pair(lData->length, lFetchData);

#ifdef ENABLE_TASK_PROFILING
    if(pMemSection->GetLockingTask())
        pMemSection->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(pMemSection->IsInput()?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, true);
#endif
    
	lFetchData.receiveCommand->MarkExecutionStart();
	pCommand = lFetchData.receiveCommand;
}
    
pmStatus pmLinuxMemoryManager::CopyReceivedMemory(pmMemSection* pMemSection, ulong pOffset, ulong pLength, void* pSrcMem)
{
    using namespace linuxMemManager;

    if(!pLength)
        PMTHROW(pmFatalErrorException());
    
    memSectionSpecifics& lSpecifics = GetMemSectionSpecifics(pMemSection);
    pmInFlightRegions& lMap = lSpecifics.mInFlightMemoryMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS& lLock = lSpecifics.mInFlightLock;

    FINALIZE_RESOURCE_PTR(dInFlightLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lLock, Lock(), Unlock());
    
    void* lDestMem = pMemSection->GetMem();
    char* lAddr = (char*)lDestMem + pOffset;
    
    pmInFlightRegions::iterator lIter = lMap.find(lAddr);
    if((lIter != lMap.end()) && (lIter->second.first == pLength))
    {
        std::pair<size_t, regionFetchData>& lPair = lIter->second;
        memcpy((void*)lAddr, pSrcMem, pLength);

        regionFetchData& lData = lPair.second;
        pMemSection->AcquireOwnershipImmediate(pOffset, lPair.first);

    #ifdef ENABLE_TASK_PROFILING
        pmTask* lLockingTask = pMemSection->GetLockingTask();
        if(lLockingTask)
            lLockingTask->GetTaskProfiler()->RecordProfileEvent(pMemSection->IsInput()?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, false);
    #endif

    #ifdef ENABLE_MEM_PROFILING
        if(pLength)
            pMemSection->RecordMemReceive(pLength);
    #endif

        delete (pmCommunicatorCommand::memoryTransferRequest*)(lData.sendCommand->GetData());
        lData.receiveCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(lData.receiveCommand));

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
            pMemSection->AcquireOwnershipImmediate(lOffset, lPair.first);
            
        #ifdef ENABLE_TASK_PROFILING
            pmTask* lLockingTask = pMemSection->GetLockingTask();
            if(lLockingTask)
                pMemSection->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(pMemSection->IsInput()?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, false);
        #endif

        #ifdef ENABLE_MEM_PROFILING
            if(pLength)
                pMemSection->RecordMemReceive(pLength);
        #endif

            delete (pmCommunicatorCommand::memoryTransferRequest*)(lData.sendCommand->GetData());
            lData.receiveCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(lData.receiveCommand));

            lSpecifics.mInFlightMemoryMap.erase(lBaseIter);
        }
        else
        {            
            // Make partial receive entry
            lPartialReceiveRecordMap[lRecvAddr] = pLength;
            
            memcpy((void*)lAddr, pSrcMem, pLength);

            size_t lOffset = lRecvAddr - reinterpret_cast<size_t>(lDestMem);
            pMemSection->AcquireOwnershipImmediate(lOffset, pLength);
        }
    }
    
    return pmSuccess;
}
    
#ifdef SUPPORT_LAZY_MEMORY
pmStatus pmLinuxMemoryManager::SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed)
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
    
    return pmSuccess;
}

pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, void* pLazyMemAddr, uint pForwardPrefetchPageCount)
{
	size_t lPageSize = GetVirtualMemoryPageSize();
    size_t lBytesToBeFetched = (1 + pForwardPrefetchPageCount) * lPageSize;

	size_t lStartAddr = reinterpret_cast<size_t>(pMemSection->GetMem());
	size_t lLength = pMemSection->GetLength();
	size_t lLastAddr = lStartAddr + lLength;

	size_t lMemAddr = reinterpret_cast<size_t>(pLazyMemAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);

	size_t lOffset = lPageAddr - lStartAddr;
	size_t lLeftoverLength = lLastAddr - lPageAddr;

	if(lLeftoverLength > lBytesToBeFetched)
		lLeftoverLength = lBytesToBeFetched;

    pmTask* lLockingTask = pMemSection->GetLockingTask();
    ushort lPriority = (lLockingTask ? lLockingTask->GetPriority() : MAX_CONTROL_PRIORITY);
    
    if(!pMemSection->IsRegionLocallyOwned(lOffset, ((lLeftoverLength > lPageSize) ? lPageSize : lLeftoverLength)))
    {
        // We want to fetch lazy memory page and prefetch pages collectively. But we do want this thread to resume execution as soon as
        // lazy memory page is fetched without waiting for prefetch pages to come. To do this, we make two FetchMemory requests - first with
        // lazy memory page and prefetch pages and second with lazy memory page only. The in-flight memory system will piggy back second
        // request onto the first one. The current thread only waits on commands returned by second FetchMemory statement.
        std::vector<pmCommunicatorCommandPtr> lCommandVector;
        if(pForwardPrefetchPageCount)
            FetchMemoryRegion(pMemSection, lPriority, lOffset, lLeftoverLength, lCommandVector);

        lCommandVector.clear();
        FetchMemoryRegion(pMemSection, lPriority, lOffset, ((lLeftoverLength > lPageSize) ? lPageSize : lLeftoverLength), lCommandVector);

        pStub->WaitForNetworkFetch(lCommandVector);
    }

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, void* pLazyMemAddr)
{
    return LoadLazyMemoryPage(pStub, pSubtaskId, pMemSection, pLazyMemAddr, pMemSection->GetLazyForwardPrefetchPageCount());
}

pmStatus pmLinuxMemoryManager::CopyLazyInputMemPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, void* pFaultAddr)
{
#ifdef _DEBUG
    if(pMemSection->IsOutput() || !pMemSection->IsLazy())
        PMTHROW(pmFatalErrorException());
#endif

	size_t lPageSize = GetVirtualMemoryPageSize();
	size_t lMemAddr = reinterpret_cast<size_t>(pFaultAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
    size_t lOffset = (lPageAddr - reinterpret_cast<size_t>(pMemSection->GetReadOnlyLazyMemoryMapping()));
    
    void* lDestAddr = reinterpret_cast<void*>(lPageAddr);
    void* lSrcAddr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pMemSection->GetMem()) + lOffset);

    LoadLazyMemoryPage(pStub, pSubtaskId, pMemSection, lSrcAddr);
    SetLazyProtection(lDestAddr, lPageSize, true, true);    // we may actually not allow writes here at all and abort if a write access is done to RO memory
    
    return pmSuccess;
}

pmStatus pmLinuxMemoryManager::CopyShadowMemPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr)
{
#ifdef _DEBUG
    if(pMemSection->IsInput() || !pMemSection->IsLazyReadWrite())
        PMTHROW(pmFatalErrorException());
#endif
    
	size_t lPageSize = GetVirtualMemoryPageSize();
	size_t lMemAddr = reinterpret_cast<size_t>(pFaultAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
    size_t lOffset = pShadowMemOffset + (lPageAddr - reinterpret_cast<size_t>(pShadowMemBaseAddr));
    size_t lSrcMemBaseAddr = reinterpret_cast<size_t>(pMemSection->GetMem());
    
    void* lDestAddr = reinterpret_cast<void*>(lPageAddr);
    void* lSrcAddr = reinterpret_cast<void*>(lSrcMemBaseAddr + lOffset);

    LoadLazyMemoryPage(pStub, pSubtaskId, pMemSection, lSrcAddr);
    SetLazyProtection(lDestAddr, lPageSize, true, true);
    
    size_t lMaxSrcAddr = lSrcMemBaseAddr + pMemSection->GetLength();
    size_t lMaxCopyAddr = reinterpret_cast<size_t>(lSrcAddr) + lPageSize;
    if(lMaxCopyAddr > lMaxSrcAddr)
        ::memcpy(lDestAddr, lSrcAddr, lMaxSrcAddr - reinterpret_cast<size_t>(lSrcAddr));
    else
        ::memcpy(lDestAddr, lSrcAddr, lPageSize);
    
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
    ACCUMULATION_TIMER(Timer_ACC, "SegFaultHandler");
    
    const std::pair<void*, void*>& lPair = TLS_IMPLEMENTATION_CLASS::GetTls()->GetThreadLocalStoragePair(TLS_EXEC_STUB, TLS_CURRENT_SUBTASK_ID);
    pmExecutionStub* lStub = static_cast<pmExecutionStub*>(lPair.first);
    void* lSubtaskPtr = lPair.second;
    if(!lStub || !lSubtaskPtr)
        abort();
    
    ulong lSubtaskId = *(static_cast<ulong*>(lSubtaskPtr));
    
    subscription::pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(lStub, lSubtaskId);

    try
    {
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
    
        /* Check if the address belongs to a lazy input memory */
        pmMemSection* lMemSection = pmMemSection::FindMemSectionContainingLazyAddress((void*)(pSigInfo->si_addr));
        if(lMemSection)
        {
            if(lMemoryManager->CopyLazyInputMemPage(lStub, lSubtaskId, lMemSection, (void*)(pSigInfo->si_addr)) != pmSuccess)
                abort();
        }
        else    /* Check if the address belongs to a lazy output memory */
        {
            lMemSection = pmSubscriptionManager::FindMemSectionContainingShadowAddr((void*)(pSigInfo->si_addr), lShadowMemOffset, lShadowMemBaseAddr);
            if(lMemSection && lShadowMemBaseAddr)
            {
                if(lMemSection->IsLazyReadWrite())
                {
                    if(lMemoryManager->CopyShadowMemPage(lStub, lSubtaskId, lMemSection, lShadowMemOffset, lShadowMemBaseAddr, (void*)(pSigInfo->si_addr)) != pmSuccess)
                        abort();
                }
                else
                {
                    pmTask* lTask = lMemSection->GetLockingTask();
                    if(!lTask)
                        abort();
                
                	size_t lPageSize = lMemoryManager->GetVirtualMemoryPageSize();
                    size_t lMemAddr = reinterpret_cast<size_t>((void*)(pSigInfo->si_addr));
                    size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
                    size_t lOffset = lShadowMemOffset + (lPageAddr - reinterpret_cast<size_t>(lShadowMemBaseAddr));

                    lMemoryManager->SetLazyProtection(reinterpret_cast<void*>(lPageAddr), lPageSize, true, true);
                    lTask->GetSubscriptionManager().InitializeWriteOnlyLazyMemory(lStub, lSubtaskId, lOffset, reinterpret_cast<void*>(lPageAddr), lPageSize);
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
}
#endif

linuxMemManager::memSectionSpecifics::memSectionSpecifics()
    : mInFlightLock __LOCK_NAME__("linuxMemManager::memSectionSpecifics::mInFlightLock")
{
}

linuxMemManager::regionFetchData::regionFetchData()
{
    accumulatedPartialReceivesLength = 0;
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


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

#include <string.h>

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
    
	size_t lPageSize = GetVirtualMemoryPageSize();

	void* lPtr = NULL;
	void** lRef = (void**)(&lPtr);

	if(::posix_memalign(lRef, lPageSize, pLength) != 0)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MEM_ALIGN_FAILED));

	if(!lPtr)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::ALLOCATION_FAILED));
    
    return lPtr;
}
    
void pmLinuxMemoryManager::CreateMemSectionSpecifics(pmMemSection* pMemSection)
{
    FINALIZE_RESOURCE_PTR(dMemSectionSpecificsMapLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mMemSectionSpecificsMapLock, Lock(), Unlock());

    if(mMemSectionSpecificsMap.find(pMemSection) != mMemSectionSpecificsMap.end())
        PMTHROW(pmFatalErrorException());
    
    mMemSectionSpecificsMap[pMemSection].mInFlightMemoryMap.size();
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
    void* lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);
    
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalAllocatedMemory += pLength;
        ++mTotalAllocations;
    }
#endif

    if(pMemSection)
        CreateMemSectionSpecifics(pMemSection);
    
	return lPtr;
}

#ifdef SUPPORT_LAZY_MEMORY
void* pmLinuxMemoryManager::AllocateLazyMemory(pmMemSection* pMemSection, size_t& pLength, size_t& pPageCount)
{   
    void* lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);

#ifdef MACOS
    if(pPageCount < 31)
    {
    #if 0   //def _DEBUG
        std::cout << "WARNING: Less than 31 pages of lazy memory allocated on Mac OS X " << std::endl << std::flush;
    #endif
        
        ::free(lPtr);
        pLength = 31 * GetVirtualMemoryPageSize();
        
        lPtr = AllocatePageAlignedMemoryInternal(pLength, pPageCount);
    }
#endif

    // Input memory protection: Read/Write not allowed
    // Output memory protection: Read/Write allowed (Shadow mem will be marked lazy)
    // Shadow memory protection: Read/Write not allowed
    bool lIsInputOrShadow = (!pMemSection || pMemSection->IsInput());
    SetLazyProtection(lPtr, pLength, !lIsInputOrShadow, !lIsInputOrShadow);

#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        mTotalLazyMemory += pLength;
        ++mTotalAllocations;
    }
#endif

    if(pMemSection)
        CreateMemSectionSpecifics(pMemSection);
    
	return lPtr;
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
    
	::free(pMemSection->GetMem());

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

    MEM_REQ_DUMP(pMemSection, pMem, pOffset, pOwnerOffset, pLength, (uint)(*pOwnerMachine));
        
	lFetchData.receiveCommand = pmCommunicatorCommand::CreateSharedPtr(pPriority, pmCommunicatorCommand::RECEIVE, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host, pmCommunicatorCommand::BYTE, NULL, 0, NULL, 0);	// Dummy command just to allow threads to wait on it
    
	char* lAddr = (char*)pMem + lData->receiverOffset;
    pInFlightMap[lAddr] = std::make_pair(lData->length, lFetchData);

#ifdef ENABLE_TASK_PROFILING
    if(pMemSection->GetLockingTask())
        pMemSection->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(dynamic_cast<pmInputMemSection*>(pMemSection)?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, true);
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
        
        if(pMemSection->IsLazy() && pMemSection->IsInput())
        {
        #ifdef SUPPORT_LAZY_MEMORY
            SetLazyProtection((void*)lAddr, (size_t)pLength, false, true);
            memcpy((void*)lAddr, pSrcMem, pLength);
            SetLazyProtection((void*)lAddr, (size_t)pLength, true, true);
        #endif
        }
        else
        {
            memcpy((void*)lAddr, pSrcMem, pLength);
        }
        
        regionFetchData& lData = lPair.second;
        pMemSection->AcquireOwnershipImmediate(pOffset, lPair.first);

#ifdef ENABLE_TASK_PROFILING
        pmTask* lLockingTask = pMemSection->GetLockingTask();
        if(lLockingTask)
            lLockingTask->GetTaskProfiler()->RecordProfileEvent(dynamic_cast<pmInputMemSection*>(pMemSection)?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, false);
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
            if(pMemSection->IsLazy() && pMemSection->IsInput())
            {
#ifdef SUPPORT_LAZY_MEMORY                
                SetLazyProtection((void*)lAddr, pLength, false, true);
                memcpy((void*)lAddr, pSrcMem, pLength);
                SetLazyProtection(lBaseIter->first, lPair.first, true, true);
                
                typedef std::map<void*, std::vector<char> > partialPageReceiveBufferType;
                partialPageReceiveBufferType& lPartialPageReceiveBufferMap = lData.partialPageReceiveBufferMap;

                partialPageReceiveBufferType::iterator lBegin = lPartialPageReceiveBufferMap.begin();
                partialPageReceiveBufferType::iterator lEnd = lPartialPageReceiveBufferMap.end();
                
                for(; lBegin != lEnd; ++lBegin)
                    memcpy(lBegin->first, (void*)(&(lBegin->second[0])), lBegin->second.size());
#endif
            }
            else
            {
                memcpy((void*)lAddr, pSrcMem, pLength);                
            }

            size_t lOffset = lStartAddr - reinterpret_cast<size_t>(lDestMem);
            pMemSection->AcquireOwnershipImmediate(lOffset, lPair.first);
            
#ifdef ENABLE_TASK_PROFILING        
            pmTask* lLockingTask = pMemSection->GetLockingTask();
            if(lLockingTask)
                pMemSection->GetLockingTask()->GetTaskProfiler()->RecordProfileEvent(dynamic_cast<pmInputMemSection*>(pMemSection)?pmTaskProfiler::INPUT_MEMORY_TRANSFER:pmTaskProfiler::OUTPUT_MEMORY_TRANSFER, false);
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
            
            if(pMemSection->IsLazy() && pMemSection->IsInput())
            {
#ifdef SUPPORT_LAZY_MEMORY
                typedef std::map<void*, std::vector<char> > partialPageReceiveBufferType;
                partialPageReceiveBufferType& lPartialPageReceiveBufferMap = lData.partialPageReceiveBufferMap;

                size_t lPageSize = GetVirtualMemoryPageSize();

                size_t lPageStartAddr = GET_VM_PAGE_START_ADDRESS(lRecvAddr, lPageSize);
                if(lPageStartAddr < lRecvAddr)
                    lPageStartAddr += lPageSize;
                
                if(lPageStartAddr != lRecvAddr)
                {
                    lPartialPageReceiveBufferMap[lAddr].resize(lPageStartAddr - lRecvAddr);
                    memcpy((void*)(&(lPartialPageReceiveBufferMap[lAddr][0])), pSrcMem, lPageStartAddr - lRecvAddr);
                }
                
                for(; lPageStartAddr + lPageSize <= lRecvAddr + pLength; lPageStartAddr += lPageSize)
                {
                    void* lSrcAddr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSrcMem) + lPageStartAddr - lRecvAddr);
                    SetLazyProtection(reinterpret_cast<void*>(lPageStartAddr), lPageSize, false, true);
                    memcpy(reinterpret_cast<void*>(lPageStartAddr), lSrcAddr, lPageSize);
                    SetLazyProtection(reinterpret_cast<void*>(lPageStartAddr), lPageSize, true, true);

                    size_t lOffset = lPageStartAddr - reinterpret_cast<size_t>(lDestMem);
                    pMemSection->AcquireOwnershipImmediate(lOffset, lPageSize);
                }
                
                if(lPageStartAddr != lRecvAddr + pLength)
                {
                    void* lTempAddr = reinterpret_cast<void*>(lPageStartAddr);
                    void* lSrcAddr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSrcMem) + lPageStartAddr - lRecvAddr);
                    
                    lPartialPageReceiveBufferMap[lTempAddr].resize(lRecvAddr + pLength - lPageStartAddr);
                    memcpy((void*)(&(lPartialPageReceiveBufferMap[lTempAddr][0])), lSrcAddr, lRecvAddr + pLength - lPageStartAddr);                    
                }
#endif                
            }
            else
            {
                memcpy((void*)lAddr, pSrcMem, pLength);

                size_t lOffset = lRecvAddr - reinterpret_cast<size_t>(lDestMem);
                pMemSection->AcquireOwnershipImmediate(lOffset, pLength);
            }
        }
    }
    
    return pmSuccess;
}
    
#ifdef SUPPORT_LAZY_MEMORY
pmStatus pmLinuxMemoryManager::SetLazyProtection(void* pAddr, size_t pLength, bool pReadAllowed, bool pWriteAllowed)
{
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
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalLazySegFaults;
    }
#endif

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

    // We want to fetch lazy memory page and prefetch pages collectively. But we do want this thread to resume execution as soon as
    // lazy memory page is fetched without waiting for prefetch pages to come. To do this, we make two FetchMemory requests - first with
    // lazy memory page and prefetch pages and second with lazy memory page only. The in-flight memory system will piggy back second
    // request onto first one. The current thread only waits on commands returned by second FetchMemory statement.
    std::vector<pmCommunicatorCommandPtr> lCommandVector;
    if(pForwardPrefetchPageCount)
        FetchMemoryRegion(pMemSection, lPriority, lOffset, lLeftoverLength, lCommandVector);

    lCommandVector.clear();
	FetchMemoryRegion(pMemSection, lPriority, lOffset, ((lLeftoverLength > lPageSize) ? lPageSize : lLeftoverLength), lCommandVector);
    
    pmStatus lStatus;
    std::vector<pmCommunicatorCommandPtr>::const_iterator lIter = lCommandVector.begin(), lEndIter = lCommandVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const pmCommunicatorCommandPtr& lCommand = *lIter;
        if(lCommand.get())
        {
            while((lStatus = lCommand->GetStatus()) == pmStatusUnavailable)
            {
                if(lCommand->WaitWithTimeOut(GetIntegralCurrentTimeInSecs() + MEMORY_TRANSFER_TIMEOUT))
                {
                    if(pStub->RequiresPrematureExit(pSubtaskId))
                        PMTHROW_NODUMP(pmPrematureExitException());
                }
            }
        
            if(lStatus != pmSuccess)
                PMTHROW(pmMemoryFetchException());
        }
    }

	return pmSuccess;
}

pmStatus pmLinuxMemoryManager::LoadLazyMemoryPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, void* pLazyMemAddr)
{
    return LoadLazyMemoryPage(pStub, pSubtaskId, pMemSection, pLazyMemAddr, pMemSection->GetLazyForwardPrefetchPageCount());
}

pmStatus pmLinuxMemoryManager::CopyShadowMemPage(pmExecutionStub* pStub, ulong pSubtaskId, pmMemSection* pMemSection, size_t pShadowMemOffset, void* pShadowMemBaseAddr, void* pFaultAddr)
{
#ifdef TRACK_MEMORY_ALLOCATIONS
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dTrackLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTrackLock, Lock(), Unlock());
        ++mTotalLazySegFaults;
    }
#endif
    
	size_t lPageSize = GetVirtualMemoryPageSize();
	size_t lMemAddr = reinterpret_cast<size_t>(pFaultAddr);
	size_t lPageAddr = GET_VM_PAGE_START_ADDRESS(lMemAddr, lPageSize);
    size_t lOffset = pShadowMemOffset + (lPageAddr - reinterpret_cast<size_t>(pShadowMemBaseAddr));
    
    void* lDestAddr = reinterpret_cast<void*>(lPageAddr);
    void* lSrcAddr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pMemSection->GetMem()) + lOffset);

    LoadLazyMemoryPage(pStub, pSubtaskId, pMemSection, lSrcAddr);
    SetLazyProtection(lDestAddr, lPageSize, true, true);    
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
    pmExecutionStub* lStub = (pmExecutionStub*)(TLS_IMPLEMENTATION_CLASS::GetTls()->GetThreadLocalStorage(TLS_EXEC_STUB));
    void* lSubtaskPtr = TLS_IMPLEMENTATION_CLASS::GetTls()->GetThreadLocalStorage(TLS_CURRENT_SUBTASK_ID);
    if(!lStub || !lSubtaskPtr)
        abort();
    
    ulong lSubtaskId = *((ulong*)lSubtaskPtr);
        
    subscription::pmSubtaskTerminationCheckPointAutoPtr lSubtaskTerminationCheckPointAutoPtr(lStub, lSubtaskId);

    try
    {
        size_t lShadowMemOffset = 0;
        void* lShadowMemBaseAddr = NULL;

        pmLinuxMemoryManager* lMemoryManager = dynamic_cast<pmLinuxMemoryManager*>(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager());

        pmMemSection* lMemSection = pmSubscriptionManager::FindMemSectionContainingShadowAddr((void*)(pSigInfo->si_addr), lShadowMemOffset, lShadowMemBaseAddr);
        if(lMemSection)
        {
            if(!lMemSection->IsLazy() || !lShadowMemBaseAddr)
                abort();
        
            if(lMemoryManager->CopyShadowMemPage(lStub, lSubtaskId, lMemSection, lShadowMemOffset, lShadowMemBaseAddr, (void*)(pSigInfo->si_addr)) != pmSuccess)
                abort();
        }
        else
        {
            lMemSection = pmMemSection::FindMemSectionContainingAddress((void*)(pSigInfo->si_addr));
        
            if(!lMemSection || !lMemSection->IsLazy())
                abort();
        
            if(lMemoryManager->LoadLazyMemoryPage(lStub, lSubtaskId, lMemSection, (void*)(pSigInfo->si_addr)) != pmSuccess)
                abort();
        }
    }
    catch(pmPrematureExitException&)
    {
    }
}
#endif

linuxMemManager::regionFetchData::regionFetchData()
{
    accumulatedPartialReceivesLength = 0;
}

}

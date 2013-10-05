
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

#include "pmUtility.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmNetwork.h"
#include "pmCommunicator.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <dlfcn.h>	// For dlopen/dlclose/dlsym

namespace pm
{

STATIC_ACCESSOR(pmUtility::fileMappingsMapType, pmUtility, GetFileMappingsMap)
STATIC_ACCESSOR(pmUtility::pendingResponsesMapType, pmUtility, GetFileMappingPendingResponsesMap)
STATIC_ACCESSOR(pmUtility::pendingResponsesMapType, pmUtility, GetFileUnmappingPendingResponsesMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmUtility::ResourceLock"), pmUtility, GetResourceLock)
    
void pmUtility::MapFileOnAllMachines(const char* pPath)
{
    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::string lStr(pPath);
        pendingResponsesMapType& lPendingResponses = GetFileMappingPendingResponsesMap();
        if(lPendingResponses.find(lStr) != lPendingResponses.end())
            PMTHROW(pmFatalErrorException());
        
        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());
        lPendingResponses[lStr] = std::make_pair(lCount, lSignalWaitSharedPtr);
    }

    for(uint i = 0; i < lCount; ++i)
    {
        const pmMachine* lMachine = pmMachinePool::GetMachinePool()->GetMachine(i);
    
        if(lMachine == PM_LOCAL_MACHINE)
        {
            pmUtility::MapFile(pPath);
            RegisterFileMappingResponse(pPath);
        }
        else
        {
            finalize_ptr<communicator::fileOperationsStruct> lFileOperationsData(new communicator::fileOperationsStruct(pPath, communicator::MMAP_FILE, NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId()));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::fileOperationsStruct>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::FILE_OPERATIONS_TAG, lMachine, communicator::FILE_OPERATIONS_STRUCT, lFileOperationsData, 1);

            pmCommunicator::GetCommunicator()->Send(lCommand, false);
        }
    }
    
    lSignalWaitSharedPtr->Wait();
}

void pmUtility::UnmapFileOnAllMachines(const char* pPath)
{
    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::string lStr(pPath);
        pendingResponsesMapType& lPendingResponses = GetFileUnmappingPendingResponsesMap();
        if(lPendingResponses.find(lStr) != lPendingResponses.end())
            PMTHROW(pmFatalErrorException());
        
        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());
        lPendingResponses[lStr] = std::make_pair(lCount, lSignalWaitSharedPtr);
    }
    
    for(uint i = 0; i < lCount; ++i)
    {
        const pmMachine* lMachine = pmMachinePool::GetMachinePool()->GetMachine(i);
    
        if(lMachine == PM_LOCAL_MACHINE)
        {
            pmUtility::UnmapFile(pPath);
            RegisterFileUnmappingResponse(pPath);
        }
        else
        {
            finalize_ptr<communicator::fileOperationsStruct> lFileOperationsData(new communicator::fileOperationsStruct(pPath, communicator::MUNMAP_FILE, NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId()));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::fileOperationsStruct>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::FILE_OPERATIONS_TAG, lMachine, communicator::FILE_OPERATIONS_STRUCT, lFileOperationsData, 1);

            pmCommunicator::GetCommunicator()->Send(lCommand, false);
        }
    }

    lSignalWaitSharedPtr->Wait();
}

void pmUtility::RegisterFileMappingResponse(const char* pPath)
{
    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
    std::string lStr(pPath);
    pendingResponsesMapType& lPendingResponses = GetFileMappingPendingResponsesMap();

    pendingResponsesMapType::iterator lIter = lPendingResponses.find(lStr);
    if(lIter == lPendingResponses.end())
        PMTHROW(pmFatalErrorException());
    
    pendingResponsesMapType::mapped_type& lPair = lPendingResponses[lStr];
    --lPair.first;

    if(lPair.first == 0)
    {
        lPair.second->Signal();
        lPendingResponses.erase(lIter);
    }
}
    
void pmUtility::RegisterFileUnmappingResponse(const char* pPath)
{
    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
    std::string lStr(pPath);
    pendingResponsesMapType& lPendingResponses = GetFileUnmappingPendingResponsesMap();

    pendingResponsesMapType::iterator lIter = lPendingResponses.find(lStr);
    if(lIter == lPendingResponses.end())
        PMTHROW(pmFatalErrorException());
    
    pendingResponsesMapType::mapped_type& lPair = lPendingResponses[lStr];
    --lPair.first;
    
    if(lPair.first == 0)
    {
        lPair.second->Signal();
        lPendingResponses.erase(lIter);
    }
}

void pmUtility::SendFileMappingAcknowledgement(const char* pPath, const pmMachine* pSourceHost)
{
    finalize_ptr<communicator::fileOperationsStruct> lFileOperationsData(new communicator::fileOperationsStruct(pPath, communicator::MMAP_ACK, *pSourceHost));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::fileOperationsStruct>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::FILE_OPERATIONS_TAG, pSourceHost, communicator::FILE_OPERATIONS_STRUCT, lFileOperationsData, 1);

    pmCommunicator::GetCommunicator()->Send(lCommand, false);
}
    
void pmUtility::SendFileUnmappingAcknowledgement(const char* pPath, const pmMachine* pSourceHost)
{
    finalize_ptr<communicator::fileOperationsStruct> lFileOperationsData(new communicator::fileOperationsStruct(pPath, communicator::MUNMAP_ACK, *pSourceHost));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::fileOperationsStruct>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::FILE_OPERATIONS_TAG, pSourceHost, communicator::FILE_OPERATIONS_STRUCT, lFileOperationsData, 1);

    pmCommunicator::GetCommunicator()->Send(lCommand, false);
}

void pmUtility::MapFile(const char* pPath)
{
    std::string lStr(pPath);

    struct stat lStatBuf;
    int lFileDescriptor = -1;

    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

    fileMappingsMapType& lFileMappings = GetFileMappingsMap();
    if(lFileMappings.find(lStr) != lFileMappings.end())
        PMTHROW(pmFatalErrorException());

    if(((lFileDescriptor = open(pPath, O_RDONLY)) < 0) || (fstat(lFileDescriptor, &lStatBuf) < 0))
        PMTHROW(pmFatalErrorException());
    
    void* lMem = mmap(NULL, lStatBuf.st_size, PROT_READ, MAP_SHARED, lFileDescriptor, 0);
    if(lMem == MAP_FAILED || !lMem)
		PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MMAP_FAILED));
    
    if(close(lFileDescriptor) < 0)
        PMTHROW(pmFatalErrorException());

    lFileMappings[lStr].first = lMem;
    lFileMappings[lStr].second = lStatBuf.st_size;
}
    
void* pmUtility::GetMappedFile(const char* pPath)
{
    std::string lStr(pPath);

    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
    fileMappingsMapType& lFileMappings = GetFileMappingsMap();
    if(lFileMappings.find(lStr) == lFileMappings.end())
        PMTHROW(pmFatalErrorException());

    return lFileMappings[lStr].first;
}

void pmUtility::UnmapFile(const char* pPath)
{
    std::string lStr(pPath);

    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
    fileMappingsMapType& lFileMappings = GetFileMappingsMap();
    if(lFileMappings.find(lStr) == lFileMappings.end())
        PMTHROW(pmFatalErrorException());

    if(munmap(lFileMappings[lStr].first, lFileMappings[lStr].second) != 0)
        PMTHROW(pmVirtualMemoryException(pmVirtualMemoryException::MUNMAP_FAILED));
    
    lFileMappings.erase(lStr);
}

void* pmUtility::OpenLibrary(char* pPath)
{
	return dlopen(pPath, RTLD_LAZY | RTLD_LOCAL);
}

pmStatus pmUtility::CloseLibrary(void* pLibHandle)
{
	if(pLibHandle)
	{
		if(dlclose(pLibHandle) != 0)
			PMTHROW(pmIgnorableException(pmIgnorableException::LIBRARY_CLOSE_FAILURE));
	}

	return pmSuccess;
}

void* pmUtility::GetExportedSymbol(void* pLibHandle, char* pSymbol)
{
    static std::map<void*, std::map<std::string, void*>> sSymbolMap;    // map of libHandle versus map of symbol versus address in dynamic library

	if(!pLibHandle || !pSymbol)
		return NULL;

    decltype(sSymbolMap)::iterator lIter = sSymbolMap.find(pLibHandle);
    if(lIter == sSymbolMap.end())
        lIter = sSymbolMap.emplace(std::make_pair(pLibHandle, std::map<std::string, void*>())).first;
    
    std::string lSymbolStr(pSymbol);
    
    std::map<std::string, void*>::iterator lInnerIter = lIter->second.find(lSymbolStr);
    if(lInnerIter == lIter->second.end())
        lInnerIter = lIter->second.emplace(lSymbolStr, dlsym(pLibHandle, pSymbol)).first;
    
	return lInnerIter->second;
}

} // end namespace pm




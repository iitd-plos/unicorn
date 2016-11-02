
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

#include "pmUtility.h"
#include "pmHardware.h"
#include "pmDevicePool.h"
#include "pmNetwork.h"
#include "pmCommunicator.h"
#include "pmHeavyOperations.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <dlfcn.h>	// For dlopen/dlclose/dlsym

namespace pm
{

STATIC_ACCESSOR(ulong, pmUtility, GetMultiFileOperationsId)
STATIC_ACCESSOR(pmUtility::multiFileOperationsMapType, pmUtility, GetMultiFileOperationsMap)
STATIC_ACCESSOR(pmUtility::fileMappingsMapType, pmUtility, GetFileMappingsMap)
STATIC_ACCESSOR(pmUtility::pendingResponsesMapType, pmUtility, GetFileMappingPendingResponsesMap)
STATIC_ACCESSOR(pmUtility::pendingResponsesMapType, pmUtility, GetFileUnmappingPendingResponsesMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmUtility::ResourceLock"), pmUtility, GetResourceLock)
    
void pmUtility::MapFileOnAllMachines(const char* pPath)
{
    EXCEPTION_ASSERT(strlen(pPath) <= MAX_FILE_SIZE_LEN - 1);

    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::string lStr(pPath);
        pendingResponsesMapType& lPendingResponses = GetFileMappingPendingResponsesMap();
        if(lPendingResponses.find(lStr) != lPendingResponses.end())
            PMTHROW(pmFatalErrorException());
        
        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
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
    EXCEPTION_ASSERT(strlen(pPath) <= MAX_FILE_SIZE_LEN - 1);

    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        std::string lStr(pPath);
        pendingResponsesMapType& lPendingResponses = GetFileUnmappingPendingResponsesMap();
        if(lPendingResponses.find(lStr) != lPendingResponses.end())
            PMTHROW(pmFatalErrorException());
        
        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
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
    
void pmUtility::MapFilesOnAllMachines(const char* const* pPath, uint pFileCount)
{
    if(!pFileCount)
        return;

    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
    uint lTotalLength = 0;

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;
    
    std::vector<ushort> lLengthsVector;
    std::vector<char> lNamesVector;
    
    lLengthsVector.reserve(pFileCount);
    
    ulong lMultiFileOperationsId = 0;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        lMultiFileOperationsId = GetMultiFileOperationsId()++;
        
        multiFileOperationsMapType& lMultiFileOperationsMap = GetMultiFileOperationsMap();
        EXCEPTION_ASSERT(lMultiFileOperationsMap.find(lMultiFileOperationsId) == lMultiFileOperationsMap.end());

        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
        lMultiFileOperationsMap.emplace(std::piecewise_construct, std::forward_as_tuple(lMultiFileOperationsId), std::forward_as_tuple(lCount, lSignalWaitSharedPtr));

        for(uint i = 0; i < pFileCount; ++i)
        {
            EXCEPTION_ASSERT(strlen(pPath[i]) <= MAX_FILE_SIZE_LEN - 1);

            std::string lStr(pPath[i]);
            
            lTotalLength += lStr.size();

            lLengthsVector.push_back(lStr.size());
            std::copy(lStr.begin(), lStr.end(), std::back_inserter(lNamesVector));
        }
    }
    
    for(uint i = 0; i < lCount; ++i)
    {
        const pmMachine* lMachine = pmMachinePool::GetMachinePool()->GetMachine(i);
    
        if(lMachine == PM_LOCAL_MACHINE)
        {
            for(uint i = 0; i < pFileCount; ++i)
                pmUtility::MapFile(pPath[i]);
        }
        else
        {
            finalize_ptr<ushort, deleteArrayDeallocator<ushort>> lFileNameLengthsArray(&lLengthsVector[0], false);
            finalize_ptr<char, deleteArrayDeallocator<char>> lFileNames(&lNamesVector[0], false);

            finalize_ptr<communicator::multiFileOperationsPacked> lPackedData(new communicator::multiFileOperationsPacked(communicator::MMAP_FILE, NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId(), lMultiFileOperationsId, pFileCount, lTotalLength, std::move(lFileNameLengthsArray), std::move(lFileNames)));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::multiFileOperationsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::MULTI_FILE_OPERATIONS_TAG, lMachine, communicator::MULTI_FILE_OPERATIONS_PACKED, lPackedData, 1);

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
        }
    }
    
    RegisterMultiFileMappingResponse(lMultiFileOperationsId);
    lSignalWaitSharedPtr->Wait();
}

void pmUtility::UnmapFilesOnAllMachines(const char* const* pPath, uint pFileCount)
{
    if(!pFileCount)
        return;

    uint lCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
    uint lTotalLength = 0;

    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitSharedPtr;
    
    std::vector<ushort> lLengthsVector;
    std::vector<char> lNamesVector;
    
    lLengthsVector.reserve(pFileCount);
    
    ulong lMultiFileOperationsId = 0;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

        lMultiFileOperationsId = GetMultiFileOperationsId()++;
        
        multiFileOperationsMapType& lMultiFileOperationsMap = GetMultiFileOperationsMap();
        EXCEPTION_ASSERT(lMultiFileOperationsMap.find(lMultiFileOperationsId) == lMultiFileOperationsMap.end());

        lSignalWaitSharedPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
        lMultiFileOperationsMap.emplace(std::piecewise_construct, std::forward_as_tuple(lMultiFileOperationsId), std::forward_as_tuple(lCount, lSignalWaitSharedPtr));

        for(uint i = 0; i < pFileCount; ++i)
        {
            EXCEPTION_ASSERT(strlen(pPath[i]) <= MAX_FILE_SIZE_LEN - 1);

            std::string lStr(pPath[i]);
            
            lTotalLength += lStr.size();

            lLengthsVector.push_back(lStr.size());
            std::copy(lStr.begin(), lStr.end(), std::back_inserter(lNamesVector));
        }
    }

    for(uint i = 0; i < lCount; ++i)
    {
        const pmMachine* lMachine = pmMachinePool::GetMachinePool()->GetMachine(i);
    
        if(lMachine == PM_LOCAL_MACHINE)
        {
            for(uint i = 0; i < pFileCount; ++i)
                pmUtility::UnmapFile(pPath[i]);
        }
        else
        {
            finalize_ptr<ushort, deleteArrayDeallocator<ushort>> lFileNameLengthsArray(&lLengthsVector[0], false);
            finalize_ptr<char, deleteArrayDeallocator<char>> lFileNames(&lNamesVector[0], false);
    
            finalize_ptr<communicator::multiFileOperationsPacked> lPackedData(new communicator::multiFileOperationsPacked(communicator::MUNMAP_FILE, NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId(), lMultiFileOperationsId, pFileCount, lTotalLength, std::move(lFileNameLengthsArray), std::move(lFileNames)));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::multiFileOperationsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::MULTI_FILE_OPERATIONS_TAG, lMachine, communicator::MULTI_FILE_OPERATIONS_PACKED, lPackedData, 1);

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
        }
    }
    
    RegisterMultiFileUnmappingResponse(lMultiFileOperationsId);
    lSignalWaitSharedPtr->Wait();
}
    
void pmUtility::RegisterMultiFileMappingResponse(ulong pMultiFileOperationsId)
{
    FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
    multiFileOperationsMapType& lMultiFileOperationsMap = GetMultiFileOperationsMap();
    auto lIter = lMultiFileOperationsMap.find(pMultiFileOperationsId);
    EXCEPTION_ASSERT(lIter != lMultiFileOperationsMap.end());
    
    pendingResponsesMapType::mapped_type& lPair = lMultiFileOperationsMap[pMultiFileOperationsId];
    --lPair.first;

    if(lPair.first == 0)
    {
        lPair.second->Signal();
        lMultiFileOperationsMap.erase(lIter);
    }
}
    
void pmUtility::RegisterMultiFileUnmappingResponse(ulong pMultiFileOperationsId)
{
    return RegisterMultiFileMappingResponse(pMultiFileOperationsId);  // Different operation ids are used for map and unmap operations
}

void pmUtility::SendMultiFileMappingAcknowledgement(ulong pMultiFileOperationsId, const pmMachine* pSourceHost)
{
    finalize_ptr<communicator::multiFileOperationsPacked> lPackedData(new communicator::multiFileOperationsPacked(communicator::MMAP_ACK, *pSourceHost, pMultiFileOperationsId));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::multiFileOperationsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::MULTI_FILE_OPERATIONS_TAG, pSourceHost, communicator::MULTI_FILE_OPERATIONS_PACKED, lPackedData, 1);

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
}

void pmUtility::SendMultiFileUnmappingAcknowledgement(ulong pMultiFileOperationsId, const pmMachine* pSourceHost)
{
    finalize_ptr<communicator::multiFileOperationsPacked> lPackedData(new communicator::multiFileOperationsPacked(communicator::MUNMAP_ACK, *pSourceHost, pMultiFileOperationsId));

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<communicator::multiFileOperationsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, communicator::SEND, communicator::MULTI_FILE_OPERATIONS_TAG, pSourceHost, communicator::MULTI_FILE_OPERATIONS_PACKED, lPackedData, 1);

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
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

void* pmUtility::GetExportedSymbol(void* pLibHandle, const char* pSymbol)
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
    
bool pmUtility::IsReadOnly(pmMemType pMemType)
{
    return (pMemType == READ_ONLY || pMemType == READ_ONLY_LAZY);
}

bool pmUtility::IsWritable(pmMemType pMemType)
{
    return (pMemType == WRITE_ONLY || pMemType == READ_WRITE || pMemType == WRITE_ONLY_LAZY || pMemType == READ_WRITE_LAZY);
}

bool pmUtility::IsWriteOnly(pmMemType pMemType)
{
    return (pMemType == WRITE_ONLY || pMemType == WRITE_ONLY_LAZY);
}

bool pmUtility::IsReadWrite(pmMemType pMemType)
{
    return (pMemType == READ_WRITE || pMemType == READ_WRITE_LAZY);
}
    
bool pmUtility::IsLazy(pmMemType pMemType)
{
    return ((pMemType == READ_ONLY_LAZY) || (pMemType == READ_WRITE_LAZY) || (pMemType == WRITE_ONLY_LAZY));
}

bool pmUtility::IsLazyWriteOnly(pmMemType pMemType)
{
    return (pMemType == WRITE_ONLY_LAZY);
}

bool pmUtility::IsLazyReadWrite(pmMemType pMemType)
{
    return (pMemType == READ_WRITE_LAZY);
}

} // end namespace pm




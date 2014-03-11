
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

#ifndef __PM_UTILITY__
#define __PM_UTILITY__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"

#include <string>
#include <map>
#include <memory>

namespace pm
{

class pmMachine;
    
class pmUtility : public pmBase
{
    typedef std::map<std::string, std::pair<void*, size_t> > fileMappingsMapType;
    typedef std::map<std::string, std::pair<size_t, std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS>>> pendingResponsesMapType;
    typedef std::map<ulong, std::pair<size_t, std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS>>> multiFileOperationsMapType;

public:
    static void MapFileOnAllMachines(const char* pPath);
    static void UnmapFileOnAllMachines(const char* pPath);
    static void MapFilesOnAllMachines(const char* const* pPath, uint pFileCount);
    static void UnmapFilesOnAllMachines(const char* const* pPath, uint pFileCount);
    static void* GetMappedFile(const char* pPath);

    static void MapFile(const char* pPath);
    static void UnmapFile(const char* pPath);
    
    static void SendFileMappingAcknowledgement(const char* pPath, const pmMachine* pSourceHost);
    static void SendFileUnmappingAcknowledgement(const char* pPath, const pmMachine* pSourceHost);

    static void RegisterFileMappingResponse(const char* pPath);
    static void RegisterFileUnmappingResponse(const char* pPath);

    static void SendMultiFileMappingAcknowledgement(ulong pMultiFileOperationsId, const pmMachine* pSourceHost);
    static void SendMultiFileUnmappingAcknowledgement(ulong pMultiFileOperationsId, const pmMachine* pSourceHost);

    static void RegisterMultiFileMappingResponse(ulong pMultiFileOperationsId);
    static void RegisterMultiFileUnmappingResponse(ulong pMultiFileOperationsId);

    static void* OpenLibrary(char* pPath);
    static pmStatus CloseLibrary(void* pLibHandle);
    static void* GetExportedSymbol(void* pLibHandle, char* pSymbol);

    static bool IsReadOnly(pmMemType pMemType);
    static bool IsWritable(pmMemType pMemType);
    static bool IsWriteOnly(pmMemType pMemType);
    static bool IsReadWrite(pmMemType pMemType);
    static bool IsLazy(pmMemType pMemType);
    static bool IsLazyWriteOnly(pmMemType pMemType);
    static bool IsLazyReadWrite(pmMemType pMemType);

private:
    static ulong& GetMultiFileOperationsId();
    static multiFileOperationsMapType& GetMultiFileOperationsMap();
    static pendingResponsesMapType& GetFileMappingPendingResponsesMap();
    static pendingResponsesMapType& GetFileUnmappingPendingResponsesMap();
    static fileMappingsMapType& GetFileMappingsMap();
    static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
};

} // end namespace pm

#endif

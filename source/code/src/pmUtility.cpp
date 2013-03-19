
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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

namespace pm
{

STATIC_ACCESSOR(pmUtility::fileMappingsMapType, pmUtility, GetFileMappingsMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmUtility::ResourceLock"), pmUtility, GetResourceLock)
    
void* pmUtility::MapFile(const char* pPath)
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
    
    return NULL;
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

} // end namespace pm




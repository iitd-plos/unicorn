
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
    static void* GetExportedSymbol(void* pLibHandle, const char* pSymbol);

    static bool IsReadOnly(pmMemType pMemType);
    static bool IsWritable(pmMemType pMemType);
    static bool IsWriteOnly(pmMemType pMemType);
    static bool IsReadWrite(pmMemType pMemType);
    static bool IsLazy(pmMemType pMemType);
    static bool IsLazyWriteOnly(pmMemType pMemType);
    static bool IsLazyReadWrite(pmMemType pMemType);

    template<typename T>
    static std::shared_ptr<T> CompressForSentinel(const T* pMem, T pSentinel, ulong pCount, ulong& pCompressedLength)
    {
    #ifdef USE_OMP_FOR_REDUCTION
        std::vector<ulong> lSentinelLocationsVector;
        lSentinelLocationsVector.reserve(pCount/2);
    #endif

        std::shared_ptr<T> lMemPtr(new T[pCount]);
        T* lMem = lMemPtr.get();

        bool lOngoingSentinels = false;
        uint lIndex = 0;

    #ifdef USE_OMP_FOR_REDUCTION
         if(pMem[0] != pSentinel)
        {
            lSentinelLocationsVector.emplace_back(0);
            lSentinelLocationsVector.emplace_back(0);
        }

        for(uint i = 0; i < pCount; ++i)
        {
            if(pMem[i] == pSentinel)
            {
                lOngoingSentinels = true;
            }
            else
            {
                if(lOngoingSentinels)
                {
                    lSentinelLocationsVector.emplace_back(lIndex);
                    lSentinelLocationsVector.emplace_back(i);
                    
                    lOngoingSentinels = false;
                }
                
                lMem[lIndex++] = pMem[i];
            }
        }
    #else
        for(uint i = 0; i < pCount; ++i)
        {
            if(pMem[i] == pSentinel)
            {
                lOngoingSentinels = true;
            }
            else
            {
                if(lOngoingSentinels)
                {
                    if(lIndex + 1 >= pCount)
                        return std::shared_ptr<T>();

                    lMem[lIndex++] = pSentinel;
                    *((uint*)(&lMem[lIndex])) = i;
                    
                    ++lIndex;
                    
                    lOngoingSentinels = false;
                }
                
                if(lIndex >= pCount)
                    return std::shared_ptr<T>();

                lMem[lIndex++] = pMem[i];
            }
        }
    #endif

    #ifdef USE_OMP_FOR_REDUCTION
        uint lFirstSentinelLoc = lIndex;
        if(lIndex + 1 + lSentinelLocationsVector.size() >= pCount)
            return std::shared_ptr<T>();
        
        for_each(lSentinelLocationsVector, [&] (uint pLocation)
        {
            *((uint*)(&lMem[lIndex])) = pLocation;
            ++lIndex;
        });
        
        *((uint*)(&lMem[lIndex])) = lFirstSentinelLoc;
        ++lIndex;
    #endif

        pCompressedLength = lIndex * sizeof(T);

    #ifdef DUMP_DATA_COMPRESSION_STATISTICS
        pmCompressionDataRecorder::RecordCompressionData(pCount * sizeof(T), pCompressedLength, true);
    #endif
        
        return lMemPtr;
    }

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

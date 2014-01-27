
#include <stdio.h>
#include <iostream>

#include "commonAPI.h"
#include "pageRank.h"

#include <string.h>
#include <math.h>

#include <map>

namespace pageRank
{

unsigned int gTotalWebPages;
unsigned int gMaxOutlinksPerWebPage;
unsigned int gWebPagesPerFile;
unsigned int gWebPagesPerSubtask;

PAGE_RANK_DATA_TYPE* gSerialOutput;
PAGE_RANK_DATA_TYPE* gParallelOutput;

void readWebMetaData(char* pBasePath)
{
	char path[1024];

	strcpy(path, pBasePath);
	strcat(path, "/web_dump_metadata");

	FILE* fp = fopen(path, "rb");
	if(fp == NULL)
	{
		std::cout << "Error in opening metadata file " << path << std::endl;
		exit(1);
	}

	if(fread((void*)(&gTotalWebPages), 4, 1, fp) != 1)
		exit(1);

	if(fread((void*)(&gMaxOutlinksPerWebPage), 4, 1, fp) != 1)
		exit(1);

	if(fread((void*)(&gWebPagesPerFile), 4, 1, fp) != 1)
		exit(1);

	std::cout << "Configuration: " << "Pages = " << gTotalWebPages << "; Max outlinks/page = " << gMaxOutlinksPerWebPage << "; Web pages per file = " << gWebPagesPerFile << std::endl;

	fclose(fp);
}

void readWebPagesFile(char* pBasePath, unsigned int pTotalWebPages, unsigned int pMaxOutlinksPerWebPage, unsigned int pWebPagesPerFile, unsigned int pStartPageNum, unsigned int pPageCount, unsigned int* pData)
{
	char filePath[1024];
	char buf[12];
    
    unsigned int lFileNum = 1 + pStartPageNum;
    
	sprintf(buf, "%u", lFileNum);
    strcpy(filePath, pBasePath);
    strcat(filePath, "/web/page_");
    strcat(filePath, buf);

	FILE* fp = fopen(filePath, "rb");
	if(fp == NULL)
	{
		std::cout << "Error in opening page file " << filePath << std::endl;
        exit(1);
	}

	unsigned int lIndex = 0;
    unsigned int lOutlinkCount = 0;
    unsigned int lEndPageNum = pStartPageNum + pPageCount;

    for(unsigned int lPage = (pStartPageNum+1); lPage <= lEndPageNum; ++lPage)
    {
        //std::cout << "Page " << lPage+1 << std::endl;

        if(fread((void*)(&lOutlinkCount), 4, 1, fp) != 1)
            exit(1);

        pData[lIndex] = lOutlinkCount;
        
        //std::cout << "Web Page: " << lPage+1 << " Outlink Count: " << lOutlinkCount << std::endl;

        for(unsigned int j = 0; j < lOutlinkCount; ++j)
        {
            unsigned int lOutlinkPage = pTotalWebPages;
            if(fread((void*)(&lOutlinkPage), 4, 1, fp) != 1)
                exit(1);
        
            if(lOutlinkPage > pTotalWebPages)
            {
                std::cout << "Error: Invalid outlink " << lOutlinkPage << " on page " << pStartPageNum+1 << std::endl;
                exit(1);
            }

            pData[lIndex + 1 + j] = lOutlinkPage;
            //std::cout << lOutlinkPage << " ";
        }

        if(pMaxOutlinksPerWebPage != lOutlinkCount)
        {
            if(fseek(fp, (pMaxOutlinksPerWebPage - lOutlinkCount) * 4, SEEK_CUR) != 0)
                exit(1);
        }

        lIndex += (pMaxOutlinksPerWebPage + 1);

        //std::cout << std::endl;
    }

	fclose(fp);
}
    
void mapAllFiles(char* pBasePath)
{
    char filePath[1024];
    char buf[12];

    unsigned int lTotalFiles = gTotalWebPages / gWebPagesPerFile;
    for(unsigned int i = 0; i < lTotalFiles; ++i)
    {
        unsigned int lFileNum = 1 + i * gWebPagesPerFile;
        
        sprintf(buf, "%u", lFileNum);
        strcpy(filePath, pBasePath);
        strcat(filePath, "/web/page_");
        strcat(filePath, buf);

        if(pmMapFile(filePath) != pmSuccess)
            exit(1);
    }
}

void unMapAllFiles(char* pBasePath)
{
    char filePath[1024];
    char buf[12];

    unsigned int lTotalFiles = gTotalWebPages / gWebPagesPerFile;
    for(unsigned int i = 0; i < lTotalFiles; ++i)
    {
        unsigned int lFileNum = 1 + i * gWebPagesPerFile;
        
        sprintf(buf, "%u", lFileNum);
        strcpy(filePath, pBasePath);
        strcat(filePath, "/web/page_");
        strcat(filePath, buf);
    
        if(pmUnmapFile(filePath) != pmSuccess)
            exit(1);
    }
}
    
void initializePageRankArray(PAGE_RANK_DATA_TYPE* pPageRankArray, PAGE_RANK_DATA_TYPE pVal, unsigned int pCount)
{
	for(unsigned int i = 0; i < pCount; ++i)
		pPageRankArray[i] = pVal;
}

void serialPageRank(unsigned int* pWebDump)
{
	PAGE_RANK_DATA_TYPE* lGlobalPageRankArray = NULL;
	PAGE_RANK_DATA_TYPE* lLocalPageRankArray = NULL;

	for(int i = 0; i < PAGE_RANK_ITERATIONS; ++i)
	{
		if(i != 0)
        {
            delete[] lLocalPageRankArray;
			lLocalPageRankArray = lGlobalPageRankArray;
        }

        if(i == PAGE_RANK_ITERATIONS - 1)
            lGlobalPageRankArray = gSerialOutput;
        else
            lGlobalPageRankArray = new PAGE_RANK_DATA_TYPE[gTotalWebPages];

        memset(lGlobalPageRankArray, 0, gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE));
        //initializePageRankArray(lGlobalPageRankArray, 1.0 - DAMPENING_FACTOR, gTotalWebPages);

		unsigned int index = 0;
		for(unsigned int j = 0; j < gTotalWebPages; ++j)
		{
			unsigned int lOutlinks = pWebDump[index++];
            PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((i == 0) ? INITIAL_PAGE_RANK : lLocalPageRankArray[j])/(float)lOutlinks);

			for(unsigned int k = 0; k < lOutlinks; ++k)
			{
				unsigned int lRefLink = pWebDump[index + k];

				lGlobalPageRankArray[lRefLink - 1] += lIncr;
			}
        
			index += gMaxOutlinksPerWebPage;
		}
	}

	delete[] lLocalPageRankArray;
}
    
bool IsValidSplitOrNormalSubtask(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
    if(!pSubtaskInfo.splitInfo.splitCount)
        return true;
    
    pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    bool lPartialLastSubtask = (lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask));
    unsigned int lWebPages = (unsigned int)(lPartialLastSubtask ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    
    unsigned int lAllowedSplits = pSubtaskInfo.splitInfo.splitCount;
    if(lWebFiles < pSubtaskInfo.splitInfo.splitCount)
        lAllowedSplits = lWebFiles;
    
    if(pSubtaskInfo.splitInfo.splitId >= lAllowedSplits)
        return false;

    return true;
}
    
bool GetSplitFilesAndPages(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, unsigned int& pSplitStartFile, unsigned int& pSplitFileCount, unsigned int& pSplitStartPage, unsigned int& pSplitPageCount)
{
    if(pSubtaskInfo.splitInfo.splitCount)
    {
        pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);

        bool lPartialLastSubtask = (lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask));
        unsigned int lWebPages = (unsigned int)(lPartialLastSubtask ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
        unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
        unsigned int lFirstWebFile = (unsigned int)pSubtaskInfo.subtaskId * lWebFiles;
        
        unsigned int lAllowedSplits = pSubtaskInfo.splitInfo.splitCount;
        if(lWebFiles < pSubtaskInfo.splitInfo.splitCount)
            lAllowedSplits = lWebFiles;
        
        if(pSubtaskInfo.splitInfo.splitId >= lAllowedSplits)
            return false;
        
        pSplitFileCount = lWebFiles / lAllowedSplits;
        pSplitStartFile = lFirstWebFile + pSplitFileCount * pSubtaskInfo.splitInfo.splitId;
        
        if(pSubtaskInfo.splitInfo.splitId + 1 == lAllowedSplits)
            pSplitFileCount = lWebFiles - pSplitFileCount * pSubtaskInfo.splitInfo.splitId;
        
        pSplitStartPage = pSplitStartFile * lTaskConf->webPagesPerFile;
        pSplitPageCount = pSplitFileCount * lTaskConf->webPagesPerFile;
        
        if(lPartialLastSubtask && pSubtaskInfo.splitInfo.splitId + 1 == lAllowedSplits)
            pSplitPageCount = lWebPages - pSplitPageCount * pSubtaskInfo.splitInfo.splitId;
    }
    
    return true;
}

pmStatus pageRankDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    bool lPartialLastSubtask = (lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask));
    unsigned int lStartPage = (unsigned int)(pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask);
    unsigned int lWebPages = (unsigned int)(lPartialLastSubtask ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);

	// Subscribe to an input memory partition (no data is required for first iteration)
    if(lTaskConf->iteration != 0)
    {
        unsigned int lSplitStartFile, lSplitFileCount;
        unsigned int lSplitStartPage = lStartPage;
        unsigned int lSplitPageCount = lWebPages;
        
        if(!GetSplitFilesAndPages(pTaskInfo, pSubtaskInfo, lSplitStartFile, lSplitFileCount, lSplitStartPage, lSplitPageCount))
            return pmSuccess;
        
        pmSubscriptionInfo lSubscriptionInfo(lSplitStartPage * sizeof(PAGE_RANK_DATA_TYPE), sizeof(PAGE_RANK_DATA_TYPE) * lSplitPageCount);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);
    }

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
    {
        size_t lReservedMem = lWebPages * (lTaskConf->maxOutlinksPerWebPage + 1) * sizeof(PAGE_RANK_DATA_TYPE);
        pmReserveCudaGlobalMem(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lReservedMem);

        size_t lPerPageStorageSize = sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + lTaskConf->maxOutlinksPerWebPage * sizeof(unsigned int);   // no. of outlinks, value (i.e. increment) and keys
        size_t lScratchBufferSize = lWebPages * lPerPageStorageSize;
        pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, SUBTASK_TO_POST_SUBTASK, lScratchBufferSize, NULL);
    }
#endif

	return pmSuccess;
}
    
void** LoadMappedFiles(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    ulong lSubtaskId = pSubtaskInfo.subtaskId;

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((lSubtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (lSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)lSubtaskId * lWebFiles;
    
    unsigned int lSplitStartPage;
    if(!GetSplitFilesAndPages(pTaskInfo, pSubtaskInfo, lFirstWebFile, lWebFiles, lSplitStartPage, lWebPages))
    {
        std::cout << "Problem in splitting code !!!" << std::endl;
        exit(1);
    }
    
    unsigned int lLastWebFile = lFirstWebFile + lWebFiles;
    
    void** lWebFilePtrs = new void*[lWebFiles];
    for(unsigned int fileIndex = lFirstWebFile; fileIndex < lLastWebFile; ++fileIndex)
    {
        char filePath[1024];
        char buf[12];
        
        unsigned int lFileNum = 1 + fileIndex * lTaskConf->webPagesPerFile;
        
        sprintf(buf, "%u", lFileNum);
        strcpy(filePath, lTaskConf->basePath);
        strcat(filePath, "/web/page_");
        strcat(filePath, buf);
        
        lWebFilePtrs[fileIndex - lFirstWebFile] = pmGetMappedFile(filePath);
    }
    
    return lWebFilePtrs;
}

pmStatus pageRank_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    ulong lSubtaskId = pSubtaskInfo.subtaskId;

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((lSubtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (lSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)pSubtaskInfo.subtaskId * lWebFiles;

    unsigned int lSplitStartPage;
    if(!GetSplitFilesAndPages(pTaskInfo, pSubtaskInfo, lFirstWebFile, lWebFiles, lSplitStartPage, lWebPages))
        return pmSuccess;

	PAGE_RANK_DATA_TYPE* lLocalArray = ((lTaskConf->iteration == 0) ? NULL : (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[MEM_INDEX].ptr);

    void** lWebFilePtrs = LoadMappedFiles(pTaskInfo, pSubtaskInfo);
    
    /* The format of storage for every page (input to this subtask) in scratch buffer is -
     * <No. of outlinks> <Incr> <Key 1> <Key 2> ... <Key (no. of outlinks)>
     */

    // Value for every key here is same (i.e. input page rank / outlinks); so no need to store key value pairs
    size_t lPerPageStorageSize = sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + lTaskConf->maxOutlinksPerWebPage * sizeof(unsigned int);   // no. of outlinks, value (i.e. increment) and keys
    size_t lScratchBufferSize = lWebPages * lPerPageStorageSize;
    void* lScratchBuffer = pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, SUBTASK_TO_POST_SUBTASK, lScratchBufferSize, NULL);

    size_t lScratchBufferWriteOffsetInBytes = 0;
    
    unsigned int lTotalFiles = (lTaskConf->totalWebPages / lTaskConf->webPagesPerFile) + ((lTaskConf->totalWebPages % lTaskConf->webPagesPerFile) ? 1 : 0);
    for(unsigned int i = 0; i < lWebFiles; ++i)
    {
        unsigned int* lMappedFile = (unsigned int*)(lWebFilePtrs[i]);

        unsigned int index = 0;
        unsigned int lPagesInFile = lTaskConf->webPagesPerFile;
        if(i + lFirstWebFile == lTotalFiles - 1)
            lPagesInFile = lTaskConf->totalWebPages - (i + lFirstWebFile) * lTaskConf->webPagesPerFile;

        for(unsigned int j = 0; j < lPagesInFile; ++j)
        {
            unsigned int lPageNum = i * lTaskConf->webPagesPerFile + j;

            unsigned int lOutlinks = lMappedFile[index++];
            PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((lTaskConf->iteration == 0) ? lTaskConf->initialPageRank : lLocalArray[lPageNum])/(float)lOutlinks);
            
            char* lWriteLocation = (char*)lScratchBuffer + lScratchBufferWriteOffsetInBytes;
            
            ((unsigned int*)lWriteLocation)[0] = lOutlinks;
            ((PAGE_RANK_DATA_TYPE*)(lWriteLocation + sizeof(unsigned int)))[0] = lIncr;

            for(unsigned int k = 0; k < lOutlinks; ++k)
                ((unsigned int*)(lWriteLocation + sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + k * sizeof(unsigned int)))[0] = lMappedFile[index + k] - 1;
        
            index += lTaskConf->maxOutlinksPerWebPage;
            lScratchBufferWriteOffsetInBytes += lPerPageStorageSize;
        }
    }

    delete[] lWebFilePtrs;

	return pmSuccess;
}
    
void LoadSubtaskBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer, pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    ulong lSubtaskId = pSubtaskInfo.subtaskId;

    size_t lReadOffset = 0;
    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((lSubtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (lSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    size_t lPerPageStorageSize = sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + lTaskConf->maxOutlinksPerWebPage * sizeof(unsigned int);   // no. of outlinks, value (i.e. increment) and keys

    unsigned int lFirstWebFile, lWebFiles, lSplitStartPage;
    if(!GetSplitFilesAndPages(pTaskInfo, pSubtaskInfo, lFirstWebFile, lWebFiles, lSplitStartPage, lWebPages))
    {
        std::cout << "Problem in splitting ..." << std::endl;
        exit(1);
    }

    for(unsigned int i = 0; i < lWebPages; ++i)
    {
        unsigned int lOutlinks = ((unsigned int*)(pBuffer + lReadOffset))[0];
        PAGE_RANK_DATA_TYPE lIncr = ((PAGE_RANK_DATA_TYPE*)(pBuffer + lReadOffset + sizeof(unsigned int)))[0];
        
        for(unsigned int j = 0; j < lOutlinks; ++j)
        {
            unsigned int lKey = ((unsigned int*)(pBuffer + lReadOffset + sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + j * sizeof(unsigned int)))[0];
            
            auto lIter = pMap.find(lKey);
            if(lIter == pMap.end())
                pMap.emplace(lKey, lIncr);
            else
                lIter->second += lIncr;
        }
        
        lReadOffset += lPerPageStorageSize;
    }
}

void LoadReductionBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer)
{
    size_t lKeyValuePairs = ((unsigned int*)pBuffer)[0];
    size_t lKeyValueSize = (sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE));
    
    pBuffer += sizeof(unsigned int);
    
    for(unsigned int i = 0; i < lKeyValuePairs; ++i)
    {
        char* lKeyValueLocation = (pBuffer + i * lKeyValueSize);
        unsigned int lKey = ((unsigned int*)lKeyValueLocation)[0];
        PAGE_RANK_DATA_TYPE lValue = ((PAGE_RANK_DATA_TYPE*)(lKeyValueLocation + sizeof(unsigned int)))[0];
        
        auto lIter = pMap.find(lKey);
        if(lIter == pMap.end())
            pMap.emplace(lKey, lValue);
        else
            lIter->second += lValue;
    }
}
    
void PlaceMapIntoSubtaskReductionBuffer(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap)
{
    pmReleaseScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, REDUCTION_TO_REDUCTION);

    size_t lBufferSize = pMap.size() * (sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE)) + sizeof(unsigned int); // count of key value pairs and key value pairs
    char* lOutputBuffer = (char*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, REDUCTION_TO_REDUCTION, lBufferSize, NULL);

    ((unsigned int*)lOutputBuffer)[0] = (unsigned int)pMap.size();  // Store the number of key value pairs
    size_t lWriteOffset = sizeof(unsigned int);

    auto lIter = pMap.begin(), lEndIter = pMap.end();
    for(; lIter != lEndIter; ++lIter)
    {
        ((unsigned int*)(lOutputBuffer + lWriteOffset))[0] = lIter->first;
        lWriteOffset += sizeof(unsigned int);
        
        ((PAGE_RANK_DATA_TYPE*)(lOutputBuffer + lWriteOffset))[0] = lIter->second;
        lWriteOffset += sizeof(PAGE_RANK_DATA_TYPE);
    }
}

pmStatus pageRankDataReduction(pmTaskInfo pTaskInfo, pmDeviceInfo pDevice1Info, pmSubtaskInfo pSubtask1Info, pmDeviceInfo pDevice2Info, pmSubtaskInfo pSubtask2Info)
{
    bool lIsValid1 = IsValidSplitOrNormalSubtask(pTaskInfo, pSubtask1Info);
    bool lIsValid2 = IsValidSplitOrNormalSubtask(pTaskInfo, pSubtask2Info);

    char* lScratchBuffer1 = (char*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDevice1Info.deviceHandle, pSubtask1Info.subtaskId, pSubtask1Info.splitInfo, SUBTASK_TO_POST_SUBTASK, 0, NULL);

    char* lScratchBuffer2 = (char*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDevice2Info.deviceHandle, pSubtask2Info.subtaskId, pSubtask2Info.splitInfo, SUBTASK_TO_POST_SUBTASK, 0, NULL);

    // The first entry in reduction buffer is the no. of key value pairs following it
    char* lRBuffer1 = (char*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDevice1Info.deviceHandle, pSubtask1Info.subtaskId, pSubtask1Info.splitInfo, REDUCTION_TO_REDUCTION, 0, NULL);
    char* lRBuffer2 = (char*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDevice2Info.deviceHandle, pSubtask2Info.subtaskId, pSubtask2Info.splitInfo, REDUCTION_TO_REDUCTION, 0, NULL);

    if((lIsValid1 && !lRBuffer1 && !lScratchBuffer1) || (lIsValid2 && !lRBuffer2 && !lScratchBuffer2))
    {
        std::cout << "Scratch buffer not available !!!" << std::endl;
        exit(1);
    }

    std::map<unsigned int, PAGE_RANK_DATA_TYPE> lMap;

    if(lRBuffer2)
        LoadReductionBufferInMap(lMap, lRBuffer2);
    else if(lIsValid2)
        LoadSubtaskBufferInMap(lMap, lScratchBuffer2, pTaskInfo, pSubtask2Info);

    if(lRBuffer1)
        LoadReductionBufferInMap(lMap, lRBuffer1);
    else if(lIsValid1)
        LoadSubtaskBufferInMap(lMap, lScratchBuffer1, pTaskInfo, pSubtask1Info);

    pmReleaseScratchBuffer(pTaskInfo.taskHandle, pDevice1Info.deviceHandle, pSubtask1Info.subtaskId, pSubtask1Info.splitInfo, SUBTASK_TO_POST_SUBTASK);

    PlaceMapIntoSubtaskReductionBuffer(pTaskInfo, pDevice1Info, pSubtask1Info, lMap);

    return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	char* lBasePath = DEFAULT_BASE_PATH; \
	FETCH_STR_ARG(lBasePath, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	unsigned int* lWebDump = new unsigned int[gTotalWebPages * (gMaxOutlinksPerWebPage + 1)];

    unsigned int lTotalFiles = (gTotalWebPages / gWebPagesPerFile);
    unsigned int i = 0;
    for(; i < lTotalFiles; ++i)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));
    
    if((gTotalWebPages % gWebPagesPerFile) != 0)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gTotalWebPages - i*gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));

	double lStartTime = getCurrentTimeInSecs();

	serialPageRank(lWebDump);

	double lEndTime = getCurrentTimeInSecs();

    delete[] lWebDump;

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	pageRankTaskConf lTaskConf;
	lTaskConf.totalWebPages = gTotalWebPages;
    lTaskConf.maxOutlinksPerWebPage = gMaxOutlinksPerWebPage;
    lTaskConf.webPagesPerFile = gWebPagesPerFile;
    lTaskConf.webPagesPerSubtask = gWebPagesPerSubtask;
    lTaskConf.initialPageRank = INITIAL_PAGE_RANK;
    strcpy(lTaskConf.basePath, lBasePath);

	unsigned int* lWebDump = new unsigned int[gTotalWebPages * (gMaxOutlinksPerWebPage + 1)];
    
    unsigned int lTotalFiles = (gTotalWebPages / gWebPagesPerFile);
    unsigned int i = 0;
    for(; i < lTotalFiles; ++i)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));
    
    if((gTotalWebPages % gWebPagesPerFile) != 0)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gTotalWebPages - i*gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));
    
	double lStartTime = getCurrentTimeInSecs();

	if(singleGpuPageRank(lTaskConf, lWebDump, gParallelOutput) != 0)
        return 0;

	double lEndTime = getCurrentTimeInSecs();

    delete[] lWebDump;

	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

void CopyReductionBufferIntoAddressSpace(char* pBuffer, pmMemHandle pMemHandle, unsigned int pIteration)
{
    pmRawMemPtr lRawMemPtr;
    pmGetRawMemPtr(pMemHandle, &lRawMemPtr);

    size_t lKeyValuePairs = ((unsigned int*)pBuffer)[0];
    size_t lKeyValueSize = (sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE));
    
    unsigned int lLastKey = 0;  // Becuase pBuffer is created out of std::map, it is sorted

    pBuffer += sizeof(unsigned int);

    if(pIteration == 0) // Address space is first touched only after 0th iteration
    {
        // Initialize all entries in the address space
        for(unsigned int i = 0; i < lKeyValuePairs; ++i)
        {
            char* lKeyValueLocation = (pBuffer + i * lKeyValueSize);
            unsigned int lKey = ((unsigned int*)lKeyValueLocation)[0];
            PAGE_RANK_DATA_TYPE lValue = ((PAGE_RANK_DATA_TYPE*)(lKeyValueLocation + sizeof(unsigned int)))[0];
            
            if(i != 0 && lLastKey + 1 != lKey)
                memset((((PAGE_RANK_DATA_TYPE*)lRawMemPtr) + lLastKey + 1), 0, (lKey - lLastKey - 1) * sizeof(PAGE_RANK_DATA_TYPE));

            ((PAGE_RANK_DATA_TYPE*)lRawMemPtr)[lKey] = lValue;
            
            lLastKey = lKey;
        }
    }
    else
    {
        // Only modify entries which are in pBuffer
        // In other words, pages with 0 inlinks are not updated here
        for(unsigned int i = 0; i < lKeyValuePairs; ++i)
        {
            char* lKeyValueLocation = (pBuffer + i * lKeyValueSize);
            unsigned int lKey = ((unsigned int*)lKeyValueLocation)[0];
            PAGE_RANK_DATA_TYPE lValue = ((PAGE_RANK_DATA_TYPE*)(lKeyValueLocation + sizeof(unsigned int)))[0];

            ((PAGE_RANK_DATA_TYPE*)lRawMemPtr)[lKey] = lValue;
        }
    }
}

bool ParallelPageRankIteration(pmMemHandle pMemHandle, pageRankTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    unsigned long lSubtasks = (gTotalWebPages / gWebPagesPerSubtask) + ((gTotalWebPages % gWebPagesPerSubtask) ? 1 : 0);
    
    if(lSubtasks == 1)
    {
        // CopyReductionBufferIntoAddressSpace expects sorted buffer
        // for one subtask there will be no reduction and hence no sorting
        std::cout << "One subtask case is not implemented !!!" << std::endl;
        exit(1);
    }

	CREATE_TASK(lSubtasks, pCallbackHandle, pSchedulingPolicy)

    pmTaskMem lTaskMem[1];
    
    if(pTaskConf->iteration != 0)
    {
        lTaskMem[0] = {pMemHandle, READ_ONLY, SUBSCRIPTION_NATURAL};
        lTaskDetails.taskMem = (pmTaskMem*)(lTaskMem);
        lTaskDetails.taskMemCount = 1;
    }
    else
    {
        lTaskDetails.taskMemCount = 0;
    }
    
    lTaskDetails.canSplitCpuSubtasks = true;
    
	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(pageRankTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return (double)-1.0;
    }

    char* lBuffer = (char*)pmGetLastReductionScratchBuffer(lTaskHandle);
    CopyReductionBufferIntoAddressSpace(lBuffer, pMemHandle, pTaskConf->iteration);

    pmReleaseTask(lTaskHandle);

    return true;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS
    
    mapAllFiles(lBasePath);

	// Input Mem contains page rank of each page before the iteration
	// Output Mem contains the page rank of each page after the iteration
	// Number of subtasks is equal to the number of rows
	size_t lMemSize = gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE);
    
    pmMemHandle lMemHandle = NULL;
    pmCreateMemory(lMemSize, &lMemHandle);

	pageRankTaskConf lTaskConf;
	lTaskConf.totalWebPages = gTotalWebPages;
    lTaskConf.maxOutlinksPerWebPage = gMaxOutlinksPerWebPage;
    lTaskConf.webPagesPerFile = gWebPagesPerFile;
    lTaskConf.webPagesPerSubtask = gWebPagesPerSubtask;
    lTaskConf.initialPageRank = INITIAL_PAGE_RANK;
    strcpy(lTaskConf.basePath, lBasePath);

	double lStartTime = getCurrentTimeInSecs();

    for(int i = 0; i < PAGE_RANK_ITERATIONS; ++i)
    {
        lTaskConf.iteration = i;
		if(!ParallelPageRankIteration(lMemHandle, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
			return (double)-1.0;
    }
    
	double lEndTime = getCurrentTimeInSecs();

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lMemHandle) );

        pmRawMemPtr lRawOutputPtr;
        pmGetRawMemPtr(lMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    }

    pmReleaseMemory(lMemHandle);

    unMapAllFiles(lBasePath);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = pageRankDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = pageRank_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = pageRank_cudaLaunchFunc;
#endif

    lCallbacks.dataReduction = pageRankDataReduction;

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
    if(strlen(lBasePath) >= MAX_BASE_PATH_LENGTH)
    {
        std::cout << "Base path too long" << std::endl;
        exit(1);
    }

    readWebMetaData(lBasePath);
    
    if(gWebPagesPerFile > WEB_PAGES_PER_SUBTASK)
    {
        gWebPagesPerSubtask = gWebPagesPerFile;
    }
    else
    {
        if((WEB_PAGES_PER_SUBTASK % gWebPagesPerFile) == 0)
            gWebPagesPerSubtask = WEB_PAGES_PER_SUBTASK;
        else
            gWebPagesPerSubtask = (WEB_PAGES_PER_SUBTASK / gWebPagesPerFile) * gWebPagesPerFile;
    }

	gSerialOutput = new PAGE_RANK_DATA_TYPE[gTotalWebPages];
	gParallelOutput = new PAGE_RANK_DATA_TYPE[gTotalWebPages];
    
	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
	delete[] gSerialOutput;
	delete[] gParallelOutput;

	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	for(unsigned int i = 0; i < gTotalWebPages; ++i)
	{
		if((int)(fabs(gSerialOutput[i] - gParallelOutput[i])) > 0)
		{
			std::cout << "Mismatch index " << i << " Serial Value = " << gSerialOutput[i] << " Parallel Value = " << gParallelOutput[i] << std::endl;
			return 1;
		}
	}

	return 0;
}

/**	Non-common args
 *	1. Matrix Dimension
 */
int main(int argc, char** argv)
{
    callbackStruct lStruct[1] = { {DoSetDefaultCallbacks, "PAGERANK"} };

	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 1);

	commonFinish();

	return 0;
}

}

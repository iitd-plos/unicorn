
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

#include <stdio.h>
#include <iostream>

#include "commonAPI.h"
#include "pageRankInvertedIndex.h"

#include <string.h>
#include <math.h>

#include <memory>

namespace pageRankInvertedIndex
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
    unsigned int lTotalFiles = gTotalWebPages / gWebPagesPerFile;
    
    char** filePaths = new char*[lTotalFiles];
    char buf[12];

    for(unsigned int i = 0; i < lTotalFiles; ++i)
    {
        filePaths[i] = new char[1024];
        unsigned int lFileNum = 1 + i * gWebPagesPerFile;
        
        sprintf(buf, "%u", lFileNum);
        strcpy(filePaths[i], pBasePath);
        strcat(filePaths[i], "/web/page_");
        strcat(filePaths[i], buf);
    }

    if(pmMapFiles(filePaths, lTotalFiles) != pmSuccess)
        exit(1);

    for(unsigned int i = 0; i < lTotalFiles; ++i)
        delete[] filePaths[i];
    
    delete[] filePaths;
}

void unMapAllFiles(char* pBasePath)
{
    unsigned int lTotalFiles = gTotalWebPages / gWebPagesPerFile;
    
    char** filePaths = new char*[lTotalFiles];
    char buf[12];

    for(unsigned int i = 0; i < lTotalFiles; ++i)
    {
        filePaths[i] = new char[1024];
        unsigned int lFileNum = 1 + i * gWebPagesPerFile;
        
        sprintf(buf, "%u", lFileNum);
        strcpy(filePaths[i], pBasePath);
        strcat(filePaths[i], "/web/page_");
        strcat(filePaths[i], buf);
    }

    if(pmUnmapFiles(filePaths, lTotalFiles) != pmSuccess)
        exit(1);

    for(unsigned int i = 0; i < lTotalFiles; ++i)
        delete[] filePaths[i];
    
    delete[] filePaths;
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
    
void** LoadMappedFiles(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	invertIndexTaskConf* lTaskConf = (invertIndexTaskConf*)(pTaskInfo.taskConf);
    ulong lSubtaskId = pSubtaskInfo.subtaskId;

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((lSubtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (lSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)lSubtaskId * lWebFiles;
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

pmStatus invertIndexDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	invertIndexTaskConf* lTaskConf = (invertIndexTaskConf*)(pTaskInfo.taskConf);

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)pSubtaskInfo.subtaskId * lWebFiles;
    unsigned int lStartPage = lFirstWebFile * lTaskConf->webPagesPerFile;

    // Inverted index address space
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INVERTED_INDEX_MEM_INDEX, WRITE_SUBSCRIPTION, pmSubscriptionInfo(lStartPage * lTaskConf->maxOutlinksPerWebPage * sizeof(invertedIndex), lWebPages * lTaskConf->maxOutlinksPerWebPage * sizeof(invertedIndex)));

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
    {
    }
#endif

	return pmSuccess;
}
    
pmStatus invertIndex_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	invertIndexTaskConf* lTaskConf = (invertIndexTaskConf*)(pTaskInfo.taskConf);

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)pSubtaskInfo.subtaskId * lWebFiles;
    unsigned int lStartPage = lFirstWebFile * lTaskConf->webPagesPerFile;

	invertedIndex* lInvertedIndex = (invertedIndex*)pSubtaskInfo.memInfo[INVERTED_INDEX_MEM_INDEX].ptr;
    void** lWebFilePtrs = LoadMappedFiles(pTaskInfo, pSubtaskInfo);
    
	unsigned int* lIndicesPerSubtask = (unsigned int*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, SUBTASK_TO_POST_SUBTASK, pTaskInfo.subtaskCount * sizeof(unsigned int), NULL);
    
    memset(lIndicesPerSubtask, 0, pTaskInfo.subtaskCount * sizeof(unsigned int));

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
            unsigned int lOutlinks = lMappedFile[index++];

            for(unsigned int k = 0; k < lOutlinks; ++k)
            {
                unsigned int lPotentialSubtask = (lMappedFile[index + k] - 1) / lTaskConf->webPagesPerSubtask; // The subtask that will process this inlink in the page rank computation tasks
                ++lIndicesPerSubtask[lPotentialSubtask];
            }

            index += lTaskConf->maxOutlinksPerWebPage;
        }
    }
    
    unsigned int* lSubtaskIndexLocation = new unsigned int[pTaskInfo.subtaskCount];
    lSubtaskIndexLocation[0] = lIndicesPerSubtask[0] - 1;   // Highest index where this subtask's indices will be written
    for(unsigned int i = 1; i < (unsigned int)pTaskInfo.subtaskCount; ++i)
        lSubtaskIndexLocation[i] = lSubtaskIndexLocation[i - 1] + lIndicesPerSubtask[i];

    for(unsigned int i = 0; i < lWebFiles; ++i)
    {
        unsigned int* lMappedFile = (unsigned int*)(lWebFilePtrs[i]);

        unsigned int index = 0;
        unsigned int lPagesInFile = lTaskConf->webPagesPerFile;
        if(i + lFirstWebFile == lTotalFiles - 1)
            lPagesInFile = lTaskConf->totalWebPages - (i + lFirstWebFile) * lTaskConf->webPagesPerFile;

        for(unsigned int j = 0; j < lPagesInFile; ++j)
        {
            unsigned int lPageNum = lStartPage + i * lTaskConf->webPagesPerFile + j;

            unsigned int lOutlinks = lMappedFile[index++];

            for(unsigned int k = 0; k < lOutlinks; ++k)
            {
                unsigned int lPotentialSubtask = (lMappedFile[index + k] - 1) / lTaskConf->webPagesPerSubtask; // The subtask that will process this inlink in the page rank computation tasks

                invertedIndex* lInvertedIndexPtr = lInvertedIndex + lSubtaskIndexLocation[lPotentialSubtask];
                
                lInvertedIndexPtr->srcPage = lPageNum;
                lInvertedIndexPtr->destPage = lMappedFile[index + k] - 1;
                lInvertedIndexPtr->srcPageOutlinks = lOutlinks;
                
                --lSubtaskIndexLocation[lPotentialSubtask];
            }

            index += lTaskConf->maxOutlinksPerWebPage;
        }
    }

    delete[] lSubtaskIndexLocation;
    delete[] lWebFilePtrs;

	return pmSuccess;
}
    
pmStatus invertIndexDataRedistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	unsigned int* lIndicesPerSubtask = (unsigned int*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, SUBTASK_TO_POST_SUBTASK, 0, NULL);

    unsigned int lCount = 0;
    size_t lInvertedIndexOffset = 0;
    for(unsigned int i = 0; i < (unsigned int)pTaskInfo.subtaskCount; ++i)
    {
        pmRedistributeData(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INVERTED_INDEX_MEM_INDEX, lInvertedIndexOffset, lIndicesPerSubtask[i] * sizeof(invertedIndex), i);

        lCount += lIndicesPerSubtask[i];
        lInvertedIndexOffset += lIndicesPerSubtask[i] * sizeof(invertedIndex);
    }

    return pmSuccess;
}
    
pmStatus pageRankDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    subtaskMetadata& lSubtaskMetaData = ((subtaskMetadata*)((char*)lTaskConf + sizeof(pageRankTaskConf)))[pSubtaskInfo.subtaskId];
    
    unsigned int lFirstPage = (unsigned int)pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask;
    unsigned int lPageCount = lTaskConf->webPagesPerSubtask;
    
    if(lFirstPage + lPageCount > lTaskConf->totalWebPages)
        lPageCount = lTaskConf->totalWebPages - lFirstPage;

    // Inverted index address space
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INVERTED_LINKS_MEM_INDEX, READ_SUBSCRIPTION, pmSubscriptionInfo(lSubtaskMetaData.inlinksStartOffset, lSubtaskMetaData.inlinksCount * sizeof(invertedIndex)));

    // Output page rank address space
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_PAGE_RANKS_MEM_INDEX, WRITE_SUBSCRIPTION, pmSubscriptionInfo(lFirstPage * sizeof(PAGE_RANK_DATA_TYPE), lPageCount * sizeof(PAGE_RANK_DATA_TYPE)));

    // Input page rank address space
    if(lTaskConf->iteration != 0)
    {
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_PAGE_RANKS_MEM_INDEX, READ_SUBSCRIPTION, pmSubscriptionInfo(0, lTaskConf->totalWebPages * sizeof(PAGE_RANK_DATA_TYPE)));

//        invertedIndex* lInvertedIndex = (invertedIndex*)((char*)pSubtaskInfo.memInfo[INVERTED_LINKS_MEM_INDEX].readPtr + lSubtaskMetaData.inlinksStartOffset);
//        for(unsigned int i = 0; i < lSubtaskMetaData.inlinksCount ; ++i)
//            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_PAGE_RANKS_MEM_INDEX, READ_SUBSCRIPTION, pmSubscriptionInfo(lInvertedIndex[i].srcPage * sizeof(PAGE_RANK_DATA_TYPE), sizeof(PAGE_RANK_DATA_TYPE)));
    }
    
#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
    {
    }
#endif

	return pmSuccess;
}

pmStatus pageRank_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    subtaskMetadata& lSubtaskMetaData = ((subtaskMetadata*)((char*)lTaskConf + sizeof(pageRankTaskConf)))[pSubtaskInfo.subtaskId];
    
    unsigned int lFirstPage = (unsigned int)pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask;
    unsigned int lPageCount = lTaskConf->webPagesPerSubtask;
    
    if(lFirstPage + lPageCount > lTaskConf->totalWebPages)
        lPageCount = lTaskConf->totalWebPages - lFirstPage;

    PAGE_RANK_DATA_TYPE* lOutputPagerank = (PAGE_RANK_DATA_TYPE*)((char*)pSubtaskInfo.memInfo[OUTPUT_PAGE_RANKS_MEM_INDEX].ptr);
    
    invertedIndex* lInvertedIndex = (invertedIndex*)((char*)pSubtaskInfo.memInfo[INVERTED_LINKS_MEM_INDEX].readPtr);
    unsigned int lFirstSrcPage = 0; //lInvertedIndex[0].srcPage;

//    for(unsigned int i = 0; i < lSubtaskMetaData.inlinksCount; ++i)
//    {
//        if(lInvertedIndex[i].srcPage < lFirstSrcPage)
//            lFirstSrcPage = lInvertedIndex[i].srcPage;
//    }

    memset(lOutputPagerank, 0, lPageCount * sizeof(PAGE_RANK_DATA_TYPE));
    
    if(lTaskConf->iteration == 0)
    {
        for(unsigned int i = 0; i < lSubtaskMetaData.inlinksCount ; ++i)
            lOutputPagerank[lInvertedIndex[i].destPage - lFirstPage] += DAMPENING_FACTOR * (lTaskConf->initialPageRank / lInvertedIndex[i].srcPageOutlinks);
    }
    else
    {
        PAGE_RANK_DATA_TYPE* lInputPagerank = (PAGE_RANK_DATA_TYPE*)((char*)pSubtaskInfo.memInfo[INPUT_PAGE_RANKS_MEM_INDEX].ptr);

        for(unsigned int i = 0; i < lSubtaskMetaData.inlinksCount ; ++i)
            lOutputPagerank[lInvertedIndex[i].destPage - lFirstPage] += DAMPENING_FACTOR * (lInputPagerank[lInvertedIndex[i].srcPage - lFirstSrcPage] / lInvertedIndex[i].srcPageOutlinks);
    }

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	char* lBasePath = DEFAULT_BASE_PATH; \
	FETCH_STR_ARG(lBasePath, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	unsigned int* lWebDump = new unsigned int[gTotalWebPages * (gMaxOutlinksPerWebPage + 1)];

    unsigned int lTotalFiles = (gTotalWebPages / gWebPagesPerFile);
    unsigned int i = 0;
    for(; i < lTotalFiles; ++i)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));
    
    if((gTotalWebPages % gWebPagesPerFile) != 0)
        readWebPagesFile(lBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gTotalWebPages - i*gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));

	serialPageRank(lWebDump);

    delete[] lWebDump;

	double lEndTime = getCurrentTimeInSecs();

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
    
bool InvertIndexTask(invertIndexTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, unsigned long pSubtasks, pmMemHandle pInvertedIndexMemHandle, std::unique_ptr<pageRankTaskConf>* pPageRankTaskConf)
{
	CREATE_TASK(pSubtasks, pCallbackHandle, pSchedulingPolicy)

    pmTaskMem lTaskMem[1] = {{pInvertedIndexMemHandle, WRITE_ONLY}};
    lTaskDetails.taskMem = (pmTaskMem*)(lTaskMem);
    lTaskDetails.taskMemCount = 1;
    
    lTaskDetails.taskConf = (void*)pTaskConf;
    lTaskDetails.taskConfLength = sizeof(invertIndexTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return false;
    }

    unsigned long lCount = 0;
    pmRedistributionMetadata* lMetaData = pmGetRedistributionMetadata(lTaskHandle, INVERTED_INDEX_MEM_INDEX, &lCount);
    
    if(lCount != pSubtasks)
    {
        std::cout << "Redistribution count mismatch !!!" << std::endl;
        exit(1);
    }

    pPageRankTaskConf->reset((pageRankTaskConf*)new char[sizeof(pageRankTaskConf) + sizeof(subtaskMetadata) * lCount]);
    
    pageRankTaskConf* lPageRankTaskConf = pPageRankTaskConf->get();
    lPageRankTaskConf->totalWebPages = gTotalWebPages;
    lPageRankTaskConf->maxOutlinksPerWebPage = gMaxOutlinksPerWebPage;
    lPageRankTaskConf->webPagesPerFile = gWebPagesPerFile;
    lPageRankTaskConf->webPagesPerSubtask = gWebPagesPerSubtask;
    lPageRankTaskConf->initialPageRank = INITIAL_PAGE_RANK;
    
    unsigned long lOffset = 0;
    subtaskMetadata* lSubtaskMetadata = (subtaskMetadata*)((char*)lPageRankTaskConf + sizeof(pageRankTaskConf));
    for(unsigned long i = 0; i < lCount; ++i)
    {
        unsigned int lInlinksCount = lMetaData->count / sizeof(invertedIndex);
        lSubtaskMetadata->inlinksStartOffset =  lOffset;
        lSubtaskMetadata->inlinksCount = lInlinksCount;
        
        lOffset += lMetaData->count;
        ++lSubtaskMetadata;
        ++lMetaData;
    }

    pmReleaseTask(lTaskHandle);

    return true;
}

bool ParallelPageRankIteration(pageRankTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, unsigned long pSubtasks, pmMemHandle pInvertedIndexMemHandle, pmMemHandle* pPageRanksMemHandle)
{
	CREATE_TASK(pSubtasks, pCallbackHandle, pSchedulingPolicy)

    size_t lMemSize = gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE);
    pmMemHandle lOutputPageRanksMemHandle = NULL;
    pmCreateMemory(lMemSize, &lOutputPageRanksMemHandle);

//    pmTaskMem lTaskMem[3] = {{pInvertedIndexMemHandle, READ_ONLY_LAZY}, {lOutputPageRanksMemHandle, WRITE_ONLY}, {*pPageRanksMemHandle, READ_ONLY}};
    pmTaskMem lTaskMem[3] = {{pInvertedIndexMemHandle, READ_ONLY}, {lOutputPageRanksMemHandle, WRITE_ONLY}, {*pPageRanksMemHandle, READ_ONLY}};
    lTaskDetails.taskMem = (pmTaskMem*)(lTaskMem);
    lTaskDetails.taskMemCount = (*pPageRanksMemHandle) ? 3 : 2;
    
//    if(pTaskConf->iteration == 0)
//        lTaskMem[0].memType = READ_ONLY;

	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(pageRankTaskConf) + sizeof(subtaskMetadata) * (unsigned int)pSubtasks;

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );

    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return false;
    }
    
    if(*pPageRanksMemHandle)
        pmReleaseMemory(*pPageRanksMemHandle);

    *pPageRanksMemHandle = lOutputPageRanksMemHandle;

    pmReleaseTask(lTaskHandle);

    return true;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS
    
	double lStartTime = getCurrentTimeInSecs();

    mapAllFiles(lBasePath);
    
    unsigned long lSubtasks = (gTotalWebPages / gWebPagesPerSubtask) + ((gTotalWebPages % gWebPagesPerSubtask) ? 1 : 0);

    pmMemHandle lInvertedIndexMemHandle = NULL;
    pmCreateMemory(gTotalWebPages * gMaxOutlinksPerWebPage * sizeof(invertedIndex), &lInvertedIndexMemHandle);

    size_t lMemSize = gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE);
    pmMemHandle lPageRanksMemHandle = NULL;

    invertIndexTaskConf lInvertIndexTaskConf;
	lInvertIndexTaskConf.totalWebPages = gTotalWebPages;
    lInvertIndexTaskConf.maxOutlinksPerWebPage = gMaxOutlinksPerWebPage;
    lInvertIndexTaskConf.webPagesPerFile = gWebPagesPerFile;
    lInvertIndexTaskConf.webPagesPerSubtask = gWebPagesPerSubtask;
    strcpy(lInvertIndexTaskConf.basePath, lBasePath);

    std::unique_ptr<pageRankTaskConf> lPageRankTaskConf;
    
    if(!InvertIndexTask(&lInvertIndexTaskConf, pCallbackHandle[0], pSchedulingPolicy, lSubtasks, lInvertedIndexMemHandle, &lPageRankTaskConf))
        return (double)-1.0;

#if 0
    SAFE_PM_EXEC( pmFetchMemory(lInvertedIndexMemHandle) );

    pmRawMemPtr lRawOutputPtr;
    pmGetRawMemPtr(lInvertedIndexMemHandle, &lRawOutputPtr);

    invertedIndex* lInvertedIndex = (invertedIndex*)lRawOutputPtr;
    for(int i = 0; i < gTotalWebPages * gMaxOutlinksPerWebPage; ++i, ++lInvertedIndex)
        std::cout << "Src Page = " << lInvertedIndex->srcPage << " Dest Page = " << lInvertedIndex->destPage << " Outlinks = " << lInvertedIndex->srcPageOutlinks << std::endl;
#endif

    for(int i = 0; i < PAGE_RANK_ITERATIONS; ++i)
    {
        lPageRankTaskConf->iteration = i;
		if(!ParallelPageRankIteration(lPageRankTaskConf.get(), pCallbackHandle[1], pSchedulingPolicy, lSubtasks, lInvertedIndexMemHandle, &lPageRanksMemHandle))
			return (double)-1.0;
    }
    
    unMapAllFiles(lBasePath);

	double lEndTime = getCurrentTimeInSecs();

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lPageRanksMemHandle) );

        pmRawMemPtr lRawOutputPtr;
        pmGetRawMemPtr(lPageRanksMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    }

    pmReleaseMemory(lInvertedIndexMemHandle);
    pmReleaseMemory(lPageRanksMemHandle);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = invertIndexDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = invertIndex_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = invertIndex_cudaLaunchFunc;
#endif
    
    lCallbacks.dataRedistribution = invertIndexDataRedistribution;

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks2()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = pageRankDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = pageRank_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = pageRank_cudaLaunchFunc;
#endif

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
    
    // Each subtask accesses a whole number of web files
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
    callbackStruct lStruct[2] = { {DoSetDefaultCallbacks, "INVERTINDEX"}, {DoSetDefaultCallbacks2, "PAGERANK"} };

	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 2);

	commonFinish();

	return 0;
}

}

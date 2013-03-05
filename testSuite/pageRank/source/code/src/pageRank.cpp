
#include <stdio.h>
#include <iostream>

#include "commonAPI.h"
#include "pageRank.h"

#include <string.h>
#include <math.h>

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

void readWebPagesFile(char* pBasePath, unsigned int pTotalWebPages, unsigned int pMaxOutlinksPerWebPage, unsigned int pWebPagesPerFile, unsigned int pStartPageNum, unsigned int pPageCount, PAGE_RANK_DATA_TYPE* pData)
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

        for(unsigned int j=0; j<lOutlinkCount; ++j)
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
    
        lIndex += (pMaxOutlinksPerWebPage + 1);

        //std::cout << std::endl;
    }

	fclose(fp);
}
    
void initializePageRankArray(PAGE_RANK_DATA_TYPE* pPageRankArray, PAGE_RANK_DATA_TYPE pVal, unsigned int pCount)
{
	for(unsigned int i=0; i<pCount; ++i)
		pPageRankArray[i] = pVal;
}

PAGE_RANK_DATA_TYPE normalizePageRank(PAGE_RANK_DATA_TYPE pPageRank, unsigned int pTotalWebPages)
{
	return pPageRank*pTotalWebPages;
}

PAGE_RANK_DATA_TYPE denormalizePageRank(PAGE_RANK_DATA_TYPE pPageRank, unsigned int pTotalWebPages)
{
	return pPageRank/pTotalWebPages;
}
    
void serialPageRank(char* pBasePath)
{
	PAGE_RANK_DATA_TYPE* lWebDump = new PAGE_RANK_DATA_TYPE[gTotalWebPages * (gMaxOutlinksPerWebPage + 1)];

    unsigned int lTotalFiles = (gTotalWebPages / gWebPagesPerFile);
    unsigned int i = 0;
    for(; i<lTotalFiles; ++i)
        readWebPagesFile(pBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));
    
    if((gTotalWebPages % gWebPagesPerFile) != 0)
        readWebPagesFile(pBasePath, gTotalWebPages, gMaxOutlinksPerWebPage, gWebPagesPerFile, i*gWebPagesPerFile, gTotalWebPages - i*gWebPagesPerFile, lWebDump + (i * gWebPagesPerFile * (gMaxOutlinksPerWebPage + 1)));

	PAGE_RANK_DATA_TYPE* lGlobalPageRankArray = gSerialOutput;
	PAGE_RANK_DATA_TYPE* lLocalPageRankArray = new PAGE_RANK_DATA_TYPE[gTotalWebPages];

	initializePageRankArray(lLocalPageRankArray, normalizePageRank(INITIAL_PAGE_RANK, gTotalWebPages), gTotalWebPages);

	for(int i=0; i<PAGE_RANK_ITERATIONS; ++i)
	{
		if(i!=0)
			memcpy((void*)lLocalPageRankArray, (void*)lGlobalPageRankArray, sizeof(PAGE_RANK_DATA_TYPE)*gTotalWebPages);

        memset(lGlobalPageRankArray, 0, gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE));
        //initializePageRankArray(lGlobalPageRankArray, normalizePageRank(1 - DAMPENING_FACTOR, gTotalWebPages), gTotalWebPages);

		unsigned int index=0;
		for(unsigned int j=0; j<gTotalWebPages; ++j)
		{
			unsigned int lOutlinks = lWebDump[index++];
            PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * lLocalPageRankArray[j]/(float)lOutlinks);
        
			for(unsigned int k=0; k<lOutlinks; ++k)
			{
				unsigned int lRefLink = lWebDump[index+k];
				
				lGlobalPageRankArray[lRefLink - 1] += lIncr;
			}

			index += gMaxOutlinksPerWebPage;
		}
	}

	delete[] lLocalPageRankArray;
	delete[] lWebDump;
}

#ifdef BUILD_CUDA
pmCudaLaunchConf GetCudaLaunchConf(unsigned int pWebPages)
{
	pmCudaLaunchConf lCudaLaunchConf;

    if(pWebPages > 512)
    {
        lCudaLaunchConf.blocksX = pWebPages/512 + ((pWebPages%512) ? 1 : 0);
        lCudaLaunchConf.threadsX = 512;
    }
    else
    {
        lCudaLaunchConf.blocksX = 1;
        lCudaLaunchConf.threadsX = pWebPages;
    }

	return lCudaLaunchConf;
}
#endif

pmStatus pageRankDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    bool lPartialLastSubtask = (lTaskConf->totalWebPages < ((pSubtaskId + 1) * lTaskConf->webPagesPerSubtask));
    unsigned int lStartPage = (unsigned int)(pSubtaskId * lTaskConf->webPagesPerSubtask);
    unsigned int lWebPages = (unsigned int)(lPartialLastSubtask ? (lTaskConf->totalWebPages - (pSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);

	// Subscribe to an input memory partition (no data is required for first iteration)
    if(lTaskConf->iteration != 0)
    {
        lSubscriptionInfo.offset = lStartPage * sizeof(PAGE_RANK_DATA_TYPE);
        lSubscriptionInfo.length = sizeof(PAGE_RANK_DATA_TYPE) * lWebPages;

        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    }

	// Subscribe to entire output matrix

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, GetCudaLaunchConf(lWebPages));
#endif

	PAGE_RANK_DATA_TYPE* lSubtaskWebDump = (PAGE_RANK_DATA_TYPE*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, PRE_SUBTASK_TO_SUBTASK, lWebPages * (lTaskConf->maxOutlinksPerWebPage + 1) * sizeof(PAGE_RANK_DATA_TYPE), NULL);

    unsigned int lFiles = (lWebPages / lTaskConf->webPagesPerFile);
    if(lPartialLastSubtask)
        lFiles -= 1;
    
    unsigned int i=0;
    for(; i<lFiles; ++i)
        readWebPagesFile(lTaskConf->basePath, lTaskConf->totalWebPages, lTaskConf->maxOutlinksPerWebPage, lTaskConf->webPagesPerFile, lStartPage + i * lTaskConf->webPagesPerFile, lTaskConf->webPagesPerFile, lSubtaskWebDump + i * lTaskConf->webPagesPerFile * (lTaskConf->maxOutlinksPerWebPage + 1));
    
    if(lPartialLastSubtask)
        readWebPagesFile(lTaskConf->basePath, lTaskConf->totalWebPages, lTaskConf->maxOutlinksPerWebPage, lTaskConf->webPagesPerFile, lStartPage + i * lTaskConf->webPagesPerFile, lWebPages - i * lTaskConf->webPagesPerFile, lSubtaskWebDump + i * lTaskConf->webPagesPerFile * (lTaskConf->maxOutlinksPerWebPage + 1));

	return pmSuccess;
}

pmStatus pageRank_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);

	PAGE_RANK_DATA_TYPE* lSubtaskWebDump = (PAGE_RANK_DATA_TYPE*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, PRE_SUBTASK_TO_SUBTASK, 0, NULL);
	PAGE_RANK_DATA_TYPE* lGlobalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.outputMem;
	PAGE_RANK_DATA_TYPE* lLocalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.inputMem;
    
    memset(lGlobalArray, 0, pSubtaskInfo.outputMemLength);

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    
	unsigned int index=0;
	for(unsigned int j=0; j<lWebPages; ++j)
	{
		unsigned int lOutlinks = lSubtaskWebDump[index++];
        PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((lTaskConf->iteration == 0) ? lTaskConf->initialPageRank : lLocalArray[j])/(float)lOutlinks);

		for(unsigned int k=0; k<lOutlinks; ++k)
		{
			unsigned int lRefLink = lSubtaskWebDump[index+k];
	
			lGlobalArray[lRefLink - 1] += lIncr;
		}

		index += lTaskConf->maxOutlinksPerWebPage;
	}

	return pmSuccess;
}
    
pmStatus pageRankDataReduction(pmTaskInfo pTaskInfo, pmDeviceInfo pDevice1Info, pmSubtaskInfo pSubtask1Info, pmDeviceInfo pDevice2Info, pmSubtaskInfo pSubtask2Info)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);

	PAGE_RANK_DATA_TYPE* lGlobalArray1 = (PAGE_RANK_DATA_TYPE*)pSubtask1Info.outputMem;
	PAGE_RANK_DATA_TYPE* lGlobalArray2 = (PAGE_RANK_DATA_TYPE*)pSubtask2Info.outputMem;

    for(unsigned int i=0; i<lTaskConf->totalWebPages; ++i)
        lGlobalArray1[i] += lGlobalArray2[i];
    
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

	serialPageRank(lBasePath);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

bool ParallelPageRankIteration(pmMemHandle pInputMemHandle, pmMemHandle* pOutputMemHandle, pageRankTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	size_t lMemSize = gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE);
    unsigned long lSubtasks = (gTotalWebPages / gWebPagesPerSubtask) + ((gTotalWebPages % gWebPagesPerSubtask) ? 1 : 0);

	CREATE_TASK(0, lMemSize, lSubtasks, pCallbackHandle, pSchedulingPolicy)
    
    lTaskDetails.inputMemHandle = pInputMemHandle;
	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(pageRankTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return (double)-1.0;
    }

    *pOutputMemHandle = lTaskDetails.outputMemHandle;

    pmReleaseTask(lTaskHandle);

    return true;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS

	// Input Mem contains page rank of each page before the iteration
	// Output Mem contains the page rank of each page after the iteration
	// Number of subtasks is equal to the number of rows
	size_t lMemSize = gTotalWebPages * sizeof(PAGE_RANK_DATA_TYPE);

    pmMemHandle lInputMemHandle = NULL, lOutputMemHandle = NULL;

	pageRankTaskConf lTaskConf;
	lTaskConf.totalWebPages = gTotalWebPages;
    lTaskConf.maxOutlinksPerWebPage = gMaxOutlinksPerWebPage;
    lTaskConf.webPagesPerFile = gWebPagesPerFile;
    lTaskConf.webPagesPerSubtask = gWebPagesPerSubtask;
    lTaskConf.initialPageRank = normalizePageRank(INITIAL_PAGE_RANK, gTotalWebPages);
    strcpy(lTaskConf.basePath, lBasePath);

	double lStartTime = getCurrentTimeInSecs();

    for(int i=0; i<PAGE_RANK_ITERATIONS; ++i)
    {
		if(i != 0)
		{
            if(lInputMemHandle)
                pmReleaseMemory(lInputMemHandle);

			lInputMemHandle = lOutputMemHandle;
		}
    
        lTaskConf.iteration = i;
		if(!ParallelPageRankIteration(lInputMemHandle, &lOutputMemHandle, &lTaskConf, pCallbackHandle, pSchedulingPolicy))
			return (double)-1.0;
    }
    
	double lEndTime = getCurrentTimeInSecs();

	SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );

    pmRawMemPtr lRawOutputPtr;
    pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);

	memcpy(gParallelOutput, lRawOutputPtr, lMemSize);

    if(lInputMemHandle)
        pmReleaseMemory(lInputMemHandle);

	pmReleaseMemory(lOutputMemHandle);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = pageRankDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = pageRank_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = pageRank_cudaFunc;
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
	for(unsigned int i=0; i<gTotalWebPages; ++i)
	{
		if((int)(fabs(gSerialOutput[i] - gParallelOutput[i])) > 0)
		{
			std::cout << "Mismatch index " << i << " Serial Value = " << gSerialOutput[i] << " Parallel Value = " << gParallelOutput[i] << std::endl;
			return 1;
		}
	}

	std::cout << "Perfect match against serial execution" << std::endl;
	return 0;
}

/**	Non-common args
 *	1. Matrix Dimension
 */
int main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "PAGERANK");

	commonFinish();

	return 0;
}

}

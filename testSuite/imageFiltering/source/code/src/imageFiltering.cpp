
#include <stdio.h>
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "imageFiltering.h"

namespace imageFiltering
{
    
void* gSampleInput;
void* gSerialOutput;
void* gParallelOutput;
    
#ifdef DO_MULTIPLE_CONVOLUTIONS
char gFilter[MAX_ITERATIONS][MAX_FILTER_DIM][MAX_FILTER_DIM];
#else
char gFilter[MAX_FILTER_DIM][MAX_FILTER_DIM];
#endif
    
size_t gImageWidth, gImageHeight, gImageOffset, gImageBytesPerLine;

#ifndef GENERATE_RANDOM_IMAGE_IN_MEMORY
void readImageMetaData(char* pImagePath)
{
	FILE* fp = fopen(pImagePath, "rb");
	if(fp == NULL)
	{
		std::cout << "Error in opening image file " << pImagePath << std::endl;
		exit(1);
	}

    bitmapHeader fileHeader;
    
	if(fread((void*)(&fileHeader), sizeof(fileHeader), 1, fp) != 1)
		exit(1);
    
    if(fileHeader.identifier[0] != 'B' || fileHeader.identifier[1] != 'M')
        exit(1);
    
    if(fileHeader.reserved[0] != 0 || fileHeader.reserved[1] != 0)
        exit(1);

#if 0
    if(fileHeader.headersize != 54 || fileHeader.infoSize != 40)
        exit(1);

    if(fileHeader.imageSize + fileHeader.headersize != fileHeader.filesize)
        exit(1);
#endif

    if(fileHeader.bitPlanes != 1 || fileHeader.bitCount != 24 || fileHeader.compression != 0)
        exit(1);
    
    gImageBytesPerLine = (fileHeader.filesize - fileHeader.headersize) / fileHeader.height;
    gImageOffset = fileHeader.headersize;
    
    gImageWidth = fileHeader.width;
    gImageHeight = fileHeader.height;

	fclose(fp);
}

// If pInverted is true, the image read in pImageData is inverted otherwise non-inverted
void readImage(char* pImagePath, void* pImageData, bool pInverted)
{
	FILE* fp = fopen(pImagePath, "rb");
	if(fp == NULL)
	{
		std::cout << "Error in opening image file " << pImagePath << std::endl;
		exit(1);
	}

    if(fseek(fp, gImageOffset, SEEK_CUR) != 0)
        exit(1);
    
    char lColor[PIXEL_COUNT];
    unsigned int lSeekOffset = gImageBytesPerLine - (gImageWidth * PIXEL_COUNT);

    for(size_t i = 0; i < gImageHeight; ++i)
    {
        char* lRow = ((char*)pImageData) + ((pInverted ? i : (gImageHeight - i - 1)) * gImageWidth * PIXEL_COUNT);
        for(size_t j = 0; j < gImageWidth; ++j)
        {
            if(fread((void*)(&lColor), sizeof(lColor), 1, fp) != 1)
                exit(1);

            lRow[PIXEL_COUNT * j] = (pInverted ? lColor[0] : lColor[2]);
            lRow[PIXEL_COUNT * j + 1] = lColor[1];
            lRow[PIXEL_COUNT * j + 2] = (pInverted ? lColor[2] : lColor[0]);
        }

        if(lSeekOffset)
        {
            if(fseek(fp, lSeekOffset, SEEK_CUR) != 0)
                exit(1);
        }
    }
    
	fclose(fp);
}
#endif
    
void serialImageFilter(void* pImageData, size_t pFilterRadius, char* pSerialOutput, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM])
{
    char* lImageData = (char*)pImageData;
    
    int lDimMinX, lDimMaxX, lDimMinY, lDimMaxY;

    for(size_t i = 0; i < gImageHeight; ++i)
    {
        for(size_t j = 0; j < gImageWidth; ++j)
        {
            lDimMinX = j - pFilterRadius;
            lDimMaxX = j + pFilterRadius;
            lDimMinY = i - pFilterRadius;
            lDimMaxY = i + pFilterRadius;
            
            char lRedVal = 0, lGreenVal = 0, lBlueVal = 0;

        #ifdef USE_ELLIPTICAL_FILTER
            unsigned int lSemiMajorAxis = 1, lSemiMinorAxis = 1;
            float lSemiMajorAxisSquare = 1, lSemiMinorAxisSquare = 1;
            
            unsigned int lTotalSubtasks = ((unsigned int)gImageWidth/TILE_DIM + ((unsigned int)gImageWidth%TILE_DIM ? 1 : 0)) * ((unsigned int)gImageHeight/TILE_DIM + ((unsigned int)gImageHeight%TILE_DIM ? 1 : 0));
            unsigned int lSubtasksPerRow = ((unsigned int)gImageWidth/TILE_DIM + ((unsigned int)gImageWidth%TILE_DIM ? 1 : 0));
            ulong lSubtaskId = (i / TILE_DIM) * lSubtasksPerRow + (j / TILE_DIM);
            ulong lSubtasksLeft = lTotalSubtasks - lSubtaskId;  // This is always greater than 1

            lSemiMajorAxis = pFilterRadius * ((float)lSubtaskId / lTotalSubtasks);
            lSemiMinorAxis = pFilterRadius * ((float)lSubtasksLeft / lTotalSubtasks);
            
            lSemiMajorAxisSquare = lSemiMajorAxis * lSemiMajorAxis;
            lSemiMinorAxisSquare = lSemiMinorAxis * lSemiMinorAxis;
        #endif

            for(int k = lDimMinY; k <= lDimMaxY; ++k)
            {
                for(int l = lDimMinX; l <= lDimMaxX; ++l)
                {
                    int m = ((k < 0) ? 0 : (((size_t)k >= gImageHeight) ? (gImageHeight - 1) : k));
                    int n = ((l < 0) ? 0 : (((size_t)l >= gImageWidth) ? (gImageWidth - 1) : l));
                    
                #ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
                    // The allocated address space is of size imageWidth * imageHeight. Every row is not aligned at imageBytesPerLine offset.
                    size_t lInvertedIndex = ((gImageHeight - 1 - m) * gImageWidth + n) * PIXEL_COUNT;
                #else
                    size_t lInvertedIndex = (gImageHeight - 1 - m) * lTaskConf->imageBytesPerLine + n * PIXEL_COUNT;
                #endif
                    
                #ifdef USE_ELLIPTICAL_FILTER
                    float x = l - (lDimMinX + lDimMaxX)/2;
                    float y = k - (lDimMinY + lDimMaxY)/2;
                    
                    if((x * x) / lSemiMajorAxisSquare + (y * y) / lSemiMinorAxisSquare < 1.0)   // Inside Ellipse
                    {
                        lBlueVal += lImageData[lInvertedIndex] * pFilter[k - lDimMinY][l - lDimMinX];
                        lGreenVal += lImageData[lInvertedIndex + 1] * pFilter[k - lDimMinY][l - lDimMinX];
                        lRedVal += lImageData[lInvertedIndex + 2] * pFilter[k - lDimMinY][l - lDimMinX];
                    }
                    else    // Outside Ellipse
                    {
                    }
                #else
                    lBlueVal += lImageData[lInvertedIndex] * pFilter[k - lDimMinY][l - lDimMinX];
                    lGreenVal += lImageData[lInvertedIndex + 1] * pFilter[k - lDimMinY][l - lDimMinX];
                    lRedVal += lImageData[lInvertedIndex + 2] * pFilter[k - lDimMinY][l - lDimMinX];
                #endif
                }
            }
            
            size_t lOffset = (i * gImageWidth + j) * PIXEL_COUNT;
            pSerialOutput[lOffset] = lRedVal;
            pSerialOutput[lOffset + 1] = lGreenVal;
            pSerialOutput[lOffset + 2] = lBlueVal;
        }
    }
}
    
bool parallelImageFilter(pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, const char* pImagePath, int pFilterRadius, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM])
{
	// Number of subtasks is equal to the number of tiles in the image
    unsigned int lSubtasks = ((unsigned int)gImageWidth/TILE_DIM + ((unsigned int)gImageWidth%TILE_DIM ? 1 : 0)) * ((unsigned int)gImageHeight/TILE_DIM + ((unsigned int)gImageHeight%TILE_DIM ? 1 : 0));

    pmTaskMem lTaskMem[MAX_MEM_INDICES];

#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    CREATE_TASK(lSubtasks, pCallbackHandle[0], pSchedulingPolicy)
    
    lTaskMem[INPUT_MEM_INDEX] = {pInputMemHandle, READ_ONLY, SUBSCRIPTION_OPTIMAL};
#else
	CREATE_TASK(lSubtasks, pCallbackHandle[0], pSchedulingPolicy)
#endif
    
    lTaskMem[OUTPUT_MEM_INDEX] = {pOutputMemHandle, WRITE_ONLY, SUBSCRIPTION_OPTIMAL};

    lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    lTaskDetails.taskMem = (pmTaskMem*)(lTaskMem);

	imageFilterTaskConf lTaskConf;
    strcpy(lTaskConf.imagePath, pImagePath);
    lTaskConf.imageWidth = gImageWidth;
    lTaskConf.imageHeight = gImageHeight;
    lTaskConf.imageOffset = gImageOffset;
    lTaskConf.imageBytesPerLine = gImageBytesPerLine;
    lTaskConf.filterRadius = pFilterRadius;
    
    int lFilterDim = 2 * pFilterRadius + 1;
    for(int i = 0; i < lFilterDim; ++i)
        for(int j = 0;  j < lFilterDim; ++j)
            lTaskConf.filter[i][j] = pFilter[i][j];

	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);
    
    lTaskDetails.canSplitCpuSubtasks = true;

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return false;
    }
    
    pmReleaseTask(lTaskHandle);
    
    return true;
}
    
bool GetSubtaskSubscription(imageFilterTaskConf* pTaskConf, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, int* pStartCol, int* pEndCol, int* pStartRow, int* pEndRow)
{
    unsigned int lTilesPerRow = (pTaskConf->imageWidth/TILE_DIM + (pTaskConf->imageWidth%TILE_DIM ? 1 : 0));

	// Subscribe to one tile of the output matrix
    *pStartCol = (int)((pSubtaskId % lTilesPerRow) * TILE_DIM);
    *pEndCol = *pStartCol + TILE_DIM;
    *pStartRow = (int)((pSubtaskId / lTilesPerRow) * TILE_DIM);
    *pEndRow = *pStartRow + TILE_DIM;
    
    if((size_t)*pEndCol > pTaskConf->imageWidth)
        *pEndCol = pTaskConf->imageWidth;

    if((size_t)*pEndRow > pTaskConf->imageHeight)
        *pEndRow = pTaskConf->imageHeight;

    // If it is a split subtask, then subscribe to a smaller number of rows within the tile
    if(pSplitInfo.splitCount)
    {
        int lSplitCount = pSplitInfo.splitCount;
        int lRows = (*pEndRow - *pStartRow);

        if(lSplitCount > lRows)
            lSplitCount = lRows;
        
        // No subscription required
        if((int)pSplitInfo.splitId >= lSplitCount)
            return false;
        
        int lRowFactor = lRows / lSplitCount;

        *pStartRow += pSplitInfo.splitId * lRowFactor;
        if((int)pSplitInfo.splitId != lSplitCount - 1)
            *pEndRow = (*pStartRow + lRowFactor);
    }
    
    return true;
}

pmStatus imageFilterDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pmSubscriptionInfo lSubscriptionInfo;
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    int lSubscriptionStartCol, lSubscriptionEndCol, lSubscriptionStartRow, lSubscriptionEndRow;
    if(GetSubtaskSubscription(lTaskConf, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, &lSubscriptionStartCol, &lSubscriptionEndCol, &lSubscriptionStartRow, &lSubscriptionEndRow))
    {
        size_t lRowSize = lTaskConf->imageWidth * PIXEL_COUNT;
        
    #ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
        int lFirstRow = ((lSubscriptionStartRow - (int)lTaskConf->filterRadius < 0) ? 0 : (lSubscriptionStartRow - (int)lTaskConf->filterRadius));
        int lLastRow = ((lSubscriptionEndRow - 1 + (int)lTaskConf->filterRadius >= lTaskConf->imageHeight) ? (int)(lTaskConf->imageHeight - 1) : (lSubscriptionEndRow - 1 + (int)lTaskConf->filterRadius));
        int lFirstCol = ((lSubscriptionStartCol - (int)lTaskConf->filterRadius < 0) ? 0 : (lSubscriptionStartCol - (int)lTaskConf->filterRadius));
        int lLastCol = ((lSubscriptionEndCol - 1 + (int)lTaskConf->filterRadius >= lTaskConf->imageWidth) ? (int)(lTaskConf->imageWidth - 1) : (lSubscriptionEndCol - 1 + (int)lTaskConf->filterRadius));
        
        int lFirstInvertedRow = (int)(lTaskConf->imageHeight - 1 - lFirstRow);
        int lLastInvertedRow = (int)(lTaskConf->imageHeight - 1 - lLastRow);
        
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, pmScatteredSubscriptionInfo((lLastInvertedRow * lRowSize) + (lFirstCol * PIXEL_COUNT), (lLastCol - lFirstCol + 1) * PIXEL_COUNT, lRowSize, lFirstInvertedRow - lLastInvertedRow + 1));
    #endif

        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, pmScatteredSubscriptionInfo((lSubscriptionStartRow * lRowSize) + (lSubscriptionStartCol * PIXEL_COUNT), (lSubscriptionEndCol - lSubscriptionStartCol) * PIXEL_COUNT, lRowSize, lSubscriptionEndRow - lSubscriptionStartRow));

    #ifdef BUILD_CUDA
        // Reserve CUDA Global Mem
        if(pDeviceInfo.deviceType == pm::GPU_CUDA)
        {
            size_t lReservedMem = computeSubtaskReservedMemRequirement(pTaskInfo, pDeviceInfo, pSubtaskInfo.subtaskId, lSubscriptionStartCol, lSubscriptionEndCol, lSubscriptionStartRow, lSubscriptionEndRow);
            pmReserveCudaGlobalMem(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lReservedMem);
        }
    #endif
    }

	return pmSuccess;
}

pmStatus imageFilter_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    char* lOutput = (char*)(pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);

    int lSubscriptionStartCol, lSubscriptionEndCol, lSubscriptionStartRow, lSubscriptionEndRow;
    if(!GetSubtaskSubscription(lTaskConf, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, &lSubscriptionStartCol, &lSubscriptionEndCol, &lSubscriptionStartRow, &lSubscriptionEndRow))
        return pmSuccess;

#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    if(pSubtaskInfo.memInfo[INPUT_MEM_INDEX].visibilityType != SUBSCRIPTION_NATURAL)
        exit(1);

    char* lInvertedImageData = (char*)(pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr);
    int lFirstRow = ((lSubscriptionStartRow - (int)lTaskConf->filterRadius < 0) ? 0 : (lSubscriptionStartRow - (int)lTaskConf->filterRadius));
    int lFirstCol = ((lSubscriptionStartCol - (int)lTaskConf->filterRadius < 0) ? 0 : (lSubscriptionStartCol - (int)lTaskConf->filterRadius));
    int lLastRow = ((lSubscriptionEndRow - 1 + (int)lTaskConf->filterRadius >= lTaskConf->imageHeight) ? (int)(lTaskConf->imageHeight - 1) : (lSubscriptionEndRow - 1 + (int)lTaskConf->filterRadius));
    
    int lTotalRows = lLastRow - lFirstRow + 1;
#else
    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset;
#endif

    int lDimMinX, lDimMaxX, lDimMinY, lDimMaxY;
    size_t lWidth = ((pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->imageWidth : (lSubscriptionEndCol - lSubscriptionStartCol));

#ifdef USE_ELLIPTICAL_FILTER
    unsigned int lSemiMajorAxis = 1, lSemiMinorAxis = 1;
    float lSemiMajorAxisSquare = 1, lSemiMinorAxisSquare = 1;
    
    ulong lSubtasksLeft = pTaskInfo.subtaskCount - pSubtaskInfo.subtaskId;  // This is always greater than 1

    lSemiMajorAxis = lTaskConf->filterRadius * ((float)pSubtaskInfo.subtaskId / pTaskInfo.subtaskCount);
    lSemiMinorAxis = lTaskConf->filterRadius * ((float)lSubtasksLeft / pTaskInfo.subtaskCount);
    
    lSemiMajorAxisSquare = lSemiMajorAxis * lSemiMajorAxis;
    lSemiMinorAxisSquare = lSemiMinorAxis * lSemiMinorAxis;
#endif

    for(int i = lSubscriptionStartRow; i < lSubscriptionEndRow; ++i)
    {
        for(int j = lSubscriptionStartCol; j < lSubscriptionEndCol; ++j)
        {
            lDimMinX = j - (int)lTaskConf->filterRadius;
            lDimMaxX = j + (int)lTaskConf->filterRadius;
            lDimMinY = i - (int)lTaskConf->filterRadius;
            lDimMaxY = i + (int)lTaskConf->filterRadius;
            
            char lRedVal = 0, lGreenVal = 0, lBlueVal = 0;
            for(int k = lDimMinY; k <= lDimMaxY; ++k)
            {
                for(int l = lDimMinX; l <= lDimMaxX; ++l)
                {
                    int m = ((k < 0) ? 0 : (((size_t)k >= lTaskConf->imageHeight) ? (lTaskConf->imageHeight - 1) : k));
                    int n = ((l < 0) ? 0 : (((size_t)l >= lTaskConf->imageWidth) ? (lTaskConf->imageWidth - 1) : l));
                    
                #ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
                    // The allocated address space is of size imageWidth * imageHeight. Every row is not aligned at imageBytesPerLine offset.
                    size_t lInvertedIndex = ((lTotalRows - 1 - (m - lFirstRow)) * lTaskConf->imageWidth + (n - lFirstCol)) * PIXEL_COUNT;
                #else
                    size_t lInvertedIndex = (lTaskConf->imageHeight - 1 - m) * lTaskConf->imageBytesPerLine + n * PIXEL_COUNT;
                #endif
                    
                #ifdef USE_ELLIPTICAL_FILTER
                    float x = l - (lDimMinX + lDimMaxX)/2;
                    float y = k - (lDimMinY + lDimMaxY)/2;
                    
                    if((x * x) / lSemiMajorAxisSquare + (y * y) / lSemiMinorAxisSquare < 1.0)   // Inside Ellipse
                    {
                        lBlueVal += lInvertedImageData[lInvertedIndex] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                        lGreenVal += lInvertedImageData[lInvertedIndex + 1] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                        lRedVal += lInvertedImageData[lInvertedIndex + 2] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                    }
                    else    // Outside Ellipse
                    {
                    }
                #else
                    lBlueVal += lInvertedImageData[lInvertedIndex] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                    lGreenVal += lInvertedImageData[lInvertedIndex + 1] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                    lRedVal += lInvertedImageData[lInvertedIndex + 2] * lTaskConf->filter[k - lDimMinY][l - lDimMinX];
                #endif
                }
            }
            
            size_t lOffset = ((i - lSubscriptionStartRow) * lWidth + (j - lSubscriptionStartCol)) * PIXEL_COUNT;
            lOutput[lOffset] = lRedVal;
            lOutput[lOffset + 1] = lGreenVal;
            lOutput[lOffset + 2] = lBlueVal;
        }
    }
    
	return pmSuccess;
}
    
#ifdef DO_MULTIPLE_CONVOLUTIONS

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    
#define READ_NON_COMMON_ARGS \
    int lFilterRadius = DEFAULT_FILTER_RADIUS; \
    int lFilterRadiusStep = DEFAULT_FILTER_RADIUS_STEP; \
    gImageWidth = DEFAULT_IMAGE_WIDTH; \
    gImageHeight = DEFAULT_IMAGE_HEIGHT; \
    int lIterations = DEFAULT_ITERATION_COUNT; \
    FETCH_INT_ARG(lFilterRadius, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lFilterRadiusStep, pCommonArgs + 1, argc, argv); \
    FETCH_INT_ARG(gImageWidth, pCommonArgs + 2, argc, argv); \
    FETCH_INT_ARG(gImageHeight, pCommonArgs + 3, argc, argv); \
    FETCH_INT_ARG(lIterations, pCommonArgs + 4, argc, argv);

#else
    
#define READ_NON_COMMON_ARGS \
    int lFilterRadius = DEFAULT_FILTER_RADIUS; \
    int lFilterRadiusStep = DEFAULT_FILTER_RADIUS_STEP; \
	char* lImagePath = DEFAULT_IMAGE_PATH; \
    int lIterations = DEFAULT_ITERATION_COUNT; \
    FETCH_INT_ARG(lFilterRadius, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lFilterRadiusStep, pCommonArgs + 1, argc, argv); \
    FETCH_STR_ARG(lImagePath, pCommonArgs + 2, argc, argv); \
    FETCH_INT_ARG(lIterations, pCommonArgs + 3, argc, argv);
    
#endif
    
#else

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    
#define READ_NON_COMMON_ARGS \
    int lFilterRadius = DEFAULT_FILTER_RADIUS; \
    gImageWidth = DEFAULT_IMAGE_WIDTH; \
    gImageHeight = DEFAULT_IMAGE_HEIGHT; \
    FETCH_INT_ARG(lFilterRadius, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(gImageWidth, pCommonArgs + 1, argc, argv); \
    FETCH_INT_ARG(gImageHeight, pCommonArgs + 2, argc, argv); \

#else
    
#define READ_NON_COMMON_ARGS \
    int lFilterRadius = DEFAULT_FILTER_RADIUS; \
	char* lImagePath = DEFAULT_IMAGE_PATH; \
    FETCH_INT_ARG(lFilterRadius, pCommonArgs, argc, argv); \
	FETCH_STR_ARG(lImagePath, pCommonArgs + 1, argc, argv);
    
#endif

#endif

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

    void* lImageData = malloc(IMAGE_SIZE);

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    memcpy(lImageData, gSampleInput, IMAGE_SIZE);
#else
    readImage(lImagePath, lImageData, true);
#endif
    
	double lStartTime = getCurrentTimeInSecs();

#ifdef DO_MULTIPLE_CONVOLUTIONS
    for(uint i = 0; i < lIterations; ++i)
    {
        if(i != 0)
            std::swap(lImageData, gSerialOutput);

        serialImageFilter(lImageData, lFilterRadius + i * lFilterRadiusStep, (char*)gSerialOutput, gFilter[i]);
    }
    
    if(lIterations % 2 == 0)
        std::swap(lImageData, gSerialOutput);
#else
	serialImageFilter(lImageData, lFilterRadius, (char*)gSerialOutput, gFilter);
#endif

	double lEndTime = getCurrentTimeInSecs();

    free(lImageData);
	return (lEndTime - lStartTime);
}

// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

    void* lImageData = malloc(IMAGE_SIZE);

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    memcpy(lImageData, gSampleInput, IMAGE_SIZE);
#else
    readImage(lImagePath, lImageData, true);
#endif

	double lStartTime = getCurrentTimeInSecs();

#ifdef DO_MULTIPLE_CONVOLUTIONS
    for(uint i = 0; i < lIterations; ++i)
    {
        if(i != 0)
            std::swap(lImageData, gParallelOutput);

        if(singleGpuImageFilter(lImageData, gImageWidth, gImageHeight, gFilter[i], lFilterRadius + i * lFilterRadiusStep, gImageBytesPerLine, gParallelOutput) != 0)
            return 0;
    }

    if(lIterations % 2 == 0)
        std::swap(lImageData, gParallelOutput);
#else
    if(singleGpuImageFilter(lImageData, gImageWidth, gImageHeight, gFilter, lFilterRadius, gImageBytesPerLine, gParallelOutput) != 0)
        return 0;
#endif

	double lEndTime = getCurrentTimeInSecs();

    free(lImageData);
	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

    pmMemHandle lInputMemHandle = NULL, lOutputMemHandle = NULL;

#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    pmRawMemPtr lRawInputPtr;

    CREATE_MEM_2D(gImageHeight, (size_t)gImageWidth * PIXEL_COUNT, lInputMemHandle);

    pmGetRawMemPtr(lInputMemHandle, &lRawInputPtr);

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    const char* lImagePath = "";
    memcpy(lRawInputPtr, gSampleInput, IMAGE_SIZE);
#else
    readImage(lImagePath, lRawInputPtr, true);
#endif

    DistributeMemory(lInputMemHandle, BLOCK_DIST_2D_RANDOM, TILE_DIM, (unsigned int)gImageWidth, (unsigned int)gImageHeight, PIXEL_COUNT, false);
#else
    if(pmMapFile(lImagePath) != pmSuccess)
        exit(1);
#endif

    CREATE_MEM_2D(gImageHeight, (size_t)gImageWidth * PIXEL_COUNT, lOutputMemHandle);

    double lStartTime = getCurrentTimeInSecs();

#ifdef DO_MULTIPLE_CONVOLUTIONS
    for(uint i = 0; i < lIterations; ++i)
    {
        if(i != 0)
            std::swap(lInputMemHandle, lOutputMemHandle);

        if(!parallelImageFilter(pCallbackHandle, pSchedulingPolicy, lInputMemHandle, lOutputMemHandle, lImagePath, lFilterRadius + i * lFilterRadiusStep, gFilter[i]))
            exit(1);
    }
    
    if(lIterations % 2 == 0)
        std::swap(lInputMemHandle, lOutputMemHandle);
#else
    if(!parallelImageFilter(pCallbackHandle, pSchedulingPolicy, lInputMemHandle, lOutputMemHandle, lImagePath, lFilterRadius, gFilter))
        exit(1);
#endif
    
    double lEndTime = getCurrentTimeInSecs();

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );

        pmRawMemPtr lRawOutputPtr;
        pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, IMAGE_SIZE);
    }
    
    pmReleaseMemory(lOutputMemHandle);

#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    pmReleaseMemory(lInputMemHandle);
#else
    if(pmUnmapFile(lImagePath) != pmSuccess)
        exit(1);
#endif

    return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = imageFilterDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = imageFilter_cpu;

	#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = imageFilter_cudaLaunchFunc;
	#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

#ifdef DO_MULTIPLE_CONVOLUTIONS
    if(lIterations > MAX_ITERATIONS)
        exit(1);
#endif

    size_t lImageSize = IMAGE_SIZE;

    gSerialOutput = malloc(lImageSize);
	gParallelOutput = malloc(lImageSize);

    srand((unsigned int)time(NULL));

#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    gSampleInput = malloc(lImageSize);
    gImageBytesPerLine = gImageWidth * PIXEL_COUNT;
    gImageOffset = 0;

	for(size_t i = 0; i < lImageSize; ++i)
		((char*)gSampleInput)[i] = rand();
#else
    readImageMetaData(lImagePath);

    if(strlen(lImagePath) >= MAX_IMAGE_PATH_LENGTH)
    {
        std::cout << "Image path too long" << std::endl;
        exit(1);
    }
#endif
    
#ifdef DO_MULTIPLE_CONVOLUTIONS
    for(uint k = 0; k < lIterations; ++k)
    {
        if((lFilterRadius + k * lFilterRadiusStep) < MIN_FILTER_RADIUS || (lFilterRadius + k * lFilterRadiusStep) > MAX_FILTER_RADIUS)
        {
            std::cout << "Filter radius must be between " << MIN_FILTER_RADIUS << " and " << MAX_FILTER_RADIUS << std::endl;
            exit(1);
        }

        int lFilterDim = 2 * (lFilterRadius + k * lFilterRadiusStep) + 1;
        for(int i = 0; i < lFilterDim; ++i)
            for(int j = 0; j < lFilterDim; ++j)
                gFilter[k][i][j] = (((rand() % 2) ? 1 : -1) * rand());
    }
#else
    if(lFilterRadius < MIN_FILTER_RADIUS || lFilterRadius > MAX_FILTER_RADIUS)
    {
        std::cout << "Filter radius must be between " << MIN_FILTER_RADIUS << " and " << MAX_FILTER_RADIUS << std::endl;
        exit(1);
    }

    int lFilterDim = 2 * lFilterRadius + 1;

    for(int i = 0; i < lFilterDim; ++i)
        for(int j = 0; j < lFilterDim; ++j)
            gFilter[i][j] = (((rand() % 2) ? 1 : -1) * rand());
#endif
    
	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
    free(gSampleInput);
#endif

	free(gSerialOutput);
	free(gParallelOutput);

	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
    char* lSerialOutput = (char*)gSerialOutput;
    char* lParallelOutput = (char*)gParallelOutput;

    size_t lImageSize = IMAGE_SIZE;
	for(size_t i = 0; i < lImageSize; ++i)
	{
		if(lSerialOutput[i] != lParallelOutput[i])
		{
			std::cout << "Mismatch index " << i << " Serial Value = " << (int)(lSerialOutput[i]) << " Parallel Value = " << (int)(lParallelOutput[i]) << std::endl;
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
    callbackStruct lStruct[1] = { {DoSetDefaultCallbacks, "IMAGEFILTER"} };
    
	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 1);

	commonFinish();

	return 0;
}

}

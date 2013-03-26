
#include <stdio.h>
#include <string.h>

#include "commonAPI.h"
#include "imageFiltering.h"

namespace imageFiltering
{
    
void* gSampleInput;
void* gSerialOutput;
void* gParallelOutput;
    
unsigned int gImageWidth, gImageHeight, gImageOffset, gImageBytesPerLine;

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
    
    if(fileHeader.headersize != 54 || fileHeader.infoSize != 40)
        exit(1);
    
    if(fileHeader.imageSize + fileHeader.headersize != fileHeader.filesize)
        exit(1);
    
    if(fileHeader.bitPlanes != 1 || fileHeader.bitCount != 24 || fileHeader.compression != 0)
        exit(1);
    
    gImageBytesPerLine = (fileHeader.filesize - fileHeader.headersize) / fileHeader.height;
    gImageOffset = fileHeader.headersize;
    
    gImageWidth = fileHeader.width;
    gImageHeight = fileHeader.height;

	fclose(fp);
}

void readImage(char* pImagePath, void* pImageData)
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

    for(unsigned int i = 0; i < gImageHeight; ++i)
    {
        char* lRow = ((char*)pImageData) + ((gImageHeight - i - 1) * gImageWidth * PIXEL_COUNT);
        for(unsigned int j = 0; j < gImageWidth; ++j)
        {
            if(fread((void*)(&lColor), sizeof(lColor), 1, fp) != 1)
                exit(1);
        
            lRow[PIXEL_COUNT * j] = lColor[2];
            lRow[PIXEL_COUNT * j + 1] = lColor[1];
            lRow[PIXEL_COUNT * j + 2] = lColor[0];
        }

        if(lSeekOffset)
        {
            if(fseek(fp, lSeekOffset, SEEK_CUR) != 0)
                exit(1);
        }
    }
    
	fclose(fp);
}
    
void serialImageFilter(void* pImageData)
{
}

#ifdef BUILD_CUDA
pmCudaLaunchConf GetCudaLaunchConf(int pMatrixDim)
{
	pmCudaLaunchConf lCudaLaunchConf;

	int lMaxThreadsPerBlock = 512;	// Max. 512 threads allowed per block

	lCudaLaunchConf.threadsX = pMatrixDim;
	if(lCudaLaunchConf.threadsX > lMaxThreadsPerBlock)
	{
		lCudaLaunchConf.threadsX = lMaxThreadsPerBlock;
		lCudaLaunchConf.blocksX = pMatrixDim/lCudaLaunchConf.threadsX;
        if(lCudaLaunchConf.blocksX * lCudaLaunchConf.threadsX < pMatrixDim)
            ++lCudaLaunchConf.blocksX;
	}

	return lCudaLaunchConf;
}
#endif

pmStatus imageFilterDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    unsigned int lTilesPerRow = (lTaskConf->imageWidth/TILE_DIM + (lTaskConf->imageWidth%TILE_DIM ? 1 : 0));
    //unsigned int lTilesPerCol = (lTaskConf->imageHeight/TILE_DIM + (lTaskConf->imageHeight%TILE_DIM ? 1 : 0));
    
	// Subscribe to one tile of the output matrix
    unsigned int lSubscriptionStartCol = (unsigned int)((pSubtaskId % lTilesPerRow) * TILE_DIM);
    unsigned int lSubscriptionEndCol = lSubscriptionStartCol + TILE_DIM;
    unsigned int lSubscriptionStartRow = (unsigned int)((pSubtaskId / lTilesPerRow) * TILE_DIM);
    unsigned int lSubscriptionEndRow = lSubscriptionStartRow + TILE_DIM;
    
    if(lSubscriptionEndCol > lTaskConf->imageWidth)
        lSubscriptionEndCol = lTaskConf->imageWidth;

    if(lSubscriptionEndRow > lTaskConf->imageHeight)
        lSubscriptionEndRow = lTaskConf->imageHeight;

    unsigned int lRowSize = lTaskConf->imageWidth * PIXEL_COUNT;
    lSubscriptionInfo.length = (lSubscriptionEndCol - lSubscriptionStartCol - 1) * PIXEL_COUNT;
    for(unsigned int i = lSubscriptionStartRow; i < lSubscriptionEndRow; ++i)
    {
        lSubscriptionInfo.offset = (i * lRowSize) + (lSubscriptionStartCol * PIXEL_COUNT);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);
    }

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, lTaskConf->cudaLaunchConf);
#endif

	return pmSuccess;
}

pmStatus imageFilter_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);
    
    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset;
    
	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	char* lImagePath = DEFAULT_IMAGE_PATH; \
	FETCH_STR_ARG(lImagePath, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

    void* lImageData = malloc(IMAGE_SIZE);
    readImage(lImagePath, lImageData);
    
	double lStartTime = getCurrentTimeInSecs();

	serialImageFilter(lImageData);

	double lEndTime = getCurrentTimeInSecs();

    free(lImageData);
	return (lEndTime - lStartTime);
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS

    if(pmMapFile(lImagePath) != pmSuccess)
        exit(1);

	// Output Mem contains the result image
	// Number of subtasks is equal to the number of rows
    unsigned int lSubtasks = (gImageWidth/TILE_DIM + (gImageWidth%TILE_DIM ? 1 : 0)) * (gImageHeight/TILE_DIM + (gImageHeight%TILE_DIM ? 1 : 0));
	CREATE_TASK(0, IMAGE_SIZE, lSubtasks, pCallbackHandle, pSchedulingPolicy)

	imageFilterTaskConf lTaskConf;
    strcpy(lTaskConf.imagePath, lImagePath);
    lTaskConf.imageWidth = gImageWidth;
    lTaskConf.imageHeight = gImageHeight;
    lTaskConf.imageOffset = gImageOffset;
    lTaskConf.imageBytesPerLine = gImageBytesPerLine;
	#ifdef BUILD_CUDA
		lTaskConf.cudaLaunchConf = GetCudaLaunchConf(lMatrixDim);
	#endif

	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	double lStartTime = getCurrentTimeInSecs();

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return (double)-1.0;
    }
    
	double lEndTime = getCurrentTimeInSecs();

	SAFE_PM_EXEC( pmFetchMemory(lTaskDetails.outputMemHandle) );

    pmRawMemPtr lRawOutputPtr;
    pmGetRawMemPtr(lTaskDetails.outputMemHandle, &lRawOutputPtr);
	memcpy(gParallelOutput, lRawOutputPtr, IMAGE_SIZE);

	FREE_TASK_AND_RESOURCES

    if(pmUnmapFile(lImagePath) != pmSuccess)
        exit(1);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = imageFilterDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = imageFilter_cpu;

	#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = imageFilter_cudaFunc;
	#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

    if(strlen(lImagePath) >= MAX_IMAGE_PATH_LENGTH)
    {
        std::cout << "Image path too long" << std::endl;
        exit(1);
    }
    
    readImageMetaData(lImagePath);

	gSerialOutput = malloc(IMAGE_SIZE);
	gParallelOutput = malloc(IMAGE_SIZE);

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
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
	for(size_t i=0; i<lImageSize; ++i)
	{
		if(lSerialOutput[i] != lParallelOutput[i])
		{
			std::cout << "Mismatch index " << i << " Serial Value = " << lSerialOutput[i] << " Parallel Value = " << lParallelOutput[i] << std::endl;
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
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "IMAGEFILTER");

	commonFinish();

	return 0;
}

}

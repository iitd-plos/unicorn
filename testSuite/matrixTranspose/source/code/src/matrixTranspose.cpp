
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "matrixTranspose.h"

MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;

size_t getBlockSizeRows(size_t pMatrixDimRows)
{
    return (pMatrixDimRows <= MAX_BLOCK_SIZE) ? pMatrixDimRows : MAX_BLOCK_SIZE;
}

size_t getBlockSizeCols(size_t pMatrixDimCols)
{
    return (pMatrixDimCols <= MAX_BLOCK_SIZE) ? pMatrixDimCols : MAX_BLOCK_SIZE;
}

void setBit(char* pFlags, size_t pBit)
{
    size_t lByte = pBit / 8;
    pFlags[lByte] |= (0x1 << (pBit % 8));
}

bool testBit(char* pFlags, size_t pBit)
{
    size_t lByte = pBit / 8;
    return (pFlags[lByte] & (0x1 << (pBit % 8)));
}

bool getNextCycleStart(char* pFlags, size_t pStartBit, size_t pBitCount, size_t* pNextCycleStart)
{
    for(size_t i = pStartBit; i < pBitCount; ++i)
    {
        if(!testBit(pFlags, i))
        {
            *pNextCycleStart = i;
            return true;
        }
    }
    
    return false;
}

size_t getBitIndex(size_t pIndexInBlock, size_t pInputDimY, size_t pBlockSizeY)
{
    size_t lBlockRow = (pIndexInBlock / pInputDimY);
    return (lBlockRow * pBlockSizeY) + (pIndexInBlock - (lBlockRow * pInputDimY));
}

size_t getBlockIndexFromBitIndex(size_t pBitIndex, size_t pInputDimY, size_t pBlockSizeY)
{
    size_t lBlockRow = (pBitIndex / pBlockSizeY);
    return (lBlockRow * pInputDimY) + (pBitIndex - (lBlockRow * pBlockSizeY));
}

void transposeMatrixBlock(MATRIX_DATA_TYPE* pInputBlock, MATRIX_DATA_TYPE* pOutputBlock, size_t pInputDimRows, size_t pInputDimCols, size_t pBlockSizeRows, size_t pBlockSizeCols, bool pIsFirstBlock, bool pIsLastBlock, unsigned long pSubtaskId)
{
    size_t lBitCount = pBlockSizeRows * pBlockSizeCols;
    size_t lByteCount = (lBitCount / 8) + ((lBitCount % 8) ? 1 : 0);
    char* lFlags = (char*)malloc(sizeof(char) * lByteCount);
    
    memset(lFlags, 0, lByteCount);
    if(pIsFirstBlock)
        setBit(lFlags, 0);
    
    if(pIsLastBlock)
        setBit(lFlags, lBitCount - 1);

    size_t lBlockCountRows = pInputDimRows / pBlockSizeRows;
    size_t lBlockCountCols = pInputDimCols / pBlockSizeCols;
    size_t lBlockIdRow = (int)(pSubtaskId / lBlockCountCols);
    size_t lBlockIdCol = (int)(pSubtaskId % lBlockCountCols);
    size_t lInputBlockIndex = (lBlockIdCol * pBlockSizeCols) + (lBlockIdRow * lBlockCountCols * pBlockSizeRows * pBlockSizeCols);
    size_t lOutputBlockIndex = (lBlockIdRow * pBlockSizeRows) + (lBlockIdCol * lBlockCountRows * pBlockSizeRows * pBlockSizeCols);

    MATRIX_DATA_TYPE lCurrVal, lTempVal;
    size_t lStartGlobalIndex, lInputGlobalIndex, lOutputGlobalIndex, lLocalIndex, lBitIndex;
    size_t lTestIndex = 0;
    size_t lFirstRow = (lBlockIdRow * pBlockSizeRows);
    size_t lFirstCol = (lBlockIdCol * pBlockSizeCols);
    size_t lLastRow = lFirstRow + pBlockSizeRows;
    size_t lLastCol = lFirstCol + pBlockSizeCols;
    while(getNextCycleStart(lFlags, lTestIndex, lBitCount, &lBitIndex))
    {
        bool lFlag = true;
        lTestIndex = lBitIndex + 1;

        lLocalIndex = getBlockIndexFromBitIndex(lBitIndex, pInputDimCols, pBlockSizeCols);
        lStartGlobalIndex = lInputGlobalIndex = lInputBlockIndex + lLocalIndex;
    
        lCurrVal = pInputBlock[lLocalIndex];

        do
        {
            setBit(lFlags, getBitIndex(lLocalIndex, pInputDimCols, pBlockSizeCols));
            lOutputGlobalIndex = (lInputGlobalIndex * pInputDimRows) % (pInputDimRows * pInputDimCols - 1);
            lTempVal = pOutputBlock[lOutputGlobalIndex - lOutputBlockIndex];
            pOutputBlock[lOutputGlobalIndex - lOutputBlockIndex] = lCurrVal;
        
            size_t lGlobalRow = (lOutputGlobalIndex / pInputDimCols);
            size_t lGlobalCol = lOutputGlobalIndex - (lGlobalRow * pInputDimCols);

            lFlag = false;
            if(lGlobalRow >= lFirstRow && lGlobalRow < lLastRow && lGlobalCol >= lFirstCol && lGlobalCol < lLastCol)
            {
                if(lOutputGlobalIndex != lStartGlobalIndex && !testBit(lFlags, getBitIndex(lOutputGlobalIndex - lInputBlockIndex, pInputDimCols, pBlockSizeCols)))
                {
                    lLocalIndex = lOutputGlobalIndex - lInputBlockIndex;
                    lInputGlobalIndex = lOutputGlobalIndex;
                    lCurrVal = lTempVal;
                    lFlag = true;
                }
            }
        } while(lFlag);
    }

    free(lFlags);
}

void serialmatrixTranspose(MATRIX_DATA_TYPE* pMatrix, size_t pInputDimRows, size_t pInputDimCols)
{
    size_t lBitCount = pInputDimRows * pInputDimCols;
    size_t lByteCount = (lBitCount / 8) + ((lBitCount % 8) ? 1 : 0);
    char* lFlags = (char*)malloc(sizeof(char) * lByteCount);
    
    memset(lFlags, 0, lByteCount);
    setBit(lFlags, 0);
    setBit(lFlags, lBitCount - 1);
    
    size_t lStartIndex, lIndex, lNextIndex, lVal;
    size_t lTestIndex = 1;
    while(getNextCycleStart(lFlags, lTestIndex, lBitCount, &lIndex))
    {
        lStartIndex = lIndex;
        lTestIndex = lStartIndex + 1;
    
        while(1)
        {
            setBit(lFlags, lIndex);

            lNextIndex = (lIndex * pInputDimCols) % (lBitCount - 1);
        
            if(lNextIndex == lStartIndex)
                break;
        
            lVal = pMatrix[lIndex];
            pMatrix[lIndex] = pMatrix[lNextIndex];
            pMatrix[lNextIndex] = lVal;
        
            lIndex = lNextIndex;
        }
    }
    
    free(lFlags);
}

#ifdef BUILD_CUDA
pmCudaLaunchConf GetCudaLaunchConf(size_t pMatrixDim)
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

pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
    size_t i;
	pmSubscriptionInfo lSubscriptionInfo;
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);
    
    size_t lBlockCountRows = lTaskConf->matrixDimRows / lTaskConf->blockSizeRows;
    size_t lBlockCountCols = lTaskConf->matrixDimCols / lTaskConf->blockSizeCols;
    size_t lBlockIdRow = (int)(pSubtaskId / lBlockCountCols);
    size_t lBlockIdCol = (int)(pSubtaskId % lBlockCountCols);
    
	// Subscribe to one block of the output matrix for reading and it's transposed block for writing
    size_t lBlockElemCount = lTaskConf->blockSizeRows * lTaskConf->blockSizeCols;
    size_t lBlockOffset = ((lBlockIdCol * lTaskConf->blockSizeCols) + (lBlockIdRow * lBlockCountCols * lBlockElemCount)) * sizeof(MATRIX_DATA_TYPE);
    for(i = 0; i < lTaskConf->blockSizeRows; ++i)
    {
        lSubscriptionInfo.offset = lBlockOffset;
        lSubscriptionInfo.length = lTaskConf->blockSizeCols * sizeof(MATRIX_DATA_TYPE);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    
        lBlockOffset += lTaskConf->matrixDimCols * sizeof(MATRIX_DATA_TYPE);
    }

    lBlockOffset = ((lBlockIdRow * lTaskConf->blockSizeRows) + (lBlockIdCol * lBlockCountRows * lBlockElemCount)) * sizeof(MATRIX_DATA_TYPE);
    for(i = 0; i < lTaskConf->blockSizeCols; ++i)
    {
        lSubscriptionInfo.offset = lBlockOffset;
        lSubscriptionInfo.length = lTaskConf->blockSizeRows * sizeof(MATRIX_DATA_TYPE);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);
    
        lBlockOffset += lTaskConf->matrixDimRows * sizeof(MATRIX_DATA_TYPE);
    }
    
#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, lTaskConf->cudaLaunchConf);
#endif
    
	return pmSuccess;
}

pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);

    transposeMatrixBlock((MATRIX_DATA_TYPE*)pSubtaskInfo.outputMemRead, (MATRIX_DATA_TYPE*)pSubtaskInfo.outputMemWrite, lTaskConf->matrixDimRows, lTaskConf->matrixDimCols, lTaskConf->blockSizeRows, lTaskConf->blockSizeCols, (pSubtaskInfo.subtaskId == 0), (pSubtaskInfo.subtaskId == pTaskInfo.subtaskCount - 1), pSubtaskInfo.subtaskId);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
    size_t lPowRows = DEFAULT_POW_ROWS; \
    size_t lPowCols = DEFAULT_POW_COLS; \
    FETCH_INT_ARG(lPowRows, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lPowCols, pCommonArgs + 1, argc, argv); \
    size_t lMatrixDimRows = 1 << lPowRows; \
    size_t lMatrixDimCols = 1 << lPowCols;

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
	double lStartTime = getCurrentTimeInSecs();
    
	serialmatrixTranspose(gSerialOutput, lMatrixDimRows, lMatrixDimCols);
    
	double lEndTime = getCurrentTimeInSecs();
    
	return (lEndTime - lStartTime);
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS
    
    size_t lBlockSizeRows = getBlockSizeRows(lMatrixDimRows);
    size_t lBlockSizeCols = getBlockSizeCols(lMatrixDimCols);
    size_t lBlocks = (lMatrixDimRows /lBlockSizeRows) * (lMatrixDimCols / lBlockSizeCols);

	// Input Mem is null
	// Output Mem contains the input and inplace result matrix
	// Number of subtasks is equal to the number of blocks the matrix is divided into
    
	pmMemHandle lMemHandle;
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
	size_t lMemSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);
    
	CREATE_MEM(lMemSize, lMemHandle);

	CREATE_TASK(0, 0, lBlocks, pCallbackHandle, pSchedulingPolicy)
    
    lTaskDetails.outputMemHandle = lMemHandle;
    lTaskDetails.outputMemInfo = OUTPUT_MEM_READ_WRITE;

    pmRawMemPtr lRawOutputPtr;
    pmGetRawMemPtr(lTaskDetails.outputMemHandle, &lRawOutputPtr);
    
	memcpy(lRawOutputPtr, gParallelOutput, lMemSize);
    
	matrixTransposeTaskConf lTaskConf;
	lTaskConf.matrixDimRows = lMatrixDimRows;
	lTaskConf.matrixDimCols = lMatrixDimCols;
    lTaskConf.blockSizeRows = lBlockSizeRows;
    lTaskConf.blockSizeCols = lBlockSizeCols;
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
    
	SAFE_PM_EXEC( pmFetchMemory(lMemHandle) );
    
	pmGetRawMemPtr(lMemHandle, &lRawOutputPtr);
	memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    
	pmReleaseMemory(lMemHandle);
    pmReleaseTask(lTaskHandle);
    
	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;
    
	lCallbacks.dataDistribution = matrixTransposeDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixTranspose_cpu;
    
#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = matrixTranspose_cuda;
#endif
    
	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
	srand((unsigned int)time(NULL));
    
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];
    
	for(size_t i=0; i<lMatrixElems; ++i)
		gSerialOutput[i] = gParallelOutput[i] = i;  //(MATRIX_DATA_TYPE)rand(); // i;
    
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
	READ_NON_COMMON_ARGS
    
    size_t i;
    
#if 0
	for(i=0; i<lMatrixDimCols; ++i)
    {
        for(size_t j=0; j<lMatrixDimRows; ++j)
        {
            std::cout << gSerialOutput[i * lMatrixDimRows + j] << "(" << gParallelOutput[i * lMatrixDimRows + j] << ") ";
        }

        std::cout << std::endl;
    }
#endif
    
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
	for(i=0; i<lMatrixElems; ++i)
    {
		if(gSerialOutput[i] != gParallelOutput[i])
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
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "MATRIXMUL");
    
	commonFinish();
    
	return 0;
}



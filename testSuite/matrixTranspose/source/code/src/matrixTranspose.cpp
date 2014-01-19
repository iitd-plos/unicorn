
/* Non-Square Matrix Transpose */

#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "matrixTranspose.h"

namespace matrixTranspose
{

MATRIX_DATA_TYPE* gSampleInput;
MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;

size_t getMin(size_t pVal1, size_t pVal2)
{
    return ((pVal1 < pVal2) ? pVal1 : pVal2);
}

void getBlockSize(size_t pMatrixDimRows, size_t pMatrixDimCols, size_t* pBlockSizeRows, size_t* pBlockSizeCols)
{
#ifdef USE_SQUARE_BLOCKS
    *pBlockSizeRows = *pBlockSizeCols = getMin(getMin(pMatrixDimRows, pMatrixDimCols), MAX_BLOCK_SIZE);
#else
    *pBlockSizeRows = ((pMatrixDimRows <= MAX_BLOCK_SIZE) ? pMatrixDimRows : MAX_BLOCK_SIZE);
    *pBlockSizeCols = ((pMatrixDimCols <= MAX_BLOCK_SIZE) ? pMatrixDimCols : MAX_BLOCK_SIZE);
#endif
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

#ifdef USE_SQUARE_BLOCKS
void transposeMatrixBlock(MATRIX_DATA_TYPE* pBlock, size_t pBlockDim)
{
    for(size_t i = 0; i < pBlockDim; ++i)
    {
        for(size_t j = i+1; j < pBlockDim; ++j)
        {
            MATRIX_DATA_TYPE lTemp = pBlock[j + i*pBlockDim];
            pBlock[j + i*pBlockDim] = pBlock[i + j*pBlockDim];
            pBlock[i + j*pBlockDim] = lTemp;
        }
    }
}

void transposeMatrixBlock(MATRIX_DATA_TYPE* pInputBlock, MATRIX_DATA_TYPE* pOutputBlock, size_t pBlockDim)
{
    for(size_t i = 0; i < pBlockDim; ++i)
        for(size_t j = 0; j < pBlockDim; ++j)
            pOutputBlock[j + i*pBlockDim] = pInputBlock[i + j*pBlockDim];
}
#else
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

void transposeMatrixBlock(MATRIX_DATA_TYPE* pInputBlock, MATRIX_DATA_TYPE* pOutputBlock, size_t pInputDimRows, size_t pInputDimCols, size_t pBlockSizeRows, size_t pBlockSizeCols, bool pIsFirstBlock, bool pIsLastBlock, unsigned long pSubtaskId, bool pInplace)
{
    MATRIX_DATA_TYPE* lInputBlock = pInplace ? pOutputBlock : pInputBlock;
    
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
    
    if(!pInplace)
    {
        if(pIsFirstBlock)
            pOutputBlock[0] = lInputBlock[0];

        if(pIsLastBlock)
        {
            lLocalIndex = getBlockIndexFromBitIndex(lBitCount - 1, pInputDimCols, pBlockSizeCols);
            pOutputBlock[lLocalIndex + lInputBlockIndex - lOutputBlockIndex] = lInputBlock[lLocalIndex];
        }
    }

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
            lTempVal = lInputBlock[lOutputGlobalIndex - (pInplace ? lOutputBlockIndex : lInputBlockIndex)];
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
#endif

void serialMatrixTranspose(bool pInplace, MATRIX_DATA_TYPE* pInputMatrix, MATRIX_DATA_TYPE* pOutputMatrix, size_t pInputDimRows, size_t pInputDimCols)
{
    MATRIX_DATA_TYPE* lInputMatrix = pInplace ? pOutputMatrix : pInputMatrix;

    size_t lBitCount = pInputDimRows * pInputDimCols;
    size_t lByteCount = (lBitCount / 8) + ((lBitCount % 8) ? 1 : 0);
    char* lFlags = (char*)malloc(sizeof(char) * lByteCount);
    
    memset(lFlags, 0, lByteCount);
    setBit(lFlags, 0);
    setBit(lFlags, lBitCount - 1);

    if(!pInplace)
    {
        pOutputMatrix[0] = lInputMatrix[0];
        pOutputMatrix[lBitCount - 1] = lInputMatrix[lBitCount - 1];
    }
    
    MATRIX_DATA_TYPE lVal;
    size_t lStartIndex, lIndex, lNextIndex;
    size_t lTestIndex = 1;
    
    if(pInplace)
    {
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
                
                lVal = pOutputMatrix[lIndex];
                pOutputMatrix[lIndex] = pOutputMatrix[lNextIndex];
                pOutputMatrix[lNextIndex] = lVal;
            
                lIndex = lNextIndex;
            }
        }
    }
    else
    {
        while(getNextCycleStart(lFlags, lTestIndex, lBitCount, &lIndex))
        {
            lStartIndex = lIndex;
            lTestIndex = lStartIndex + 1;
        
            while(1)
            {
                setBit(lFlags, lIndex);

                lNextIndex = (lIndex * pInputDimCols) % (lBitCount - 1);
                pOutputMatrix[lIndex] = lInputMatrix[lNextIndex];
            
                if(lNextIndex == lStartIndex)
                    break;
                
                lIndex = lNextIndex;
            }
        }
    }

    free(lFlags);
}

pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pmSubscriptionInfo lSubscriptionInfo;
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);
    
    size_t lBlockCountRows = lTaskConf->matrixDimRows / lTaskConf->blockSizeRows;
    size_t lBlockCountCols = lTaskConf->matrixDimCols / lTaskConf->blockSizeCols;
    size_t lBlockIdRow = (int)(pSubtaskInfo.subtaskId / lBlockCountCols);
    size_t lBlockIdCol = (int)(pSubtaskInfo.subtaskId % lBlockCountCols);

    unsigned int lInputMemIndex = (lTaskConf->inplace ? (unsigned int)INPLACE_MEM_INDEX : (unsigned int)INPUT_MEM_INDEX);
    unsigned int lOutputMemIndex = (lTaskConf->inplace ? (unsigned int)INPLACE_MEM_INDEX : (unsigned int)OUTPUT_MEM_INDEX);
    
	// Subscribe to one block of the output matrix for reading and it's transposed block for writing
    size_t lBlockElemCount = lTaskConf->blockSizeRows * lTaskConf->blockSizeCols;

    size_t lBlockOffset = ((lBlockIdCol * lTaskConf->blockSizeCols) + (lBlockIdRow * lBlockCountCols * lBlockElemCount)) * sizeof(MATRIX_DATA_TYPE);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lInputMemIndex, READ_SUBSCRIPTION, pmScatteredSubscriptionInfo(lBlockOffset, lTaskConf->blockSizeCols * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDimCols * sizeof(MATRIX_DATA_TYPE), lTaskConf->blockSizeRows));

    lBlockOffset = ((lBlockIdRow * lTaskConf->blockSizeRows) + (lBlockIdCol * lBlockCountRows * lBlockElemCount)) * sizeof(MATRIX_DATA_TYPE);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lOutputMemIndex, WRITE_SUBSCRIPTION, pmScatteredSubscriptionInfo(lBlockOffset, lTaskConf->blockSizeRows * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDimRows * sizeof(MATRIX_DATA_TYPE), lTaskConf->blockSizeCols));
    
#ifdef BUILD_CUDA
	// Reserve CUDA Global Mem
	if(lTaskConf->inplace && pDeviceInfo.deviceType == pm::GPU_CUDA)
    {
        size_t lBlockSize = sizeof(MATRIX_DATA_TYPE) * lTaskConf->blockSizeRows * lTaskConf->blockSizeRows;
		pmReserveCudaGlobalMem(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lBlockSize);
    }
#endif

	return pmSuccess;
}

pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);

    MATRIX_DATA_TYPE* lInputMem = lTaskConf->inplace ? (MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].readPtr : (MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr;
    pmSubscriptionVisibilityType lInputMemVisibilityType = lTaskConf->inplace ? pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].visibilityType : pSubtaskInfo.memInfo[INPUT_MEM_INDEX].visibilityType;
    pmSubscriptionVisibilityType lOutputMemVisibilityType = lTaskConf->inplace ? pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].visibilityType : pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType;
    
#ifdef USE_SQUARE_BLOCKS
    if(lTaskConf->inplace)
    {
        if(lInputMemVisibilityType != SUBSCRIPTION_NATURAL || lOutputMemVisibilityType != SUBSCRIPTION_NATURAL)
            exit(1);

        size_t lBlockDim = lTaskConf->blockSizeRows;
        size_t lBlockDimSize = sizeof(MATRIX_DATA_TYPE) * lBlockDim;
        size_t lBlockSize = lBlockDimSize * lBlockDim;
        MATRIX_DATA_TYPE* lBlock = (MATRIX_DATA_TYPE*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo,  PRE_SUBTASK_TO_SUBTASK, lBlockSize, NULL);
        
        size_t i;
        for(i = 0; i < lBlockDim; ++i)
            memcpy(lBlock + i*lBlockDim, lInputMem + i*lTaskConf->matrixDimCols, lBlockDimSize);
        
        transposeMatrixBlock(lBlock, lBlockDim);
        
        unsigned int lOutputMemIndex = (lTaskConf->inplace ? (unsigned int)INPLACE_MEM_INDEX : (unsigned int)OUTPUT_MEM_INDEX);
        
        for(i = 0; i < lBlockDim; ++i)
            memcpy(((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[lOutputMemIndex].writePtr) + i * lTaskConf->matrixDimRows, lBlock + i * lBlockDim, lBlockDimSize);
    }
    else
    {
        if(lInputMemVisibilityType != SUBSCRIPTION_COMPACT || lOutputMemVisibilityType != SUBSCRIPTION_COMPACT)
            exit(1);

        unsigned int lOutputMemIndex = (lTaskConf->inplace ? (unsigned int)INPLACE_MEM_INDEX : (unsigned int)OUTPUT_MEM_INDEX);

        MATRIX_DATA_TYPE* lOutputMem = (MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[lOutputMemIndex].writePtr;
        transposeMatrixBlock(lInputMem, lOutputMem, lTaskConf->blockSizeRows);
    }
#else
    if(lInputMemVisibilityType != SUBSCRIPTION_NATURAL || lOutputMemVisibilityType != SUBSCRIPTION_NATURAL)
        exit(1);

    transposeMatrixBlock(lInputMem, (MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[lOutputMemIndex].writePtr, lTaskConf->matrixDimRows, lTaskConf->matrixDimCols, lTaskConf->blockSizeRows, lTaskConf->blockSizeCols, (pSubtaskInfo.subtaskId == 0), (pSubtaskInfo.subtaskId == pTaskInfo.subtaskCount - 1), pSubtaskInfo.subtaskId, lTaskConf->inplace);
#endif

	return pmSuccess;
}

double parallelMatrixTranspose(size_t pPowRows, size_t pPowCols, size_t pMatrixDimRows, size_t pMatrixDimCols, pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemType pInputMemType, pmMemType pOutputMemType)
{
    bool lInplace = (pInputMemHandle == pOutputMemHandle);
    
    size_t lBlockSizeRows, lBlockSizeCols;
    getBlockSize(pMatrixDimRows, pMatrixDimCols, &lBlockSizeRows, &lBlockSizeCols);
    size_t lBlocks = (pMatrixDimRows / lBlockSizeRows) * (pMatrixDimCols / lBlockSizeCols);

	CREATE_TASK(lBlocks, pCallbackHandle, pSchedulingPolicy)

    pmTaskMem lTaskMem[MAX_MEM_INDICES];

    if(lInplace)
    {
        lTaskMem[INPLACE_MEM_INDEX] = {pOutputMemHandle, pOutputMemType, SUBSCRIPTION_OPTIMAL};
        
        lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
        lTaskDetails.taskMemCount = INPLACE_MAX_MEM_INDICES;
    }
    else
    {
    #ifdef USE_SQUARE_BLOCKS
        lTaskMem[INPUT_MEM_INDEX] = {pInputMemHandle, pInputMemType, SUBSCRIPTION_COMPACT};
        lTaskMem[OUTPUT_MEM_INDEX] = {pOutputMemHandle, pOutputMemType, SUBSCRIPTION_COMPACT};
    #else
        lTaskMem[INPUT_MEM_INDEX] = {pInputMemHandle, pInputMemType, SUBSCRIPTION_NATURAL};
        lTaskMem[OUTPUT_MEM_INDEX] = {pOutputMemHandle, pOutputMemType, SUBSCRIPTION_NATURAL};
    #endif
        
        lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
        lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    }
    
	matrixTransposeTaskConf lTaskConf;
	lTaskConf.matrixDimRows = pMatrixDimRows;
	lTaskConf.matrixDimCols = pMatrixDimCols;
    lTaskConf.blockSizeRows = lBlockSizeRows;
    lTaskConf.blockSizeCols = lBlockSizeCols;
    lTaskConf.inplace = lInplace;

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
    
    pmReleaseTask(lTaskHandle);

    return (lEndTime - lStartTime);
}

#define READ_NON_COMMON_ARGS \
    size_t lPowRows = DEFAULT_POW_ROWS; \
    size_t lPowCols = DEFAULT_POW_COLS; \
    bool lInplace = (bool)DEFAULT_INPLACE_VALUE; \
    FETCH_INT_ARG(lPowRows, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lPowCols, pCommonArgs + 1, argc, argv); \
    FETCH_BOOL_ARG(lInplace, pCommonArgs + 2, argc, argv); \
    size_t lMatrixDimRows = 1 << lPowRows; \
    size_t lMatrixDimCols = 1 << lPowCols;

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
	double lStartTime = getCurrentTimeInSecs();
    
	serialMatrixTranspose(lInplace, gSampleInput, gSerialOutput, lMatrixDimRows, lMatrixDimCols);
    
	double lEndTime = getCurrentTimeInSecs();
    
	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();
    
	singleGpuMatrixTranspose(lInplace, gSampleInput, gParallelOutput, lMatrixDimRows, lMatrixDimCols);
    
	double lEndTime = getCurrentTimeInSecs();
    
	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

	// Input Mem is null for inplace operation and contains input data otherwise
	// Output Mem contains the input and inplace result matrix for inplace operation and only result matrix otherwise
	// Number of subtasks is equal to the number of blocks the matrix is divided into
    
	pmMemHandle lInputMemHandle, lOutputMemHandle;
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
	size_t lMemSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);

	CREATE_MEM(lMemSize, lOutputMemHandle);
    
    if(lInplace)
        lInputMemHandle = lOutputMemHandle;
    else
        CREATE_MEM(lMemSize, lInputMemHandle);

    pmRawMemPtr lRawInputPtr;
    pmGetRawMemPtr(lInputMemHandle, &lRawInputPtr);
	memcpy(lRawInputPtr, (lInplace ? gParallelOutput : gSampleInput), lMemSize);

    double lTime = parallelMatrixTranspose(lPowRows, lPowCols, lMatrixDimRows, lMatrixDimCols, lInputMemHandle, lOutputMemHandle, pCallbackHandle[0], pSchedulingPolicy, READ_ONLY, (lInplace ? READ_WRITE : WRITE_ONLY));
    
    if(lTime != -1.0 && pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );
        
        pmRawMemPtr lRawOutputPtr;
        pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);

        memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    }
    
    pmReleaseMemory(lOutputMemHandle);
    
    if(!lInplace)
        pmReleaseMemory(lInputMemHandle);
    
	return lTime;
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;
    
	lCallbacks.dataDistribution = matrixTransposeDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixTranspose_cpu;
    
#ifdef BUILD_CUDA
#ifdef USE_SQUARE_BLOCKS
	lCallbacks.subtask_gpu_custom = matrixTranspose_cudaLaunchFunc;
#endif
#endif
    
	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
	srand((unsigned int)time(NULL));
    
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
    gSampleInput = (lInplace ? NULL : (new MATRIX_DATA_TYPE[lMatrixElems]));
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];

#if MATRIX_DATA_TYPE_INT == 1
    if(lInplace)
    {
        for(size_t i = 0; i < lMatrixElems; ++i)
            gSerialOutput[i] = gParallelOutput[i] = (MATRIX_DATA_TYPE)rand(); // i;
    }
    else
    {
        for(size_t i = 0; i < lMatrixElems; ++i)
            gSampleInput[i] = (MATRIX_DATA_TYPE)rand(); // i;
    }
#else
#endif

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
    delete[] gSampleInput;
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
            std::cout << gSerialOutput[i * lMatrixDimRows + j] << " (" << gParallelOutput[i * lMatrixDimRows + j] << ") ";
        }

        std::cout << std::endl;
    }
#endif
    
    size_t lMatrixElems = lMatrixDimRows * lMatrixDimCols;
	for(i=0; i<lMatrixElems; ++i)
    {
#if MATRIX_DATA_TYPE_INT == 1
		if(gSerialOutput[i] != gParallelOutput[i])
        {
			std::cout << "Mismatch index " << i << " Serial Value = " << gSerialOutput[i] << " Parallel Value = " << gParallelOutput[i] << std::endl;
			return 1;
        }
#else
#endif
    }

	return 0;
}

/**	Non-common args
 *	1. log 2 (no. of rows)
 *	2. log 2 (no. of cols)
 */
int main(int argc, char** argv)
{
    callbackStruct lStruct[1] = { {DoSetDefaultCallbacks, "MATRIXTRANSPOSE"} };
    
	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 1);
    
	commonFinish();
    
	return 0;
}

}

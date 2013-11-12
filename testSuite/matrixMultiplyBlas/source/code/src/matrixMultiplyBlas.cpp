
#include <time.h>
#include <string.h>
#include <math.h>

#include "commonAPI.h"
#include "matrixMultiplyBlas.h"

namespace matrixMultiplyBlas
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CBLAS_GEMM cblas_sgemm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CBLAS_GEMM cblas_dgemm
#endif

MATRIX_DATA_TYPE* gSampleInput;
MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;

size_t getBlockSize(size_t pMatrixDim)
{
    return ((pMatrixDim < (size_t)BLOCK_DIM) ? pMatrixDim : (size_t)BLOCK_DIM);
}

// pMatrixA is pDim1 * pDim2
// pMatrixB is pDim2 * pDim3
// pMatrixC is pDim1 * pDim3
void serialMatrixMultiply(MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, size_t pDim1, size_t pDim2, size_t pDim3, size_t pRowStepElems1, size_t pRowStepElems2, size_t pRowStepElems3)
{
    CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)pDim1, (int)pDim3, (int)pDim2, 1.0f, pMatrixA, (int)pRowStepElems1, pMatrixB, (int)pRowStepElems2, 0.0f, pMatrixC, (int)pRowStepElems3);
}

bool GetSplitData(size_t* pBlockOffset, size_t* pBlockHeight, matrixMultiplyTaskConf* pTaskConf, pmSplitInfo& pSplitInfo)
{
    *pBlockOffset = 0;
    *pBlockHeight = pTaskConf->blockDim;
    if(pSplitInfo.splitCount)
    {
        size_t lSplitCount = ((pTaskConf->blockDim < pSplitInfo.splitCount) ? pTaskConf->blockDim : pSplitInfo.splitCount);
        
        if(pSplitInfo.splitId > lSplitCount - 1)
            return false;
        
        size_t lSplittedBlockSize = (pTaskConf->blockDim / lSplitCount);
        *pBlockOffset = pSplitInfo.splitId * lSplittedBlockSize;

        if(pSplitInfo.splitId == lSplitCount - 1)
            *pBlockHeight = (pTaskConf->blockDim - (pSplitInfo.splitId * lSplittedBlockSize));
        else
            *pBlockHeight = lSplittedBlockSize;
    }
    
    return true;
}
    
pmStatus matrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

    // Subtask no. increases vertically in output matrix (for increased locality)
    size_t lBlocksPerDim = (lTaskConf->matrixDim / lTaskConf->blockDim);
    size_t lBlockRow = (pSubtaskInfo.subtaskId % lBlocksPerDim);
    size_t lBlockCol = (pSubtaskInfo.subtaskId / lBlocksPerDim);
    
    size_t lBlockOffset, lBlockHeight;
    if(!GetSplitData(&lBlockOffset, &lBlockHeight, lTaskConf, pSubtaskInfo.splitInfo))
        return pmSuccess;

	// Subscribe to entire lBlockRow of the first matrix (with equal split)
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX1_MEM_INDEX, READ_SUBSCRIPTION, pmScatteredSubscriptionInfo((lBlockRow * lTaskConf->blockDim + lBlockOffset) * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE), lBlockHeight));

	// Subscribe to entire lBlockCol of the second matrix
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX2_MEM_INDEX, READ_SUBSCRIPTION, pmScatteredSubscriptionInfo((lBlockCol * lTaskConf->blockDim) * sizeof(MATRIX_DATA_TYPE), lTaskConf->blockDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim));

	// Subscribe to one block of the output matrix (with equal split)
    SUBSCRIBE_BLOCK(lBlockRow, lBlockCol, lBlockOffset, lBlockHeight, lTaskConf->blockDim, lTaskConf->matrixDim, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MATRIX_MEM_INDEX, WRITE_SUBSCRIPTION)

	return pmSuccess;
}

pmStatus matrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    
    size_t lBlockOffset, lBlockHeight;
    if(!GetSplitData(&lBlockOffset, &lBlockHeight, lTaskConf, pSubtaskInfo.splitInfo))
        return pmSuccess;
        
    MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lMatrix2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lMatrix3 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

    size_t lSpanMatrix2 = (pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;
    size_t lSpanMatrix3 = (pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;

	serialMatrixMultiply(lMatrix1, lMatrix2, lMatrix3, lBlockHeight, lTaskConf->matrixDim, lTaskConf->blockDim, lTaskConf->matrixDim, lSpanMatrix2, lSpanMatrix3);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	size_t lPowMatrixDim = DEFAULT_POW_MATRIX_DIM; \
	FETCH_INT_ARG(lPowMatrixDim, pCommonArgs, argc, argv); \
    size_t lMatrixDim = (1 << lPowMatrixDim); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim; \
    lMatrixElems = lMatrixElems;        // Supress unused variable warning

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	serialMatrixMultiply(gSampleInput, gSampleInput + lMatrixElems, gSerialOutput, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	singleGpuMatrixMultiply(gSampleInput, gParallelOutput, lMatrixDim);

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

	size_t lMatrixSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);
    size_t lBlockSize = getBlockSize(lMatrixDim);

	// Input Mem 1 contains first input matrix
    // Input Mem 2 contains second input matrix
	// Output Mem contains the result matrix
	// The number of subtasks is equal to the number of blocks in the output matrix
    unsigned long lSubtaskCount = (lMatrixDim / lBlockSize) * (lMatrixDim / lBlockSize);

	CREATE_TASK(lSubtaskCount, pCallbackHandle[0], pSchedulingPolicy)

    pmMemHandle lInputMem1, lInputMem2, lOutputMem;
    CREATE_MEM(lMatrixSize, lInputMem1)
    CREATE_MEM(lMatrixSize, lInputMem2)
    CREATE_MEM(lMatrixSize, lOutputMem)

    pmRawMemPtr lRawInputPtr1, lRawInputPtr2, lRawOutputPtr;
    pmGetRawMemPtr(lInputMem1, &lRawInputPtr1);
    pmGetRawMemPtr(lInputMem2, &lRawInputPtr2);
    
	memcpy(lRawInputPtr1, gSampleInput, lMatrixSize);
	memcpy(lRawInputPtr2, gSampleInput + lMatrixElems, lMatrixSize);

    pmTaskMem lTaskMem[MAX_MEM_INDICES];
    lTaskMem[INPUT_MATRIX1_MEM_INDEX] = {lInputMem1, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[INPUT_MATRIX2_MEM_INDEX] = {lInputMem2, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[OUTPUT_MATRIX_MEM_INDEX] = {lOutputMem, WRITE_ONLY, SUBSCRIPTION_OPTIMAL};

    lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
    lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    
	matrixMultiplyTaskConf lTaskConf;
	lTaskConf.matrixDim = lMatrixDim;
    lTaskConf.blockDim = lBlockSize;

	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

    lTaskDetails.canSplitCpuSubtasks = true;
    
	double lStartTime = getCurrentTimeInSecs();

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
    {
        FREE_TASK_AND_RESOURCES
        return (double)-1.0;
    }
    
	double lEndTime = getCurrentTimeInSecs();

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lOutputMem) );

        pmGetRawMemPtr(lOutputMem, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lMatrixSize);
    }

	FREE_TASK_AND_RESOURCES

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = matrixMultiplyDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixMultiply_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = matrixMultiply_cudaLaunchFunc;
#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	srand((unsigned int)time(NULL));

	size_t lInputSize = 2 * lMatrixElems;

	gSampleInput = new MATRIX_DATA_TYPE[lInputSize];
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];

	for(size_t i = 0; i < lInputSize; ++i)
		gSampleInput[i] = (MATRIX_DATA_TYPE)(int)rand(); // (MATRIX_DATA_TYPE)i;

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

	for(size_t i = 0; i < lMatrixElems; ++i)
	{
		if((fabs(gSerialOutput[i] - gParallelOutput[i])/gSerialOutput[i]) > 1e-5)
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
    callbackStruct lStruct[1] = { {DoSetDefaultCallbacks, "MATRIXMUL"} };
    
	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 1);

	commonFinish();

	return 0;
}

}


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
void serialMatrixMultiply(MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, size_t pDim1, size_t pDim2, size_t pDim3, size_t pRowStepElems)
{
    CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)pDim1, (int)pDim3, (int)pDim2, 1.0f, pMatrixA, (int)pRowStepElems, pMatrixB, (int)pRowStepElems, 0.0f, pMatrixC, (int)pRowStepElems);
}

pmStatus matrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    
    size_t lBlocksPerDim = (lTaskConf->matrixDim / lTaskConf->blockDim);
    size_t lBlockRow = (pSubtaskId / lBlocksPerDim);
    size_t lBlockCol = (pSubtaskId % lBlocksPerDim);

	// Subscribe to entire lBlockRow of the first matrix
    for(size_t i = 0; i < lTaskConf->blockDim; ++i)
    {
        lSubscriptionInfo.offset = ((lBlockRow * lTaskConf->blockDim) + i) * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
        lSubscriptionInfo.length = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    }

	// Subscribe to entire lBlockCol of the second matrix
    for(size_t i = 0; i < lTaskConf->matrixDim; ++i)
    {
        lSubscriptionInfo.offset = (lTaskConf->matrixDim * lTaskConf->matrixDim + i * lTaskConf->matrixDim + lBlockCol * lTaskConf->blockDim) * sizeof(MATRIX_DATA_TYPE);
        lSubscriptionInfo.length = lTaskConf->blockDim * sizeof(MATRIX_DATA_TYPE);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    }

	// Subscribe to one block of the output matrix
    SUBSCRIBE_BLOCK(lBlockRow, lBlockCol, lTaskConf->blockDim, lTaskConf->matrixDim, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION)

	return pmSuccess;
}

pmStatus matrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    
    size_t lBlocksPerDim = (lTaskConf->matrixDim / lTaskConf->blockDim);
    size_t lBlockRow = (pSubtaskInfo.subtaskId / lBlocksPerDim);
    size_t lBlockCol = (pSubtaskInfo.subtaskId % lBlocksPerDim);

    MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem);
    MATRIX_DATA_TYPE* lMatrix2 = ((MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem)) + (lTaskConf->matrixDim * lTaskConf->matrixDim + lBlockCol * lTaskConf->blockDim) - (lBlockRow * lTaskConf->blockDim * lTaskConf->matrixDim);
    MATRIX_DATA_TYPE* lMatrix3 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.outputMem);
    
	serialMatrixMultiply(lMatrix1, lMatrix2, lMatrix3, lTaskConf->blockDim, lTaskConf->matrixDim, lTaskConf->blockDim, lTaskConf->matrixDim);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	size_t lPowMatrixDim = DEFAULT_POW_MATRIX_DIM; \
	FETCH_INT_ARG(lPowMatrixDim, pCommonArgs, argc, argv); \
    size_t lMatrixDim = (1 << lPowMatrixDim); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim;

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	serialMatrixMultiply(gSampleInput, gSampleInput + lMatrixElems, gSerialOutput, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim);

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

	// Input Mem contains both input matrices one after the other
	// Output Mem contains the result matrix
	// The number of subtasks is equal to the number of blocks in the output matrix
	size_t lInputMemSize = 2 * lMatrixSize;
	size_t lOutputMemSize = lMatrixSize;
    
    size_t lBlockSize = getBlockSize(lMatrixDim);
    unsigned long lSubtaskCount = (lMatrixDim / lBlockSize) * (lMatrixDim / lBlockSize);

	CREATE_TASK(lInputMemSize, lOutputMemSize, lSubtaskCount, pCallbackHandle[0], pSchedulingPolicy)

    pmRawMemPtr lRawInputPtr, lRawOutputPtr;
    pmGetRawMemPtr(lTaskDetails.inputMemHandle, &lRawInputPtr);
    
	memcpy(lRawInputPtr, gSampleInput, lInputMemSize);

	matrixMultiplyTaskConf lTaskConf;
	lTaskConf.matrixDim = lMatrixDim;
    lTaskConf.blockDim = lBlockSize;

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

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lTaskDetails.outputMemHandle) );

        pmGetRawMemPtr(lTaskDetails.outputMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lOutputMemSize);
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
		gSampleInput[i] = (MATRIX_DATA_TYPE)(int)rand(); // i;

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

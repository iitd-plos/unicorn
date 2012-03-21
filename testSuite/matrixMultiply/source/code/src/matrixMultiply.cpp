
#include <time.h>

#include "commonAPI.h"
#include "matrixMultiply.h"

MATRIX_DATA_TYPE* gSampleInput;
MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;

void serialMatrixMultiply(MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, int pDim1, int pDim2, int pDim3)
{
	int i, j, k;

	for(i = 0; i < pDim1; i++)
		for(j = 0; j < pDim3; j++)
			for(k = 0; k < pDim2; k++)
				pMatrixC[i*pDim3 + j] += pMatrixA[i*pDim2 + k] * pMatrixB[k*pDim3 + j];
}

pmStatus matrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	// Subscribe to one row of the first input matrix
	lSubscriptionInfo.offset = pSubtaskId * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	lSubscriptionInfo.blockLength = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, INPUT_MEM, lSubscriptionInfo);

	// Subscribe to entire second input matrix
	lSubscriptionInfo.offset = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	lSubscriptionInfo.blockLength = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, INPUT_MEM, lSubscriptionInfo);

	// Subscribe to one row of the output matrix
	lSubscriptionInfo.offset = pSubtaskId * lTaskConf->matrixDim;
	lSubscriptionInfo.blockLength = lTaskConf->matrixDim;
	pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, OUTPUT_MEM, lSubscriptionInfo);

	return pmSuccess;
}

pmStatus matrixMultiply_cpu(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	serialMatrixMultiply((MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem), (MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem) + (lTaskConf->matrixDim * lTaskConf->matrixDim), 
		(MATRIX_DATA_TYPE*)(pSubtaskInfo.outputMem), 1, lTaskConf->matrixDim, lTaskConf->matrixDim);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	int lMatrixDim = DEFAULT_MATRIX_DIM; \
	FETCH_INT_ARG(lMatrixDim, pCommonArgs, argc, argv); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim; \
	size_t lMatrixSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	serialMatrixMultiply(gSampleInput, gSampleInput + lMatrixElems, gSerialOutput, lMatrixDim, lMatrixDim, lMatrixDim);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbacks pCallbacks)
{
	double lExecTime = (double)0;

	READ_NON_COMMON_ARGS

	// Input Mem contains both input matrices one after the other
	// Output Mem contains the result matrix
	// Number of subtasks is equal to the number of rows
	size_t lInputMemSize = 2 * lMatrixSize;
	size_t lOutputMemSize = lMatrixSize;

	CREATE_TASK(lInputMemSize, lOutputMemSize, lMatrixDim, "MATRIXMUL", pCallbacks)

	memcpy(lTaskDetails.inputMem, gSampleInput, lInputMemSize);
	memset(lTaskDetails.outputMem, 0, lOutputMemSize);

	matrixMultiplyTaskConf lTaskConf;
	lTaskConf.matrixDim = lMatrixDim;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, lTaskHandle) );
	SAFE_PM_EXEC( pmGetTaskExecutionTimeInSecs(lTaskHandle, &lExecTime) );

	memcpy(gParallelOutput, lTaskDetails.outputMem, lOutputMemSize);

	FREE_TASK_AND_RESOURCES

	return lExecTime;
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = matrixMultiplyDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixMultiply_cpu;
	lCallbacks.subtask_gpu_cuda = matrixMultiply_cuda;

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	srand(time(NULL));

	size_t lInputSize = 2*lMatrixElems;

	gSampleInput = new MATRIX_DATA_TYPE[lInputSize];
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];

	for(size_t i=0; i<lInputSize; ++i)
		gSampleInput[i] = (MATRIX_DATA_TYPE)rand();

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

	for(size_t i=0; i<lMatrixElems; ++i)
	{
		if(gSerialOutput[i] != gParallelOutput[i])
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
void main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy);

	commonFinish();
}



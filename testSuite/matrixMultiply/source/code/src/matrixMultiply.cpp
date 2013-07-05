
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "matrixMultiply.h"

namespace matrixMultiply
{
    
MATRIX_DATA_TYPE* gSampleInput;
MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;

void serialMatrixMultiply(MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, int pDim1, int pDim2, int pDim3)
{
	int i, j, k;

	for(i = 0; i < pDim1; ++i)
		for(j = 0; j < pDim3; ++j)
			for(k = 0; k < pDim2; ++k)
				pMatrixC[i*pDim3 + j] += pMatrixA[i*pDim2 + k] * pMatrixB[k*pDim3 + j];
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

pmStatus matrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	// Subscribe to entire first and second input matrix (default PMLIB behaviour)

	// Subscribe to one row of the output matrix
	lSubscriptionInfo.offset = pSubtaskId * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	lSubscriptionInfo.length = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, lTaskConf->cudaLaunchConf);
#endif

	return pmSuccess;
}

pmStatus matrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	memset(pSubtaskInfo.outputMem, 0, lTaskConf->matrixDim*sizeof(MATRIX_DATA_TYPE));

	serialMatrixMultiply((MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem) + (pSubtaskInfo.subtaskId * lTaskConf->matrixDim), (MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem) + (lTaskConf->matrixDim * lTaskConf->matrixDim), (MATRIX_DATA_TYPE*)(pSubtaskInfo.outputMem), 1, lTaskConf->matrixDim, lTaskConf->matrixDim);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	int lMatrixDim = DEFAULT_MATRIX_DIM; \
	FETCH_INT_ARG(lMatrixDim, pCommonArgs, argc, argv); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim;

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
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

	size_t lMatrixSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);

	// Input Mem contains both input matrices one after the other
	// Output Mem contains the result matrix
	// Number of subtasks is equal to the number of rows
	size_t lInputMemSize = 2 * lMatrixSize;
	size_t lOutputMemSize = lMatrixSize;

	CREATE_TASK(lInputMemSize, lOutputMemSize, lMatrixDim, pCallbackHandle, pSchedulingPolicy)

    pmRawMemPtr lRawInputPtr, lRawOutputPtr;
    pmGetRawMemPtr(lTaskDetails.inputMemHandle, &lRawInputPtr);
    
	memcpy(lRawInputPtr, gSampleInput, lInputMemSize);

	matrixMultiplyTaskConf lTaskConf;
	lTaskConf.matrixDim = lMatrixDim;
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
	lCallbacks.subtask_gpu_cuda = matrixMultiply_cudaFunc;
	#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	srand((unsigned int)time(NULL));

	size_t lInputSize = 2*lMatrixElems;

	gSampleInput = new MATRIX_DATA_TYPE[lInputSize];
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];

	for(size_t i=0; i<lInputSize; ++i)
		gSampleInput[i] = (MATRIX_DATA_TYPE)rand(); // i;

	memset(gSerialOutput, 0, lMatrixElems*sizeof(MATRIX_DATA_TYPE));

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
int main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "MATRIXMUL");

	commonFinish();

	return 0;
}

}

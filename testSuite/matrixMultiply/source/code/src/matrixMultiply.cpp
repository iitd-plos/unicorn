
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

pmStatus matrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	pmSubscriptionInfo lSubscriptionInfo;
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	// Subscribe to one row of the first input matrix
    lSubscriptionInfo.offset = pSubtaskInfo.subtaskId * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
    lSubscriptionInfo.length = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);
    
    // Subscribe to entire second input matrix
    lSubscriptionInfo.offset = 0;
    lSubscriptionInfo.length = lTaskConf->matrixDim * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX2_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);

	// Subscribe to one row of the output matrix
	lSubscriptionInfo.offset = pSubtaskInfo.subtaskId * lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	lSubscriptionInfo.length = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
	pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MATRIX_MEM_INDEX, WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	// Set CUDA Launch Configuration
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lTaskConf->cudaLaunchConf);
#endif

	return pmSuccess;
}

pmStatus matrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

	memset(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr, 0, lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE));

	serialMatrixMultiply((MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr), (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr), (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr), 1, lTaskConf->matrixDim, lTaskConf->matrixDim);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	int lMatrixDim = DEFAULT_MATRIX_DIM; \
	FETCH_INT_ARG(lMatrixDim, pCommonArgs, argc, argv); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim; \
    lMatrixElems = lMatrixElems;    // To suppress unused variable warning

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
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

	size_t lMatrixSize = lMatrixElems * sizeof(MATRIX_DATA_TYPE);

	// Input Mem 1 contains first input matrix
    // Input Mem 2 contains second input matrix
	// Output Mem contains the result matrix
	// Number of subtasks is equal to the number of rows
	CREATE_TASK(lMatrixDim, pCallbackHandle[0], pSchedulingPolicy)
    
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
    lTaskMem[INPUT_MATRIX1_MEM_INDEX] = {lInputMem1, READ_ONLY};
    lTaskMem[INPUT_MATRIX2_MEM_INDEX] = {lInputMem2, READ_ONLY};
    lTaskMem[OUTPUT_MATRIX_MEM_INDEX] = {lOutputMem, WRITE_ONLY};

    lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
    lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    
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

	for(size_t i = 0; i < lInputSize; ++i)
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

	for(size_t i = 0; i < lMatrixElems; ++i)
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
    callbackStruct lStruct[1] = { {DoSetDefaultCallbacks, "MATRIXMUL"} };
    
	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 1);

	commonFinish();

	return 0;
}

}

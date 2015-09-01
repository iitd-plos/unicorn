
#include <time.h>
#include <string.h>
#include <math.h>

#include "commonAPI.h"
#include "sparseSolver.h"

#include <vector>
#include <set>
#include <algorithm>

namespace sparseSolver
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CBLAS_GEMM cblas_sgemm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CBLAS_GEMM cblas_dgemm
#endif

INDICES_TYPE* gSampleInputRowIndices;
INDICES_TYPE* gSampleInputColIndices;
COUNT_TYPE* gSampleInputNnz1;
MATRIX_DATA_TYPE* gSampleInputData;

MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;
    
COUNT_TYPE gCountNnz1, gCountNnz2;


// pMatrixA is pDim1 * pDim2
// pMatrixB is pDim2 * pDim3
// pMatrixC is pDim1 * pDim3
void serialSparseMatrixMultiply(MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, size_t pDim1, size_t pDim2, size_t pDim3, size_t pRowStepElems1, size_t pRowStepElems2, size_t pRowStepElems3)
{
//    CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)pDim1, (int)pDim3, (int)pDim2, 1.0f, pMatrixA, (int)pRowStepElems1, pMatrixB, (int)pRowStepElems2, 0.0f, pMatrixC, (int)pRowStepElems3);
}

bool GetSplitData(size_t* pRowOffset, size_t* pRowCount, sparseMatrixMultiplyTaskConf* pTaskConf, pmSplitInfo& pSplitInfo)
{
    *pRowOffset = 0;
    *pRowCount = MATRIX_ROWS_PER_SUBTASK;

    if(pSplitInfo.splitCount)
    {
        unsigned int lEffectiveSplitCount = std::min(pSplitInfo.splitCount, (unsigned int)MATRIX_ROWS_PER_SUBTASK);
        
        size_t lRowsPerSplit = MATRIX_ROWS_PER_SUBTASK / lEffectiveSplitCount;
        size_t lLeftoverRows = MATRIX_ROWS_PER_SUBTASK - lRowsPerSplit * lEffectiveSplitCount;
        
        if(pSplitInfo.splitId > lEffectiveSplitCount - 1)
            return false;

        *pRowCount = lRowsPerSplit + ((lLeftoverRows > pSplitInfo.splitId) ? 1 : 0);
        *pRowOffset = pSplitInfo.splitId * lRowsPerSplit + std::min(lLeftoverRows, (size_t)pSplitInfo.splitId);
    }
    
    return true;
}

// Finds the data range (indices in the input matrix) that corresponds to a subtask's rows
bool GetRowIndicesFromSplitData(size_t pFirstRow, size_t pRowOffset, size_t pRowCount, int* pDistributionData, INDICES_TYPE* pRowIndices1, int& pStartIndex, int& pEndIndex, pmSplitInfo& pSplitInfo)
{
    if(pSplitInfo.splitCount)
    {
        size_t lStartRow = pFirstRow + pRowOffset;
        size_t lEndRow = lStartRow + pRowCount;

        if(pRowIndices1[pStartIndex] > lEndRow)
            return false;
        
        for(int i = pStartIndex; i<= pEndIndex; ++i)
        {
            if(pRowIndices1[i] >= lStartRow)
            {
                pStartIndex = i;
                break;
            }
        }

        if(pStartIndex != pEndIndex)
        {
            for(int i = pStartIndex + 1; i<= pEndIndex; ++i)
            {
                if(pRowIndices1[i] > lEndRow)
                {
                    pEndIndex = i;
                    break;
                }
            }
        }
    }
    
    return true;
}

pmStatus sparseMatrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	sparseMatrixMultiplyTaskConf* lTaskConf = (sparseMatrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    int* lDistributionData = (int*)(((char*)lTaskConf) + sizeof(sparseMatrixMultiplyTaskConf));

    int lStartIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId];
    int lEndIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId + 1];
    
    if(lStartIndexForSubtask != -1 && lEndIndexForSubtask != -1)
    {
        size_t lRowOffset, lRowCount;
        if(!GetSplitData(&lRowOffset, &lRowCount, lTaskConf, pSubtaskInfo.splitInfo))
            return pmSuccess;

        size_t lRowSize = lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE);
        lRowOffset += MATRIX_ROWS_PER_SUBTASK * pSubtaskInfo.subtaskId;
        
        size_t lStartIndexInBytes = lStartIndexForSubtask * sizeof(INDICES_TYPE);
        size_t lEndIndexInBytes = lEndIndexForSubtask * sizeof(INDICES_TYPE);
        
        pmSubscriptionInfo lSubscriptionInfo1(lStartIndexInBytes, lEndIndexInBytes);
        pmSubscriptionInfo lSubscriptionInfo2(0, NON_SPARSE_ELEMENT_COUNT((lTaskConf->matrixDim * lTaskConf->matrixDim)) * sizeof(MATRIX_DATA_TYPE));
        
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo1);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_ROW_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo1);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_COL_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo1);
        
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo2);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_ROW_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo2);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_COL_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo2);
        
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_NNZ1_INDEX, READ_SUBSCRIPTION, pmSubscriptionInfo(lRowOffset * sizeof(COUNT_TYPE), lRowCount * sizeof(COUNT_TYPE)));

        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MATRIX_MEM_INDEX, WRITE_SUBSCRIPTION, pmSubscriptionInfo(lRowOffset * lRowSize, lRowCount * lRowSize));

    #ifdef BUILD_CUDA
        // Reserve CUDA Global Mem
        if(pDeviceInfo.deviceType == pm::GPU_CUDA)
        {
            size_t lReservedMem = (2 * (lRowCount + 1)  + (lTaskConf->matrixDim + 1)) * sizeof(INDICES_TYPE);
            pmReserveCudaGlobalMem(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lReservedMem);
        }
    #endif
    }

	return pmSuccess;
}

pmStatus sparseMatrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	sparseMatrixMultiplyTaskConf* lTaskConf = (sparseMatrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    int* lDistributionData = (int*)(((char*)lTaskConf) + sizeof(sparseMatrixMultiplyTaskConf));
    
    int lStartIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId];
    int lEndIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId + 1];
    
    if(lStartIndexForSubtask != -1 && lEndIndexForSubtask != -1)
    {
        size_t lRowOffset, lRowCount;
        if(!GetSplitData(&lRowOffset, &lRowCount, lTaskConf, pSubtaskInfo.splitInfo))
            return pmSuccess;

        INDICES_TYPE* lRowIndices1 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_ROW_INDICES1_MEM_INDEX].ptr);
        INDICES_TYPE* lRowIndices2 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_ROW_INDICES2_MEM_INDEX].ptr);
        INDICES_TYPE* lColIndices1 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_COL_INDICES1_MEM_INDEX].ptr);
        INDICES_TYPE* lColIndices2 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_COL_INDICES2_MEM_INDEX].ptr);
        COUNT_TYPE* lNnz1 = (COUNT_TYPE*)(pSubtaskInfo.memInfo[INPUT_MEM_NNZ1_INDEX].ptr);

        if(!GetRowIndicesFromSplitData(MATRIX_ROWS_PER_SUBTASK * pSubtaskInfo.subtaskId, lRowOffset, lRowCount, lDistributionData, lRowIndices1, lStartIndexForSubtask, lEndIndexForSubtask, pSubtaskInfo.splitInfo))
            return pmSuccess;

        MATRIX_DATA_TYPE* lSparseMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lSparseMatrix2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lOutputMatrix = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

//        size_t lSpanMatrix2 = (pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;
//        size_t lSpanMatrix3 = (pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;
//
//        serialSparseMatrixMultiply(lMatrix1, lMatrix2, lMatrix3, lBlockHeight, lTaskConf->matrixDim, lTaskConf->blockDim, lTaskConf->matrixDim, lSpanMatrix2, lSpanMatrix3);
    }

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
    size_t lMatrixDim = DEFAULT_MATRIX_DIM; \
    FETCH_INT_ARG(lMatrixDim, pCommonArgs, argc, argv); \
	size_t lMatrixElems = lMatrixDim * lMatrixDim; \
    lMatrixElems = lMatrixElems;        // Supress unused variable warning

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	serialSparseMatrixMultiply(gSampleInputData, gSampleInputData + lMatrixElems, gSerialOutput, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim, lMatrixDim);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	singleGpuSparseMatrixMultiply(gSampleInputRowIndices, gSampleInputColIndices, gSampleInputData, gParallelOutput, gCountNnz1, gCountNnz2, lMatrixDim);

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

    size_t lInputSize = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);
    size_t lInputSizeInBytes = lInputSize * sizeof(MATRIX_DATA_TYPE);
    size_t lIndicesSizeInBytes = lInputSize * sizeof(INDICES_TYPE);

	// Input Mem 1 contains first input matrix
    // Input Mem 2 contains second input matrix
	// Output Mem contains the result matrix
	// The number of subtasks is equal to the number of blocks in the output matrix
    unsigned long lSubtaskCount = lMatrixDim / MATRIX_ROWS_PER_SUBTASK;

	CREATE_TASK(lSubtaskCount, pCallbackHandle[0], pSchedulingPolicy)

    pmMemHandle lInputMem1, lInputMem2, lInputMemRowIndices1, lInputMemRowIndices2, lInputMemColIndices1, lInputMemColIndices2, lInputMemNnz1, lOutputMem;
    CREATE_MEM(lInputSizeInBytes, lInputMem1)
    CREATE_MEM(lInputSizeInBytes, lInputMem2)
    CREATE_MEM(lIndicesSizeInBytes, lInputMemRowIndices1)
    CREATE_MEM(lIndicesSizeInBytes, lInputMemRowIndices2)
    CREATE_MEM(lIndicesSizeInBytes, lInputMemColIndices1)
    CREATE_MEM(lIndicesSizeInBytes, lInputMemColIndices2)
    CREATE_MEM((lMatrixDim * sizeof(COUNT_TYPE)), lInputMemNnz1)
    CREATE_MEM(lInputSizeInBytes, lOutputMem)

    pmRawMemPtr lRawInputPtr1, lRawInputPtr2, lRawInputRowIndicesPtr1, lRawInputRowIndicesPtr2, lRawInputColIndicesPtr1, lRawInputColIndicesPtr2, lRawInputMemNnzPtr1, lRawOutputPtr;
    pmGetRawMemPtr(lInputMem1, &lRawInputPtr1);
    pmGetRawMemPtr(lInputMem2, &lRawInputPtr2);
    pmGetRawMemPtr(lInputMemRowIndices1, &lRawInputRowIndicesPtr1);
    pmGetRawMemPtr(lInputMemRowIndices2, &lRawInputRowIndicesPtr2);
    pmGetRawMemPtr(lInputMemColIndices1, &lRawInputColIndicesPtr1);
    pmGetRawMemPtr(lInputMemColIndices2, &lRawInputColIndicesPtr2);
    pmGetRawMemPtr(lInputMemNnz1, &lRawInputMemNnzPtr1);

	memcpy(lRawInputPtr1, gSampleInputData, lInputSizeInBytes);
	memcpy(lRawInputPtr2, gSampleInputData + lInputSize, lInputSizeInBytes);
    memcpy(lRawInputRowIndicesPtr1, gSampleInputRowIndices, lInputSizeInBytes);
    memcpy(lRawInputRowIndicesPtr2, gSampleInputRowIndices + lInputSize, lInputSizeInBytes);
    memcpy(lRawInputColIndicesPtr1, gSampleInputColIndices, lInputSizeInBytes);
    memcpy(lRawInputColIndicesPtr2, gSampleInputColIndices + lInputSize, lInputSizeInBytes);
    memcpy(lRawInputMemNnzPtr1, gSampleInputNnz1, lMatrixDim * sizeof(COUNT_TYPE));
    
    DistributeMemory(lInputMem1, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(MATRIX_DATA_TYPE), true);
    DistributeMemory(lInputMem2, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(MATRIX_DATA_TYPE), true);
    DistributeMemory(lInputMemRowIndices1, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemRowIndices2, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemColIndices1, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemColIndices2, LINEAR_DIST, 0, (unsigned int)lInputSize, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemNnz1, LINEAR_DIST, 0, (unsigned int)lMatrixDim, 0, sizeof(COUNT_TYPE), true);

    pmTaskMem lTaskMem[MAX_MEM_INDICES];
    lTaskMem[INPUT_MATRIX1_MEM_INDEX] = {lInputMem1, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_MATRIX2_MEM_INDEX] = {lInputMem2, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_ROW_INDICES1_MEM_INDEX] = {lInputMemRowIndices1, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_ROW_INDICES2_MEM_INDEX] = {lInputMemRowIndices2, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_COL_INDICES1_MEM_INDEX] = {lInputMemColIndices1, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_COL_INDICES2_MEM_INDEX] = {lInputMemColIndices2, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[INPUT_MEM_NNZ1_INDEX] = {lInputMemNnz1, READ_ONLY, SUBSCRIPTION_NATURAL};
    lTaskMem[OUTPUT_MATRIX_MEM_INDEX] = {lOutputMem, WRITE_ONLY, SUBSCRIPTION_NATURAL};

    lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
    lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    
     // 2 entries per subtask (one for start index and other for end index in all three input data arrays; -1 is sentinel)
    unsigned int lTaskConfLength = sizeof(sparseMatrixMultiplyTaskConf) + 2 * sizeof(unsigned int) * (unsigned int)lSubtaskCount;
    std::vector<char> lTaskConf(lTaskConfLength);

    sparseMatrixMultiplyTaskConf* lSparseTaskConf = (sparseMatrixMultiplyTaskConf*)(&lTaskConf[0]);
    int* lDistributionData = (int*)((char*)(&lTaskConf[0]) + sizeof(sparseMatrixMultiplyTaskConf));

    lSparseTaskConf->matrixDim = lMatrixDim;
    lSparseTaskConf->nnz2 = gCountNnz2;

	lTaskDetails.taskConf = (void*)(&lTaskConf[0]);
    lTaskDetails.taskConfLength = lTaskConfLength;

    lTaskDetails.canSplitCpuSubtasks = true;
    
	double lStartTime = getCurrentTimeInSecs();

    unsigned int lCurrentIndex = 0;
    for(unsigned int i = 0; i < (unsigned int)lSubtaskCount; ++i)
    {
        unsigned int lLastRowForSubtask = (i + 1) * MATRIX_ROWS_PER_SUBTASK - 1;
        
        if(gSampleInputRowIndices[lCurrentIndex] > lLastRowForSubtask)
        {
            lDistributionData[2 * i] = lDistributionData[2 * i + 1] = -1;
        }
        else
        {
            lDistributionData[2 * i] = lCurrentIndex;
            
            while(gSampleInputRowIndices[++lCurrentIndex] <= lLastRowForSubtask);

            lDistributionData[2 * i + 1] = lCurrentIndex - 1;
        }
    }

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
        memcpy(gParallelOutput, lRawOutputPtr, lInputSizeInBytes);
    }

	FREE_TASK_AND_RESOURCES

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = sparseMatrixMultiplyDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = sparseMatrixMultiply_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = sparseMatrixMultiply_cudaLaunchFunc;
#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
    if(lMatrixDim % MATRIX_ROWS_PER_SUBTASK != 0)
    {
        std::cout << "Matrix dimension not divisible by MATRIX_ROWS_PER_SUBTASK" << std::endl;
        exit(1);
    }

	srand((unsigned int)time(NULL));

	size_t lInputSize = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);

    gSampleInputRowIndices = new INDICES_TYPE[2 * lInputSize];
    gSampleInputColIndices = new INDICES_TYPE[2 * lInputSize];
    gSampleInputNnz1 = new COUNT_TYPE[lMatrixDim];   // Stores number of non-zero elements in the first input matrix
    gSampleInputData = new MATRIX_DATA_TYPE[2 * lInputSize];

	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];
    
    memset(gSampleInputNnz1, 0, 2 * lMatrixDim * sizeof(COUNT_TYPE));
    
    std::set<std::pair<INDICES_TYPE, INDICES_TYPE>> lLocationsMatrix1, lLocationsMatrix2;

	for(size_t i = 0; i < lInputSize; ++i)
    {
        lLocationsMatrix1.emplace(rand() % lMatrixDim, rand() % lMatrixDim);
        lLocationsMatrix2.emplace(rand() % lMatrixDim, rand() % lMatrixDim);
    }

    INDICES_TYPE index = 0;
    std::for_each(lLocationsMatrix1.begin(), lLocationsMatrix1.end(), [&] (const std::pair<INDICES_TYPE, INDICES_TYPE>& pPair)
    {
        gSampleInputRowIndices[index] = pPair.first;
        gSampleInputColIndices[index] = pPair.second;
        
        MATRIX_DATA_TYPE lVal = (MATRIX_DATA_TYPE)(int)rand();
        
        if(lVal)
        {
            ++gSampleInputNnz1[pPair.first];
            ++gCountNnz1;
        }

		gSampleInputData[index] = lVal;
        ++index;
    });

    index = 0;
    std::for_each(lLocationsMatrix2.begin(), lLocationsMatrix2.end(), [&] (const std::pair<INDICES_TYPE, INDICES_TYPE>& pPair)
    {
        size_t lMatrixIndex = lInputSize + index;

        gSampleInputRowIndices[lMatrixIndex] = pPair.first;
        gSampleInputColIndices[lMatrixIndex] = pPair.second;

        MATRIX_DATA_TYPE lVal = (MATRIX_DATA_TYPE)(int)rand();

        if(lVal)
            ++gCountNnz2;

        gSampleInputData[lMatrixIndex] = lVal;
        ++index;
    });
    
    return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
    delete[] gSampleInputNnz1;
    delete[] gSampleInputRowIndices;
    delete[] gSampleInputColIndices;
    delete[] gSampleInputData;

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

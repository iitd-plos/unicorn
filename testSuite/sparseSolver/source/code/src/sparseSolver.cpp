
#include <time.h>
#include <string.h>
#include <math.h>

#include "commonAPI.h"
#include "sparseSolver.h"

#include <vector>
#include <set>
#include <algorithm>

#include <mkl_spblas.h>

namespace sparseSolver
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define MKL_SPARSE_COOMM mkl_scoomm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define MKL_SPARSE_COOMM mkl_dcoomm
#endif

INDICES_TYPE* gSampleInputRowIndices;
INDICES_TYPE* gSampleInputColIndices;
MATRIX_DATA_TYPE* gSampleInputData1;
COUNT_TYPE* gSampleInputNnz1;

MATRIX_DATA_TYPE* gSampleInputData2;

MATRIX_DATA_TYPE* gSerialOutput;
MATRIX_DATA_TYPE* gParallelOutput;
    
COUNT_TYPE gCountNnz1;

size_t getBlockSize(size_t pMatrixDim)
{
    return ((pMatrixDim < (size_t)BLOCK_DIM) ? pMatrixDim : (size_t)BLOCK_DIM);
}

// pMatrixA is pDim1 * pDim2
// pMatrixB is pDim2 * pDim3
// pMatrixC is pDim1 * pDim3
void serialSparseMatrixMultiply(INDICES_TYPE* pRowIndicesA, INDICES_TYPE* pColIndicesA, int pNnzA, MATRIX_DATA_TYPE* pMatrixA, MATRIX_DATA_TYPE* pMatrixB, MATRIX_DATA_TYPE* pMatrixC, int pDim1, int pDim2, int pDim3, int pRowStepElems2, int pRowStepElems3)
{
    float lAlpha = 1.0;
    float lBeta = 0.0;
    
    MKL_SPARSE_COOMM((const char*)"N", &pDim1, &pDim3, &pDim2, &lAlpha, (const char*)"GXXC", pMatrixA, pRowIndicesA, pColIndicesA, &pNnzA, pMatrixB, &pRowStepElems2, &lBeta, pMatrixC, &pRowStepElems3);
}

bool GetSplitData(size_t* pRowOffset, size_t* pRowCount, sparseMatrixMultiplyTaskConf* pTaskConf, pmSplitInfo& pSplitInfo)
{
    *pRowOffset = 0;
    *pRowCount = pTaskConf->blockDim;

    if(pSplitInfo.splitCount)
    {
        unsigned int lEffectiveSplitCount = std::min(pSplitInfo.splitCount, (unsigned int)pTaskConf->blockDim);
        
        if(pSplitInfo.splitId > lEffectiveSplitCount - 1)
            return false;

        size_t lRowsPerSplit = pTaskConf->blockDim / lEffectiveSplitCount;
        size_t lLeftoverRows = pTaskConf->blockDim - lRowsPerSplit * lEffectiveSplitCount;
        
        *pRowCount = lRowsPerSplit + ((lLeftoverRows > pSplitInfo.splitId) ? 1 : 0);
        *pRowOffset = pSplitInfo.splitId * lRowsPerSplit + std::min(lLeftoverRows, (size_t)pSplitInfo.splitId);
    }
    
    return true;
}

// Finds the data range (indices in the input matrix) that corresponds to a subtask's rows; pStartIndex and pEndIndex are inclusive
bool GetRowIndicesFromSplitData(size_t pFirstRow, size_t pRowOffset, size_t pRowCount, int* pDistributionData, INDICES_TYPE* pRowIndices1, int pStartIndex, int pEndIndex, pmSplitInfo& pSplitInfo, int& pUpdatedStartIndex)
{
    if(pSplitInfo.splitCount)
    {
        pUpdatedStartIndex = -1;

        INDICES_TYPE lStartRow = (INDICES_TYPE)(pFirstRow + pRowOffset);
        INDICES_TYPE lEndRow = (INDICES_TYPE)(lStartRow + pRowCount);

        if(pRowIndices1[pStartIndex] > lEndRow)
            return false;
        
        for(int i = pStartIndex; i<= pEndIndex; ++i)
        {
            if(pRowIndices1[i] >= lStartRow)
            {
                pUpdatedStartIndex = i;
                break;
            }
        }
        
        if(pUpdatedStartIndex == -1)
            return false;
    }
    else
    {
        pUpdatedStartIndex = 0;
    }
    
    return true;
}

pmStatus sparseMatrixMultiplyDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	sparseMatrixMultiplyTaskConf* lTaskConf = (sparseMatrixMultiplyTaskConf*)(pTaskInfo.taskConf);

    size_t lRowOffset, lRowCount;
    if(!GetSplitData(&lRowOffset, &lRowCount, lTaskConf, pSubtaskInfo.splitInfo))
        return pmSuccess;

    // Subtask no. increases vertically in output matrix (for increased locality)
    size_t lBlocksPerDim = (lTaskConf->matrixDim / lTaskConf->blockDim);
    size_t lBlockRow = (pSubtaskInfo.subtaskId % lBlocksPerDim);
    size_t lBlockCol = (pSubtaskInfo.subtaskId / lBlocksPerDim);

    int* lDistributionData = (int*)(((char*)lTaskConf) + sizeof(sparseMatrixMultiplyTaskConf));
    int lStartIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId];
    int lEndIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId + 1];

    if(lStartIndexForSubtask != -1 && lEndIndexForSubtask != -1)
    {
        // Subscribing to entire unsplitted range even if subtask is splitted (for the sparse matrix)
        // This is because the start and end of the set of rows chosen for a subtask can only be found from InputMemRowIndices1 which can not be read in this callback
        // Secondly, sparse matrix data is generally small. So, it won't be a big issue to subscribe to entire unsplitted data as RO. This might increasing sharing among splits.
        pmSubscriptionInfo lSubscriptionInfo1(lStartIndexForSubtask * sizeof(MATRIX_DATA_TYPE), (lEndIndexForSubtask - lStartIndexForSubtask + 1) * sizeof(MATRIX_DATA_TYPE));
        pmSubscriptionInfo lSubscriptionInfo2(lStartIndexForSubtask * sizeof(INDICES_TYPE), (lEndIndexForSubtask - lStartIndexForSubtask + 1) * sizeof(INDICES_TYPE));
        
        // lSubscriptionInfo3 has correct values for split subtasks
        pmSubscriptionInfo lSubscriptionInfo3((lRowOffset + lTaskConf->blockDim * lBlockRow) * sizeof(COUNT_TYPE), lRowCount * sizeof(COUNT_TYPE));
        
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo1);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_ROW_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo2);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_COL_INDICES1_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo2);

        // Only required for split subtasks
        if(pDeviceInfo.deviceType != pm::GPU_CUDA)
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_NNZ1_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo3);

        // Subscribe to entire lBlockCol of the second matrix
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MATRIX2_MEM_INDEX, READ_SUBSCRIPTION, pmScatteredSubscriptionInfo((lBlockCol * lTaskConf->blockDim) * sizeof(MATRIX_DATA_TYPE), lTaskConf->blockDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim * sizeof(MATRIX_DATA_TYPE), lTaskConf->matrixDim));

    #ifdef BUILD_CUDA
        // Reserve CUDA Global Mem
        if(pDeviceInfo.deviceType == pm::GPU_CUDA)
        {
            size_t lReservedMem = (lRowCount + 1) * sizeof(INDICES_TYPE);
            pmReserveCudaGlobalMem(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lReservedMem);
        }
    #endif
    }

    // Subscribe to one block of the output matrix (with equal split)
    SUBSCRIBE_BLOCK(lBlockRow, lBlockCol, lRowOffset, lRowCount, lTaskConf->blockDim, lTaskConf->matrixDim, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MATRIX_MEM_INDEX, WRITE_SUBSCRIPTION)

	return pmSuccess;
}

pmStatus sparseMatrixMultiply_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	sparseMatrixMultiplyTaskConf* lTaskConf = (sparseMatrixMultiplyTaskConf*)(pTaskInfo.taskConf);

    size_t lRowOffset, lRowCount;
    if(!GetSplitData(&lRowOffset, &lRowCount, lTaskConf, pSubtaskInfo.splitInfo))
        return pmSuccess;
    
    int* lDistributionData = (int*)(((char*)lTaskConf) + sizeof(sparseMatrixMultiplyTaskConf));

    bool lZeroOut = true;

    int lStartIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId];
    int lEndIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId + 1];

    if(lStartIndexForSubtask != -1 && lEndIndexForSubtask != -1)
    {
        INDICES_TYPE* lRowIndices1 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_ROW_INDICES1_MEM_INDEX].ptr);
        INDICES_TYPE* lColIndices1 = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_COL_INDICES1_MEM_INDEX].ptr);
        COUNT_TYPE* lNnz1 = (COUNT_TYPE*)(pSubtaskInfo.memInfo[INPUT_MEM_NNZ1_INDEX].ptr);

        size_t lBlocksPerDim = (lTaskConf->matrixDim / lTaskConf->blockDim);
        size_t lBlockRow = (pSubtaskInfo.subtaskId % lBlocksPerDim);

        // lStartIndexForSubtask is for unsplitted subtasks. The below call updates it for split subtasks.
        int lUpdatedStartIndex = -1;
        if(GetRowIndicesFromSplitData(lBlockRow * lTaskConf->blockDim, lRowOffset, lRowCount, lDistributionData, lRowIndices1, 0, (lEndIndexForSubtask - lStartIndexForSubtask), pSubtaskInfo.splitInfo, lUpdatedStartIndex))
        {
            lZeroOut = false;
            
            MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
            MATRIX_DATA_TYPE* lMatrix2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
            MATRIX_DATA_TYPE* lOutputMatrix = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

            int lSpanMatrix2 = (int)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? (int)lTaskConf->matrixDim : (int)lTaskConf->blockDim;
            int lSpanMatrix3 = (int)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? (int)lTaskConf->matrixDim : (int)lTaskConf->blockDim;
            
            lRowIndices1 += lUpdatedStartIndex;
            lColIndices1 += lUpdatedStartIndex;
            lMatrix1 += lUpdatedStartIndex;

            size_t lCountNnz1 = 0;
            for(size_t i = 0; i < lRowCount; ++i)
                lCountNnz1 += lNnz1[i];

            if(lCountNnz1 == 0)
            {
                std::cout << "Error in data computation for subtask " << pSubtaskInfo.subtaskId << " split " << pSubtaskInfo.splitInfo.splitId << " of " << pSubtaskInfo.splitInfo.splitCount << std::endl;
                exit(1);
            }

            std::vector<INDICES_TYPE> lTempRowIndices;
            lTempRowIndices.reserve(lCountNnz1);
            for(size_t i = 0; i < lCountNnz1; ++i)
                lTempRowIndices.emplace_back((lRowIndices1[i] % lTaskConf->blockDim) - lRowOffset);

            serialSparseMatrixMultiply(&lTempRowIndices[0], lColIndices1, (int)lCountNnz1, lMatrix1, lMatrix2, lOutputMatrix, (int)lRowCount, (int)lTaskConf->blockDim, (int)lTaskConf->blockDim, lSpanMatrix2, lSpanMatrix3);
        }
    }
    
    if(lZeroOut)
    {
        // Zero out the rows (for this split subtask) in the output matrix block
        MATRIX_DATA_TYPE* lOutputMatrix = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);
        
        int lSpanMatrix3 = (int)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? (int)lTaskConf->matrixDim : (int)lTaskConf->blockDim;

        for(size_t i = 0; i < lRowCount; ++i)
            memset(lOutputMatrix + i * lSpanMatrix3, 0, lTaskConf->blockDim * sizeof(MATRIX_DATA_TYPE));
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

    serialSparseMatrixMultiply(gSampleInputRowIndices, gSampleInputColIndices, (int)gCountNnz1, gSampleInputData1, gSampleInputData2, gSerialOutput, (int)lMatrixDim, (int)lMatrixDim, (int)lMatrixDim, (int)lMatrixDim, (int)lMatrixDim);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	singleGpuSparseMatrixMultiply(gSampleInputRowIndices, gSampleInputColIndices, gSampleInputData1, gSampleInputData2, gParallelOutput, gCountNnz1, lMatrixDim);

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

    size_t lSparseMatrixElems = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);
    size_t lSparseMatrixSizeInBytes = lSparseMatrixElems * sizeof(MATRIX_DATA_TYPE);
    size_t lSparseIndicesSizeInBytes = lSparseMatrixElems * sizeof(INDICES_TYPE);
    size_t lSparseNnzSizeInBytes = lMatrixDim * sizeof(COUNT_TYPE);

    size_t lDenseMatrixSizeInBytes = lMatrixElems * sizeof(MATRIX_DATA_TYPE);
    size_t lBlockSize = getBlockSize(lMatrixDim);

    // Input Mem 1, Input mem Row/Col Indices 1, Input Mem Nnz 1 contain the first input matrix (sparse)
    // Input Mem 2 contains second input matrix (dense)
	// Output Mem contains the result matrix
	// The number of subtasks is equal to the number of blocks in the output matrix
    unsigned long lSubtaskCount = (lMatrixDim / lBlockSize) * (lMatrixDim / lBlockSize);

	CREATE_TASK(lSubtaskCount, pCallbackHandle[0], pSchedulingPolicy)

    pmMemHandle lInputMem1, lInputMem2, lInputMemRowIndices1, lInputMemColIndices1, lInputMemNnz1, lOutputMem;
    CREATE_MEM(lSparseMatrixSizeInBytes, lInputMem1)
    CREATE_MEM(lSparseIndicesSizeInBytes, lInputMemRowIndices1)
    CREATE_MEM(lSparseIndicesSizeInBytes, lInputMemColIndices1)
    CREATE_MEM(lSparseNnzSizeInBytes, lInputMemNnz1)
    CREATE_MEM(lDenseMatrixSizeInBytes, lInputMem2)
    CREATE_MEM(lDenseMatrixSizeInBytes, lOutputMem)

    pmRawMemPtr lRawInputPtr1, lRawInputPtr2, lRawInputRowIndicesPtr1, lRawInputColIndicesPtr1, lRawInputMemNnzPtr1, lRawOutputPtr;
    pmGetRawMemPtr(lInputMem1, &lRawInputPtr1);
    pmGetRawMemPtr(lInputMem2, &lRawInputPtr2);
    pmGetRawMemPtr(lInputMemRowIndices1, &lRawInputRowIndicesPtr1);
    pmGetRawMemPtr(lInputMemColIndices1, &lRawInputColIndicesPtr1);
    pmGetRawMemPtr(lInputMemNnz1, &lRawInputMemNnzPtr1);
    pmGetRawMemPtr(lOutputMem, &lRawOutputPtr);

	memcpy(lRawInputPtr1, gSampleInputData1, lSparseMatrixSizeInBytes);
	memcpy(lRawInputPtr2, gSampleInputData2, lDenseMatrixSizeInBytes);
    memcpy(lRawInputRowIndicesPtr1, gSampleInputRowIndices, lSparseIndicesSizeInBytes);
    memcpy(lRawInputColIndicesPtr1, gSampleInputColIndices, lSparseIndicesSizeInBytes);
    memcpy(lRawInputMemNnzPtr1, gSampleInputNnz1, lSparseNnzSizeInBytes);
    
    DistributeMemory(lInputMem1, LINEAR_DIST, 0, (unsigned int)lSparseMatrixElems, 0, sizeof(MATRIX_DATA_TYPE), true);
    DistributeMemory(lInputMemRowIndices1, LINEAR_DIST, 0, (unsigned int)lSparseMatrixElems, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemColIndices1, LINEAR_DIST, 0, (unsigned int)lSparseMatrixElems, 0, sizeof(INDICES_TYPE), true);
    DistributeMemory(lInputMemNnz1, LINEAR_DIST, 0, (unsigned int)lMatrixDim, 0, sizeof(COUNT_TYPE), true);

    DistributeMemory(lInputMem2, BLOCK_DIST_1D_COL, BLOCK_DIM, (unsigned int)lMatrixDim, (unsigned int)lMatrixDim, sizeof(MATRIX_DATA_TYPE), true);

    pmTaskMem lTaskMem[MAX_MEM_INDICES];
    lTaskMem[INPUT_MATRIX1_MEM_INDEX] = {lInputMem1, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[INPUT_MATRIX2_MEM_INDEX] = {lInputMem2, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[INPUT_ROW_INDICES1_MEM_INDEX] = {lInputMemRowIndices1, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[INPUT_COL_INDICES1_MEM_INDEX] = {lInputMemColIndices1, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[INPUT_MEM_NNZ1_INDEX] = {lInputMemNnz1, READ_ONLY, SUBSCRIPTION_OPTIMAL};
    lTaskMem[OUTPUT_MATRIX_MEM_INDEX] = {lOutputMem, WRITE_ONLY, SUBSCRIPTION_OPTIMAL};

    lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
    lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    
     // 2 entries per subtask (one for start index and other for end index in all three input data arrays (both indices inclusive); -1 is sentinel)
    unsigned int lTaskConfLength = sizeof(sparseMatrixMultiplyTaskConf) + 2 * sizeof(unsigned int) * (unsigned int)lSubtaskCount;
    std::vector<char> lTaskConf(lTaskConfLength);

    sparseMatrixMultiplyTaskConf* lSparseTaskConf = (sparseMatrixMultiplyTaskConf*)(&lTaskConf[0]);
    int* lDistributionData = (int*)((char*)(&lTaskConf[0]) + sizeof(sparseMatrixMultiplyTaskConf));

    lSparseTaskConf->matrixDim = lMatrixDim;
    lSparseTaskConf->blockDim = lBlockSize;

	lTaskDetails.taskConf = (void*)(&lTaskConf[0]);
    lTaskDetails.taskConfLength = lTaskConfLength;

    lTaskDetails.canSplitCpuSubtasks = true;
    
    unsigned int lBlockRows = (unsigned int)(lMatrixDim / lBlockSize);

    double lStartTime = getCurrentTimeInSecs();
    
    unsigned int lCurrentIndex = 0;
    for(unsigned int i = 0; i < (unsigned int)lSubtaskCount; ++i)
    {
        // Subtask numbers increase vertically in the output matrix
        unsigned int lBlockRow = i % lBlockRows;
        if(lBlockRow == 0)
            lCurrentIndex = 0;

        INDICES_TYPE lLastRowForSubtask = (INDICES_TYPE)((lBlockRow + 1) * lBlockSize - 1);

        if(lCurrentIndex >= lSparseMatrixElems || gSampleInputRowIndices[lCurrentIndex] > lLastRowForSubtask)
        {
            lDistributionData[2 * i] = lDistributionData[2 * i + 1] = -1;
        }
        else
        {
            lDistributionData[2 * i] = lCurrentIndex;
            
            while(1)
            {
                if(lCurrentIndex == lSparseMatrixElems - 1)
                {
                    lDistributionData[2 * i + 1] = lCurrentIndex++;
                    break;
                }
                else
                {
                    ++lCurrentIndex;
                    if(gSampleInputRowIndices[lCurrentIndex] > lLastRowForSubtask)
                    {
                        lDistributionData[2 * i + 1] = lCurrentIndex - 1;
                        break;
                    }
                }
            }
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
        memcpy(gParallelOutput, lRawOutputPtr, lDenseMatrixSizeInBytes);
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

    size_t lBlockSize = getBlockSize(lMatrixDim);

    if(lMatrixDim % lBlockSize != 0)
    {
        std::cout << "Matrix dimension not divisible by " << lBlockSize << std::endl;
        exit(1);
    }
    
#ifdef BUILD_CUDA
    if(lMatrixDim < 32 || lBlockSize < 32)
    {
        std::cout << "Matrix Dim and Block Dim must be 32 or more !!! " << std::endl;
        exit(1);
    }
#endif

	srand((unsigned int)time(NULL));

	size_t lSparseMatrixElems = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);

    // Matrix 1 (Sparse)
    gSampleInputRowIndices = new INDICES_TYPE[lSparseMatrixElems];
    gSampleInputColIndices = new INDICES_TYPE[lSparseMatrixElems];
    gSampleInputData1 = new MATRIX_DATA_TYPE[lSparseMatrixElems];
    gSampleInputNnz1 = new COUNT_TYPE[lMatrixDim];   // Stores number of non-zero elements in every row of the first input matrix

    // Matrix 2 (Dense)
    gSampleInputData2 = new MATRIX_DATA_TYPE[lMatrixElems];

    // Output Matrix (Dense)
	gSerialOutput = new MATRIX_DATA_TYPE[lMatrixElems];
	gParallelOutput = new MATRIX_DATA_TYPE[lMatrixElems];
    
    memset(gSampleInputNnz1, 0, lMatrixDim * sizeof(COUNT_TYPE));

    std::set<std::pair<INDICES_TYPE, INDICES_TYPE>> lLocationsMatrix1;
	while(lLocationsMatrix1.size() != lSparseMatrixElems)
        lLocationsMatrix1.emplace(rand() % lMatrixDim, rand() % lMatrixDim);

    INDICES_TYPE index = 0;
    std::for_each(lLocationsMatrix1.begin(), lLocationsMatrix1.end(), [&] (const std::pair<INDICES_TYPE, INDICES_TYPE>& pPair)
    {
        gSampleInputRowIndices[index] = pPair.first;
        gSampleInputColIndices[index] = pPair.second;
        
        MATRIX_DATA_TYPE lVal = (MATRIX_DATA_TYPE)(int)rand();
        
        while(!lVal)
            lVal = (MATRIX_DATA_TYPE)(int)rand();

        ++gSampleInputNnz1[pPair.first];
        ++gCountNnz1;

        gSampleInputData1[index] = lVal;
        ++index;
    });
    
	for(size_t i = 0; i < lMatrixElems; ++i)
        gSampleInputData2[i] = (MATRIX_DATA_TYPE)i + 1;    //rand(); // (MATRIX_DATA_TYPE)i;

    return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
#ifdef BUILD_CUDA
    FreeCusparseHandles();
#endif

    delete[] gSampleInputRowIndices;
    delete[] gSampleInputColIndices;
    delete[] gSampleInputData1;
    delete[] gSampleInputNnz1;

    delete[] gSampleInputData2;

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
		if((fabs(gSerialOutput[i] - gParallelOutput[i]) / gSerialOutput[i]) > 1e-5)
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

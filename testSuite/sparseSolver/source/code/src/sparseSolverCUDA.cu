
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "sparseSolver.h"
#include "commonAPI.h"
#include "cusparse_v2.h"

#include <iostream>
#include <map>

namespace sparseSolver
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CUSPARSE_GEMM cusparseScsrgemm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CUSPARSE_GEMM cusparseDcsrgemm
#endif
    
#define CUSPARSE_ERROR_CHECK(name, x) \
{ \
    cusparseStatus_t dStatus = x; \
    if(dStatus != CUSPARSE_STATUS_SUCCESS) \
    { \
        std::cout << name << " failed with error " << dStatus << std::endl; \
        exit(1); \
    } \
}
    
class cusparseHandleWrapper
{
public:
    cusparseHandleWrapper()
    {
        CUSPARSE_ERROR_CHECK("cusparseCreate", cusparseCreate(&mHandle));
    }
    
    ~cusparseHandleWrapper()
    {
        CUSPARSE_ERROR_CHECK("cusparseDestroy", cusparseDestroy(mHandle));
    }
    
    cusparseHandle_t GetHandle()
    {
        return mHandle;
    }
    
private:
    cusparseHandle_t mHandle;
};

class cusparseHandleManager
{
public:
    static cusparseHandle_t GetCusparseHandle(pmDeviceHandle pDeviceHandle)
    {
        return mMap[pDeviceHandle].GetHandle();
    }

private:
    static std::map<pmDeviceHandle, cusparseHandleWrapper> mMap;
};
    
std::map<pmDeviceHandle, cusparseHandleWrapper> cusparseHandleManager::mMap;

pmStatus sparseMatrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cusparseHandle_t lCusparseHandle = cusparseHandleManager::GetCusparseHandle(pDeviceInfo.deviceHandle);

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
        
        int lCountNnz1 = 0;
        for(int i = 0; i < lRowCount; ++i)
            lCountNnz1 += lNnz1[i];

        int lSubtaskRowSpan = lEndIndexForSubtask - lStartIndexForSubtask + 1;
        
        MATRIX_DATA_TYPE* lSparseMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lSparseMatrix2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lOutputMatrix = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

        CUSPARSE_ERROR_CHECK("cusparseSetStream", cusparseSetStream(lCusparseHandle, (cudaStream_t)pCudaStream));

        INDICES_TYPE* lOutputMemRowIndicesCudaPtr = (INDICES_TYPE*)(pSubtaskInfo.gpuContext.reservedGlobalMem);  // Size: sizeof(INDICES_TYPE) * (lSubtaskRowSpan + 1)
        INDICES_TYPE* lCsrRowPtrA = lOutputMemRowIndicesCudaPtr + (lSubtaskRowSpan + 1);    // Size: sizeof(INDICES_TYPE) * (lSubtaskRowSpan + 1)
        INDICES_TYPE* lCsrRowPtrB = lCsrRowPtrA + (lSubtaskRowSpan + 1);    // Size: sizeof(INDICES_TYPE) * (lTaskConf->matrixDim + 1)

        CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lRowIndices1, lCountNnz1, lSubtaskRowSpan, lCsrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
        CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lRowIndices2, lTaskConf->nnz2, lTaskConf->matrixDim, lCsrRowPtrB, CUSPARSE_INDEX_BASE_ZERO));

        cusparseMatDescr_t lDescr = 0;

        CUSPARSE_ERROR_CHECK("cusparseCreateMatDescr", cusparseCreateMatDescr(&lDescr));
        CUSPARSE_ERROR_CHECK("cusparseSetMatType", cusparseSetMatType(lDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR_CHECK("cusparseSetMatIndexBase", cusparseSetMatIndexBase(lDescr, CUSPARSE_INDEX_BASE_ZERO));
        CUSPARSE_ERROR_CHECK("cusparseSetPointerMode", cusparseSetPointerMode(lCusparseHandle, CUSPARSE_POINTER_MODE_HOST));

        int lTotalNonZeroOutputElements = 0;
        int* lTotalNonZeroOutputElementsPtr = &lTotalNonZeroOutputElements;

        CUSPARSE_ERROR_CHECK("cusparseXcsrgemmNnz", cusparseXcsrgemmNnz(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, lSubtaskRowSpan, lTaskConf->matrixDim, lTaskConf->matrixDim, lDescr, lCountNnz1, lCsrRowPtrA, lColIndices1, lDescr, lTaskConf->nnz2, lCsrRowPtrB, lColIndices2, lDescr, lOutputMemRowIndicesCudaPtr, lTotalNonZeroOutputElementsPtr));
        
        if(lTotalNonZeroOutputElementsPtr != NULL)
        {
            lTotalNonZeroOutputElements = *lTotalNonZeroOutputElementsPtr;
        }
        else
        {
            int baseOutputMatrix = 0;
            CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(&lTotalNonZeroOutputElements, (INDICES_TYPE*)lOutputMemRowIndicesCudaPtr + lTaskConf->matrixDim, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)pCudaStream));
            CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(&baseOutputMatrix, lOutputMemRowIndicesCudaPtr, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)pCudaStream));

            lTotalNonZeroOutputElements -= baseOutputMatrix;
        }
        
        INDICES_TYPE* lOutputMemColIndicesCudaPtr = NULL;
        MATRIX_DATA_TYPE* lOutputMemCudaPtr = NULL;
        
        CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemColIndicesCudaPtr, sizeof(INDICES_TYPE) * lTotalNonZeroOutputElements));
        CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, sizeof(MATRIX_DATA_TYPE) * lTotalNonZeroOutputElements));

        CUSPARSE_ERROR_CHECK("cusparse_gemm", CUSPARSE_GEMM(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, lSubtaskRowSpan, lTaskConf->matrixDim, lTaskConf->matrixDim, lDescr, lCountNnz1, lSparseMatrix1, lCsrRowPtrA, lColIndices1, lDescr, lTaskConf->nnz2, lSparseMatrix2, lCsrRowPtrB, lColIndices2, lDescr, lOutputMemCudaPtr, lOutputMemRowIndicesCudaPtr, lOutputMemColIndicesCudaPtr));
    }

    return pmSuccess;
}
    
// Returns 0 on success; non-zero on failure
int singleGpuSparseMatrixMultiply(INDICES_TYPE* pSampleInputRowIndices, INDICES_TYPE* pSampleInputColIndices, MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, COUNT_TYPE pCountNnz1, COUNT_TYPE pCountNnz2, int pDim)
{
    cusparseHandleWrapper lWrapper;
    cusparseHandle_t lCusparseHandle = lWrapper.GetHandle();

    MATRIX_DATA_TYPE* lInputMemCudaPtr = NULL;
    INDICES_TYPE* lInputMemRowIndicesCudaPtr = NULL;
    INDICES_TYPE* lInputMemColIndicesCudaPtr = NULL;
    INDICES_TYPE* lOutputMemRowIndicesCudaPtr = NULL;
    int lTotalNonZeroOutputElements = 0;
    int* lTotalNonZeroOutputElementsPtr = &lTotalNonZeroOutputElements;

    size_t lMatrixElems = pDim * pDim;
    size_t lInputSize = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);
    size_t lInputSizeInBytes = lInputSize * sizeof(MATRIX_DATA_TYPE);

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr, lInputSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemRowIndicesCudaPtr, lInputSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemColIndicesCudaPtr, lInputSizeInBytes));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr, pInputMatrices, lInputSizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemRowIndicesCudaPtr, pSampleInputRowIndices, lInputSizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemColIndicesCudaPtr, pSampleInputColIndices, lInputSizeInBytes, cudaMemcpyHostToDevice));

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemRowIndicesCudaPtr, sizeof(INDICES_TYPE) * (pDim + 1)));
    
    MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)lInputMemCudaPtr;
    MATRIX_DATA_TYPE* lMatrix2 = lMatrix1 + lInputSize;
    INDICES_TYPE* lRowIndices1 = (INDICES_TYPE*)lInputMemRowIndicesCudaPtr;
    INDICES_TYPE* lRowIndices2 = lRowIndices1 + lInputSize;
    INDICES_TYPE* lColIndices1 = (INDICES_TYPE*)lInputMemColIndicesCudaPtr;
    INDICES_TYPE* lColIndices2 = lColIndices1 + lInputSize;

    size_t lRowPtrsPerMatrix = (pDim + 1);
    INDICES_TYPE* lCsrRowPtrA = NULL;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lCsrRowPtrA, 2 * lRowPtrsPerMatrix * sizeof(INDICES_TYPE)));
    
    INDICES_TYPE* lCsrRowPtrB = lCsrRowPtrA + lRowPtrsPerMatrix;

    CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lRowIndices1, lInputSize, pDim, lCsrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lRowIndices2, lInputSize, pDim, lCsrRowPtrB, CUSPARSE_INDEX_BASE_ZERO));

    cusparseMatDescr_t lDescr = 0;

    CUSPARSE_ERROR_CHECK("cusparseCreateMatDescr", cusparseCreateMatDescr(&lDescr));
    CUSPARSE_ERROR_CHECK("cusparseSetMatType", cusparseSetMatType(lDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_ERROR_CHECK("cusparseSetMatIndexBase", cusparseSetMatIndexBase(lDescr, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_ERROR_CHECK("cusparseSetPointerMode", cusparseSetPointerMode(lCusparseHandle, CUSPARSE_POINTER_MODE_HOST));

    CUSPARSE_ERROR_CHECK("cusparseXcsrgemmNnz", cusparseXcsrgemmNnz(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, pDim, pDim, pDim, lDescr, pCountNnz1, lCsrRowPtrA, lColIndices1, lDescr, pCountNnz2, lCsrRowPtrB, lColIndices2, lDescr, lOutputMemRowIndicesCudaPtr, lTotalNonZeroOutputElementsPtr));
    
    if(lTotalNonZeroOutputElementsPtr != NULL)
    {
        lTotalNonZeroOutputElements = *lTotalNonZeroOutputElementsPtr;
    }
    else
    {
        int baseOutputMatrix = 0;
        CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(&lTotalNonZeroOutputElements, (INDICES_TYPE*)lOutputMemRowIndicesCudaPtr + pDim, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(&baseOutputMatrix, lOutputMemRowIndicesCudaPtr, sizeof(int), cudaMemcpyDeviceToHost));

        lTotalNonZeroOutputElements -= baseOutputMatrix;
    }
    
    INDICES_TYPE* lOutputMemColIndicesCudaPtr = NULL;
    MATRIX_DATA_TYPE* lOutputMemCudaPtr = NULL;
    
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemColIndicesCudaPtr, sizeof(INDICES_TYPE) * lTotalNonZeroOutputElements));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, sizeof(MATRIX_DATA_TYPE) * lTotalNonZeroOutputElements));
    CUSPARSE_ERROR_CHECK("cusparse_gemm", CUSPARSE_GEMM(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, pDim, pDim, pDim, lDescr, pCountNnz1, lMatrix1, lCsrRowPtrA, lColIndices1, lDescr, pCountNnz2, lMatrix2, lCsrRowPtrB, lColIndices2, lDescr, lOutputMemCudaPtr, lOutputMemRowIndicesCudaPtr, lOutputMemColIndicesCudaPtr));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemColIndicesCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lCsrRowPtrA));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemRowIndicesCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemColIndicesCudaPtr));

    return 0;
}

}

#endif
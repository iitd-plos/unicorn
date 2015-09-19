
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "sparseSolver.h"
#include "commonAPI.h"
#include "cusparse_v2.h"

#include <iostream>
#include <map>
#include <vector>

namespace sparseSolver
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CUSPARSE_CSRMM2 cusparseScsrmm2
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CUSPARSE_CSRMM2 cusparseDcsrmm2
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

#define GPU_TILE_DIM 32
#define GPU_BLOCK_COLS 8

__global__ void squareMatrixTransposeInplace_cuda(size_t pDim, MATRIX_DATA_TYPE* pMatrix)
{
    // Code leveraged from https://devtalk.nvidia.com/default/topic/765696/efficient-in-place-transpose-of-multiple-square-float-matrices/

    __shared__ MATRIX_DATA_TYPE tile_s[GPU_TILE_DIM][GPU_TILE_DIM + 1];
    __shared__ MATRIX_DATA_TYPE tile_d[GPU_TILE_DIM][GPU_TILE_DIM + 1];

    int x = blockIdx.x * GPU_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * GPU_TILE_DIM + threadIdx.y;

    if(blockIdx.y >= blockIdx.x)
    {
        int dx = blockIdx.y * GPU_TILE_DIM + threadIdx.x;
        int dy = blockIdx.x * GPU_TILE_DIM + threadIdx.y;

        for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_COLS)
          tile_s[threadIdx.y + j][threadIdx.x] = pMatrix[(y + j) * pDim + x];

        if(blockIdx.y > blockIdx.x)
        {
            for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_COLS)
              tile_d[threadIdx.y + j][threadIdx.x] = pMatrix[(dy + j) * pDim + dx];
        }

        __syncthreads();

        for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_COLS)
          pMatrix[(dy + j) * pDim + dx] = tile_s[threadIdx.x][threadIdx.y + j];

        if(blockIdx.y > blockIdx.x)
        {
            for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_COLS)
              pMatrix[(y + j) * pDim + x] = tile_d[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void adjustRowIndices_cuda(INDICES_TYPE* pRowIndices, size_t pDim, size_t pElems)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    
    if(threadId >= pElems)
        return;

    pRowIndices[threadId] %= pDim;
}
    
void AdjustRowIndices(INDICES_TYPE* pRowIndices1CudaPtr, size_t pBlockDim, size_t pElems, cudaStream_t pCudaStream)
{
    int lMaxThreadsPerBlock = 512;
    size_t lBlocksReqd = (pElems + lMaxThreadsPerBlock - 1) / lMaxThreadsPerBlock;
    size_t lGridX = (size_t)(sqrt(lBlocksReqd));
    if(!lGridX)
        lGridX = 1;

    size_t lGridY = lGridX;

    if(lGridX * lGridY < lBlocksReqd)
        ++lGridX;

    if(lGridX * lGridY < lBlocksReqd)
        ++lGridY;

    dim3 gridConf(lGridX, lGridY, 1);
    dim3 blockConf(32, 16, 1);

    adjustRowIndices_cuda <<<gridConf, blockConf, 0, pCudaStream>>> (pRowIndices1CudaPtr, pBlockDim, pElems);
}
    
const MATRIX_DATA_TYPE gZero = (MATRIX_DATA_TYPE)0.0;
const MATRIX_DATA_TYPE gOne = (MATRIX_DATA_TYPE)1.0;

typedef std::map<pmDeviceHandle, cusparseHandle_t> cusparseHandleMapType;
cusparseHandleMapType& GetCusparseHandleMap()
{
    static cusparseHandleMapType gMap;
    
    return gMap;
}

cusparseHandle_t CreateCusparseHandle()
{
    cusparseHandle_t lCusparseHandle;
    
    CUSPARSE_ERROR_CHECK("cusparseCreate", cusparseCreate(&lCusparseHandle));
    
    return lCusparseHandle;
}

void DestroyCusparseHandle(cusparseHandle_t pCusparseHandle)
{
    CUSPARSE_ERROR_CHECK("cusparseDestroy", cusparseDestroy(pCusparseHandle));
}

cusparseHandle_t GetCusparseHandle(pmDeviceHandle pDeviceHandle)
{
    cusparseHandleMapType& lCusparseHandleMap = GetCusparseHandleMap();
    
    typename cusparseHandleMapType::iterator lIter = lCusparseHandleMap.find(pDeviceHandle);
    if(lIter == lCusparseHandleMap.end())
        lIter = lCusparseHandleMap.insert(std::make_pair(pDeviceHandle, CreateCusparseHandle())).first;
    
    return lIter->second;
}

void FreeCusparseHandles()
{
    cusparseHandleMapType& lCusparseHandleMap = GetCusparseHandleMap();
    typename cusparseHandleMapType::iterator lIter = lCusparseHandleMap.begin(), lEndIter = lCusparseHandleMap.end();
    
    for(; lIter != lEndIter; ++lIter)
        DestroyCusparseHandle(lIter->second);
    
    lCusparseHandleMap.clear();
}

class cusparseHandleWrapper
{
public:
    cusparseHandleWrapper()
    : mHandle(CreateCusparseHandle())
    {
    }
    
    ~cusparseHandleWrapper()
    {
        DestroyCusparseHandle(mHandle);
    }
    
    cusparseHandle_t GetHandle()
    {
        return mHandle;
    }
    
private:
    cusparseHandle_t mHandle;
};

pmStatus sparseMatrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
	sparseMatrixMultiplyTaskConf* lTaskConf = (sparseMatrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    
    if(pSubtaskInfo.splitInfo.splitCount != 0)
    {
        std::cout << "Subtask splitting not supported for CUDA subtasks !!!" << std::endl;
        exit(1);
    }

    cusparseHandle_t lCusparseHandle = GetCusparseHandle(pDeviceInfo.deviceHandle);
    
    int* lDistributionData = (int*)(((char*)lTaskConf) + sizeof(sparseMatrixMultiplyTaskConf));
    int lStartIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId];
    int lEndIndexForSubtask = lDistributionData[2 * pSubtaskInfo.subtaskId + 1];

    if(lStartIndexForSubtask != -1 && lEndIndexForSubtask != -1)
    {
        INDICES_TYPE* lRowIndices1CudaPtr = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_ROW_INDICES1_MEM_INDEX].ptr);
        INDICES_TYPE* lColIndices1CudaPtr = (INDICES_TYPE*)(pSubtaskInfo.memInfo[INPUT_COL_INDICES1_MEM_INDEX].ptr);
        COUNT_TYPE* lNnz1CudaPtr = (COUNT_TYPE*)(pSubtaskInfo.memInfo[INPUT_MEM_NNZ1_INDEX].ptr);

        MATRIX_DATA_TYPE* lMatrix1CudaPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lMatrix2CudaPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lOutputMatrixCudaPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

        if(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL || pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL)
        {
            std::cout << "Only compact subscription supported for CUDA subtasks !!!" << std::endl;
            exit(1);
        }
        
        int lCountNnz1 = (lEndIndexForSubtask - lStartIndexForSubtask + 1);
        if(lCountNnz1 == 0)
        {
            std::cout << "Error in data computation for subtask " << pSubtaskInfo.subtaskId << std::endl;
            exit(1);
        }

        AdjustRowIndices(lRowIndices1CudaPtr, lTaskConf->blockDim, lCountNnz1, (cudaStream_t)pCudaStream);

        INDICES_TYPE* lCsrRowPtrA = (INDICES_TYPE*)(pSubtaskInfo.gpuContext.reservedGlobalMem);  // Size: sizeof(INDICES_TYPE) * (lTaskConf->blockDim + 1)
        CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lRowIndices1CudaPtr, lCountNnz1, lTaskConf->blockDim, lCsrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));

        cusparseMatDescr_t lDescr = 0;
        CUSPARSE_ERROR_CHECK("cusparseCreateMatDescr", cusparseCreateMatDescr(&lDescr));
        CUSPARSE_ERROR_CHECK("cusparseSetMatType", cusparseSetMatType(lDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR_CHECK("cusparseSetMatIndexBase", cusparseSetMatIndexBase(lDescr, CUSPARSE_INDEX_BASE_ZERO));
        CUSPARSE_ERROR_CHECK("cusparseSetPointerMode", cusparseSetPointerMode(lCusparseHandle, CUSPARSE_POINTER_MODE_HOST));
        
        CUSPARSE_ERROR_CHECK("cusparse_csrmm2", CUSPARSE_CSRMM2(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, lTaskConf->blockDim, lTaskConf->blockDim, lTaskConf->matrixDim, lCountNnz1, &gOne, lDescr, lMatrix1CudaPtr, lCsrRowPtrA, lColIndices1CudaPtr, lMatrix2CudaPtr, lTaskConf->blockDim, &gZero, lOutputMatrixCudaPtr, lTaskConf->blockDim));

        dim3 gridConf(lTaskConf->blockDim / GPU_TILE_DIM, lTaskConf->blockDim / GPU_TILE_DIM, 1);
        dim3 blockConf(GPU_TILE_DIM, GPU_BLOCK_COLS, 1);

        squareMatrixTransposeInplace_cuda <<<gridConf, blockConf, 0, (cudaStream_t)pCudaStream>>> (lTaskConf->blockDim, lOutputMatrixCudaPtr);
    }
    else
    {
        // Zero out the rows (for this subtask) in the output matrix block
        MATRIX_DATA_TYPE* lOutputMatrixCudaPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

        CUDA_ERROR_CHECK("cudaMemsetAsync", cudaMemsetAsync(lOutputMatrixCudaPtr, 0, lTaskConf->blockDim * lTaskConf->blockDim * sizeof(MATRIX_DATA_TYPE), (cudaStream_t)pCudaStream));
    }

    return pmSuccess;
}

// Returns 0 on success; non-zero on failure
int singleGpuSparseMatrixMultiply(INDICES_TYPE* pSampleInputRowIndices, INDICES_TYPE* pSampleInputColIndices, MATRIX_DATA_TYPE* pSampleInputData1, MATRIX_DATA_TYPE* pSampleInputData2, MATRIX_DATA_TYPE* pOutputMatrix, COUNT_TYPE pCountNnz1, int pDim)
{
    cusparseHandleWrapper lWrapper;
    cusparseHandle_t lCusparseHandle = lWrapper.GetHandle();

    size_t lMatrixElems = pDim * pDim;
    size_t lMatrixSizeInBytes = lMatrixElems * sizeof(MATRIX_DATA_TYPE);

    size_t lSparseMatrixElems = NON_SPARSE_ELEMENT_COUNT(lMatrixElems);
    size_t lSparseMatrixSizeInBytes = lSparseMatrixElems * sizeof(MATRIX_DATA_TYPE);
    size_t lSparseIndicesSizeInBytes = lSparseMatrixElems * sizeof(INDICES_TYPE);

    MATRIX_DATA_TYPE* lInputMemCudaPtr1 = NULL;
    MATRIX_DATA_TYPE* lInputMemCudaPtr2 = NULL;
    INDICES_TYPE* lInputMemRowIndicesCudaPtr = NULL;
    INDICES_TYPE* lInputMemColIndicesCudaPtr = NULL;
    MATRIX_DATA_TYPE* lOutputMemCudaPtr = NULL;

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr1, lSparseMatrixSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemRowIndicesCudaPtr, lSparseIndicesSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemColIndicesCudaPtr, lSparseIndicesSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr2, lMatrixSizeInBytes));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lMatrixSizeInBytes));

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr1, pSampleInputData1, lSparseMatrixSizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemRowIndicesCudaPtr, pSampleInputRowIndices, lSparseIndicesSizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemColIndicesCudaPtr, pSampleInputColIndices, lSparseIndicesSizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr2, pSampleInputData2, lMatrixSizeInBytes, cudaMemcpyHostToDevice));

    INDICES_TYPE* lCsrRowPtrA = NULL;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lCsrRowPtrA, (pDim + 1) * sizeof(INDICES_TYPE)));
    CUSPARSE_ERROR_CHECK("cusparseXcoo2csr", cusparseXcoo2csr(lCusparseHandle, lInputMemRowIndicesCudaPtr, pCountNnz1, pDim, lCsrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));

    cusparseMatDescr_t lDescr = 0;
    CUSPARSE_ERROR_CHECK("cusparseCreateMatDescr", cusparseCreateMatDescr(&lDescr));
    CUSPARSE_ERROR_CHECK("cusparseSetMatType", cusparseSetMatType(lDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_ERROR_CHECK("cusparseSetMatIndexBase", cusparseSetMatIndexBase(lDescr, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_ERROR_CHECK("cusparseSetPointerMode", cusparseSetPointerMode(lCusparseHandle, CUSPARSE_POINTER_MODE_HOST));

    CUSPARSE_ERROR_CHECK("cusparse_csrmm2", CUSPARSE_CSRMM2(lCusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, pDim, pDim, pDim, pCountNnz1, &gOne, lDescr, lInputMemCudaPtr1, lCsrRowPtrA, lInputMemColIndicesCudaPtr, lInputMemCudaPtr2, pDim, &gZero, lOutputMemCudaPtr, pDim));

    dim3 gridConf(pDim / GPU_TILE_DIM, pDim / GPU_TILE_DIM, 1);
    dim3 blockConf(GPU_TILE_DIM, GPU_BLOCK_COLS, 1);

    squareMatrixTransposeInplace_cuda <<<gridConf, blockConf>>> (pDim, lOutputMemCudaPtr);

    CUDA_ERROR_CHECK("cudaDeviceSynchronize", cudaDeviceSynchronize());

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMatrix, lOutputMemCudaPtr, lMatrixSizeInBytes, cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lCsrRowPtrA));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr2));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemColIndicesCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemRowIndicesCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr1));

    return 0;
}

}

#endif

#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "matrixMultiply.h"

#include <iostream>

namespace matrixMultiply
{

__global__ void matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
    // Each thread computes one element of resulting matrix by accumulating results into value

    matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    unsigned int lDimension = lTaskConf->matrixDim;
    unsigned int lMatrixSize = lDimension * lDimension;


    MATRIX_DATA_TYPE value = (MATRIX_DATA_TYPE)0;
    int rowId = pSubtaskInfo.subtaskId;
    int colId = ((blockIdx.x * gridDim.y + blockIdx.y)*(blockDim.x * blockDim.y)) + (threadIdx.x * blockDim.y) + threadIdx.y;

    if(colId >= lDimension)
        return;

    MATRIX_DATA_TYPE* inputMem1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.inputMem);
    MATRIX_DATA_TYPE* inputMem2 = inputMem1 + lMatrixSize;
    MATRIX_DATA_TYPE* outputMem = (MATRIX_DATA_TYPE*)(pSubtaskInfo.outputMem);

    for(int e = 0; e < lDimension; ++e)
        value += inputMem1[rowId * lDimension + e] * inputMem2[e * lDimension + colId];

    outputMem[colId] = value;

    *pStatus = pmSuccess;
}
    
__global__ void matrixMultiply_singleGpu(MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, int pDim)
{
    unsigned int lMatrixSize = pDim * pDim;

    MATRIX_DATA_TYPE value = (MATRIX_DATA_TYPE)0;
    int rowId = blockIdx.x * blockDim.x + threadIdx.x;
    int colId = blockIdx.y * blockDim.y + threadIdx.y;

    if(rowId >= pDim || colId >= pDim)
        return;

    MATRIX_DATA_TYPE* inputMem1 = (MATRIX_DATA_TYPE*)(pInputMatrices);
    MATRIX_DATA_TYPE* inputMem2 = inputMem1 + lMatrixSize;

    for(int e = 0; e < pDim; ++e)
        value += inputMem1[rowId * pDim + e] * inputMem2[e * pDim + colId];

    pOutputMatrix[rowId * pDim + colId] = value;
}
   
// Returns 0 on success; non-zero on failure
int singleGpuMatrixMultiply(MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, int pDim)
{
    void* lInputMemCudaPtr = NULL;
    void* lOutputMemCudaPtr = NULL;

    size_t lOutputSize = sizeof(MATRIX_DATA_TYPE) * pDim * pDim;
    size_t lInputSize = 2 * lOutputSize;
    if(cudaMalloc((void**)&lInputMemCudaPtr, lInputSize) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Input Memory Allocation Failed" << std::endl;
        return 1;
    }

    if(cudaMemcpy(lInputMemCudaPtr, pInputMatrices, lInputSize, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    if(cudaMalloc((void**)&lOutputMemCudaPtr, lOutputSize) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Output Memory Allocation Failed" << std::endl;
        return 1;
    }

    size_t lMaxThreadsPerDim = 32;
    size_t lBlocksPerDim = (pDim/lMaxThreadsPerDim) + ((pDim%lMaxThreadsPerDim) ? 1 : 0);

    dim3 gridConf(lBlocksPerDim, lBlocksPerDim, 1);
    dim3 blockConf(lMaxThreadsPerDim, lMaxThreadsPerDim, 1);
    matrixMultiply_singleGpu<<<gridConf, blockConf>>>((MATRIX_DATA_TYPE*)lInputMemCudaPtr, (MATRIX_DATA_TYPE*)lOutputMemCudaPtr, pDim);
    
    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Device Synchronize Failed" << std::endl;
        return 1;
    }

    if(cudaMemcpy(pOutputMatrix, lOutputMemCudaPtr, lOutputSize, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lOutputMemCudaPtr) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Output Memory Deallocation Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lInputMemCudaPtr) != cudaSuccess)
    {
        std::cout << "Matrix Multiply: CUDA Input Memory Deallocation Failed" << std::endl;
        return 1;
    }

    return 0;
}

matrixMultiply_cudaFuncPtr matrixMultiply_cudaFunc = matrixMultiply_cuda;

}

#endif
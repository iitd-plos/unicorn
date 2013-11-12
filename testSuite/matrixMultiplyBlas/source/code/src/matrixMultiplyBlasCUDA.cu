
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "matrixMultiplyBlas.h"
#include "commonAPI.h"

#include <iostream>

namespace matrixMultiplyBlas
{

#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CUBLAS_GEMM cublasSgemm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CUBLAS_GEMM cublasDgemm
#endif

const MATRIX_DATA_TYPE gZero = (MATRIX_DATA_TYPE)0.0;
const MATRIX_DATA_TYPE gOne = (MATRIX_DATA_TYPE)1.0;

pmStatus matrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cublasHandle_t lCublasHandle = GetCublasHandle(pDeviceInfo.deviceHandle);
	matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);

    size_t lBlockOffset, lBlockHeight;
    if(!GetSplitData(&lBlockOffset, &lBlockHeight, lTaskConf, pSubtaskInfo.splitInfo))
        return pmSuccess;

    MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lMatrix2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lMatrix3 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));

    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    size_t lSpanMatrix2 = (pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;
    size_t lSpanMatrix3 = (pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : lTaskConf->blockDim;

    CUBLAS_ERROR_CHECK("cublas_gemm", CUBLAS_GEMM(lCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, (int)lBlockHeight, (int)lTaskConf->blockDim, (int)lTaskConf->matrixDim, &gOne, lMatrix2, (int)lSpanMatrix2, lMatrix1, (int)lTaskConf->matrixDim, &gZero, lMatrix3, (int)lSpanMatrix3));

    return pmSuccess;
}
    
// Returns 0 on success; non-zero on failure
int singleGpuMatrixMultiply(MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, int pDim)
{
    cublasHandleManager lManager;
    cublasHandle_t lCublasHandle = lManager.GetHandle();

    void* lInputMemCudaPtr = NULL;
    void* lOutputMemCudaPtr = NULL;

    size_t lOutputSize = sizeof(MATRIX_DATA_TYPE) * pDim * pDim;
    size_t lInputSize = 2 * lOutputSize;

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr, lInputSize));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr, pInputMatrices, lInputSize, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lOutputSize));
    
    MATRIX_DATA_TYPE* lMatrix1 = (MATRIX_DATA_TYPE*)lInputMemCudaPtr;
    MATRIX_DATA_TYPE* lMatrix2 = lMatrix1 + pDim * pDim;
    MATRIX_DATA_TYPE* lMatrix3 = (MATRIX_DATA_TYPE*)lOutputMemCudaPtr;

    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    CUBLAS_ERROR_CHECK("cublas_gemm", CUBLAS_GEMM(lCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, pDim, pDim, pDim, &gOne, lMatrix2, pDim, lMatrix1, pDim, &gZero, lMatrix3, pDim));
    
    CUDA_ERROR_CHECK("cudaDeviceSynchronize", cudaDeviceSynchronize());

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMatrix, lOutputMemCudaPtr, lOutputSize, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));

    return 0;
}

}

#endif

#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "commonAPI.h"
#include "luDecomposition.h"

#include <iostream>

namespace luDecomposition
{
    
#if defined(MATRIX_DATA_TYPE_FLOAT)
#define CUBLAS_SCAL cublasSscal
#define CUBLAS_GER cublasSger
#define CUBLAS_TRSM cublasStrsm
#define CUBLAS_GEMM cublasSgemm
#elif defined(MATRIX_DATA_TYPE_DOUBLE)
#define CUBLAS_SCAL cublasDscal
#define CUBLAS_GER cublasDger
#define CUBLAS_TRSM cublasDtrsm
#define CUBLAS_GEMM cublasDgemm
#endif

const MATRIX_DATA_TYPE gOne = (MATRIX_DATA_TYPE)1.0;
const MATRIX_DATA_TYPE gMinusOne = (MATRIX_DATA_TYPE)-1.0;
    
__global__ void findDiagonalElemReciprocal(MATRIX_DATA_TYPE* pDiagonalElem, MATRIX_DATA_TYPE* pMatrix, size_t pDim, size_t pIndex)
{
    if(threadIdx.x == 0)
        *pDiagonalElem = (MATRIX_DATA_TYPE)1.0/pMatrix[pIndex + pIndex * pDim];
}
    
pmStatus luDecomposition_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cublasHandle_t lCublasHandle = GetCublasHandle(pDeviceInfo.deviceHandle);
	luTaskConf* lTaskConf = (luTaskConf*)(pTaskInfo.taskConf);
    MATRIX_DATA_TYPE* lMatrix = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
    
    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));

    MATRIX_DATA_TYPE* lDiagonalElemPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.gpuContext.reservedGlobalMem);
    
    for(size_t i = 0; i < BLOCK_DIM - 1; ++i)
    {
        findDiagonalElemReciprocal<<<1, 1, 0, (cudaStream_t)pCudaStream>>>(lDiagonalElemPtr, lMatrix, lTaskConf->matrixDim, i);
        
        cudaError_t lCudaError = cudaGetLastError();
        if(lCudaError != cudaSuccess)
        {
            std::cout << "findDiagonalElemReciprocal Failed " << cudaGetErrorString(lCudaError) << std::endl;
            exit(1);
        }
        
        CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_ERROR_CHECK("cublas_scal", CUBLAS_SCAL(lCublasHandle, (int)(BLOCK_DIM - i - 1), lDiagonalElemPtr, lMatrix + i + (i + 1) * lTaskConf->matrixDim, (int)lTaskConf->matrixDim));

        CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_ERROR_CHECK("cublas_ger", CUBLAS_GER(lCublasHandle, (int)(BLOCK_DIM - i - 1), (int)(BLOCK_DIM - i - 1), &gMinusOne, lMatrix + (i + 1) + i * lTaskConf->matrixDim, 1, lMatrix + i + (i + 1) * lTaskConf->matrixDim, (int)lTaskConf->matrixDim, lMatrix + (i + 1) + (i + 1) * lTaskConf->matrixDim, (int)lTaskConf->matrixDim));
    }
    
    return pmSuccess;
}

pmStatus horizVertComp_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cublasHandle_t lCublasHandle = GetCublasHandle(pDeviceInfo.deviceHandle);
	luTaskConf* lTaskConf = (luTaskConf*)(pTaskInfo.taskConf);

    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));
    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    bool lUpperTriangularComputation = (pSubtaskInfo.subtaskId < (pTaskInfo.subtaskCount/2));
    if(lUpperTriangularComputation)   // Upper Triangular Matrix (Solve A10 = L00 * U01)
    {
        size_t lOffsetElems = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId + 1 + pSubtaskInfo.subtaskId, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId, lTaskConf->matrixDim);
        
        MATRIX_DATA_TYPE* lL00 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lU01 = lL00 + lOffsetElems;
        
        CUBLAS_ERROR_CHECK("cublas_trsm", CUBLAS_TRSM(lCublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, BLOCK_DIM, BLOCK_DIM, &gOne, lL00, (int)lTaskConf->matrixDim, lU01, (int)lTaskConf->matrixDim));

    }
    else    // Lower Triangular Matrix (Solve A01 = L10 * U00)
    {
        size_t lStartingSubtask = (pTaskInfo.subtaskCount/2);
        
        size_t lOffsetElems = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + pSubtaskInfo.subtaskId - lStartingSubtask, pTaskInfo.taskId, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId, lTaskConf->matrixDim);
        
        MATRIX_DATA_TYPE* lU00 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lL10 = lU00 + lOffsetElems;
        
        CUBLAS_ERROR_CHECK("cublas_trsm", CUBLAS_TRSM(lCublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, BLOCK_DIM, BLOCK_DIM, &gOne, lU00, (int)lTaskConf->matrixDim, lL10, (int)lTaskConf->matrixDim));
    }
    
	return pmSuccess;
}

pmStatus diagComp_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cublasHandle_t lCublasHandle = GetCublasHandle(pDeviceInfo.deviceHandle);
	luTaskConf* lTaskConf = (luTaskConf*)(pTaskInfo.taskConf);

    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));
    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    size_t lDim = sqrtl(pTaskInfo.subtaskCount);
    size_t lRow = (pSubtaskInfo.subtaskId / lDim);
    size_t lCol = (pSubtaskInfo.subtaskId % lDim);
    
    size_t lOffsetElems1 = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId + 1 + lCol, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId, lTaskConf->matrixDim);
    
    size_t lOffsetElems2 = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId + 1 + lCol, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId, lTaskConf->matrixDim);
    
    MATRIX_DATA_TYPE* lL10 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lU01 = lL10 + lOffsetElems1;
    MATRIX_DATA_TYPE* lA11 = lL10 + lOffsetElems2;
    
    // Solve A11 = A11 - L10 * U01
    CUBLAS_ERROR_CHECK("cublas_gemm", CUBLAS_GEMM(lCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, &gMinusOne, lL10, (int)lTaskConf->matrixDim, lU01, (int)lTaskConf->matrixDim, &gOne, lA11, (int)lTaskConf->matrixDim));
    
	return pmSuccess;
}
    
int singleGpuLUDecomposition(MATRIX_DATA_TYPE* pMatrix, size_t pDim)
{
    cublasHandleManager lManager;
    cublasHandle_t lCublasHandle = lManager.GetHandle();

    void* lDevicePtr = NULL;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)(&lDevicePtr), (int)(pDim * pDim * sizeof(MATRIX_DATA_TYPE))));

    MATRIX_DATA_TYPE* lMatrix = (MATRIX_DATA_TYPE*)lDevicePtr;
    
    CUBLAS_ERROR_CHECK("cublasSetMatrix", cublasSetMatrix(pDim, pDim, sizeof(MATRIX_DATA_TYPE), pMatrix, pDim, lDevicePtr, pDim));
    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    MATRIX_DATA_TYPE lDiagonalElem = 1.0;

    for(size_t i = 0; i < pDim - 1; ++i)
    {
        CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(&lDiagonalElem, lMatrix + i + i * pDim, sizeof(MATRIX_DATA_TYPE), cudaMemcpyDeviceToHost));

        MATRIX_DATA_TYPE lReciprocal = 1.0/lDiagonalElem;
	
        CUBLAS_ERROR_CHECK("cublas_scal", CUBLAS_SCAL(lCublasHandle, (int)(pDim - i - 1), &lReciprocal, lMatrix + i + (i + 1) * pDim, (int)pDim));
        CUBLAS_ERROR_CHECK("cublas_ger", CUBLAS_GER(lCublasHandle, (int)(pDim - i - 1), (int)(pDim - i - 1), &gMinusOne, lMatrix + (i + 1) + i * pDim, 1, lMatrix + i + (i + 1) * pDim, (int)pDim, lMatrix + (i + 1) + (i + 1) * pDim, (int)pDim));
    }

    CUBLAS_ERROR_CHECK("cublasGetMatrix", cublasGetMatrix(pDim, pDim, sizeof(MATRIX_DATA_TYPE), lDevicePtr, pDim, pMatrix, pDim));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lDevicePtr));

   return 0;
}
    
}

#endif


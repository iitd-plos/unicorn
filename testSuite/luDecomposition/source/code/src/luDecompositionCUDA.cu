
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
    
    size_t lColStepElems = ((pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->matrixDim : BLOCK_DIM);

    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));

    MATRIX_DATA_TYPE* lDiagonalElemPtr = (MATRIX_DATA_TYPE*)(pSubtaskInfo.gpuContext.reservedGlobalMem);
    
    for(size_t i = 0; i < BLOCK_DIM - 1; ++i)
    {
        findDiagonalElemReciprocal<<<1, 1, 0, (cudaStream_t)pCudaStream>>>(lDiagonalElemPtr, lMatrix, lColStepElems, i);
        
        cudaError_t lCudaError = cudaGetLastError();
        if(lCudaError != cudaSuccess)
        {
            std::cout << "findDiagonalElemReciprocal Failed " << cudaGetErrorString(lCudaError) << std::endl;
            exit(1);
        }
        
        CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_DEVICE));
        CUBLAS_ERROR_CHECK("cublas_scal", CUBLAS_SCAL(lCublasHandle, (int)(BLOCK_DIM - i - 1), lDiagonalElemPtr, lMatrix + i + (i + 1) * lColStepElems, (int)lColStepElems));

        CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_ERROR_CHECK("cublas_ger", CUBLAS_GER(lCublasHandle, (int)(BLOCK_DIM - i - 1), (int)(BLOCK_DIM - i - 1), &gMinusOne, lMatrix + (i + 1) + i * lColStepElems, 1, lMatrix + i + (i + 1) * lColStepElems, (int)lColStepElems, lMatrix + (i + 1) + (i + 1) * lColStepElems, (int)lColStepElems));
    }
    
    return pmSuccess;
}

pmStatus horizVertComp_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    cublasHandle_t lCublasHandle = GetCublasHandle(pDeviceInfo.deviceHandle);
	luTaskConf* lTaskConf = (luTaskConf*)(pTaskInfo.taskConf);

    CUBLAS_ERROR_CHECK("cublasSetStream", cublasSetStream(lCublasHandle, (cudaStream_t)pCudaStream));
    CUBLAS_ERROR_CHECK("cublasSetPointerMode", cublasSetPointerMode(lCublasHandle, CUBLAS_POINTER_MODE_HOST));

    size_t lOffsetElems = 0;
    int lSpanElems = 0;

    bool lUpperTriangularComputation = (pSubtaskInfo.subtaskId < (pTaskInfo.subtaskCount/2));
    if(lUpperTriangularComputation)   // Upper Triangular Matrix (Solve A10 = L00 * U01)
    {
        if(pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL)
        {
            lOffsetElems = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId + 1 + pSubtaskInfo.subtaskId, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId, lTaskConf->matrixDim);
            lSpanElems = (int)lTaskConf->matrixDim;
        }
        else
        {
            lOffsetElems = BLOCK_DIM * BLOCK_DIM;
            lSpanElems = (int)(BLOCK_DIM);
        }
        
        MATRIX_DATA_TYPE* lL00 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lU01 = lL00 + lOffsetElems;
        
        CUBLAS_ERROR_CHECK("cublas_trsm", CUBLAS_TRSM(lCublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, BLOCK_DIM, BLOCK_DIM, &gOne, lL00, lSpanElems, lU01, lSpanElems));
    }
    else    // Lower Triangular Matrix (Solve A01 = L10 * U00)
    {
        size_t lStartingSubtask = (pTaskInfo.subtaskCount/2);
        
        if(pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL)
        {
            lOffsetElems = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + pSubtaskInfo.subtaskId - lStartingSubtask, pTaskInfo.taskId, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId, lTaskConf->matrixDim);
            lSpanElems = (int)lTaskConf->matrixDim;
        }
        else
        {
            lOffsetElems = BLOCK_DIM;
            lSpanElems = (int)(2 * BLOCK_DIM);
        }
        
        MATRIX_DATA_TYPE* lU00 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
        MATRIX_DATA_TYPE* lL10 = lU00 + lOffsetElems;
        
        CUBLAS_ERROR_CHECK("cublas_trsm", CUBLAS_TRSM(lCublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, BLOCK_DIM, BLOCK_DIM, &gOne, lU00, lSpanElems, lL10, lSpanElems));
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
    
    size_t lOffsetElems1 = 0, lOffsetElems2 = 0;
    int lSpanElems1 = 0, lSpanElems2 = 0, lSpanElems3 = 0;

    if(pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL)
    {
        lOffsetElems1 = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId, pTaskInfo.taskId + 1 + lCol, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId, lTaskConf->matrixDim);
        
        lOffsetElems2 = BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId + 1 + lCol, lTaskConf->matrixDim) - BLOCK_OFFSET_IN_ELEMS(pTaskInfo.taskId + 1 + lRow, pTaskInfo.taskId, lTaskConf->matrixDim);

        lSpanElems1 = lSpanElems2 = lSpanElems3 = (int)lTaskConf->matrixDim;
    }
    else
    {
        lOffsetElems1 = BLOCK_DIM * BLOCK_DIM;
        lOffsetElems2 = BLOCK_DIM * BLOCK_DIM + BLOCK_DIM;
        
        lSpanElems1 = (int)(BLOCK_DIM);
        lSpanElems2 = lSpanElems3 = (int)(2 * BLOCK_DIM);
    }
    
    MATRIX_DATA_TYPE* lL10 = ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* lU01 = lL10 + lOffsetElems1;
    MATRIX_DATA_TYPE* lA11 = lL10 + lOffsetElems2;
    
    // Solve A11 = A11 - L10 * U01
    CUBLAS_ERROR_CHECK("cublas_gemm", CUBLAS_GEMM(lCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM, &gMinusOne, lL10, lSpanElems1, lU01, lSpanElems2, &gOne, lA11, lSpanElems3));
    
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


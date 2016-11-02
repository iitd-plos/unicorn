
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
#include "matrixMultiply.h"
#include "commonAPI.h"

#include <iostream>

namespace matrixMultiply
{

__global__ void matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
    // Each thread computes one element of resulting matrix by accumulating results into value

    matrixMultiplyTaskConf* lTaskConf = (matrixMultiplyTaskConf*)(pTaskInfo.taskConf);
    unsigned int lDimension = lTaskConf->matrixDim;

    MATRIX_DATA_TYPE value = (MATRIX_DATA_TYPE)0;
    int colId = ((blockIdx.x * gridDim.y + blockIdx.y)*(blockDim.x * blockDim.y)) + (threadIdx.x * blockDim.y) + threadIdx.y;

    if(colId >= lDimension)
        return;

    MATRIX_DATA_TYPE* inputMem1 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX1_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* inputMem2 = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[INPUT_MATRIX2_MEM_INDEX].ptr);
    MATRIX_DATA_TYPE* outputMem = (MATRIX_DATA_TYPE*)(pSubtaskInfo.memInfo[OUTPUT_MATRIX_MEM_INDEX].ptr);

    for(int e = 0; e < lDimension; ++e)
        value += inputMem1[e] * inputMem2[e * lDimension + colId];

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

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr, lInputSize));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr, pInputMatrices, lInputSize, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lOutputSize));

    size_t lMaxThreadsPerDim = 32;
    size_t lBlocksPerDim = (pDim/lMaxThreadsPerDim) + ((pDim%lMaxThreadsPerDim) ? 1 : 0);

    dim3 gridConf(lBlocksPerDim, lBlocksPerDim, 1);
    dim3 blockConf(lMaxThreadsPerDim, lMaxThreadsPerDim, 1);
    matrixMultiply_singleGpu<<<gridConf, blockConf>>>((MATRIX_DATA_TYPE*)lInputMemCudaPtr, (MATRIX_DATA_TYPE*)lOutputMemCudaPtr, pDim);
    
    CUDA_ERROR_CHECK("cudaDeviceSynchronize", cudaDeviceSynchronize());

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMatrix, lOutputMemCudaPtr, lOutputSize, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));

    return 0;
}

matrixMultiply_cudaFuncPtr matrixMultiply_cudaFunc = matrixMultiply_cuda;

}

#endif

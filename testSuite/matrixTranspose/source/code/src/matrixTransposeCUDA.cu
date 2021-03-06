
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

#include "pmPublicDefinitions.h"
#include "commonAPI.h"
#include "matrixTranspose.h"

#include <iostream>

#ifdef BUILD_CUDA
#ifdef USE_SQUARE_BLOCKS

namespace matrixTranspose
{

__global__ void matrixTranspose_cuda(size_t pInputMemCols, size_t pSubtaskRows, void* pInputMem, void* pOutputBlock)
{
    __shared__ MATRIX_DATA_TYPE lTile[GPU_TILE_DIM][GPU_TILE_DIM + 1];

    int lBlockId_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int lBlockId_y = blockIdx.x;
    
    int lIndexX = lBlockId_x * GPU_TILE_DIM + threadIdx.x;
    int lIndexY = lBlockId_y * GPU_TILE_DIM + threadIdx.y;
    int lInputIndex = lIndexX + (lIndexY * pInputMemCols);

    lIndexX = lBlockId_y * GPU_TILE_DIM + threadIdx.x;
    lIndexY = lBlockId_x * GPU_TILE_DIM + threadIdx.y;
    int lOutputIndex = lIndexX + (lIndexY * pSubtaskRows);

    int i, lStride = (GPU_TILE_DIM/GPU_ELEMS_PER_THREAD);
    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        lTile[threadIdx.y + i][threadIdx.x] = ((MATRIX_DATA_TYPE*)pInputMem)[lInputIndex + i * pInputMemCols];

    __syncthreads();

    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        ((MATRIX_DATA_TYPE*)pOutputBlock)[lOutputIndex + i * pSubtaskRows] = lTile[threadIdx.x][threadIdx.y + i];
}

__global__ void matrixCopy_cuda(matrixTransposeTaskConf pTaskConf, pmSubtaskInfo pSubtaskInfo, void* pOutputBlock)
{
    int lIndexX = blockIdx.x * GPU_TILE_DIM + threadIdx.x;
    int lIndexY = blockIdx.y * GPU_TILE_DIM + threadIdx.y;
    int lIndex = lIndexX + (lIndexY * pTaskConf.blockSizeRows);
    
    unsigned int lOutputMemIndex = (pTaskConf.inplace ? (unsigned int)INPLACE_MEM_INDEX : (unsigned int)OUTPUT_MEM_INDEX);

    int i, lStride = (GPU_TILE_DIM/GPU_ELEMS_PER_THREAD);
    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        ((MATRIX_DATA_TYPE*)pSubtaskInfo.memInfo[lOutputMemIndex].writePtr)[lIndex + i * pTaskConf.blockSizeRows] = ((MATRIX_DATA_TYPE*)pOutputBlock)[lIndex + i * pTaskConf.blockSizeRows];
}
    
__global__ void matrixTranspose_singleGpu(size_t pInputMemCols, size_t pSubtaskRows, void* pInputMem, void* pOutputBlock, size_t pMaxBlocksX, size_t pMaxBlocksY)
{
    __shared__ MATRIX_DATA_TYPE lTile[GPU_TILE_DIM][GPU_TILE_DIM + 1];

    int lBlockId_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int lBlockId_y = blockIdx.x;
    
    if(lBlockId_x >= pMaxBlocksX || lBlockId_y >= pMaxBlocksY)
        return;
    
    int lIndexX = lBlockId_x * GPU_TILE_DIM + threadIdx.x;
    int lIndexY = lBlockId_y * GPU_TILE_DIM + threadIdx.y;
    int lInputIndex = lIndexX + (lIndexY * pInputMemCols);

    lIndexX = lBlockId_y * GPU_TILE_DIM + threadIdx.x;
    lIndexY = lBlockId_x * GPU_TILE_DIM + threadIdx.y;
    int lOutputIndex = lIndexX + (lIndexY * pSubtaskRows);

    int i, lStride = (GPU_TILE_DIM/GPU_ELEMS_PER_THREAD);
    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        lTile[threadIdx.y + i][threadIdx.x] = ((MATRIX_DATA_TYPE*)pInputMem)[lInputIndex + i * pInputMemCols];

    __syncthreads();

    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        ((MATRIX_DATA_TYPE*)pOutputBlock)[lOutputIndex + i * pSubtaskRows] = lTile[threadIdx.x][threadIdx.y + i];
}

pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);

    pmSubscriptionVisibilityType lInputMemVisibilityType = lTaskConf->inplace ? pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].visibilityType : pSubtaskInfo.memInfo[INPUT_MEM_INDEX].visibilityType;
    pmSubscriptionVisibilityType lOutputMemVisibilityType = lTaskConf->inplace ? pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].visibilityType : pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType;

    if(lInputMemVisibilityType != SUBSCRIPTION_COMPACT || lOutputMemVisibilityType != SUBSCRIPTION_COMPACT)
        exit(1);

    dim3 gridConf(lTaskConf->blockSizeRows / GPU_TILE_DIM, lTaskConf->blockSizeRows / GPU_TILE_DIM, 1);
    dim3 blockConf(GPU_TILE_DIM, GPU_TILE_DIM / GPU_ELEMS_PER_THREAD, 1);

    cudaStream_t lCudaStream = (cudaStream_t)pCudaStream;
    
    if(lTaskConf->inplace)
    {
        void* lBlockCudaPtr = pSubtaskInfo.gpuContext.reservedGlobalMem;

        matrixTranspose_cuda <<<gridConf, blockConf, 0, lCudaStream>>> (lTaskConf->blockSizeRows, lTaskConf->blockSizeRows, pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].readPtr, lBlockCudaPtr);
        matrixCopy_cuda <<<gridConf, blockConf, 0, lCudaStream>>> (*lTaskConf, pSubtaskInfo, lBlockCudaPtr);    // because transpose is inplace, this has to be a post step
    }
    else
    {
        matrixTranspose_cuda <<<gridConf, blockConf, 0, lCudaStream>>> (lTaskConf->blockSizeRows, lTaskConf->blockSizeRows, pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr, pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr);
    }

    return pmSuccess;
}
    
int singleGpuMatrixTranspose(bool pInplace, MATRIX_DATA_TYPE* pInputMatrix, MATRIX_DATA_TYPE* pOutputMatrix, size_t pInputDimRows, size_t pInputDimCols)
{
    MATRIX_DATA_TYPE* lInputMatrix = pInplace ? pOutputMatrix : pInputMatrix;
    
    void* lInputMemCudaPtr = NULL;
    void* lOutputMemCudaPtr = NULL;
    
    size_t lSize = sizeof(MATRIX_DATA_TYPE) * pInputDimRows * pInputDimCols;

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr, lSize));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr, lInputMatrix, lSize, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lSize));

    size_t lGridDim = std::max(pInputDimRows / GPU_TILE_DIM, pInputDimCols / GPU_TILE_DIM);

    dim3 gridConf(lGridDim, lGridDim, 1);
    dim3 blockConf(GPU_TILE_DIM, GPU_TILE_DIM / GPU_ELEMS_PER_THREAD, 1);
    matrixTranspose_singleGpu <<<gridConf, blockConf>>> (pInputDimCols, pInputDimRows, lInputMemCudaPtr, lOutputMemCudaPtr, pInputDimCols / GPU_TILE_DIM, pInputDimRows / GPU_TILE_DIM);

    CUDA_ERROR_CHECK("cudaDeviceSynchronize", cudaDeviceSynchronize());

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMatrix, lOutputMemCudaPtr, lSize, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    
    return 0;
}

}

#endif
#endif


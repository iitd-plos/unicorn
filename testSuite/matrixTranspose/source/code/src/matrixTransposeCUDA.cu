
#ifdef BUILD_CUDA
#ifdef USE_SQUARE_BLOCKS

#include "pmPublicDefinitions.h"
#include "matrixTranspose.h"

#include <iostream>

namespace matrixTranspose
{

__global__ void matrixTranspose_cuda(matrixTransposeTaskConf pTaskConf, pmSubtaskInfo pSubtaskInfo, void* pOutputBlock)
{
    __shared__ MATRIX_DATA_TYPE lTile[GPU_TILE_DIM][GPU_TILE_DIM + 1];

    int lBlockId_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    int lBlockId_y = blockIdx.x;
    
    int lIndexX = lBlockId_x * GPU_TILE_DIM + threadIdx.x;
    int lIndexY = lBlockId_y * GPU_TILE_DIM + threadIdx.y;
    int lInputIndex = lIndexX + (lIndexY * pTaskConf.matrixDimCols);

    lIndexX = lBlockId_y * GPU_TILE_DIM + threadIdx.x;
    lIndexY = lBlockId_x * GPU_TILE_DIM + threadIdx.y;
    int lOutputIndex = lIndexX + (lIndexY * pTaskConf.blockSizeRows);

    int i, lStride = (GPU_TILE_DIM/GPU_ELEMS_PER_THREAD);
    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        lTile[threadIdx.y + i][threadIdx.x] = ((MATRIX_DATA_TYPE*)pSubtaskInfo.outputMemRead)[lInputIndex + i * pTaskConf.matrixDimCols];

    __syncthreads();

    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        ((MATRIX_DATA_TYPE*)pOutputBlock)[lOutputIndex + i * pTaskConf.blockSizeRows] = lTile[threadIdx.x][threadIdx.y + i];
}

__global__ void matrixCopy_cuda(matrixTransposeTaskConf pTaskConf, pmSubtaskInfo pSubtaskInfo, void* pOutputBlock)
{
    int lIndexX = blockIdx.x * GPU_TILE_DIM + threadIdx.x;
    int lIndexY = blockIdx.y * GPU_TILE_DIM + threadIdx.y;
    int lInputIndex = lIndexX + (lIndexY * pTaskConf.blockSizeRows);
    int lOutputIndex = lIndexX + (lIndexY * pTaskConf.matrixDimRows);
    
    int i, lStride = (GPU_TILE_DIM/GPU_ELEMS_PER_THREAD);
    for(i = 0; i < GPU_TILE_DIM; i += lStride)
        ((MATRIX_DATA_TYPE*)pSubtaskInfo.outputMemWrite)[lOutputIndex + i * pTaskConf.matrixDimRows] = ((MATRIX_DATA_TYPE*)pOutputBlock)[lInputIndex + i * pTaskConf.blockSizeRows];
}

pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	matrixTransposeTaskConf* lTaskConf = (matrixTransposeTaskConf*)(pTaskInfo.taskConf);

    void* lBlockCudaPtr;
    size_t lBlockSize = sizeof(MATRIX_DATA_TYPE) * lTaskConf->blockSizeRows * lTaskConf->blockSizeRows;
    if(cudaMalloc((void**)&lBlockCudaPtr, lBlockSize) != cudaSuccess)
    {
        std::cout << "Matrix Transpose: CUDA Memory Allocation Failed" << std::endl;
        return pmUserError;
    }

    dim3 gridConf(lTaskConf->blockSizeRows / GPU_TILE_DIM, lTaskConf->blockSizeRows / GPU_TILE_DIM, 1);
    dim3 blockConf(GPU_TILE_DIM, GPU_TILE_DIM / GPU_ELEMS_PER_THREAD, 1);
    matrixTranspose_cuda <<<gridConf, blockConf>>> (*lTaskConf, pSubtaskInfo, lBlockCudaPtr);
    matrixCopy_cuda <<<gridConf, blockConf>>> (*lTaskConf, pSubtaskInfo, lBlockCudaPtr);
    
    if(cudaFree(lBlockCudaPtr) != cudaSuccess)
    {
        std::cout << "Matrix Transpose: CUDA Memory Deallocation Failed" << std::endl;
        return pmUserError;
    }
    
    return pmSuccess;
}

}

#endif
#endif


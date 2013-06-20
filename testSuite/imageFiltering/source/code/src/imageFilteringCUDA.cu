
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"
#include "imageFiltering.h"

#include <iostream>
#include <string.h>
#include <stdlib.h>

namespace imageFiltering
{

#define TEXEL_TYPE char

texture<TEXEL_TYPE, cudaTextureType2D, cudaReadModeElementType> gInvertedTextureRef;

__global__ void imageFilter_cuda(int pOffsetX, int pOffsetY, int pSubImageWidth, int pSubImageHeight, size_t pAlignmentOffset, void* pOutputMem, int pImageWidth, int pEffectiveTextureWidth, char* pFilter, int pFilterRadius)
{
    extern __shared__ char4 lImage[];   // [GPU_BLOCK_DIM + 2 * pFilterRadius][GPU_BLOCK_DIM + 2 * pFilterRadius];

    size_t lSharedMemDim = GPU_BLOCK_DIM + 2 * pFilterRadius;

    int lProcessibleImageWidth = min(GPU_BLOCK_DIM, pSubImageWidth - blockIdx.x * blockDim.x);
    int lProcessibleImageHeight = min(GPU_BLOCK_DIM, pSubImageHeight - blockIdx.y * blockDim.y);
    
    int lThreadsReqdX = max(2 * pFilterRadius, lProcessibleImageWidth);
    int lThreadsReqdY = max(2 * pFilterRadius, lProcessibleImageHeight);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(threadIdx.x > lThreadsReqdX || threadIdx.y > lThreadsReqdY)
        return;

    char4 lData;
    
    if(threadIdx.x < lProcessibleImageWidth && threadIdx.y < lProcessibleImageHeight)
    {
        int lX = PIXEL_COUNT * (x + pOffsetX) + pAlignmentOffset;
        int lY = y + pOffsetY;

        lData.z = tex2D(gInvertedTextureRef, lX, lY);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lY);
        lImage[(threadIdx.y + pFilterRadius) * lSharedMemDim + (threadIdx.x + pFilterRadius)] = lData;
    }
    
    // Load the top and bottom rows
    if(threadIdx.x < lProcessibleImageWidth && threadIdx.y < 2 * pFilterRadius)
    {
        int lX = PIXEL_COUNT * (x + pOffsetX) + pAlignmentOffset;
        int lY = y - threadIdx.y + pOffsetY;

        int lRow = ((threadIdx.y < pFilterRadius) ? (lY - (pFilterRadius - threadIdx.y)) : (lY + lProcessibleImageHeight + (threadIdx.y - pFilterRadius)));

        lData.z = tex2D(gInvertedTextureRef, lX, lRow);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lRow);

        int lSharedMemRow = ((threadIdx.y < pFilterRadius) ? threadIdx.y : (pFilterRadius + (threadIdx.y - pFilterRadius) + lProcessibleImageHeight));
        lImage[lSharedMemRow * lSharedMemDim + (threadIdx.x + pFilterRadius)] = lData;
    }

    // Load the left and right cols
    if(threadIdx.x < 2 * pFilterRadius && threadIdx.y < lProcessibleImageHeight)
    {
        int lX = (x - threadIdx.x + pOffsetX);
        int lY = y + pOffsetY;

        int lCol = pAlignmentOffset + PIXEL_COUNT * ((threadIdx.x < pFilterRadius) ? (lX - (pFilterRadius - threadIdx.x)) : (lX + lProcessibleImageWidth + (threadIdx.x - pFilterRadius)));

        if(lCol < 0) lCol = 0;
        if(lCol >= pEffectiveTextureWidth) lCol = pEffectiveTextureWidth - PIXEL_COUNT;
        
        lData.z = tex2D(gInvertedTextureRef, lCol, lY);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lY);

        int lSharedMemCol = ((threadIdx.x < pFilterRadius) ? threadIdx.x : (pFilterRadius + (threadIdx.x - pFilterRadius) + lProcessibleImageWidth));
        lImage[(threadIdx.y + pFilterRadius) * lSharedMemDim + lSharedMemCol] = lData;
    }

    // Load the four corners
    if(threadIdx.x < 2 * pFilterRadius && threadIdx.y < 2 * pFilterRadius)
    {
        int lX = (x - threadIdx.x + pOffsetX);
        int lY = y - threadIdx.y + pOffsetY;
        
        int lRow = ((threadIdx.y < pFilterRadius) ? (lY - (pFilterRadius - threadIdx.y)) : (lY + lProcessibleImageHeight + (threadIdx.y - pFilterRadius)));
        int lCol = pAlignmentOffset + PIXEL_COUNT * ((threadIdx.x < pFilterRadius) ? (lX - (pFilterRadius - threadIdx.x)) : (lX + lProcessibleImageWidth + (threadIdx.x - pFilterRadius)));

        if(lCol < 0) lCol = 0;
        if(lCol >= pEffectiveTextureWidth) lCol = pEffectiveTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        
        int lSharedMemRow = ((threadIdx.y < pFilterRadius) ? threadIdx.y : (pFilterRadius + (threadIdx.y - pFilterRadius) + lProcessibleImageHeight));
        int lSharedMemCol = ((threadIdx.x < pFilterRadius) ? threadIdx.x : (pFilterRadius + (threadIdx.x - pFilterRadius) + lProcessibleImageWidth));
        lImage[lSharedMemRow * lSharedMemDim + lSharedMemCol] = lData;
    }

    if(threadIdx.x >= lProcessibleImageWidth || threadIdx.y >= lProcessibleImageHeight)
        return;

    __syncthreads();

    char lRedVal = 0, lGreenVal = 0, lBlueVal = 0;
    int lFilterDim = 2 * pFilterRadius + 1;
    for(int k = 0; k < lFilterDim; ++k)
    {
        for(int l = 0; l < lFilterDim; ++l)
        {
            lData = lImage[(k + threadIdx.y) * lSharedMemDim + (l + threadIdx.x)];
            lRedVal += lData.x * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
            lGreenVal += lData.y * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
            lBlueVal += lData.z * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
        }
    }

    int lUninvertedRow = pSubImageHeight - y - 1;
    size_t lOffset = (lUninvertedRow * pImageWidth + x) * PIXEL_COUNT;

    ((char*)pOutputMem)[lOffset] = lRedVal;
    ((char*)pOutputMem)[lOffset + 1] = lGreenVal;
    ((char*)pOutputMem)[lOffset + 2] = lBlueVal;
}
    
int PrepareForLaunch(int pTextureWidth, int pTextureHeight, char* pInvertedImageData, int pImageBytesPerLine, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], int pFilterRadius, void* pOutputMem, int pImageWidth, int pOffsetX, int pOffsetY, int pCols, int pRows)
{
    int lEffectiveTextureWidth = pTextureWidth * PIXEL_COUNT;

    void* lTextureMem = NULL;
    size_t lPitch = 0;
    if(cudaMallocPitch(&lTextureMem, &lPitch, lEffectiveTextureWidth, pTextureHeight) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memory Allocation Failed" << std::endl;
        return 1;
    }

    if(cudaMemcpy2D(lTextureMem, lPitch, pInvertedImageData, pImageBytesPerLine, lEffectiveTextureWidth, pTextureHeight, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy 2D Failed" << std::endl;
        return 1;
    }
    
    gInvertedTextureRef.addressMode[0] = cudaAddressModeClamp;
    gInvertedTextureRef.addressMode[1] = cudaAddressModeClamp;
    gInvertedTextureRef.filterMode = cudaFilterModePoint;
    gInvertedTextureRef.normalized = false;
    
    float lAlignmentOffset = 0.0;
    size_t lTextureOffset = 0;
    cudaChannelFormatDesc lChannelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    if(cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, lTextureMem, lChannelDesc, lEffectiveTextureWidth, pTextureHeight, lPitch) != cudaSuccess)
    {
        std::cout << "Image Filter: Texture Binding Failed" << std::endl;
        return 1;
    }
    
    // Mislaigned binding; rebind
    if(lTextureOffset != 0)
    {
        lAlignmentOffset = lTextureOffset / sizeof(TEXEL_TYPE);
        if(cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, lTextureMem, lChannelDesc, lEffectiveTextureWidth + lAlignmentOffset, pTextureHeight, lPitch) != cudaSuccess)
        {
            std::cout << "Image Filter: Texture Binding Failed" << std::endl;
            return 1;
        }
    }

    char* lFilterPtr = NULL;
    if(cudaMalloc((void**)&lFilterPtr, (MAX_FILTER_DIM * MAX_FILTER_DIM)) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Filter Memory Allocation Failed" << std::endl;
        return 1;
    }
    
    if(cudaMemcpy(lFilterPtr, pFilter, MAX_FILTER_DIM * MAX_FILTER_DIM, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy Failed" << std::endl;
        return 1;
    }
    
    int lBlocksX = (pCols / GPU_BLOCK_DIM) + ((pCols % GPU_BLOCK_DIM) ? 1 : 0);
    int lBlocksY = (pRows / GPU_BLOCK_DIM) + ((pRows % GPU_BLOCK_DIM) ? 1 : 0);

    size_t lSharedMemReqd = sizeof(char4) * (GPU_BLOCK_DIM + 2 * pFilterRadius) * (GPU_BLOCK_DIM + 2 * pFilterRadius);

    dim3 gridConf(lBlocksX, lBlocksY, 1);
    dim3 blockConf(GPU_BLOCK_DIM, GPU_BLOCK_DIM, 1);
    imageFilter_cuda <<<gridConf, blockConf, lSharedMemReqd>>> (pOffsetX, pOffsetY, pCols, pRows, lAlignmentOffset, pOutputMem, pImageWidth, lEffectiveTextureWidth, lFilterPtr, pFilterRadius);

    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Device Synchronize Failed" << std::endl;
        return 1;
    }

    if(cudaUnbindTexture(gInvertedTextureRef) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Unbind Texture Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lTextureMem) != cudaSuccess || cudaFree(lFilterPtr) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memory Deallocation Failed" << std::endl;
        return 1;
    }

    return 0;
}

pmStatus imageFilter_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    unsigned int lTilesPerRow = (lTaskConf->imageWidth/TILE_DIM + (lTaskConf->imageWidth%TILE_DIM ? 1 : 0));
    
    int lSubscriptionStartCol = (unsigned int)((pSubtaskInfo.subtaskId % lTilesPerRow) * TILE_DIM);
    int lSubscriptionEndCol = lSubscriptionStartCol + TILE_DIM;
    int lSubscriptionStartRow = (unsigned int)((pSubtaskInfo.subtaskId / lTilesPerRow) * TILE_DIM);
    int lSubscriptionEndRow = lSubscriptionStartRow + TILE_DIM;

    if(lSubscriptionEndCol > lTaskConf->imageWidth)
        lSubscriptionEndCol = lTaskConf->imageWidth;

    if(lSubscriptionEndRow > lTaskConf->imageHeight)
        lSubscriptionEndRow = lTaskConf->imageHeight;
    
    int lStartCol = lSubscriptionStartCol - lTaskConf->filterRadius;
    int lEndCol = lSubscriptionEndCol + lTaskConf->filterRadius;
    int lStartRow = lSubscriptionStartRow - lTaskConf->filterRadius;
    int lEndRow = lSubscriptionEndRow + lTaskConf->filterRadius;
    
    if(lStartCol < 0) lStartCol = 0;
    if(lStartRow < 0) lStartRow = 0;
    if(lEndCol > lTaskConf->imageWidth) lEndCol = lTaskConf->imageWidth;
    if(lEndRow > lTaskConf->imageHeight) lEndRow = lTaskConf->imageHeight;

    int lSubImageWidth = lEndCol - lStartCol;
    int lSubImageHeight = lEndRow - lStartRow;
    
    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset;
    lInvertedImageData += (lTaskConf->imageBytesPerLine * (lTaskConf->imageHeight - lEndRow) + lStartCol * PIXEL_COUNT);
 
    int lOffsetX = lSubscriptionStartCol - lStartCol;
    int lOffsetY = lEndRow - lSubscriptionEndRow;
    
    int lCols = lSubscriptionEndCol - lSubscriptionStartCol;
    int lRows = lSubscriptionEndRow - lSubscriptionStartRow;

    if(PrepareForLaunch(lSubImageWidth, lSubImageHeight, lInvertedImageData, lTaskConf->imageBytesPerLine, lTaskConf->filter, lTaskConf->filterRadius, pSubtaskInfo.outputMem, lTaskConf->imageWidth, lOffsetX, lOffsetY, lCols, lRows))
        return pmUserError;
    
    return pmSuccess;
}

// Returns 0 on success, non-zero on failure
int singleGpuImageFilter(void* pInvertedImageData, int pImageWidth, int pImageHeight, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], int pFilterRadius, int pImageBytesPerLine, void* pOutputMem)
{
    void* lOutputMemCudaPtr = NULL;
    size_t lImageSize = (pImageWidth * pImageHeight * PIXEL_COUNT);
    if(cudaMalloc((void**)&lOutputMemCudaPtr, lImageSize) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Output Memory Allocation Failed" << std::endl;
        return 1;
    }

    int lRetVal = PrepareForLaunch(pImageWidth, pImageHeight, (char*)pInvertedImageData, pImageWidth * PIXEL_COUNT, pFilter, pFilterRadius, lOutputMemCudaPtr, pImageWidth, 0, 0, pImageWidth, pImageHeight);
    
    if(cudaMemcpy(pOutputMem, lOutputMemCudaPtr, lImageSize, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lOutputMemCudaPtr) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memory Deallocation Failed" << std::endl;
        return 1;
    }

    return lRetVal;
}

}

#endif
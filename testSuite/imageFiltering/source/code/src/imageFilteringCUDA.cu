
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

__global__ void imageFilter_cuda(int pOffsetX, int pOffsetY, int pSubImageWidth, int pSubImageHeight, size_t pAlignmentOffset, void* pOutputMem, int pImageWidth, int pTextureWidth, char* pFilter, int pFilterRadius)
{
    extern __shared__ char4 lImage[];   // [GPU_BLOCK_DIM + 2 * pFilterRadius][GPU_BLOCK_DIM + 2 * pFilterRadius];

    size_t lSharedMemDim = GPU_BLOCK_DIM + 2 * pFilterRadius;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= pSubImageWidth || y >= pSubImageHeight)
        return;

    int lX = PIXEL_COUNT * (x + pOffsetX) + pAlignmentOffset;
    int lY = y + pOffsetY;
    char4 lData;
    
    lData.z = tex2D(gInvertedTextureRef, lX, lY);
    lData.y = tex2D(gInvertedTextureRef, lX + 1, lY);
    lData.x = tex2D(gInvertedTextureRef, lX + 2, lY);
    lImage[(threadIdx.y + pFilterRadius) * lSharedMemDim + (threadIdx.x + pFilterRadius)] = lData;

    // Load the top rows
    if(threadIdx.y < pFilterRadius)
    {
        int lRow = lY - pFilterRadius;

        lData.z = tex2D(gInvertedTextureRef, lX, lRow);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lRow);
        lImage[threadIdx.y * lSharedMemDim + (threadIdx.x + pFilterRadius)] = lData;
    }

    // Load the left cols
    if(threadIdx.x < pFilterRadius)
    {
        int lCol = lX - PIXEL_COUNT * pFilterRadius;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lY);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lY);
        lImage[(threadIdx.y + pFilterRadius) * lSharedMemDim + threadIdx.x] = lData;
    }

    // Load the bottom rows
    if((y >= (pSubImageHeight - pFilterRadius)) || (threadIdx.y >= (GPU_BLOCK_DIM - pFilterRadius)))
    {
        int lRow = lY + pFilterRadius;

        lData.z = tex2D(gInvertedTextureRef, lX, lRow);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lRow);
        lImage[(threadIdx.y + 2 * pFilterRadius) * lSharedMemDim + (threadIdx.x + pFilterRadius)] = lData;
    }

    // Load the right cols
    if((x >= (pSubImageWidth - pFilterRadius)) || (threadIdx.x >= (GPU_BLOCK_DIM - pFilterRadius)))
    {
        int lCol = lX + PIXEL_COUNT * pFilterRadius;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lY);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lY);
        lImage[(threadIdx.y + pFilterRadius) * lSharedMemDim + (threadIdx.x + 2 * pFilterRadius)] = lData;
    }

    // Load the top left corner
    if(threadIdx.y < pFilterRadius && threadIdx.x < pFilterRadius)
    {
        int lRow = lY - pFilterRadius;
        int lCol = lX - PIXEL_COUNT * pFilterRadius;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y * lSharedMemDim + threadIdx.x] = lData;
    }

    // Load the top right corner
    if(threadIdx.y < pFilterRadius && ((x >= (pSubImageWidth - pFilterRadius)) || (threadIdx.x >= (GPU_BLOCK_DIM - pFilterRadius))))
    {
        int lRow = lY - pFilterRadius;
        int lCol = lX + PIXEL_COUNT * pFilterRadius;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y * lSharedMemDim + (threadIdx.x + 2 * pFilterRadius)] = lData;
    }

    // Load the bottom left corner
    if(((y >= (pSubImageHeight - pFilterRadius)) || (threadIdx.y >= (GPU_BLOCK_DIM - pFilterRadius))) && threadIdx.x < pFilterRadius)
    {
        int lRow = lY + pFilterRadius;
        int lCol = lX - PIXEL_COUNT * pFilterRadius;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[(threadIdx.y + 2 * pFilterRadius) * lSharedMemDim + threadIdx.x] = lData;
    }

    // Load the bottom right corner
    if(((y >= (pSubImageHeight - pFilterRadius)) || (threadIdx.y >= (GPU_BLOCK_DIM - pFilterRadius))) && ((x >= (pSubImageWidth - pFilterRadius)) || (threadIdx.x >= (GPU_BLOCK_DIM - pFilterRadius))))
    {
        int lRow = lY + pFilterRadius;
        int lCol = lX + PIXEL_COUNT * pFilterRadius;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[(threadIdx.y + 2 * pFilterRadius) * lSharedMemDim + (threadIdx.x + 2 * pFilterRadius)] = lData;
    }

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

pmStatus imageFilter_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    const char* lKey = "sharedMemPerBlock=";
    char* lStartStr = strstr(pDeviceInfo.description, lKey);
    lStartStr += strlen(lKey);
    char* lEndStr = strstr(lStartStr, ";");

    size_t lDigits = (size_t)lEndStr - (size_t)lStartStr;

    const size_t lArraySize = 16;
    char lArray[lArraySize];
    
    if(lArraySize - 1 < lDigits)
        exit(1);
        
    strncpy((char*)lArray, lStartStr, lDigits);
    lArray[lDigits] = '\0';
    
    int lSharedMemSize = atoi(lArray);
    
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
    
    int lEffectiveSubImageWidth = lSubImageWidth * PIXEL_COUNT;

    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset;
    lInvertedImageData += (lTaskConf->imageBytesPerLine * (lTaskConf->imageHeight - lEndRow) + lStartCol * PIXEL_COUNT);

    void* lTextureMem = NULL;
    size_t lPitch = 0;
    if(cudaMallocPitch(&lTextureMem, &lPitch, lEffectiveSubImageWidth, lSubImageHeight) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memory Allocation Failed" << std::endl;
        return pmUserError;
    }

    if(cudaMemcpy2D(lTextureMem, lPitch, lInvertedImageData, lTaskConf->imageBytesPerLine, lEffectiveSubImageWidth, lSubImageHeight, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy 2D Failed" << std::endl;
        return pmUserError;
    }
    
    gInvertedTextureRef.addressMode[0] = cudaAddressModeClamp;
    gInvertedTextureRef.addressMode[1] = cudaAddressModeClamp;
    gInvertedTextureRef.filterMode = cudaFilterModePoint;
    gInvertedTextureRef.normalized = false;
    
    float lAlignmentOffset = 0.0;
    size_t lTextureOffset = 0;
    cudaChannelFormatDesc lChannelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    if(cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, lTextureMem, lChannelDesc, lEffectiveSubImageWidth, lSubImageHeight, lPitch) != cudaSuccess)
    {
        std::cout << "Image Filter: Texture Binding Failed" << std::endl;
        return pmUserError;
    }
    
    // Mislaigned binding; rebind
    if(lTextureOffset != 0)
    {
        lAlignmentOffset = lTextureOffset / sizeof(TEXEL_TYPE);
        if(cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, lTextureMem, lChannelDesc, lEffectiveSubImageWidth + lAlignmentOffset, lSubImageHeight, lPitch) != cudaSuccess)
        {
            std::cout << "Image Filter: Texture Binding Failed" << std::endl;
            return pmUserError;
        }
    }

    char* lFilterPtr = NULL;
    if(cudaMalloc((void**)&lFilterPtr, (MAX_FILTER_DIM * MAX_FILTER_DIM)) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Filter Memory Allocation Failed" << std::endl;
        return pmUserError;
    }
    
    if(cudaMemcpy(lFilterPtr, lTaskConf->filter, MAX_FILTER_DIM * MAX_FILTER_DIM, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy Failed" << std::endl;
        return pmUserError;
    }
    
    int lOffsetX = lSubscriptionStartCol - lStartCol;
    int lOffsetY = lEndRow - lSubscriptionEndRow;
    
    int lThreadsX = lSubscriptionEndCol - lSubscriptionStartCol;
    int lThreadsY = lSubscriptionEndRow - lSubscriptionStartRow;
    
    int lBlocksX = (lThreadsX / GPU_BLOCK_DIM) + ((lThreadsX % GPU_BLOCK_DIM) ? 1 : 0);
    int lBlocksY = (lThreadsY / GPU_BLOCK_DIM) + ((lThreadsY % GPU_BLOCK_DIM) ? 1 : 0);

    size_t lSharedMemReqd = sizeof(char4) * (GPU_BLOCK_DIM + 2 * lTaskConf->filterRadius) * (GPU_BLOCK_DIM + 2 * lTaskConf->filterRadius);
    
    dim3 gridConf(lBlocksX, lBlocksY, 1);
    dim3 blockConf(GPU_BLOCK_DIM, GPU_BLOCK_DIM, 1);
    imageFilter_cuda <<<gridConf, blockConf, lSharedMemReqd>>> (lOffsetX, lOffsetY, lThreadsX, lThreadsY, lAlignmentOffset, pSubtaskInfo.outputMem, lTaskConf->imageWidth, lEffectiveSubImageWidth, lFilterPtr, lTaskConf->filterRadius);

    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Device Synchronize Failed" << std::endl;
        return pmUserError;
    }

    if(cudaUnbindTexture(gInvertedTextureRef) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Unbind Texture Failed" << std::endl;
        return pmUserError;
    }

    if(cudaFree(lTextureMem) != cudaSuccess || cudaFree(lFilterPtr) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memory Deallocation Failed" << std::endl;
        return pmUserError;
    }

    return pmSuccess;
}

}

#endif
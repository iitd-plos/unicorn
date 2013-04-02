
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

__global__ void imageFilter_cuda(int pOffsetX, int pOffsetY, int pSubImageWidth, int pSubImageHeight, size_t pAlignmentOffset, void* pOutputMem, int pImageWidth, int pTextureWidth, char* pFilter)
{
    __shared__ char4 lImage[GPU_BLOCK_DIM + 2 * FILTER_RADIUS][GPU_BLOCK_DIM + 2 * FILTER_RADIUS];

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
    lImage[threadIdx.y + FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = lData;

    // Load the top rows
    if(threadIdx.y < FILTER_RADIUS)
    {
        int lRow = lY - FILTER_RADIUS;

        lData.z = tex2D(gInvertedTextureRef, lX, lRow);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lRow);
        lImage[threadIdx.y][threadIdx.x + FILTER_RADIUS] = lData;
    }

    // Load the left cols
    if(threadIdx.x < FILTER_RADIUS)
    {
        int lCol = lX - PIXEL_COUNT * FILTER_RADIUS;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lY);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lY);
        lImage[threadIdx.y + FILTER_RADIUS][threadIdx.x] = lData;
    }

    // Load the bottom rows
    if((y >= (pSubImageHeight - FILTER_RADIUS)) || (threadIdx.y >= (GPU_BLOCK_DIM - FILTER_RADIUS)))
    {
        int lRow = lY + FILTER_RADIUS;

        lData.z = tex2D(gInvertedTextureRef, lX, lRow);
        lData.y = tex2D(gInvertedTextureRef, lX + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lX + 2, lRow);
        lImage[threadIdx.y + 2 * FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = lData;
    }

    // Load the right cols
    if((x >= (pSubImageWidth - FILTER_RADIUS)) || (threadIdx.x >= (GPU_BLOCK_DIM - FILTER_RADIUS)))
    {
        int lCol = lX + PIXEL_COUNT * FILTER_RADIUS;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lY);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lY);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lY);
        lImage[threadIdx.y + FILTER_RADIUS][threadIdx.x + 2 * FILTER_RADIUS] = lData;
    }

    // Load the top left corner
    if(threadIdx.y < FILTER_RADIUS && threadIdx.x < FILTER_RADIUS)
    {
        int lRow = lY - FILTER_RADIUS;
        int lCol = lX - PIXEL_COUNT * FILTER_RADIUS;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y][threadIdx.x] = lData;
    }

    // Load the top right corner
    if(threadIdx.y < FILTER_RADIUS && ((x >= (pSubImageWidth - FILTER_RADIUS)) || (threadIdx.x >= (GPU_BLOCK_DIM - FILTER_RADIUS))))
    {
        int lRow = lY - FILTER_RADIUS;
        int lCol = lX + PIXEL_COUNT * FILTER_RADIUS;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y][threadIdx.x + 2 * FILTER_RADIUS] = lData;
    }

    // Load the bottom left corner
    if(((y >= (pSubImageHeight - FILTER_RADIUS)) || (threadIdx.y >= (GPU_BLOCK_DIM - FILTER_RADIUS))) && threadIdx.x < FILTER_RADIUS)
    {
        int lRow = lY + FILTER_RADIUS;
        int lCol = lX - PIXEL_COUNT * FILTER_RADIUS;
        if(lCol < 0) lCol = 0;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y + 2 * FILTER_RADIUS][threadIdx.x] = lData;
    }

    // Load the bottom right corner
    if(((y >= (pSubImageHeight - FILTER_RADIUS)) || (threadIdx.y >= (GPU_BLOCK_DIM - FILTER_RADIUS))) && ((x >= (pSubImageWidth - FILTER_RADIUS)) || (threadIdx.x >= (GPU_BLOCK_DIM - FILTER_RADIUS))))
    {
        int lRow = lY + FILTER_RADIUS;
        int lCol = lX + PIXEL_COUNT * FILTER_RADIUS;
        if(lCol >= pTextureWidth) lCol = pTextureWidth - PIXEL_COUNT;

        lData.z = tex2D(gInvertedTextureRef, lCol, lRow);
        lData.y = tex2D(gInvertedTextureRef, lCol + 1, lRow);
        lData.x = tex2D(gInvertedTextureRef, lCol + 2, lRow);
        lImage[threadIdx.y + 2 * FILTER_RADIUS][threadIdx.x + 2 * FILTER_RADIUS] = lData;
    }

    __syncthreads();

#if 0   //(FILTER_RADIUS == 1)
    char4 lData00 = lImage[threadIdx.y][threadIdx.x];
    char4 lData01 = lImage[threadIdx.y][1 + threadIdx.x];
    char4 lData02 = lImage[threadIdx.y][2 + threadIdx.x];
    char4 lData10 = lImage[1 + threadIdx.y][threadIdx.x];
    char4 lData11 = lImage[1 + threadIdx.y][1 + threadIdx.x];
    char4 lData12 = lImage[1 + threadIdx.y][2 + threadIdx.x];
    char4 lData20 = lImage[2 + threadIdx.y][threadIdx.x];
    char4 lData21 = lImage[2 + threadIdx.y][1 + threadIdx.x];
    char4 lData22 = lImage[2 + threadIdx.y][2 + threadIdx.x];
    
    char lRedVal = lData00.x * pFilter[0] + lData01.x * pFilter[1] + lData02.x * pFilter[2] + lData10.x * pFilter[3] + lData11.x * pFilter[4] + lData12.x * pFilter[5] + lData20.x * pFilter[6] + lData21.x * pFilter[7] + lData22.x * pFilter[8];
    char lGreenVal = lData00.y * pFilter[0] + lData01.y * pFilter[1] + lData02.y * pFilter[2] + lData10.y * pFilter[3] + lData11.y * pFilter[4] + lData12.y * pFilter[5] + lData20.y * pFilter[6] + lData21.y * pFilter[7] + lData22.y * pFilter[8];
    char lBlueVal = lData00.z * pFilter[0] + lData01.z * pFilter[1] + lData02.z * pFilter[2] + lData10.z * pFilter[3] + lData11.z * pFilter[4] + lData12.z * pFilter[5] + lData20.z * pFilter[6] + lData21.z * pFilter[7] + lData22.z * pFilter[8];
#else
    char lRedVal = 0, lGreenVal = 0, lBlueVal = 0;
    for(int k = 0; k < FILTER_DIM; ++k)
    {
        for(int l = 0; l < FILTER_DIM; ++l)
        {
            lData = lImage[k + threadIdx.y][l + threadIdx.x];
            lRedVal += lData.x * pFilter[k * FILTER_DIM + l];
            lGreenVal += lData.y * pFilter[k * FILTER_DIM + l];
            lBlueVal += lData.z * pFilter[k * FILTER_DIM + l];
        }
    }
#endif

    int lUninvertedRow = pSubImageHeight - y - 1;
    size_t lOffset = (lUninvertedRow * pImageWidth + threadIdx.x) * PIXEL_COUNT;

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
    
    int lStartCol = lSubscriptionStartCol - FILTER_RADIUS;
    int lEndCol = lSubscriptionEndCol + FILTER_RADIUS;
    int lStartRow = lSubscriptionStartRow - FILTER_RADIUS;
    int lEndRow = lSubscriptionEndRow + FILTER_RADIUS;
    
    if(lStartCol < 0) lStartCol = 0;
    if(lStartRow < 0) lStartRow = 0;
    if(lEndCol > lTaskConf->imageWidth) lEndCol = lTaskConf->imageWidth;
    if(lEndRow > lTaskConf->imageHeight) lEndRow = lTaskConf->imageHeight;

    int lSubImageWidth = lEndCol - lStartCol;
    int lSubImageHeight = lEndRow - lStartRow;
    
    int lEffectiveSubImageWidth = lSubImageWidth * PIXEL_COUNT;

    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset + lTaskConf->imageBytesPerLine * (lEndRow - lTaskConf->imageHeight);

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
    if(cudaMalloc((void**)&lFilterPtr, (FILTER_DIM * FILTER_DIM)) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Filter Memory Allocation Failed" << std::endl;
        return pmUserError;
    }
    
    if(cudaMemcpy(lFilterPtr, lTaskConf->filter, FILTER_DIM * FILTER_DIM, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Image Filter: CUDA Memcpy Failed" << std::endl;
        return pmUserError;
    }
    
    int lOffsetX = lSubscriptionStartCol - lStartCol;
    int lOffsetY = lSubscriptionStartRow - lStartRow;
    
    int lThreadsX = lSubscriptionEndCol - lSubscriptionStartCol;
    int lThreadsY = lSubscriptionEndRow - lSubscriptionStartRow;
    
    int lBlocksX = (lThreadsX / GPU_BLOCK_DIM) + ((lThreadsX % GPU_BLOCK_DIM) ? 1 : 0);
    int lBlocksY = (lThreadsY / GPU_BLOCK_DIM) + ((lThreadsY % GPU_BLOCK_DIM) ? 1 : 0);

    dim3 gridConf(lBlocksX, lBlocksY, 1);
    dim3 blockConf(GPU_BLOCK_DIM, GPU_BLOCK_DIM, 1);
    imageFilter_cuda <<<gridConf, blockConf>>> (lOffsetX, lOffsetY, lThreadsX, lThreadsY, lAlignmentOffset, pSubtaskInfo.outputMem, lTaskConf->imageWidth, lEffectiveSubImageWidth, lFilterPtr);

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
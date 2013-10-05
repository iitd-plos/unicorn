
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"
#include "imageFiltering.h"

#include <iostream>
#include <string.h>
#include <stdlib.h>

#include <map>

#include "commonAPI.h"

namespace imageFiltering
{

size_t getCudaAlignment()
{
    static std::map<int, size_t> sMap;  // deviceId versus textureAlignment

    int lDeviceId;
    CUDA_ERROR_CHECK("cudaGetDevice", cudaGetDevice(&lDeviceId));

    std::map<int, size_t>::iterator lIter = sMap.find(lDeviceId);
    if(lIter == sMap.end())
    {
        cudaDeviceProp lDeviceProp;
        CUDA_ERROR_CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&lDeviceProp, lDeviceId));
    
        sMap[lDeviceId] = lDeviceProp.textureAlignment;
        
        return lDeviceProp.textureAlignment;
    }
    
    return lIter->second;
}

size_t getTexturePitch(size_t pTextureWidth)
{
    size_t lAlignment = getCudaAlignment();
    size_t lHighestVal = lAlignment - 1;
    
    int lEffectiveTextureWidth = pTextureWidth * PIXEL_COUNT;
    return ((lEffectiveTextureWidth + lHighestVal) & ~lHighestVal);
}

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
    
void prepareForLaunch(int pTextureWidth, int pTextureHeight, char* pInvertedImageData, int pImageBytesPerLine, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], int pFilterRadius, void* pOutputMem, int pImageWidth, int pOffsetX, int pOffsetY, int pCols, int pRows, size_t pTexturePitch, void* pTextureMem, char* pFilterPtr, cudaStream_t pCudaStream)
{
    int lEffectiveTextureWidth = pTextureWidth * PIXEL_COUNT;

    CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(pFilterPtr, pFilter, MAX_FILTER_DIM * MAX_FILTER_DIM, cudaMemcpyHostToDevice, pCudaStream));

    CUDA_ERROR_CHECK("cudaMemcpy2DAsync", cudaMemcpy2DAsync(pTextureMem, pTexturePitch, pInvertedImageData, pImageBytesPerLine, lEffectiveTextureWidth, pTextureHeight, cudaMemcpyHostToDevice, pCudaStream));

    gInvertedTextureRef.addressMode[0] = cudaAddressModeClamp;
    gInvertedTextureRef.addressMode[1] = cudaAddressModeClamp;
    gInvertedTextureRef.filterMode = cudaFilterModePoint;
    gInvertedTextureRef.normalized = false;

    float lAlignmentOffset = 0.0;
    size_t lTextureOffset = 0;
    cudaChannelFormatDesc lChannelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    CUDA_ERROR_CHECK("cudaBindTexture2D", cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, pTextureMem, lChannelDesc, lEffectiveTextureWidth, pTextureHeight, pTexturePitch));
    
    // Mislaigned binding; rebind
    if(lTextureOffset != 0)
    {
        lAlignmentOffset = lTextureOffset / sizeof(TEXEL_TYPE);
        CUDA_ERROR_CHECK("cudaBindTexture2D", cudaBindTexture2D(&lTextureOffset, gInvertedTextureRef, pTextureMem, lChannelDesc, lEffectiveTextureWidth + lAlignmentOffset, pTextureHeight, pTexturePitch));
    }
    
    int lBlocksX = (pCols / GPU_BLOCK_DIM) + ((pCols % GPU_BLOCK_DIM) ? 1 : 0);
    int lBlocksY = (pRows / GPU_BLOCK_DIM) + ((pRows % GPU_BLOCK_DIM) ? 1 : 0);

    size_t lSharedMemReqd = sizeof(char4) * (GPU_BLOCK_DIM + 2 * pFilterRadius) * (GPU_BLOCK_DIM + 2 * pFilterRadius);

    dim3 gridConf(lBlocksX, lBlocksY, 1);
    dim3 blockConf(GPU_BLOCK_DIM, GPU_BLOCK_DIM, 1);
    imageFilter_cuda <<<gridConf, blockConf, lSharedMemReqd, pCudaStream>>> (pOffsetX, pOffsetY, pCols, pRows, lAlignmentOffset, pOutputMem, pImageWidth, lEffectiveTextureWidth, pFilterPtr, pFilterRadius);

    //CUDA_ERROR_CHECK("cudaStreamSynchronize", cudaStreamSynchronize(pCudaStream));
    CUDA_ERROR_CHECK("cudaUnbindTexture", cudaUnbindTexture(gInvertedTextureRef));
}
    
size_t computeSubtaskReservedMemRequirement(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId, int pSubscriptionStartCol, int pSubscriptionEndCol, int pSubscriptionStartRow, int pSubscriptionEndRow)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);

    int lStartCol = pSubscriptionStartCol - lTaskConf->filterRadius;
    int lEndCol = pSubscriptionEndCol + lTaskConf->filterRadius;
    int lStartRow = pSubscriptionStartRow - lTaskConf->filterRadius;
    int lEndRow = pSubscriptionEndRow + lTaskConf->filterRadius;
    
    if(lStartCol < 0) lStartCol = 0;
    if(lStartRow < 0) lStartRow = 0;
    if(lEndCol > lTaskConf->imageWidth) lEndCol = lTaskConf->imageWidth;
    if(lEndRow > lTaskConf->imageHeight) lEndRow = lTaskConf->imageHeight;

    int lSubImageWidth = lEndCol - lStartCol;
    int lSubImageHeight = lEndRow - lStartRow;
    
    return ((MAX_FILTER_DIM * MAX_FILTER_DIM) + (getTexturePitch(lSubImageWidth) * lSubImageHeight));
}

pmStatus imageFilter_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
	imageFilterTaskConf* lTaskConf = (imageFilterTaskConf*)(pTaskInfo.taskConf);
    
    int lSubscriptionStartCol, lSubscriptionEndCol, lSubscriptionStartRow, lSubscriptionEndRow;
    if(!GetSubtaskSubscription(lTaskConf, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, &lSubscriptionStartCol, &lSubscriptionEndCol, &lSubscriptionStartRow, &lSubscriptionEndRow))
        return pmSuccess;
    
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
    
    char* lFilterPtr = (char*)(pSubtaskInfo.gpuContext.reservedGlobalMem);
    void* lTextureMem  = (void*)(lFilterPtr + (MAX_FILTER_DIM * MAX_FILTER_DIM));
    
    prepareForLaunch(lSubImageWidth, lSubImageHeight, lInvertedImageData, lTaskConf->imageBytesPerLine, lTaskConf->filter, lTaskConf->filterRadius, pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr, lTaskConf->imageWidth, lOffsetX, lOffsetY, lCols, lRows, getTexturePitch(lSubImageWidth), lTextureMem, lFilterPtr, (cudaStream_t)pCudaStream);

    return pmSuccess;
}

// Returns 0 on success, non-zero on failure
int singleGpuImageFilter(void* pInvertedImageData, int pImageWidth, int pImageHeight, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], int pFilterRadius, int pImageBytesPerLine, void* pOutputMem)
{
    void* lOutputMemCudaPtr = NULL;
    size_t lImageSize = (pImageWidth * pImageHeight * PIXEL_COUNT);

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lImageSize));

    int lEffectiveTextureWidth = pImageWidth * PIXEL_COUNT;
    size_t lPitch = 0;

    char* lFilterPtr = NULL;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lFilterPtr, (MAX_FILTER_DIM * MAX_FILTER_DIM)));

    void* lTextureMem = NULL;
    CUDA_ERROR_CHECK("cudaMallocPitch", cudaMallocPitch(&lTextureMem, &lPitch, lEffectiveTextureWidth, pImageHeight));

    prepareForLaunch(pImageWidth, pImageHeight, (char*)pInvertedImageData, pImageWidth * PIXEL_COUNT, pFilter, pFilterRadius, lOutputMemCudaPtr, pImageWidth, 0, 0, pImageWidth, pImageHeight, lPitch, lTextureMem, lFilterPtr, NULL);
    
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMem, lOutputMemCudaPtr, lImageSize, cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lTextureMem));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lFilterPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));

    return 0;
}

}

#endif

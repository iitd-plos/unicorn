
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

#ifdef USE_ELLIPTICAL_FILTER
__global__ void imageFilter_cuda(int pOffsetX, int pOffsetY, int pSubImageWidth, int pSubImageHeight, size_t pAlignmentOffset, void* pOutputMem, int pOutputMemRowStep, int pEffectiveTextureWidth, char* pFilter, int pFilterRadius, bool pIsSubtaskId, ulong pSubtaskIdOrSubtasksPerRow, ulong pSubtaskCount)
#else
__global__ void imageFilter_cuda(int pOffsetX, int pOffsetY, int pSubImageWidth, int pSubImageHeight, size_t pAlignmentOffset, void* pOutputMem, int pOutputMemRowStep, int pEffectiveTextureWidth, char* pFilter, int pFilterRadius)
#endif
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

    int lUninvertedRow = pSubImageHeight - y - 1;
    size_t lOffset = (lUninvertedRow * pOutputMemRowStep + x) * PIXEL_COUNT;

#ifdef USE_ELLIPTICAL_FILTER
    ulong lSubtaskId = (pIsSubtaskId ? pSubtaskIdOrSubtasksPerRow : ((lUninvertedRow / TILE_DIM) * pSubtaskIdOrSubtasksPerRow + (x / TILE_DIM)));
    
    unsigned int lSemiMajorAxis = 1, lSemiMinorAxis = 1;
    float lSemiMajorAxisSquare = 1, lSemiMinorAxisSquare = 1;
    
    ulong lSubtasksLeft = pSubtaskCount - lSubtaskId;  // This is always greater than 1
    
    lSemiMajorAxis = pFilterRadius * ((float)lSubtaskId / pSubtaskCount);
    lSemiMinorAxis = pFilterRadius * ((float)lSubtasksLeft / pSubtaskCount);
    
    lSemiMajorAxisSquare = lSemiMajorAxis * lSemiMajorAxis;
    lSemiMinorAxisSquare = lSemiMinorAxis * lSemiMinorAxis;
#endif

    char lRedVal = 0, lGreenVal = 0, lBlueVal = 0;
    int lFilterDim = 2 * pFilterRadius + 1;
    for(int k = 0; k < lFilterDim; ++k)
    {
        for(int l = 0; l < lFilterDim; ++l)
        {
            lData = lImage[(k + threadIdx.y) * lSharedMemDim + (l + threadIdx.x)];

        #ifdef USE_ELLIPTICAL_FILTER
            float x = l - lFilterDim/2;
            float y = k - lFilterDim/2;
            
            if((x * x) / lSemiMajorAxisSquare + (y * y) / lSemiMinorAxisSquare < 1.0)   // Inside Ellipse
            {
                lRedVal += lData.x * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
                lGreenVal += lData.y * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
                lBlueVal += lData.z * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
            }
            else    // Outside Ellipse
            {
            }
        #else
            lRedVal += lData.x * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
            lGreenVal += lData.y * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
            lBlueVal += lData.z * pFilter[(lFilterDim - k - 1) * MAX_FILTER_DIM + l];
        #endif
        }
    }

    ((char*)pOutputMem)[lOffset] = lRedVal;
    ((char*)pOutputMem)[lOffset + 1] = lGreenVal;
    ((char*)pOutputMem)[lOffset + 2] = lBlueVal;
}
    
void prepareForLaunch(int pTextureWidth, int pTextureHeight, char* pInvertedImageData, int pImageBytesPerLine, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], int pFilterRadius, void* pOutputMem, int pOutputMemRowStep, int pOffsetX, int pOffsetY, int pCols, int pRows, size_t pTexturePitch, void* pTextureMem, char* pFilterPtr, cudaStream_t pCudaStream, bool pCopyImageFromHostToDevice, bool pIsSubtaskId, ulong pSubtaskIdOrSubtasksPerRow, ulong pSubtaskCount)
{
    int lEffectiveTextureWidth = pTextureWidth * PIXEL_COUNT;

    CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(pFilterPtr, pFilter, MAX_FILTER_DIM * MAX_FILTER_DIM, cudaMemcpyHostToDevice, pCudaStream));

    CUDA_ERROR_CHECK("cudaMemcpy2DAsync", cudaMemcpy2DAsync(pTextureMem, pTexturePitch, pInvertedImageData, pImageBytesPerLine, lEffectiveTextureWidth, pTextureHeight, (pCopyImageFromHostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice), pCudaStream));

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

#ifdef USE_ELLIPTICAL_FILTER
    imageFilter_cuda <<<gridConf, blockConf, lSharedMemReqd, pCudaStream>>> (pOffsetX, pOffsetY, pCols, pRows, lAlignmentOffset, pOutputMem, pOutputMemRowStep, lEffectiveTextureWidth, pFilterPtr, pFilterRadius, pIsSubtaskId, pSubtaskIdOrSubtasksPerRow, pSubtaskCount);
#else
    imageFilter_cuda <<<gridConf, blockConf, lSharedMemReqd, pCudaStream>>> (pOffsetX, pOffsetY, pCols, pRows, lAlignmentOffset, pOutputMem, pOutputMemRowStep, lEffectiveTextureWidth, pFilterPtr, pFilterRadius);
#endif

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
    
#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    char* lInvertedImageData = (char*)(pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr);
#else
    char* lInvertedImageData = ((char*)pmGetMappedFile(lTaskConf->imagePath)) + lTaskConf->imageOffset;
    lInvertedImageData += (lTaskConf->imageBytesPerLine * (lTaskConf->imageHeight - lEndRow) + lStartCol * PIXEL_COUNT);
#endif
    
    int lOffsetX = lSubscriptionStartCol - lStartCol;
    int lOffsetY = lEndRow - lSubscriptionEndRow;
    
    int lCols = lSubscriptionEndCol - lSubscriptionStartCol;
    int lRows = lSubscriptionEndRow - lSubscriptionStartRow;
    
    char* lFilterPtr = (char*)(pSubtaskInfo.gpuContext.reservedGlobalMem);
    void* lTextureMem  = (void*)(lFilterPtr + (MAX_FILTER_DIM * MAX_FILTER_DIM));
    
    size_t lWidth = ((pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->imageWidth : (lSubscriptionEndCol - lSubscriptionStartCol));
#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
    // The allocated address space is of size imageWidth * imageHeight. Every row is not aligned at imageBytesPerLine offset.
    size_t lInputWidth = ((pSubtaskInfo.memInfo[INPUT_MEM_INDEX].visibilityType == SUBSCRIPTION_NATURAL) ? lTaskConf->imageWidth : lSubImageWidth);
    prepareForLaunch(lSubImageWidth, lSubImageHeight, lInvertedImageData, lInputWidth * PIXEL_COUNT, lTaskConf->filter, lTaskConf->filterRadius, pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr, lWidth, lOffsetX, lOffsetY, lCols, lRows, getTexturePitch(lSubImageWidth), lTextureMem, lFilterPtr, (cudaStream_t)pCudaStream, false, true, pSubtaskInfo.subtaskId, pTaskInfo.subtaskCount);
#else
    prepareForLaunch(lSubImageWidth, lSubImageHeight, lInvertedImageData, lTaskConf->imageBytesPerLine, lTaskConf->filter, lTaskConf->filterRadius, pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr, lWidth, lOffsetX, lOffsetY, lCols, lRows, getTexturePitch(lSubImageWidth), lTextureMem, lFilterPtr, (cudaStream_t)pCudaStream, true, true, pSubtaskInfo.subtaskId, pTaskInfo.subtaskCount);
#endif

    return pmSuccess;
}

// Returns 0 on success, non-zero on failure
int singleGpuImageFilter(void* pInvertedImageData, size_t pImageWidth, size_t pImageHeight, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], size_t pFilterRadius, size_t pImageBytesPerLine, void* pOutputMem)
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

    unsigned int lTotalSubtasks = ((unsigned int)pImageWidth/TILE_DIM + ((unsigned int)pImageWidth%TILE_DIM ? 1 : 0)) * ((unsigned int)pImageHeight/TILE_DIM + ((unsigned int)pImageHeight%TILE_DIM ? 1 : 0));
    unsigned int lSubtasksPerRow = ((unsigned int)pImageWidth/TILE_DIM + ((unsigned int)pImageWidth%TILE_DIM ? 1 : 0));

    prepareForLaunch(pImageWidth, pImageHeight, (char*)pInvertedImageData, pImageWidth * PIXEL_COUNT, pFilter, pFilterRadius, lOutputMemCudaPtr, pImageWidth, 0, 0, pImageWidth, pImageHeight, lPitch, lTextureMem, lFilterPtr, NULL, true, false, lSubtasksPerRow, lTotalSubtasks);
    
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMem, lOutputMemCudaPtr, lImageSize, cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lTextureMem));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lFilterPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));

    return 0;
}

}

#endif

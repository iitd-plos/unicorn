
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
#include "prefixSum.h"
#include "commonAPI.h"

#include <iostream>

namespace prefixSum
{

#define MAX_ELEMS_PER_BLOCK 512     // must be a power of 2
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

unsigned int findFloorPowOfTwo(unsigned int pNum)
{
    --pNum;
    pNum |= pNum >> 1;
    pNum |= pNum >> 2;
    pNum |= pNum >> 4;
    pNum |= pNum >> 8;
    pNum |= pNum >> 16;
    ++pNum;
    
    return (pNum << 1);
}
    
bool isPowOfTwo(unsigned int pNum)
{
    return ((pNum & (pNum - 1)) == 0);
}
    
__global__ void prefixSum_cuda(PREFIX_SUM_DATA_TYPE* pInput, PREFIX_SUM_DATA_TYPE* pOutput, unsigned int pCountPerBlock, unsigned int pSubtaskElements, PREFIX_SUM_DATA_TYPE* pBlockSums, unsigned int pMaxBlocks)
{
    extern __shared__ PREFIX_SUM_DATA_TYPE temp[];
    int lThreadsPerBlock = (pCountPerBlock/2);  // = blockDim.x
    int lLinearBlockIndex = (blockIdx.y * gridDim.x + blockIdx.x);
    if(lLinearBlockIndex >= pMaxBlocks)
        return;
    
    int lSubtaskOffset = 2 * lLinearBlockIndex * lThreadsPerBlock;   // Two elements processed per thread
    int offset = 1;

    int ai = threadIdx.x;
    int bi = threadIdx.x + lThreadsPerBlock;
    
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    int indexA = lSubtaskOffset + ai;
    int indexB = lSubtaskOffset + bi;

    temp[ai + bankOffsetA] = (indexA >= pSubtaskElements) ? 0 : pInput[indexA];
    temp[bi + bankOffsetB] = (indexB >= pSubtaskElements) ? 0 : pInput[indexB];

    for(int d = (pCountPerBlock >> 1); d > 0; d >>= 1)
    {   
        __syncthreads();

        if(threadIdx.x < d)
        {  
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
        
           temp[bi] += temp[ai];
        }
        
        offset *= 2;
    }
    
    if(threadIdx.x == 0)
    {
        int lIndex = pCountPerBlock - 1 + CONFLICT_FREE_OFFSET(pCountPerBlock - 1);

        pBlockSums[lLinearBlockIndex] = temp[lIndex];
        temp[lIndex] = 0;
    }
    
    for(int d = 1; d < pCountPerBlock; d *= 2)
    {  
         offset >>= 1;
         __syncthreads();
        
         if(threadIdx.x < d)
         {  
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
             
            PREFIX_SUM_DATA_TYPE t = temp[ai];
            temp[ai] = temp[bi];  
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Convert Exclusive Scan to Inclusive Scan
    if(threadIdx.x != 0)
        pOutput[indexA - 1] = temp[ai + bankOffsetA];

    if(threadIdx.x == lThreadsPerBlock - 1)
        pOutput[indexB] = pInput[indexB] + temp[bi + bankOffsetB];

    pOutput[indexB - 1] = temp[bi + bankOffsetB];
}

__global__ void elemAdd_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
    PREFIX_SUM_DATA_TYPE lElem = ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.inputMem))[0];
    unsigned int lLength = (unsigned int)((pSubtaskInfo.outputMemLength)/sizeof(PREFIX_SUM_DATA_TYPE));
    
    unsigned int lThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lStartIndex = lThreadIndex * ELEMS_ADD_PER_CUDA_THREAD;
    unsigned int lEndIndex = lStartIndex + ELEMS_ADD_PER_CUDA_THREAD;
    if(lEndIndex > lLength)
        lEndIndex = lLength;

    for(unsigned int i = lStartIndex; i < lEndIndex; ++i)
        ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.outputMem))[i] += lElem;

    *pStatus = pmSuccess;
}
    
__global__ void arrayAdd_cuda(PREFIX_SUM_DATA_TYPE* pInput, PREFIX_SUM_DATA_TYPE* pOutput, unsigned int pOutputLength)
{
    unsigned int lLinearBlockIndex = (blockIdx.y * gridDim.x + blockIdx.x);
    unsigned int lThreadIndex = lLinearBlockIndex * blockDim.x + threadIdx.x;
    if(lThreadIndex >= pOutputLength)
        return;

    pOutput[lThreadIndex] += pInput[lLinearBlockIndex];
}

PREFIX_SUM_DATA_TYPE* prefixSum_computeInternal(PREFIX_SUM_DATA_TYPE* pInput, PREFIX_SUM_DATA_TYPE* pOutput, unsigned int pElems, unsigned int& pOutputElems, unsigned int& pElemsPerBlock)
{
    unsigned int lBlockCount = ((pElems / MAX_ELEMS_PER_BLOCK) + ((pElems % MAX_ELEMS_PER_BLOCK) ? 1 : 0));
    unsigned int lRoundedElems = ((pElems < MAX_ELEMS_PER_BLOCK && isPowOfTwo(pElems) && pElems >= 2) ? pElems : (MAX_ELEMS_PER_BLOCK * lBlockCount));
    unsigned int lElemsPerBlock = (lRoundedElems / lBlockCount);
    unsigned int lThreadsPerBlock = (lElemsPerBlock / 2);
    unsigned int lSharedMem = (lElemsPerBlock + (lElemsPerBlock / NUM_BANKS)) * sizeof(PREFIX_SUM_DATA_TYPE);
    
    PREFIX_SUM_DATA_TYPE* lBlockSumsCudaPtr;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lBlockSumsCudaPtr, lBlockCount * sizeof(PREFIX_SUM_DATA_TYPE)));

    int lBlockCountX = (lBlockCount / 65535) + ((lBlockCount % 65535) ? 1 : 0);
    int lBlockCountY = (lBlockCount < 65535) ? lBlockCount : 65535;
    
    dim3 lGridConf(lBlockCountX, lBlockCountY, 1);
    prefixSum_cuda <<<lGridConf, lThreadsPerBlock, lSharedMem>>> (pInput, pOutput, lElemsPerBlock, pElems, lBlockSumsCudaPtr, lBlockCount);

    if(cudaGetLastError() != cudaSuccess)
    {
        CUDA_ERROR_CHECK("cudaFree", cudaFree(lBlockSumsCudaPtr));
        return NULL;
    }

    pOutputElems = lBlockCount;
    pElemsPerBlock = lElemsPerBlock;
    return lBlockSumsCudaPtr;
}
    
pmStatus prefixSum_compute(PREFIX_SUM_DATA_TYPE* pInput, PREFIX_SUM_DATA_TYPE* pOutput, unsigned int pElems)
{
    unsigned int lBlocksScan1, lElemsPerBlockScan1;

    PREFIX_SUM_DATA_TYPE* lBlockLastsScan1 = prefixSum_computeInternal(pInput, pOutput, pElems, lBlocksScan1, lElemsPerBlockScan1);

    if(!lBlockLastsScan1)
        return pmUserError;

    if(lBlocksScan1 > 1)
    {
        PREFIX_SUM_DATA_TYPE* lBlockIncrsCudaPtr;
        if(cudaMalloc((void**)&lBlockIncrsCudaPtr, lBlocksScan1 * sizeof(PREFIX_SUM_DATA_TYPE)) != cudaSuccess)
        {
            std::cout << "Prefix Sum: CUDA Memory Allocation Failed" << std::endl;
            cudaFree(lBlockLastsScan1);
            return pmUserError;
        }

        if(prefixSum_compute(lBlockLastsScan1, lBlockIncrsCudaPtr, lBlocksScan1) != pmSuccess)
        {
            cudaFree(lBlockLastsScan1);
            cudaFree(lBlockIncrsCudaPtr);
            return pmUserError;
        }

        int lBlockCount = lBlocksScan1 - 1;
        int lBlockCountX = (lBlockCount / 65535) + ((lBlockCount % 65535) ? 1 : 0);
        int lBlockCountY = (lBlockCount < 65535) ? lBlockCount : 65535;
        
        dim3 lGridConf(lBlockCountX, lBlockCountY, 1);
        arrayAdd_cuda <<<lGridConf, lElemsPerBlockScan1>>> (lBlockIncrsCudaPtr, pOutput + lElemsPerBlockScan1, pElems - lElemsPerBlockScan1);

        cudaError_t lError;
        if((lError = cudaGetLastError()) != cudaSuccess)
        {
            cudaFree(lBlockLastsScan1);
            cudaFree(lBlockIncrsCudaPtr);
            std::cout << "Prefix Sum: CUDA Error " << cudaGetErrorString(lError) << std::endl;
        }

        if(cudaFree(lBlockIncrsCudaPtr) != cudaSuccess)
        {
            std::cout << "Prefix Sum: CUDA Memory Deallocation Failed" << std::endl;
            cudaFree(lBlockLastsScan1);
            cudaFree(lBlockIncrsCudaPtr);
            return pmUserError;
        }
    }
    
    if(cudaFree(lBlockLastsScan1) != cudaSuccess)
    {
        std::cout << "Prefix Sum: CUDA Memory Deallocation Failed" << std::endl;
        return pmUserError;
    }
    
    return pmSuccess;
}
    
pmStatus prefixSum_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    unsigned int lElems = (pSubtaskInfo.inputMemLength / sizeof(PREFIX_SUM_DATA_TYPE));

    return prefixSum_compute((PREFIX_SUM_DATA_TYPE*)pSubtaskInfo.inputMem, (PREFIX_SUM_DATA_TYPE*)pSubtaskInfo.outputMem, lElems);
}
    
elemAdd_cudaFuncPtr elemAdd_cudaFunc = elemAdd_cuda;

}

#endif



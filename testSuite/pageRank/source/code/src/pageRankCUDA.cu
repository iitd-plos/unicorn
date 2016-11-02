
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
#include "commonAPI.h"
#include "pageRank.h"

#include <cmath>

namespace pageRank
{
    
__global__ void pageRank_cuda(pageRankTaskConf pTaskConf, unsigned int pWebPages, PAGE_RANK_DATA_TYPE* pLocalArray, PAGE_RANK_DATA_TYPE* pGlobalArray, unsigned int* pSubtaskWebDump)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;

	if(threadId >= pWebPages)
		return;

	unsigned int index = threadId * (1 + pTaskConf.maxOutlinksPerWebPage);
    unsigned int outlinks = pSubtaskWebDump[index++];
    PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((pTaskConf.iteration == 0) ? pTaskConf.initialPageRank : pLocalArray[threadId])/(float)outlinks);

    for(unsigned int k = 0; k < outlinks; ++k)
    {
        unsigned int lRefLink = pSubtaskWebDump[index + k];
        PAGE_RANK_DATA_TYPE* lAddress = (PAGE_RANK_DATA_TYPE*)(pGlobalArray + lRefLink - 1);
        
        atomicAdd(lAddress, lIncr);
    }
}
    
__global__ void zeroInit(PAGE_RANK_DATA_TYPE* pGlobalArray, unsigned int pWebPages)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
    
	if(threadId >= pWebPages)
		return;

    pGlobalArray[threadId] = 0;
}

pmCudaLaunchConf GetCudaLaunchConf(unsigned int pWebPages)
{
    const unsigned int maxThreadsPerBlockDim = 512;
    const unsigned int maxBlocksPerGridDim = 65535;
    pmCudaLaunchConf lCudaLaunchConf;
    
    if(pWebPages > maxThreadsPerBlockDim)
    {
        lCudaLaunchConf.blocksX = pWebPages / maxThreadsPerBlockDim + ((pWebPages % maxThreadsPerBlockDim) ? 1 : 0);
        lCudaLaunchConf.threadsX = maxThreadsPerBlockDim;
    }
    else
    {
        lCudaLaunchConf.blocksX = 1;
        lCudaLaunchConf.threadsX = pWebPages;
    }
    
    lCudaLaunchConf.blocksY = 1;
    
    if(lCudaLaunchConf.blocksX > maxBlocksPerGridDim)
    {
        double lSqrt = std::sqrt((double)lCudaLaunchConf.blocksX);
        unsigned int lFloor = (unsigned int)std::floor(lSqrt);
        unsigned int lCeil = lFloor;

        if(lFloor * lFloor < lCudaLaunchConf.blocksX)
            lCeil = lFloor + 1;
        
        if(lCeil * lFloor < lCudaLaunchConf.blocksX)
            lFloor = lCeil;
        
        if(lCeil * lFloor < lCudaLaunchConf.blocksX)
            exit(1);

        lCudaLaunchConf.blocksX = lCeil;
        lCudaLaunchConf.blocksY = lFloor;
    }
    
    return lCudaLaunchConf;
}
    
pmStatus pageRank_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    ulong lSubtaskId = pSubtaskInfo.subtaskId;
    void** lWebFilePtrs = LoadMappedFiles(lTaskConf, lSubtaskId);
    
    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((lSubtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (lSubtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);
    unsigned int lWebFiles = ((lWebPages / lTaskConf->webPagesPerFile) + ((lWebPages % lTaskConf->webPagesPerFile) ? 1 : 0));
    unsigned int lFirstWebFile = (unsigned int)pSubtaskInfo.subtaskId * lWebFiles;

    unsigned int* lWebDump = (unsigned int*)(pSubtaskInfo.gpuContext.reservedGlobalMem);
    unsigned int* lWebDumpPtr = lWebDump;

    unsigned int lTotalFiles = (lTaskConf->totalWebPages / lTaskConf->webPagesPerFile) + ((lTaskConf->totalWebPages % lTaskConf->webPagesPerFile) ? 1 : 0);
    for(unsigned int i = 0; i < lWebFiles; ++i)
    {
        unsigned int* lMappedFile = (unsigned int*)(lWebFilePtrs[i]);
        
        unsigned int lPagesInFile = lTaskConf->webPagesPerFile;
        if(i + lFirstWebFile == lTotalFiles - 1)
            lPagesInFile = lTaskConf->totalWebPages - (i + lFirstWebFile) * lTaskConf->webPagesPerFile;

        CUDA_ERROR_CHECK("cudaMemcpyAsync", cudaMemcpyAsync(lWebDumpPtr, lMappedFile, sizeof(unsigned int) * lPagesInFile * (lTaskConf->maxOutlinksPerWebPage + 1), cudaMemcpyHostToDevice, (cudaStream_t)pCudaStream));
        lWebDumpPtr += lPagesInFile * (lTaskConf->maxOutlinksPerWebPage + 1);
    }
    
    delete[] lWebFilePtrs;
    
	PAGE_RANK_DATA_TYPE* lLocalArray = ((lTaskConf->iteration == 0) ? NULL : (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr);
    PAGE_RANK_DATA_TYPE* lGlobalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr;

    pmCudaLaunchConf lGlobalCudaLaunchConf = GetCudaLaunchConf(lTaskConf->totalWebPages);
    dim3 globalGridConf(lGlobalCudaLaunchConf.blocksX, lGlobalCudaLaunchConf.blocksY, 1);
    dim3 globalBlockConf(lGlobalCudaLaunchConf.threadsX, 1, 1);

    zeroInit<<<globalGridConf, globalBlockConf>>>(lGlobalArray, lTaskConf->totalWebPages);

    pmCudaLaunchConf lCudaLaunchConf = GetCudaLaunchConf(lWebPages);
    dim3 gridConf(lCudaLaunchConf.blocksX, lCudaLaunchConf.blocksY, 1);
    dim3 blockConf(lCudaLaunchConf.threadsX, 1, 1);

    pageRank_cuda<<<gridConf, blockConf, 0, (cudaStream_t)pCudaStream>>>(*lTaskConf, lWebPages, lLocalArray, lGlobalArray, lWebDump);
    
    return pmSuccess;
}
    
// Returns 0 on success, non-zero on failure
int singleGpuPageRank(pageRankTaskConf& pTaskConf, unsigned int* pWebDump, void* pOutputMem)
{
    unsigned int* lWebDumpCudaPtr = NULL;
    PAGE_RANK_DATA_TYPE* lLocalArrayCudaPtr = NULL;
    PAGE_RANK_DATA_TYPE* lGlobalArrayCudaPtr = NULL;

    size_t lOutputSize = pTaskConf.totalWebPages * sizeof(PAGE_RANK_DATA_TYPE);
    size_t lWebSize = pTaskConf.totalWebPages * (pTaskConf.maxOutlinksPerWebPage + 1) * sizeof(unsigned int);

    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lWebDumpCudaPtr, lWebSize));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lWebDumpCudaPtr, pWebDump, lWebSize, cudaMemcpyHostToDevice));
    
    pmCudaLaunchConf lCudaLaunchConf = GetCudaLaunchConf(pTaskConf.totalWebPages);
    dim3 gridConf(lCudaLaunchConf.blocksX, 1, 1);
    dim3 blockConf(lCudaLaunchConf.threadsX, 1, 1);

    for(unsigned int i = 0; i < PAGE_RANK_ITERATIONS; ++i)
    {
		if(i > 1)
            CUDA_ERROR_CHECK("cudaFree", cudaFree(lLocalArrayCudaPtr));

        lLocalArrayCudaPtr = lGlobalArrayCudaPtr;
        
        CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lGlobalArrayCudaPtr, lOutputSize));

        pTaskConf.iteration = i;
        
        zeroInit<<<gridConf, blockConf>>>(lGlobalArrayCudaPtr, pTaskConf.totalWebPages);
        pageRank_cuda<<<gridConf, blockConf>>>(pTaskConf, pTaskConf.totalWebPages, lLocalArrayCudaPtr, lGlobalArrayCudaPtr, lWebDumpCudaPtr);
    }
    
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(pOutputMem, lGlobalArrayCudaPtr, lOutputSize, cudaMemcpyDeviceToHost));

    if(lLocalArrayCudaPtr)
        CUDA_ERROR_CHECK("cudaFree", cudaFree(lLocalArrayCudaPtr));

    CUDA_ERROR_CHECK("cudaFree", cudaFree(lWebDumpCudaPtr));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lGlobalArrayCudaPtr));

    return 0;
}

}

#endif

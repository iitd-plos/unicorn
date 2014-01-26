
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"
#include "commonAPI.h"
#include "pageRank.h"

namespace pageRank
{
    
__global__ void pageRank_cuda(pageRankTaskConf pTaskConf, unsigned int pWebPages, PAGE_RANK_DATA_TYPE* pLocalArray, PAGE_RANK_DATA_TYPE* pGlobalArray, unsigned int* pSubtaskWebDump)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void pageRank_cuda_key_val_generator(pageRankTaskConf pTaskConf, unsigned int pWebPages, PAGE_RANK_DATA_TYPE* pLocalArray, char* pScratchBuffer, unsigned int* pSubtaskWebDump)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= pWebPages)
		return;

    /* The format of storage for every page (input to this subtask) in scratch buffer is -
     * <No. of outlinks> <Incr> <Key 1> <Key 2> ... <Key (no. of outlinks)>
     */

    // Value for every key here is same (i.e. input page rank / outlinks); so no need to store key value pairs
    size_t lPerPageStorageSize = sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + pTaskConf.maxOutlinksPerWebPage * sizeof(unsigned int);   // no. of outlinks, value (i.e. increment) and keys
    size_t lScratchBufferWriteOffsetInBytes = threadId * lPerPageStorageSize;

	unsigned int index = threadId * (1 + pTaskConf.maxOutlinksPerWebPage);
    unsigned int lOutlinks = pSubtaskWebDump[index++];
    PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((pTaskConf.iteration == 0) ? pTaskConf.initialPageRank : pLocalArray[threadId])/(float)lOutlinks);

    char* lWriteLocation = (char*)pScratchBuffer + lScratchBufferWriteOffsetInBytes;
    
    ((unsigned int*)lWriteLocation)[0] = lOutlinks;
    ((PAGE_RANK_DATA_TYPE*)(lWriteLocation + sizeof(unsigned int)))[0] = lIncr;

    for(unsigned int k = 0; k < lOutlinks; ++k)
        ((unsigned int*)(lWriteLocation + sizeof(unsigned int) + sizeof(PAGE_RANK_DATA_TYPE) + k * sizeof(unsigned int)))[0] = pSubtaskWebDump[index + k] - 1;
}
    
__global__ void zeroInit(PAGE_RANK_DATA_TYPE* pGlobalArray, unsigned int pWebPages)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
	if(threadId >= pWebPages)
		return;

    pGlobalArray[threadId] = (PAGE_RANK_DATA_TYPE)0;
}

pmCudaLaunchConf GetCudaLaunchConf(unsigned int pWebPages)
{
    pmCudaLaunchConf lCudaLaunchConf;
    
    if(pWebPages > 512)
    {
        lCudaLaunchConf.blocksX = pWebPages/512 + ((pWebPages%512) ? 1 : 0);
        lCudaLaunchConf.threadsX = 512;
    }
    else
    {
        lCudaLaunchConf.blocksX = 1;
        lCudaLaunchConf.threadsX = pWebPages;
    }
    
    return lCudaLaunchConf;
}
    
pmStatus pageRank_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    ulong lSubtaskId = pSubtaskInfo.subtaskId;
    void** lWebFilePtrs = LoadMappedFiles(pTaskInfo, pSubtaskInfo);
    
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
    
	PAGE_RANK_DATA_TYPE* lLocalArray = ((lTaskConf->iteration == 0) ? NULL : (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[MEM_INDEX].ptr);

    pmCudaLaunchConf lCudaLaunchConf = GetCudaLaunchConf(lWebPages);
    dim3 gridConf(lCudaLaunchConf.blocksX, 1, 1);
    dim3 blockConf(lCudaLaunchConf.threadsX, 1, 1);
    pageRank_cuda_key_val_generator<<<gridConf, blockConf, 0, (cudaStream_t)pCudaStream>>>(*lTaskConf, lWebPages, lLocalArray, (char*)pSubtaskInfo.gpuContext.scratchBuffer, lWebDump);
    
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


#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"
#include "commonAPI.h"
#include "pageRankInvertedIndex.h"

namespace pageRankInvertedIndex
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
    
__global__ void zeroInit(PAGE_RANK_DATA_TYPE* pGlobalArray, unsigned int pWebPages)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
	if(threadId >= pWebPages)
		return;

    pGlobalArray[threadId] = 0;
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

pmStatus invertIndex_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
    return pmSuccess;
}
    
pmStatus pageRank_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
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

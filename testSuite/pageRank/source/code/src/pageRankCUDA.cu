
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"
#include "commonAPI.h"
#include "pageRank.h"

namespace pageRank
{
    
__global__ void pageRank_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, unsigned int* pSubtaskWebDump)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= lWebPages)
		return;

	PAGE_RANK_DATA_TYPE* lLocalArray = ((lTaskConf->iteration == 0) ? NULL : (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr);
    PAGE_RANK_DATA_TYPE* lGlobalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr;

	unsigned int index = threadId * (1 + lTaskConf->maxOutlinksPerWebPage);
    unsigned int outlinks = pSubtaskWebDump[index++];
    PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((lTaskConf->iteration == 0) ? lTaskConf->initialPageRank : lLocalArray[threadId])/(float)outlinks);

    for(unsigned int k = 0; k < outlinks; ++k)
    {
        unsigned int lRefLink = pSubtaskWebDump[index + k];
        PAGE_RANK_DATA_TYPE* lAddress = (PAGE_RANK_DATA_TYPE*)(lGlobalArray + lRefLink - 1);
        
        atomicAdd(lAddress, lIncr);
    }
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
    
    pmCudaLaunchConf lCudaLaunchConf = GetCudaLaunchConf(lWebPages);

    dim3 gridConf(lCudaLaunchConf.blocksX, 1, 1);
    dim3 blockConf(lCudaLaunchConf.threadsX, 1, 1);
    pageRank_cuda<<<gridConf, blockConf, 0, (cudaStream_t)pCudaStream>>>(pTaskInfo, pSubtaskInfo, lWebDump);
    
    return pmSuccess;
}

}

#endif

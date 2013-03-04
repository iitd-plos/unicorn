
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pageRank.h"

namespace pageRank
{
    
__global__ void pageRank_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);

	PAGE_RANK_DATA_TYPE* lSubtaskWebDump = (PAGE_RANK_DATA_TYPE*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo->deviceHandle, pSubtaskInfo.subtaskId, PRE_SUBTASK_TO_SUBTASK, 0, &pSubtaskInfo.gpuContext);
	PAGE_RANK_DATA_TYPE* lGlobalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.outputMem;
	PAGE_RANK_DATA_TYPE* lLocalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.inputMem;

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= lWebPages)
		return;

	unsigned int index = threadId * (1 + lTaskConf->maxOutlinksPerWebPage);

	unsigned int outlinks = lSubtaskWebDump[index++];
    PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((lTaskConf->iteration == 0) ? lTaskConf->initialPageRank : lLocalArray[threadId])/(float)outlinks);
    
	for(unsigned int k=0; k<outlinks; ++k)
	{
		unsigned int lRefLink = lSubtaskWebDump[index+k];

		PAGE_RANK_DATA_TYPE* lAddress = (PAGE_RANK_DATA_TYPE*)(lGlobalArray + lRefLink - 1);

		atomicAdd(lAddress, lIncr);
	}
    
    *pStatus = pmSuccess;
}

pageRank_cudaFuncPtr pageRank_cudaFunc = pageRank_cuda;

}

#endif

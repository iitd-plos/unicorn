
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "pageRank.h"

namespace pageRank
{
    
__global__ void pageRank_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	pageRankTaskConf* lTaskConf = (pageRankTaskConf*)(pTaskInfo.taskConf);
    
	PAGE_RANK_DATA_TYPE* lSubtaskWebDump = (PAGE_RANK_DATA_TYPE*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo->deviceHandle, pSubtaskInfo.subtaskId, PRE_SUBTASK_TO_SUBTASK, 0, &pSubtaskInfo.gpuContext);
	PAGE_RANK_DATA_TYPE* lLocalArray = (PAGE_RANK_DATA_TYPE*)pSubtaskInfo.inputMem;

    unsigned int lWebPages = (unsigned int)((lTaskConf->totalWebPages < ((pSubtaskInfo.subtaskId + 1) * lTaskConf->webPagesPerSubtask)) ? (lTaskConf->totalWebPages - (pSubtaskInfo.subtaskId * lTaskConf->webPagesPerSubtask)) : lTaskConf->webPagesPerSubtask);

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= lWebPages)
		return;

	unsigned int index = threadId * (1 + lTaskConf->maxOutlinksPerWebPage);
    void* lOutputBuffer = (void*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo->deviceHandle, pSubtaskInfo.subtaskId, SUBTASK_TO_POST_SUBTASK, 0, &pSubtaskInfo.gpuContext);
    
    unsigned int* lOutlinksCountBuffer = (unsigned int*)lOutputBuffer;
    unsigned int* lKeyBuffer = lOutlinksCountBuffer + lWebPages;
    PAGE_RANK_DATA_TYPE* lValueBuffer = (PAGE_RANK_DATA_TYPE*)(lKeyBuffer + (lTaskConf->maxOutlinksPerWebPage * lWebPages));

	unsigned int outlinks = lSubtaskWebDump[index++];
    PAGE_RANK_DATA_TYPE lIncr = (PAGE_RANK_DATA_TYPE)(DAMPENING_FACTOR * ((lTaskConf->iteration == 0) ? lTaskConf->initialPageRank : lLocalArray[threadId])/(float)outlinks);

    lOutlinksCountBuffer[threadId] = outlinks;
	for(unsigned int k = 0; k < outlinks; ++k)
	{
        lKeyBuffer[k * lTaskConf->maxOutlinksPerWebPage + threadId] = lSubtaskWebDump[index + k] - 1;
        lValueBuffer[k * lTaskConf->maxOutlinksPerWebPage + threadId] = lIncr;
	}

    // Make change to library to prevent output mem section from being copied back to CPU
    
    *pStatus = pmSuccess;
}

pageRank_cudaFuncPtr pageRank_cudaFunc = pageRank_cuda;

}

#endif

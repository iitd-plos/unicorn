
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "imageFiltering.h"

namespace imageFiltering
{

__global__ void imageFiltering_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
    // Each thread computes one element of resulting matrix by accumulating results into value

    imageFilteringTaskConf* lTaskConf = (imageFilteringTaskConf*)(pTaskInfo.taskConf);

    *pStatus = pmSuccess;
}

imageFiltering_cudaFuncPtr imageFiltering_cudaFunc = imageFiltering_cuda;

}

#endif
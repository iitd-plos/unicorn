
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "radixSort.h"

__global__ void radixSort_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus)
{
	return pmSuccess;
}

#endif

namespace prefixSum
{
    
#define PREFIX_SUM_DATA_TYPE int
#define ELEMS_ADD_PER_CUDA_THREAD 4

const int DEFAULT_ARRAY_LENGTH = 10000000;
const int ELEMS_PER_SUBTASK = 1000000;

using namespace pm;
    
#ifdef BUILD_CUDA
#include <cuda.h>

pmStatus prefixSum_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

typedef void (*elemAdd_cudaFuncPtr)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
extern elemAdd_cudaFuncPtr elemAdd_cudaFunc;
#endif

typedef struct prefixSumTaskConf
{
	unsigned int arrayLen;
} prefixSumTaskConf;
    
}
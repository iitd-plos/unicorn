
namespace prefixSum
{
    
#define PREFIX_SUM_DATA_TYPE int
#define ELEMS_ADD_PER_CUDA_THREAD 4

const int DEFAULT_ARRAY_LENGTH = 10000000;
const int ELEMS_PER_SUBTASK = 1000000;

#ifdef BUILD_CUDA
void prefixSum_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct prefixSumTaskConf
{
	unsigned int arrayLen;
} prefixSumTaskConf;
    
}

const int DEFAULT_ARRAY_LENGTH = 10000;
const int ELEMS_PER_SUBTASK = 100;

const int BITS_PER_ROUND = 4;
const int TOTAL_BITS = (sizeof(int)*8);
const int TOTAL_ROUNDS = (TOTAL_BITS/BITS_PER_ROUND);

#ifdef BUILD_CUDA
void radixSort_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct radixSortTaskConf
{
	int arrayLen;
	bool firstRoundSortFromMsb;
} radixSortTaskConf;
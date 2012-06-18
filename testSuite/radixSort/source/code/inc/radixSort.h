
#define DATA_TYPE unsigned int

const int DEFAULT_ARRAY_LENGTH = 100000;
const int ELEMS_PER_SUBTASK = 1;    //000;

const int BITS_PER_ROUND = 4;
const int TOTAL_BITS = (sizeof(DATA_TYPE)*8);
const int TOTAL_ROUNDS = 1; //(TOTAL_BITS/BITS_PER_ROUND);

const int BINS_COUNT = (1 << BITS_PER_ROUND);

#ifdef BUILD_CUDA
void radixSort_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct radixSortTaskConf
{
	unsigned int arrayLen;
	bool sortFromMsb;
    int round;
} radixSortTaskConf;
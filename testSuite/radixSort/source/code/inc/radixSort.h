
#define DEFAULT_ARRAY_LENGTH 10000
#define ELEMS_PER_SUBTASK 100

#define BITS_PER_ROUND 4
#define TOTAL_BITS (sizeof(int)*8)
#define TOTAL_ROUNDS (TOTAL_BITS/4)

pmStatus radixSort_cuda(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);

typedef struct radixSortTaskConf
{
	int arrayLen;
	bool fisrtRoundSortFromMsb;
} radixSortTaskConf;

#include <iostream>
#include <time.h>

#include "commonAPI.h"

#include "radixSort.h"
#include <math.h>

int* gSampleInput;
int* gSerialOutput;
int* gParallelOutput;

void serialRadixSortRound(int* pInputArray, int* pOutputArray, int pCount, int pRightShiftBits, int pBitmask)
{
	int i;

	int lBinsCount = (1 << BITS_PER_ROUND);
	int* lBins = new int[lBinsCount];

	memset(lBins, 0, lBinsCount * sizeof(int));

	for(i=0; i<pCount; ++i)
		++lBins[((pInputArray[i] >> pRightShiftBits) & pBitmask)];

	for(i=1; i<lBinsCount; ++i)
		lBins[i] += lBins[i-1];

	for(i=pCount-1; i>=0; --i)
		pOutputArray[--lBins[((pInputArray[i] >> pRightShiftBits) & pBitmask)]] = pInputArray[i];

	delete[] lBins;
}

void serialMsbRadixSortRound(int* pInputArray, int* pOutputArray, int pCount, int pRound)
{
	int lRightShiftBits = TOTAL_BITS - ((pRound + 1) * BITS_PER_ROUND);
	int lBitmask = ~(0xffffffff << BITS_PER_ROUND);

	serialRadixSortRound(pInputArray, pOutputArray, pCount, lRightShiftBits, lBitmask);
}

void serialLsbRadixSortRound(int* pInputArray, int* pOutputArray, int pCount, int pRound)
{
	int lRightShiftBits = pRound * BITS_PER_ROUND;
	int lBitmask = ~(0xffffffff << BITS_PER_ROUND);

	serialRadixSortRound(pInputArray, pOutputArray, pCount, lRightShiftBits, lBitmask);
}

void serialLsbRadixSort(int* pInputArray, int* pOutputArray, int pCount)
{
	for(int i=0; i<TOTAL_ROUNDS; ++i)
		serialLsbRadixSortRound(pInputArray, pOutputArray, pCount, i);
}

pmStatus radixSortDataDistribution(pmTaskInfo pTaskInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

	if(lTaskConf->fisrtRoundSortFromMsb == 0)
	{
		lSubscriptionInfo.offset = pSubtaskId * ELEMS_PER_SUBTASK * sizeof(int);
		lSubscriptionInfo.blockLength = ELEMS_PER_SUBTASK * sizeof(int);

		pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, INPUT_MEM, lSubscriptionInfo);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, OUTPUT_MEM, lSubscriptionInfo);
	}
	else
	{
	}

	return pmSuccess;
}

pmStatus radixSort_cpu(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);
	
	if(lTaskConf->fisrtRoundSortFromMsb == 0)
	{
		serialMsbRadixSortRound((int*)(pSubtaskInfo.inputMem), (int*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(int), 0);
	}
	else
	{
		for(int i=0; i<TOTAL_ROUNDS-1; ++i)
			serialLsbRadixSortRound((int*)(pSubtaskInfo.inputMem), (int*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(int), i);
	}

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
	int lArrayLength = DEFAULT_ARRAY_LENGTH; \
	FETCH_INT_ARG(lArrayLength, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS;

	double lStartTime = getCurrentTimeInSecs();

	serialLsbRadixSort(gSampleInput, gSerialOutput, lArrayLength);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

pmMemHandle* FirstRoundParallelSortFromMsb(size_t pInputMemSize, size_t pOutputMemSize, int pArrayLength, pmCallbacks pCallbacks)
{
	CREATE_TASK(pInputMemSize, pOutputMemSize, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK != 0)?1:0), "KEY", pCallbacks)

	memcpy(lTaskDetails.inputMem, gSampleInput, pInputMemSize);

	radixSortTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
	lTaskConf.fisrtRoundSortFromMsb = true;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, lTaskHandle) );
	SAFE_PM_EXEC( pmWaitForTaskCompletion(lTaskHandle) );

	pmReleaseTask(lTaskHandle);
	pmReleaseCallbacks(lTaskDetails.callbackHandle);
	pmReleaseMemory(lTaskDetails.inputMem);

	return lTaskDetails.outputMem;
}

pmStatus PostFirstRoundParallelSortFromLsb(pmMemHandle* pInputMem, size_t pInputMemSize, size_t pOutputMemSize, int pArrayLength, pmCallbacks pCallbacks)
{
	CREATE_TASK(0, pOutputMemSize, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK != 0)?1:0), "KEY", pCallbacks)

	lTaskDetails.inputMem = pInputMem;

	radixSortTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
	lTaskConf.fisrtRoundSortFromMsb = false;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, lTaskHandle) );
	SAFE_PM_EXEC( pmWaitForTaskCompletion(lTaskHandle) );

	memcpy(gParallelOutput, lTaskDetails.outputMem, pOutputMemSize);

	FREE_TASK_AND_RESOURCES

	return pmSuccess;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbacks pCallbacks)
{
	READ_NON_COMMON_ARGS;

	// Input Mem contains input unsorted array
	// Output Mem contains final sorted array
	// Number of subtasks is arrayLength/ELEMS_PER_SUBTASK if arrayLength is divisible by ELEMS_PER_SUBTASK; otherwise arrayLength/ELEMS_PER_SUBTASK + 1
	size_t lInputMemSize = lArrayLength * sizeof(int);
	size_t lOutputMemSize = lInputMemSize;

	double lStartTime = getCurrentTimeInSecs();

	// MSB sort in first round
	pmMemHandle* pResultMemHandle = FirstRoundParallelSortFromMsb(lInputMemSize, lOutputMemSize, lArrayLength, pCallbacks);

	// LSB sort for all remaining rounds
	PostFirstRoundParallelSortFromLsb(pResultMemHandle, lInputMemSize, lOutputMemSize, lArrayLength, pCallbacks);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = radixSortDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = radixSort_cpu;
	lCallbacks.subtask_gpu_cuda = radixSort_cuda;

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	srand(time(NULL));

	gSampleInput = new int[lArrayLength];
	gSerialOutput = new int[lArrayLength];
	gParallelOutput = new int[lArrayLength];

	for(int i=0; i<lArrayLength; ++i)
		gSampleInput[i] = (int)rand();

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
	delete[] gSampleInput;
	delete[] gSerialOutput;
	delete[] gParallelOutput;

	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	for(int i=0; i<lArrayLength; ++i)
	{
		if(gSerialOutput[i] != gParallelOutput[i])
		{
			std::cout << "Mismatch index " << i << " Serial Value = " << gSerialOutput[i] << " Parallel Value = " << gParallelOutput[i] << std::endl;
			return 1;
		}
	}

	return 0;
}

/** Non-common args
 *	1. Array Length
 */
void main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy);

	commonFinish();
}



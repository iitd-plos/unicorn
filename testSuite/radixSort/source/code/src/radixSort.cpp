
#include <iostream>
#include <time.h>

#include "commonAPI.h"

#include "radixSort.h"
#include <math.h>
#include <string.h>

DATA_TYPE* gSampleInput;
DATA_TYPE* gSerialOutput;
DATA_TYPE* gParallelOutput;

#define BIT_MASK (~((~((DATA_TYPE)0)) << BITS_PER_ROUND))
#define LSB_RIGHT_SHIFT_BITS(round) (round * BITS_PER_ROUND)
#define MSB_RIGHT_SHIFT_BITS(round) (TOTAL_BITS - ((round + 1) * BITS_PER_ROUND))
#define RIGHT_SHIFT_BITS(isMsb, round) (isMsb ? MSB_RIGHT_SHIFT_BITS(round) : LSB_RIGHT_SHIFT_BITS(round))

void serialRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int* pBins, int pRightShiftBits, DATA_TYPE pBitmask)
{    
	int lBins[BINS_COUNT] = {0};
	for(unsigned i=0; i<pCount; ++i)
		++lBins[((pInputArray[i] >> pRightShiftBits) & pBitmask)];

	if(pBins)
		memcpy(pBins, lBins, sizeof(lBins));

	for(int k=1; k<BINS_COUNT; ++k)
		lBins[k] += lBins[k-1];

	for(int j=(int)pCount-1; j>=0; --j)
		pOutputArray[--lBins[((pInputArray[j] >> pRightShiftBits) & pBitmask)]] = pInputArray[j];
}

void serialMsbRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRound, int* pBins)
{
	serialRadixSortRound(pInputArray, pOutputArray, pCount, pBins, RIGHT_SHIFT_BITS(true, pRound), BIT_MASK);
}

void serialLsbRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRound, int* pBins)
{
	serialRadixSortRound(pInputArray, pOutputArray, pCount, pBins, RIGHT_SHIFT_BITS(false, pRound), BIT_MASK);
}

void serialLsbRadixSort(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRounds)
{
	DATA_TYPE* lTempArray = NULL;

	if(pRounds > 1)
		lTempArray = new DATA_TYPE[pCount];

	for(int i=0; i<pRounds; ++i)
	{
		DATA_TYPE* lInputArray = NULL;
		DATA_TYPE* lOutputArray = NULL;

		if(i == 0)
			lInputArray = pInputArray;
		else if(i%2 == 0)
			lInputArray = pOutputArray;
		else
			lInputArray = lTempArray;

		if(i == 0 && pRounds == 1)
			lOutputArray = pOutputArray;
		else if(i%2 == 0)
			lOutputArray = lTempArray;
		else
			lOutputArray = pOutputArray;

		serialLsbRadixSortRound(lInputArray, lOutputArray, pCount, i, NULL);

		if(i == pRounds-1 && lOutputArray == lTempArray)
			memcpy(pOutputArray, lTempArray, pCount * sizeof(DATA_TYPE));
	}

	delete[] lTempArray;
}

pmStatus radixSortDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

	if(!lTaskConf->sortFromMsb || lTaskConf->round == 0)
	{
		lSubscriptionInfo.offset = pSubtaskId * ELEMS_PER_SUBTASK * sizeof(DATA_TYPE);
		lSubscriptionInfo.length = (lTaskConf->arrayLen < (pSubtaskId+1)*ELEMS_PER_SUBTASK) ? ((lTaskConf->arrayLen - (pSubtaskId * ELEMS_PER_SUBTASK)) * sizeof(DATA_TYPE))  : (ELEMS_PER_SUBTASK * sizeof(DATA_TYPE));

		pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);
	}
	else
	{
	}

	return pmSuccess;
}

pmStatus radixSort_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

	int* lBins = (int*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, sizeof(int) * BINS_COUNT);

	if(lTaskConf->sortFromMsb)
		serialMsbRadixSortRound((DATA_TYPE*)(pSubtaskInfo.inputMem), (DATA_TYPE*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(DATA_TYPE), lTaskConf->round, lBins);
	else
		serialLsbRadixSortRound((DATA_TYPE*)(pSubtaskInfo.inputMem), (DATA_TYPE*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(DATA_TYPE), lTaskConf->round, lBins);

	return pmSuccess;
}

pmStatus radixSortDataRedistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	int* lBins = (int*)pmGetScratchBuffer(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, 0);

	size_t lOffset = 0;
	for(int k=0; k<BINS_COUNT; ++k)
	{
		pmRedistributeData(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, lOffset, lBins[k] * sizeof(DATA_TYPE), (unsigned int)(pTaskInfo.subtaskCount * k + pSubtaskInfo.subtaskId));
		lOffset += lBins[k] * sizeof(DATA_TYPE);
	}

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
unsigned int lArrayLength = DEFAULT_ARRAY_LENGTH; \
FETCH_INT_ARG(lArrayLength, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS;

	double lStartTime = getCurrentTimeInSecs();

	serialLsbRadixSort(gSampleInput, gSerialOutput, lArrayLength, TOTAL_ROUNDS);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

bool ParallelSort(bool pMsbSort, pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, unsigned int pArrayLength, int pRound, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	CREATE_TASK(0, 0, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK)?1:0), pCallbackHandle, pSchedulingPolicy)

	lTaskDetails.inputMemHandle = pInputMemHandle;
	lTaskDetails.outputMemHandle = pOutputMemHandle;

	radixSortTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
	lTaskConf.sortFromMsb = pMsbSort;
	lTaskConf.round = pRound;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
	{
		FREE_TASK_AND_RESOURCES
		return false;
	}

	pmReleaseTask(lTaskHandle);

	return true;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS;

	// Input Mem contains input unsorted array
	// Output Mem contains final sorted array
	// Number of subtasks is arrayLength/ELEMS_PER_SUBTASK if arrayLength is divisible by ELEMS_PER_SUBTASK; otherwise arrayLength/ELEMS_PER_SUBTASK + 1
	pmMemHandle lInputMemHandle, lOutputMemHandle;
	size_t lMemSize = lArrayLength * sizeof(DATA_TYPE);

	CREATE_MEM(lMemSize, lInputMemHandle);
	CREATE_MEM(lMemSize, lOutputMemHandle);

	pmRawMemPtr lRawInputPtr, lRawOutputPtr;
	pmGetRawMemPtr(lInputMemHandle, &lRawInputPtr);    
	memcpy(lRawInputPtr, gSampleInput, lMemSize);

	double lStartTime = getCurrentTimeInSecs();

	for(int i=0; i<TOTAL_ROUNDS; ++i)
	{
		if(i != 0)
		{
			pmMemHandle lTempMemHandle = lInputMemHandle;
			lInputMemHandle = lOutputMemHandle;
			lOutputMemHandle = lTempMemHandle;
		}

		if(!ParallelSort(false, lInputMemHandle, lOutputMemHandle, lArrayLength, i, pCallbackHandle, pSchedulingPolicy))
			return (double)-1.0;
	}

	double lEndTime = getCurrentTimeInSecs();

	SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );

	pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);

	memcpy(gParallelOutput, lRawOutputPtr, lMemSize);

	pmReleaseMemory(lInputMemHandle);
	pmReleaseMemory(lOutputMemHandle);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = radixSortDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = radixSort_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = radixSort_cuda;
#endif

	lCallbacks.dataRedistribution = radixSortDataRedistribution;

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	srand(time(NULL));

	gSampleInput = new DATA_TYPE[lArrayLength];
	gSerialOutput = new DATA_TYPE[lArrayLength];
	gParallelOutput = new DATA_TYPE[lArrayLength];

	for(unsigned int i=0; i<lArrayLength; ++i)
		gSampleInput[i] =  lArrayLength - i; //(DATA_TYPE)rand();

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

	/*
	std::cout << "Sample Input ... " << std::endl;
	for(unsigned int j=0; j<lArrayLength; ++j)
	std::cout << gSampleInput[j] << " ";
	std::cout << std::endl << std::endl;

	std::cout << "Serial Output ... " << std::endl;
	for(unsigned int k=0; k<lArrayLength; ++k)
	std::cout << gSerialOutput[k] << " ";
	std::cout << std::endl << std::endl;

	std::cout << "Parallel Output ... " << std::endl;
	for(unsigned int l=0; l<lArrayLength; ++l)
	std::cout << gParallelOutput[l] << " ";
	std::cout << std::endl << std::endl;
	*/

	for(unsigned int i=0; i<lArrayLength; ++i)
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
int main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "RADIXSORT");

	commonFinish();

	return 0;
}



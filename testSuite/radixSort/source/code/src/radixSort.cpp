
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

void serialLsbRadixSort(int* pInputArray, int* pOutputArray, int pCount, int pRounds)
{
    int* lTempArray = NULL;
    
    if(pRounds > 1)
        lTempArray = new int[pCount];
        
	for(int i=0; i<pRounds; ++i)
    {
        int* lInputArray = NULL;
        int* lOutputArray = NULL;
        
        if(i == 0)
            lInputArray = pInputArray;
        else if(i%2 == 0)
            lInputArray = pOutputArray;
        else
            lInputArray = lTempArray;
        
        if(i%2 == 0)
            lOutputArray = lTempArray;
        else
            lOutputArray = pOutputArray;
            
		serialLsbRadixSortRound(lInputArray, lOutputArray, pCount, i);
        
        if(i == pRounds-1 && lOutputArray == lTempArray)
            memcpy(pOutputArray, lTempArray, pCount * sizeof(int));
    }
               
   delete[] lTempArray;
}

pmStatus radixSortDataDistribution(pmTaskInfo pTaskInfo, unsigned long pSubtaskId, pmDeviceTypes pDeviceType)
{
	pmSubscriptionInfo lSubscriptionInfo;
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

	if(lTaskConf->firstRoundSortFromMsb)
	{
		lSubscriptionInfo.offset = pSubtaskId * ELEMS_PER_SUBTASK * sizeof(int);
		lSubscriptionInfo.length = (lTaskConf->arrayLen < (pSubtaskId+1)*ELEMS_PER_SUBTASK) ? ((lTaskConf->arrayLen - (pSubtaskId * ELEMS_PER_SUBTASK)) * sizeof(int))  : (ELEMS_PER_SUBTASK * sizeof(int));

		pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, true, lSubscriptionInfo);
		pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, false, lSubscriptionInfo);
	}
	else
	{
	}

	return pmSuccess;
}

pmStatus radixSort_cpu(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);
	
	if(lTaskConf->firstRoundSortFromMsb)
		serialMsbRadixSortRound((int*)(pSubtaskInfo.inputMem), (int*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(int), 0);
	else
        serialLsbRadixSort((int*)(pSubtaskInfo.inputMem), (int*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(int), TOTAL_ROUNDS-1);

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

	serialLsbRadixSort(gSampleInput, gSerialOutput, lArrayLength, TOTAL_ROUNDS);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

pmMemHandle FirstRoundParallelSortFromMsb(size_t pInputMemSize, size_t pOutputMemSize, int pArrayLength, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	CREATE_TASK(pInputMemSize, pOutputMemSize, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK != 0)?1:0), pCallbackHandle, pSchedulingPolicy)

	memcpy(lTaskDetails.inputMem, gSampleInput, pInputMemSize);

	radixSortTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
	lTaskConf.firstRoundSortFromMsb = true;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
        return NULL;

	pmReleaseTask(lTaskHandle);
	pmReleaseCallbacks(lTaskDetails.callbackHandle);
	pmReleaseMemory(lTaskDetails.inputMem);

	return lTaskDetails.outputMem;
}

bool PostFirstRoundParallelSortFromLsb(pmMemHandle pInputMem, size_t pInputMemSize, size_t pOutputMemSize, int pArrayLength, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	CREATE_TASK(0, pOutputMemSize, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK != 0)?1:0), pCallbackHandle, pSchedulingPolicy)

	lTaskDetails.inputMem = pInputMem;

	radixSortTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
	lTaskConf.firstRoundSortFromMsb = false;
	lTaskDetails.taskConf = (void*)(&lTaskConf);
	lTaskDetails.taskConfLength = sizeof(lTaskConf);

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
    if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
        return false;

	SAFE_PM_EXEC( pmFetchMemory(lTaskDetails.outputMem) );

	memcpy(gParallelOutput, lTaskDetails.outputMem, pOutputMemSize);

	FREE_TASK_AND_RESOURCES

	return true;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS;

	// Input Mem contains input unsorted array
	// Output Mem contains final sorted array
	// Number of subtasks is arrayLength/ELEMS_PER_SUBTASK if arrayLength is divisible by ELEMS_PER_SUBTASK; otherwise arrayLength/ELEMS_PER_SUBTASK + 1
	size_t lInputMemSize = lArrayLength * sizeof(int);
	size_t lOutputMemSize = lInputMemSize;

	double lStartTime = getCurrentTimeInSecs();

	// MSB sort in first round
	pmMemHandle pResultMemHandle = FirstRoundParallelSortFromMsb(lInputMemSize, lOutputMemSize, lArrayLength, pCallbackHandle, pSchedulingPolicy);
    
    if(!pResultMemHandle)
        return (double)-1.0;

	// LSB sort for all remaining rounds
	if(!PostFirstRoundParallelSortFromLsb(pResultMemHandle, lInputMemSize, lOutputMemSize, lArrayLength, pCallbackHandle, pSchedulingPolicy))
        return (double)-1.0;

	double lEndTime = getCurrentTimeInSecs();

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
    
    //lCallbacks.dataReduction = ;
    //lCallbacks.dataScatter = ;

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
		gSampleInput[i] =  lArrayLength - i; //(int)rand();

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

    std::cout << "Sample Input ... " << std::endl;
    for(int j=0; j<lArrayLength; ++j)
        std::cout << gSampleInput[j] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "Serial Output ... " << std::endl;
    for(int k=0; k<lArrayLength; ++k)
        std::cout << gSerialOutput[k] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "Parallel Output ... " << std::endl;
    for(int l=0; l<lArrayLength; ++l)
        std::cout << gParallelOutput[l] << " ";
    std::cout << std::endl << std::endl;

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
int main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "RADIXSORT");

	commonFinish();
    
    return 0;
}



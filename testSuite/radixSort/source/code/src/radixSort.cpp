
#include <iostream>
#include <time.h>

#include "commonAPI.h"

#include "radixSort.h"
#include <math.h>

DATA_TYPE* gSampleInput;
DATA_TYPE* gSerialOutput;
DATA_TYPE* gParallelOutput;

#define BIT_MASK (~((~((DATA_TYPE)0)) << BITS_PER_ROUND))
#define LSB_RIGHT_SHIFT_BITS(round) (round * BITS_PER_ROUND)
#define MSB_RIGHT_SHIFT_BITS(round) (TOTAL_BITS - ((round + 1) * BITS_PER_ROUND))
#define RIGHT_SHIFT_BITS(isMsb, round) (isMsb ? MSB_RIGHT_SHIFT_BITS(round) : LSB_RIGHT_SHIFT_BITS(round))

void serialRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRightShiftBits, DATA_TYPE pBitmask)
{    
	int lBins[BINS_COUNT] = {0};
	for(unsigned i=0; i<pCount; ++i)
		++lBins[((pInputArray[i] >> pRightShiftBits) & pBitmask)];

	for(int k=1; k<BINS_COUNT; ++k)
		lBins[k] += lBins[k-1];

	for(int j=(int)pCount-1; j>=0; --j)
		pOutputArray[--lBins[((pInputArray[j] >> pRightShiftBits) & pBitmask)]] = pInputArray[j];
}

void serialMsbRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRound)
{
	serialRadixSortRound(pInputArray, pOutputArray, pCount, RIGHT_SHIFT_BITS(true, pRound), BIT_MASK);
}

void serialLsbRadixSortRound(DATA_TYPE* pInputArray, DATA_TYPE* pOutputArray, unsigned int pCount, int pRound)
{
	serialRadixSortRound(pInputArray, pOutputArray, pCount, RIGHT_SHIFT_BITS(false, pRound), BIT_MASK);
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
            
		serialLsbRadixSortRound(lInputArray, lOutputArray, pCount, i);
        
        if(i == pRounds-1 && lOutputArray == lTempArray)
            memcpy(pOutputArray, lTempArray, pCount * sizeof(DATA_TYPE));
    }
               
   delete[] lTempArray;
}

pmStatus radixSortDataDistribution(pmTaskInfo pTaskInfo, unsigned long pSubtaskId, pmDeviceTypes pDeviceType)
{
	pmSubscriptionInfo lSubscriptionInfo;
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

	if(!lTaskConf->sortFromMsb || lTaskConf->round == 0)
	{
		lSubscriptionInfo.offset = pSubtaskId * ELEMS_PER_SUBTASK * sizeof(DATA_TYPE);
		lSubscriptionInfo.length = (lTaskConf->arrayLen < (pSubtaskId+1)*ELEMS_PER_SUBTASK) ? ((lTaskConf->arrayLen - (pSubtaskId * ELEMS_PER_SUBTASK)) * sizeof(DATA_TYPE))  : (ELEMS_PER_SUBTASK * sizeof(DATA_TYPE));

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
	
	if(lTaskConf->sortFromMsb)
		serialMsbRadixSortRound((DATA_TYPE*)(pSubtaskInfo.inputMem), (DATA_TYPE*)(pSubtaskInfo.outputMem), lTaskConf->arrayLen, lTaskConf->round);
	else
        serialLsbRadixSortRound((DATA_TYPE*)(pSubtaskInfo.inputMem), (DATA_TYPE*)(pSubtaskInfo.outputMem), (pSubtaskInfo.inputMemLength)/sizeof(DATA_TYPE), lTaskConf->round);
    
	return pmSuccess;
}

pmStatus radixSortDataRedistribution(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	radixSortTaskConf* lTaskConf = (radixSortTaskConf*)(pTaskInfo.taskConf);

    DATA_TYPE lBitmask = BIT_MASK;
    int lRightShiftBits = RIGHT_SHIFT_BITS(lTaskConf->sortFromMsb, lTaskConf->round);
    DATA_TYPE* lOutputArray = (DATA_TYPE*)(pSubtaskInfo.outputMem);
    unsigned int lCount = (pSubtaskInfo.inputMemLength)/sizeof(DATA_TYPE);
                                               
	int lBins[BINS_COUNT]= {0};
	for(unsigned int i=0; i<lCount; ++i)
		++lBins[((lOutputArray[i] >> lRightShiftBits) & lBitmask)];

    size_t lOffset = 0;
	for(int k=0; k<BINS_COUNT; ++k)
    {
        pmRedistributeData(pTaskInfo.taskHandle, pSubtaskInfo.subtaskId, lOffset, lBins[k] * sizeof(DATA_TYPE), k);
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

pmMemHandle ParallelSort(bool pMsbSort, pmMemHandle pInputMem, unsigned int pArrayLength, int pRound, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    size_t lMemSize = pArrayLength * sizeof(DATA_TYPE);
	CREATE_TASK(0, lMemSize, (pArrayLength/ELEMS_PER_SUBTASK) + ((pArrayLength%ELEMS_PER_SUBTASK)?1:0), pCallbackHandle, pSchedulingPolicy)

	lTaskDetails.inputMem = pInputMem;

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
        return NULL;
    }

	pmReleaseTask(lTaskHandle);
	pmReleaseMemory(lTaskDetails.inputMem);
    
	return lTaskDetails.outputMem;
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS;

	// Input Mem contains input unsorted array
	// Output Mem contains final sorted array
	// Number of subtasks is arrayLength/ELEMS_PER_SUBTASK if arrayLength is divisible by ELEMS_PER_SUBTASK; otherwise arrayLength/ELEMS_PER_SUBTASK + 1
    pmMemHandle lInputMem;
	size_t lMemSize = lArrayLength * sizeof(DATA_TYPE);
    
	double lStartTime = getCurrentTimeInSecs();

    CREATE_INPUT_MEM(lMemSize, lInputMem);
    memcpy(lInputMem, gSampleInput, lMemSize);

    for(int i=0; i<TOTAL_ROUNDS; ++i)
    {
        pmMemHandle lOutputMem;
        if((lOutputMem = ParallelSort(false, lInputMem, lArrayLength, i, pCallbackHandle, pSchedulingPolicy)) == NULL)
            return (double)-1.0;
        
        lInputMem = lOutputMem;
    }

	SAFE_PM_EXEC( pmFetchMemory(lInputMem) );
    
	memcpy(gParallelOutput, lInputMem, lMemSize);
    
	pmReleaseMemory(lInputMem);

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



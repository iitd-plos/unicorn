
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "prefixSum.h"

namespace prefixSum
{
    
PREFIX_SUM_DATA_TYPE* gSampleInput;
PREFIX_SUM_DATA_TYPE* gSerialOutput;
PREFIX_SUM_DATA_TYPE* gParallelOutput;

void serialPrefixSum(PREFIX_SUM_DATA_TYPE* pInputArray, PREFIX_SUM_DATA_TYPE* pOutputArray, unsigned int pLength)
{
    pOutputArray[0] = pInputArray[0];

    for(unsigned int i=1; i<pLength; ++i)
        pOutputArray[i] = pInputArray[i] + pOutputArray[i-1];
}

#ifdef BUILD_CUDA
pmCudaLaunchConf elemAddGetCudaLaunchConf(int pSubtaskElems)
{
	pmCudaLaunchConf lCudaLaunchConf;

	int lMaxThreadsPerBlock = 512;

    int lFactor = pSubtaskElems / ELEMS_ADD_PER_CUDA_THREAD;
	lCudaLaunchConf.threadsX = lFactor;
	if(lCudaLaunchConf.threadsX > lMaxThreadsPerBlock)
	{
		lCudaLaunchConf.threadsX = lMaxThreadsPerBlock;
		lCudaLaunchConf.blocksX = lFactor/lCudaLaunchConf.threadsX;
        if(lCudaLaunchConf.blocksX * lCudaLaunchConf.threadsX < lFactor)
            ++lCudaLaunchConf.blocksX;
	}

	return lCudaLaunchConf;
}
#endif

pmStatus prefixSumDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	prefixSumTaskConf* lTaskConf = (prefixSumTaskConf*)(pTaskInfo.taskConf);

    unsigned long lSubtaskElems = (lTaskConf->arrayLen < (pSubtaskId+1)*ELEMS_PER_SUBTASK) ? ((lTaskConf->arrayLen - (pSubtaskId * ELEMS_PER_SUBTASK)))  : (ELEMS_PER_SUBTASK);

    lSubscriptionInfo.offset = pSubtaskId * ELEMS_PER_SUBTASK * sizeof(PREFIX_SUM_DATA_TYPE);
    lSubscriptionInfo.length = lSubtaskElems * sizeof(PREFIX_SUM_DATA_TYPE);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);
    
	return pmSuccess;
}

pmStatus prefixSum_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
    serialPrefixSum((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.inputMem), (PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.outputMem), (unsigned int)((pSubtaskInfo.inputMemLength)/sizeof(PREFIX_SUM_DATA_TYPE)));

	return pmSuccess;
}

pmStatus elemAddDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	prefixSumTaskConf* lTaskConf = (prefixSumTaskConf*)(pTaskInfo.taskConf);

    lSubscriptionInfo.offset = pSubtaskId * sizeof(PREFIX_SUM_DATA_TYPE);
    lSubscriptionInfo.length = sizeof(PREFIX_SUM_DATA_TYPE);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);

    unsigned long lSubtaskElems = (lTaskConf->arrayLen < (pSubtaskId+2)*ELEMS_PER_SUBTASK) ? ((lTaskConf->arrayLen - ((pSubtaskId+1) * ELEMS_PER_SUBTASK)))  : (ELEMS_PER_SUBTASK);
    
    lSubscriptionInfo.offset = (pSubtaskId+1) * ELEMS_PER_SUBTASK * sizeof(PREFIX_SUM_DATA_TYPE);
    lSubscriptionInfo.length = lSubtaskElems * sizeof(PREFIX_SUM_DATA_TYPE);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

#ifdef BUILD_CUDA
	if(pDeviceInfo.deviceType == pm::GPU_CUDA)
		pmSetCudaLaunchConf(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, elemAddGetCudaLaunchConf(lSubtaskElems));
#endif

	return pmSuccess;
}

pmStatus elemAdd_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
    PREFIX_SUM_DATA_TYPE lElem = ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.inputMem))[0];
    
    unsigned int lLength = (unsigned int)((pSubtaskInfo.outputMemLength)/sizeof(PREFIX_SUM_DATA_TYPE));
    for(unsigned int i=0; i<lLength; ++i)
        ((PREFIX_SUM_DATA_TYPE*)(pSubtaskInfo.outputMem))[i] += lElem;

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
    unsigned int lArrayLength = DEFAULT_ARRAY_LENGTH; \
    FETCH_INT_ARG(lArrayLength, pCommonArgs, argc, argv);

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

	double lStartTime = getCurrentTimeInSecs();

	serialPrefixSum(gSampleInput, gSerialOutput, lArrayLength);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

	return 0;
#else
    return 0;
#endif
}

bool ParallelPrefixSum(pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, unsigned int pArrayLength, unsigned int pSubtaskCount, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	CREATE_TASK(0, 0, pSubtaskCount, pCallbackHandle, pSchedulingPolicy)

	lTaskDetails.inputMemHandle = pInputMemHandle;
	lTaskDetails.outputMemHandle = pOutputMemHandle;

	prefixSumTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
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
    
void PopulateAuxMem(pmMemHandle pSrcMemHandle, pmMemHandle pAuxMemHandle, unsigned int pSrcArrayLength, unsigned int pAuxArrayLength)
{
    unsigned int lSrcSize = pSrcArrayLength * sizeof(PREFIX_SUM_DATA_TYPE);
    
	pmRawMemPtr lRawSrcPtr, lRawAuxPtr;
	pmGetRawMemPtr(pSrcMemHandle, &lRawSrcPtr);
	pmGetRawMemPtr(pAuxMemHandle, &lRawAuxPtr);
    
    char* lDestPtr = (char*)lRawAuxPtr;

    unsigned int lSrcStep = ELEMS_PER_SUBTASK * sizeof(PREFIX_SUM_DATA_TYPE);
    for(unsigned int i = (lSrcStep - sizeof(PREFIX_SUM_DATA_TYPE)); i < lSrcSize; i += lSrcStep)
    {
        SAFE_PM_EXEC( pmFetchMemoryRange(pSrcMemHandle, i, sizeof(PREFIX_SUM_DATA_TYPE)) );
        memcpy((void*)lDestPtr, (void*)((char*)lRawSrcPtr + i), sizeof(PREFIX_SUM_DATA_TYPE));
        lDestPtr += sizeof(PREFIX_SUM_DATA_TYPE);
    }
    
    if(pSrcArrayLength % ELEMS_PER_SUBTASK != 0)
    {
        unsigned int lOffset = (pSrcArrayLength - 1) * sizeof(PREFIX_SUM_DATA_TYPE);
        SAFE_PM_EXEC( pmFetchMemoryRange(pSrcMemHandle, lOffset, sizeof(PREFIX_SUM_DATA_TYPE)) );
        memcpy((void*)lDestPtr, (void*)((char*)lRawSrcPtr + lOffset), sizeof(PREFIX_SUM_DATA_TYPE));
    }

    serialPrefixSum((PREFIX_SUM_DATA_TYPE*)lRawAuxPtr, (PREFIX_SUM_DATA_TYPE*)lRawAuxPtr, pAuxArrayLength);
}

bool ParallelAddAuxArray(pmMemHandle pSrcMemHandle, pmMemHandle pAuxMemHandle, unsigned int pArrayLength, unsigned int pSubtaskCount, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
	CREATE_TASK(0, 0, pSubtaskCount - 1, pCallbackHandle, pSchedulingPolicy)

	lTaskDetails.inputMemHandle = pAuxMemHandle;
	lTaskDetails.outputMemHandle = pSrcMemHandle;
    lTaskDetails.inputMemInfo = INPUT_MEM_READ_ONLY;
    lTaskDetails.outputMemInfo = OUTPUT_MEM_READ_WRITE;
    lTaskDetails.disjointReadWritesAcrossSubtasks = true;

	prefixSumTaskConf lTaskConf;
	lTaskConf.arrayLen = pArrayLength;
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
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS;

	// Input Mem contains input array
	// Output Mem contains final array with prefix sums
	// Number of subtasks is arrayLength/ELEMS_PER_SUBTASK if arrayLength is divisible by ELEMS_PER_SUBTASK; otherwise arrayLength/ELEMS_PER_SUBTASK + 1
	pmMemHandle lInputMemHandle, lOutputMemHandle, lAuxMemHandle = NULL;
	size_t lMemSize = lArrayLength * sizeof(PREFIX_SUM_DATA_TYPE);

	CREATE_MEM(lMemSize, lInputMemHandle);
	CREATE_MEM(lMemSize, lOutputMemHandle);

	pmRawMemPtr lRawInputPtr, lRawOutputPtr;
	pmGetRawMemPtr(lInputMemHandle, &lRawInputPtr);    
	memcpy(lRawInputPtr, gSampleInput, lMemSize);

	double lStartTime = getCurrentTimeInSecs();

    unsigned int lSubtaskCount = (lArrayLength/ELEMS_PER_SUBTASK) + ((lArrayLength%ELEMS_PER_SUBTASK)?1:0);
    if(!ParallelPrefixSum(lInputMemHandle, lOutputMemHandle, lArrayLength, lSubtaskCount, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;

    if(lSubtaskCount > 1)
    {
        CREATE_MEM(lSubtaskCount * sizeof(PREFIX_SUM_DATA_TYPE), lAuxMemHandle);
        PopulateAuxMem(lOutputMemHandle, lAuxMemHandle, lArrayLength, lSubtaskCount);

        if(!ParallelAddAuxArray(lOutputMemHandle, lAuxMemHandle, lArrayLength, lSubtaskCount, pCallbackHandle[1], pSchedulingPolicy))
            return (double)-1.0;
    }
    
	double lEndTime = getCurrentTimeInSecs();

    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );

        pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    }

    if(lAuxMemHandle)
        pmReleaseMemory(lAuxMemHandle);

	pmReleaseMemory(lInputMemHandle);
	pmReleaseMemory(lOutputMemHandle);

	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = prefixSumDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = prefixSum_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = prefixSum_cudaLaunchFunc;
#endif

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks2()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = elemAddDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = elemAdd_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_cuda = elemAdd_cudaFunc;
#endif

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
	srand((unsigned int)time(NULL));
    
	gSampleInput = new PREFIX_SUM_DATA_TYPE[lArrayLength];
	gSerialOutput = new PREFIX_SUM_DATA_TYPE[lArrayLength];
	gParallelOutput = new PREFIX_SUM_DATA_TYPE[lArrayLength];
    
	for(unsigned int i = 0; i < lArrayLength; ++i)
		gSampleInput[i] = (PREFIX_SUM_DATA_TYPE)rand();   //lArrayLength - i;
    
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

	for(unsigned int i = 0; i < lArrayLength; ++i)
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
    callbackStruct lStruct[2] = { {DoSetDefaultCallbacks, "PREFIXSUM"}, {DoSetDefaultCallbacks2, "ELEMADD"} };

	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 2);

	commonFinish();

	return 0;
}

}

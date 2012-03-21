
#include <iostream>

#include "commonAPI.h"

//#include "<sample>.h"

pmStatus sampleDataDistribution(pmTaskInfo pTaskInfo, unsigned long pSubtaskId)
{
	//pmSubscriptionInfo lSubscriptionInfo;
	//sampleTaskConf* lTaskConf = (sampleTaskConf*)(pTaskInfo.taskConf);

	// Subscribe to 
	//lSubscriptionInfo.offset = ;
	//lSubscriptionInfo.blockLength = ;
	//pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, INPUT_MEM, lSubscriptionInfo);

	// Subscribe to 
	//lSubscriptionInfo.offset = ;
	//lSubscriptionInfo.blockLength = ;
	//pmSubscribeToMemory(pTaskInfo.taskHandle, pSubtaskId, OUTPUT_MEM, lSubscriptionInfo);

	return pmSuccess;
}

pmStatus sample_cpu(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	//sampleTaskConf* lTaskConf = (sampleTaskConf*)(pTaskInfo.taskConf);

	return pmSuccess;
}

//#define READ_NON_COMMON_ARGS \
// \
//

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	//READ_NON_COMMON_ARGS;

	double lStartTime = getCurrentTimeInSecs();

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}

// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbacks pCallbacks)
{
	double lExecTime = (double)0;

	//READ_NON_COMMON_ARGS;

	// Input Mem contains
	// Output Mem contains
	// Number of subtasks 
	//size_t lInputMemSize = ;
	//size_t lOutputMemSize = ;

	// CREATE_TASK(lInputMemSize, lOutputMemSize, totalSubtasks, "KEY", pCallbacks)

	//sampleTaskConf lTaskConf;
	//lTaskConf. = ;
	//lTaskDetails.taskConf = (void*)(&lTaskConf);
	//lTaskDetails.taskConfLength = sizeof(lTaskConf);

	// SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, lTaskHandle) );
	// SAFE_PM_EXEC( pmGetTaskExecutionTimeInSecs(lTaskHandle, &lExecTime) );

	// FREE_TASK_AND_RESOURCES

	return lExecTime;
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	//lCallbacks.dataDistribution = sampleDataDistribution;
	//lCallbacks.deviceSelection = NULL;
	//lCallbacks.subtask_cpu = sample_cpu;
	//lCallbacks.subtask_gpu_cuda = sample_cuda;

	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	//READ_NON_COMMON_ARGS

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	//READ_NON_COMMON_ARGS

	return 0;
}

/** Non-common args
 */
void main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy);

	commonFinish();
}



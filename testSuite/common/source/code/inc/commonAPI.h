
#include <iostream>
#include <stdlib.h>

#include "pmPublicDefinitions.h"

using namespace pm;

bool localDeviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo);
double getCurrentTimeInSecs();

#define SUBMITTING_HOST_ID 0
#define LOCAL_DEVICE_SELECTION_CALLBACK localDeviceSelectionCallback

#define SAFE_PM_EXEC(x) \
{ \
	pmStatus dExecStatus = x; \
	if(dExecStatus != pmSuccess) \
	{ \
		std::cout << "PM Command Exited with status " << dExecStatus << std::endl; \
		exit(dExecStatus); \
	} \
}

#define FETCH_INT_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = atoi(argArray[argIndex+1]); }

typedef double (*serialProcessFunc)(int argc, char** argv, int pCommonArgs);
typedef double (*parallelProcessFunc)(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy);
typedef pmCallbacks (*callbacksFunc)();
typedef int (*initFunc)(int argc, char** argv, int pCommonArgs);
typedef int (*destroyFunc)();
typedef int (*compareFunc)(int argc, char** argv, int pCommonArgs);

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc, 
	callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey);

void commonFinish();

#define CREATE_INPUT_MEM(inputMemSize, lInputMem) SAFE_PM_EXEC( pmCreateMemory(INPUT_MEM_READ_ONLY, inputMemSize, &lInputMem) )
#define CREATE_OUTPUT_MEM(outputMemSize, lOutputMem) SAFE_PM_EXEC( pmCreateMemory(OUTPUT_MEM_READ_WRITE, outputMemSize, &lOutputMem) )

#define CREATE_TASK(inputMemSize, outputMemSize, totalSubtasks, cbHandle, schedPolicy) \
	pmTaskHandle lTaskHandle; \
	pmMemHandle lInputMemHandle; \
	pmMemHandle lOutputMemHandle; \
	pmTaskDetails lTaskDetails; \
	if(inputMemSize) \
		CREATE_INPUT_MEM(inputMemSize, lInputMemHandle); \
	if(outputMemSize) \
		CREATE_OUTPUT_MEM(outputMemSize, lOutputMemHandle); \
	lTaskDetails.inputMemHandle = lInputMemHandle; \
	lTaskDetails.outputMemHandle = lOutputMemHandle; \
	lTaskDetails.callbackHandle = cbHandle; \
	lTaskDetails.subtaskCount = totalSubtasks; \
    lTaskDetails.policy = schedPolicy; \
    lTaskDetails.autoFetchOutputMem = false;

#define FREE_TASK_AND_RESOURCES \
	SAFE_PM_EXEC( pmReleaseTask(lTaskHandle) ); \
	SAFE_PM_EXEC( pmReleaseMemory(lTaskDetails.inputMemHandle) ); \
	SAFE_PM_EXEC( pmReleaseMemory(lTaskDetails.outputMemHandle) );




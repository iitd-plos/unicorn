
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

#define FETCH_INT_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex < totalArgs) argName = atoi(argArray[argIndex]); }

typedef double (*serialProcessFunc)(int argc, char** argv, int pCommonArgs);
typedef double (*parallelProcessFunc)(int argc, char** argv, int pCommonArgs, pmCallbacks pCallbacks);
typedef pmCallbacks (*callbacksFunc)();
typedef int (*initFunc)(int argc, char** argv, int pCommonArgs);
typedef int (*destroyFunc)();
typedef int (*compareFunc)(int argc, char** argv, int pCommonArgs);

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc, 
	callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc);

void commonFinish();

#define CREATE_TASK(inputMemSize, outputMemSize, totalSubtasks, key, callbacks) \
	pmTaskHandle* lTaskHandle = NULL; \
	pmMemHandle* lInputMem = NULL; \
	pmMemHandle* lOutputMem = NULL; \
	pmCallbackHandle* lCallbackHandle = NULL; \
	pmTaskDetails lTaskDetails; \
	if(inputMemSize) \
		SAFE_PM_EXEC( pmCreateMemory(INPUT_MEM_READ_ONLY, inputMemSize, lInputMem) ); \
	if(outputMemSize) \
		SAFE_PM_EXEC( pmCreateMemory(OUTPUT_MEM_WRITE_ONLY, outputMemSize, lOutputMem) ); \
	SAFE_PM_EXEC( pmRegisterCallbacks(key, callbacks, lCallbackHandle) ); \
	lTaskDetails.inputMem = lInputMem; \
	lTaskDetails.outputMem = lOutputMem; \
	lTaskDetails.callbackHandle = lCallbackHandle; \
	lTaskDetails.subtaskCount = totalSubtasks;

#define FREE_TASK_AND_RESOURCES SAFE_PM_EXEC( pmReleaseTaskAndResources(lTaskDetails, lTaskHandle) );




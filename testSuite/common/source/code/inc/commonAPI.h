
#include <iostream>
#include <stdlib.h>

#include "pmPublicDefinitions.h"
#include "pmPublicUtilities.h"

using namespace pm;

typedef struct complex
{
	float x;
	float y;
} complex;

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
		commonFinish(); \
		exit(dExecStatus); \
	} \
}

#define FETCH_INT_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = atoi(argArray[argIndex+1]); }
#define FETCH_STR_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = argArray[argIndex+1]; }

typedef double (*serialProcessFunc)(int argc, char** argv, int pCommonArgs);
typedef double (*parallelProcessFunc)(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy);
typedef double (*parallelProcessFunc2)(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle1, pmCallbackHandle pCallbackHandle2, pmSchedulingPolicy pSchedulingPolicy);
typedef pmCallbacks (*callbacksFunc)();
typedef int (*initFunc)(int argc, char** argv, int pCommonArgs);
typedef int (*destroyFunc)();
typedef int (*compareFunc)(int argc, char** argv, int pCommonArgs);

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc, 
	callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey);

void commonStart2(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc2 pParallelFunc,
                    callbacksFunc pCallbacksFunc1, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey1,
                    callbacksFunc pCallbacksFunc2, std::string pCallbackKey2);

void commonFinish();

bool isMultiAssignEnabled();    /* by default, it's enabled */
bool isLazyMemEnabled();    /* by default, it's diabled */

#define CREATE_MEM(memSize, memHandle) SAFE_PM_EXEC( pmCreateMemory(memSize, &memHandle) )

#define CREATE_TASK(inputMemSize, outputMemSize, totalSubtasks, cbHandle, schedPolicy) \
	pmTaskHandle lTaskHandle = NULL; \
	pmMemHandle lInputMemHandle = NULL; \
	pmMemHandle lOutputMemHandle = NULL; \
	pmTaskDetails lTaskDetails; \
	if(inputMemSize) \
		CREATE_MEM(inputMemSize, lInputMemHandle); \
	if(outputMemSize) \
		CREATE_MEM(outputMemSize, lOutputMemHandle); \
	lTaskDetails.inputMemHandle = lInputMemHandle; \
	lTaskDetails.outputMemHandle = lOutputMemHandle; \
	lTaskDetails.inputMemInfo = INPUT_MEM_READ_ONLY; \
	lTaskDetails.outputMemInfo = OUTPUT_MEM_WRITE_ONLY; \
	lTaskDetails.callbackHandle = cbHandle; \
	lTaskDetails.subtaskCount = totalSubtasks; \
	lTaskDetails.policy = schedPolicy; \
	lTaskDetails.multiAssignEnabled = isMultiAssignEnabled();

#define FREE_TASK_AND_RESOURCES \
	SAFE_PM_EXEC( pmReleaseTask(lTaskHandle) ); \
    if(lTaskDetails.inputMemHandle) \
        SAFE_PM_EXEC( pmReleaseMemory(lTaskDetails.inputMemHandle) ); \
    if(lTaskDetails.outputMemHandle) \
        SAFE_PM_EXEC( pmReleaseMemory(lTaskDetails.outputMemHandle) );




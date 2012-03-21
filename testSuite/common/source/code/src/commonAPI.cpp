
#include <iostream>
#include <sys/time.h>

#include "commonAPI.h"

double ExecuteParallelTask(int argc, char** argv, int pParallelMode, parallelProcessFunc pParallelFunc, callbacksFunc pCallbacksFunc)
{
	pmCallbacks lCallbacks = pCallbacksFunc();

	switch(pParallelMode)
	{
		case 1:
		case 4:
			lCallbacks.subtask_gpu_cuda = NULL;
			break;
		case 2:
		case 5:
			lCallbacks.subtask_cpu = NULL;
			break;
	}

	if(pParallelMode <= 3)
		lCallbacks.deviceSelection = LOCAL_DEVICE_SELECTION_CALLBACK;

	return pParallelFunc(argc, argv, COMMON_ARGS, lCallbacks);
}

/** Common Arguments:
 *	1. Run Mode - [0: Don't compare to serial execution; 1: Compare to serial execution (default); 2: Only run serial]
 *	2. Parallel Task Mode - [0: All; 1: Local CPU; 2: Local GPU; 3: Local CPU + GPU; 4: Global CPU; 5: Global GPU; 6: Global CPU + GPU (default)]
 */
#define COMMON_ARGS 2
#define DEFAULT_RUN_MODE 1
#define DEFAULT_PARALLEL_MODE 6
void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc, 
	callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc)
{
	double lSerialExecTime = (double)0;
	double lParallelExecTime = (double)0;

	int lRunMode = DEFAULT_RUN_MODE;
	int lParallelMode = DEFAULT_PARALLEL_MODE;

	FETCH_INT_ARG(lRunMode, 0, argc, argv);
	FETCH_INT_ARG(lParallelMode, 1, argc, argv);

	if(lRunMode < 0 || lRunMode > 2)
		lRunMode = DEFAULT_RUN_MODE;

	if(lParallelMode < 1 || lParallelMode > 6)
		lParallelMode = DEFAULT_PARALLEL_MODE;

	SAFE_PM_EXEC( pmInitialize() );

	if(pmGetHostId() == SUBMITTING_HOST_ID)
	{
		if(pInitFunc(argc, argv, COMMON_ARGS) != 0)
		{
			std::cout << "Initialization Error" << std::endl;
			exit(1);
		}

		if(lRunMode != 0)
		{
			lSerialExecTime = pSerialFunc(argc, argv, COMMON_ARGS);
			std::cout << "Serial Task Execution Time = " << lSerialExecTime << std::endl;
		}

		if(lRunMode != 2)
		{
			// Six Parallel Execution Modes
			for(int i=1; i<=6; ++i)
			{
				if(lParallelMode == 0 || lParallelMode == i)
				{
					lParallelExecTime = ExecuteParallelTask(argc, argv, lParallelMode, pParallelFunc, pCallbacksFunc);
					std::cout << "Parallel Task " << lParallelMode << " Execution Time = " << lParallelExecTime << std::endl;

					if(lRunMode == 1)
					{
						if(pCompareFunc(argc, argv, COMMON_ARGS))
							std::cout << "Parallel Task " << i << " Test Failed" << std::endl;
						else
							std::cout << "Parallel Task " << i << " Test Passed" << std::endl;
					}
				}
			}
		}

		if(pDestroyFunc() != 0)
		{
			std::cout << "Destruction Error" << std::endl;
			exit(1);
		}
	}
}

void commonFinish()
{
	SAFE_PM_EXEC( pmFinalize() );
}

bool localDeviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo)
{
	if(pDeviceInfo.host != SUBMITTING_HOST_ID)
		return false;

	return true;
}

double getCurrentTimeInSecs()
{
	struct timeval lTimeVal;
	struct timezone lTimeZone;

	::gettimeofday(&lTimeVal, &lTimeZone);

	double lCurrentTime = ((double)(lTimeVal.tv_sec * 1000000 + lTimeVal.tv_usec))/1000000;

	return lCurrentTime;
}






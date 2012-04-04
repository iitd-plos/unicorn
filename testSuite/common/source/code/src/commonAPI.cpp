
#include <iostream>
#include <sys/time.h>
#include <string>

#include "commonAPI.h"

/** Common Arguments:
 *	1. Run Mode - [0: Don't compare to serial execution; 1: Compare to serial execution (default); 2: Only run serial]
 *	2. Parallel Task Mode - [0: All; 1: Local CPU; 2: Local GPU; 3: Local CPU + GPU; 4: Global CPU; 5: Global GPU; 6: Global CPU + GPU (default)]
 *  3. Scheduling Policy - [0: Push; 1: Pull]
 */
#define COMMON_ARGS 3
#define DEFAULT_RUN_MODE 1
#define DEFAULT_PARALLEL_MODE 6
#define DEFAULT_SCHEDULING_POLICY 0

pmCallbackHandle gCallbackHandleArray[6];

double ExecuteParallelTask(int argc, char** argv, int pParallelMode, parallelProcessFunc pParallelFunc, pmSchedulingPolicy pSchedulingPolicy)
{
	return pParallelFunc(argc, argv, COMMON_ARGS, gCallbackHandleArray[pParallelMode-1], pSchedulingPolicy);
}

void RegisterLibraryCallback(int pParallelMode, std::string pCallbackKey, pmCallbacks pCallbacks)
{
	switch(pParallelMode)
	{
		case 1:
		case 4:
			pCallbacks.subtask_gpu_cuda = NULL;
			break;
		case 2:
		case 5:
			pCallbacks.subtask_cpu = NULL;
			break;
	}
    
	if(pParallelMode <= 3)
		pCallbacks.deviceSelection = LOCAL_DEVICE_SELECTION_CALLBACK;    

    static char* lArray[6] = {"1", "2", "3", "4", "5", "6"};

    std::string lTempKey;
    lTempKey = pCallbackKey + lArray[pParallelMode-1];

    SAFE_PM_EXEC( pmRegisterCallbacks((char*)lTempKey.c_str(), pCallbacks, &gCallbackHandleArray[pParallelMode-1]) );
}

void RegisterLibraryCallbacks(std::string pCallbackKey, callbacksFunc pCallbacksFunc)
{
	pmCallbacks lCallbacks = pCallbacksFunc();

    RegisterLibraryCallback(1, pCallbackKey, lCallbacks);
    RegisterLibraryCallback(2, pCallbackKey, lCallbacks);
    RegisterLibraryCallback(3, pCallbackKey, lCallbacks);
    RegisterLibraryCallback(4, pCallbackKey, lCallbacks);
    RegisterLibraryCallback(5, pCallbackKey, lCallbacks);
    RegisterLibraryCallback(6, pCallbackKey, lCallbacks);
}

void ReleaseLibraryCallbacks()
{
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[0]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[1]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[2]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[3]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[4]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray[5]) );
}

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc, 
                 callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey)
{
	double lSerialExecTime = (double)0;
	double lParallelExecTime = (double)0;

	int lRunMode = DEFAULT_RUN_MODE;
	int lParallelMode = DEFAULT_PARALLEL_MODE;
	int lSchedulingPolicy = DEFAULT_SCHEDULING_POLICY;

	FETCH_INT_ARG(lRunMode, 0, argc, argv);
	FETCH_INT_ARG(lParallelMode, 1, argc, argv);
	FETCH_INT_ARG(lSchedulingPolicy, 2, argc, argv);

	if(lRunMode < 0 || lRunMode > 2)
		lRunMode = DEFAULT_RUN_MODE;

	if(lParallelMode < 0 || lParallelMode > 6)
		lParallelMode = DEFAULT_PARALLEL_MODE;

	if(lSchedulingPolicy < 0 || lSchedulingPolicy > 1)
		lSchedulingPolicy = DEFAULT_SCHEDULING_POLICY;

	SAFE_PM_EXEC( pmInitialize() );
    
    if(lRunMode != 2)
        RegisterLibraryCallbacks(pCallbackKey, pCallbacksFunc);

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
					lParallelExecTime = ExecuteParallelTask(argc, argv, i, pParallelFunc, (lSchedulingPolicy == 0)?SLOW_START:RANDOM_STEAL);
                    
                    if(lParallelExecTime < 0.0)
                    {
                        std::cout << "Parallel Task " << i << " Failed" << std::endl;                        
                    }
                    else
                    {
                        std::cout << "Parallel Task " << i << " Execution Time = " << lParallelExecTime << std::endl;

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

            ReleaseLibraryCallbacks();
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






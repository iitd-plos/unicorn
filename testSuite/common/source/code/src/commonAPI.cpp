
#include <iostream>
#include <sys/time.h>
#include <string>

#include "commonAPI.h"

/** Common Arguments:
 *	1. Run Mode - [0: Don't compare to serial execution; 1: Compare to serial execution (default); 2: Only run serial]
 *	2. Parallel Task Mode - [0: All; 1: Local CPU; 2: Local GPU; 3: Local CPU + GPU; 4: Global CPU; 5: Global GPU; 6: Global CPU + GPU (default); 7: (4, 5, 6)]
 *	3. Scheduling Policy - [0: Push (default); 1: Pull; 2: Equal_Static; 3: Proportional_Static]
 */
#define COMMON_ARGS 3
#define DEFAULT_RUN_MODE 1
#define DEFAULT_PARALLEL_MODE 6
#define DEFAULT_SCHEDULING_POLICY 0

pmCallbackHandle gCallbackHandleArray1[6];
pmCallbackHandle gCallbackHandleArray2[6];
int gCallbackHandleArrayCount;

double ExecuteParallelTask(int argc, char** argv, int pParallelMode, void* pParallelFunc, pmSchedulingPolicy pSchedulingPolicy)
{
    if(gCallbackHandleArrayCount == 1)
        return (*((parallelProcessFunc*)pParallelFunc))(argc, argv, COMMON_ARGS, gCallbackHandleArray1[pParallelMode-1], pSchedulingPolicy);
    
    return (*((parallelProcessFunc2*)pParallelFunc))(argc, argv, COMMON_ARGS, gCallbackHandleArray1[pParallelMode-1], gCallbackHandleArray2[pParallelMode-1], pSchedulingPolicy);
}

void RegisterLibraryCallback(int pParallelMode, std::string pCallbackKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle)
{
	switch(pParallelMode)
	{
		case 1:
		case 4:
            pCallbacks.subtask_gpu_cuda = NULL;
            pCallbacks.subtask_gpu_custom = NULL;
			break;
		case 2:
		case 5:
			pCallbacks.subtask_cpu = NULL;
			break;
	}
    
	if(pParallelMode <= 3)
		pCallbacks.deviceSelection = LOCAL_DEVICE_SELECTION_CALLBACK;    

    static const char* lArray[6] = {"1", "2", "3", "4", "5", "6"};

    std::string lTempKey;
    lTempKey = pCallbackKey + lArray[pParallelMode-1];

    SAFE_PM_EXEC( pmRegisterCallbacks((char*)lTempKey.c_str(), pCallbacks, &pCallbackHandle[pParallelMode-1]) );
}

void RegisterLibraryCallbacks(std::string pCallbackKey1, callbacksFunc pCallbacksFunc1, std::string pCallbackKey2, callbacksFunc pCallbacksFunc2)
{
	pmCallbacks lCallbacks1 = pCallbacksFunc1();

    RegisterLibraryCallback(1, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);
    RegisterLibraryCallback(2, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);
    RegisterLibraryCallback(3, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);
    RegisterLibraryCallback(4, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);
    RegisterLibraryCallback(5, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);
    RegisterLibraryCallback(6, pCallbackKey1, lCallbacks1, gCallbackHandleArray1);

    if(pCallbacksFunc2)
    {
        pmCallbacks lCallbacks2 = pCallbacksFunc2();

        RegisterLibraryCallback(1, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
        RegisterLibraryCallback(2, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
        RegisterLibraryCallback(3, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
        RegisterLibraryCallback(4, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
        RegisterLibraryCallback(5, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
        RegisterLibraryCallback(6, pCallbackKey2, lCallbacks2, gCallbackHandleArray2);
    }
    
    gCallbackHandleArrayCount = pCallbacksFunc2 ? 2 : 1;
}

void ReleaseLibraryCallbacks()
{
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[0]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[1]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[2]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[3]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[4]) );
    SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray1[5]) );

    if(gCallbackHandleArrayCount == 2)
    {
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[0]) );
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[1]) );
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[2]) );
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[3]) );
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[4]) );
        SAFE_PM_EXEC( pmReleaseCallbacks(gCallbackHandleArray2[5]) );
    }
}

void commonStartInternal(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, void* pParallelFunc,
                    callbacksFunc pCallbacksFunc1, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey1,
                    callbacksFunc pCallbacksFunc2, std::string pCallbackKey2)
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

	if(lParallelMode < 0 || lParallelMode > 7)
		lParallelMode = DEFAULT_PARALLEL_MODE;

	if(lSchedulingPolicy < 0 || lSchedulingPolicy > 3)
		lSchedulingPolicy = DEFAULT_SCHEDULING_POLICY;

	SAFE_PM_EXEC( pmInitialize() );
    
    if(lRunMode != 2)
        RegisterLibraryCallbacks(pCallbackKey1, pCallbacksFunc1, pCallbackKey2, pCallbacksFunc2);

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
			for(int i = 1; i <= 6; ++i)
			{
				if(lParallelMode == 0 || lParallelMode == i || (lParallelMode == 7 && (i == 4 || i == 5 || i == 6)))
				{
					pmSchedulingPolicy lPolicy = SLOW_START;
					if(lSchedulingPolicy == 1)
						lPolicy = RANDOM_STEAL;
					else if(lSchedulingPolicy == 2)
						lPolicy = EQUAL_STATIC;
					else if(lSchedulingPolicy == 3)
						lPolicy = PROPORTIONAL_STATIC;

                    lParallelExecTime = ExecuteParallelTask(argc, argv, i, pParallelFunc, lPolicy);
                
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

void commonStart2(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc2 pParallelFunc,
                    callbacksFunc pCallbacksFunc1, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey1,
                    callbacksFunc pCallbacksFunc2, std::string pCallbackKey2)
{
    commonStartInternal(argc, argv, pInitFunc, pSerialFunc, &pParallelFunc, pCallbacksFunc1, pCompareFunc, pDestroyFunc, pCallbackKey1, pCallbacksFunc2, pCallbackKey2);
}

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, parallelProcessFunc pParallelFunc,
                 callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey)
{
    commonStartInternal(argc, argv, pInitFunc, pSerialFunc, &pParallelFunc, pCallbacksFunc, pCompareFunc, pDestroyFunc, pCallbackKey, NULL, std::string());
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

#ifdef SAMPLE_NAME
namespace SAMPLE_NAME
{
    int main(int argc, char** argv);
}

int main(int argc, char** argv)
{
    return SAMPLE_NAME::main(argc, argv);
}
#else
#error "SAMPLE_NAME not defined"
#endif





#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <string>
#include <vector>

#include "commonAPI.h"

/** Common Arguments:
 *	1. Run Mode - [0: Don't compare to sequential execution; 1: Compare to sequential execution (default); 2: Only run sequential; 3: Only run single GPU; 4: Compare Single GPU to sequential]
 *	2. Parallel Task Mode - [0: All; 1: Local CPU; 2: Local GPU; 3: Local CPU + GPU; 4: Global CPU; 5: Global GPU; 6: Global CPU + GPU (default); 7: (4, 5, 6)]
 *	3. Scheduling Policy - [0: Push (default); 1: Pull; 2: Equal_Static; 3: Proportional_Static, 4: All]
 */
#define COMMON_ARGS 3
#define DEFAULT_RUN_MODE 1
#define DEFAULT_PARALLEL_MODE 6
#define DEFAULT_SCHEDULING_POLICY 0

pmCallbackHandle gCallbackHandleArray1[6];
pmCallbackHandle gCallbackHandleArray2[6];
int gCallbackHandleArrayCount;

struct Result
{
    size_t schedulingPolicy;
    size_t parallelMode;
    double execTime;
    bool serialComparisonResult;
};

std::vector<Result> gResultVector;
bool gSerialResultsCompared;
preSetupPostMpiInitFunc gPreSetupPostMpiInitFunc = NULL;

double ExecuteParallelTask(int argc, char** argv, int pParallelMode, void* pParallelFunc, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
    if(gCallbackHandleArrayCount == 1)
        return (*((parallelProcessFunc*)pParallelFunc))(argc, argv, COMMON_ARGS, gCallbackHandleArray1[pParallelMode-1], pSchedulingPolicy, pFetchBack);
    
    return (*((parallelProcessFunc2*)pParallelFunc))(argc, argv, COMMON_ARGS, gCallbackHandleArray1[pParallelMode-1], gCallbackHandleArray2[pParallelMode-1], pSchedulingPolicy, pFetchBack);
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

void commonStartInternal(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, singleGpuProcessFunc pSingleGpuFunc,
                    void* pParallelFunc, callbacksFunc pCallbacksFunc1, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey1,
                    callbacksFunc pCallbacksFunc2, std::string pCallbackKey2)
{
	double lSerialExecTime = (double)0;
	double lSingleGpuExecTime = (double)0;

	int lRunMode = DEFAULT_RUN_MODE;
	int lParallelMode = DEFAULT_PARALLEL_MODE;
	int lSchedulingPolicy = DEFAULT_SCHEDULING_POLICY;

	FETCH_INT_ARG(lRunMode, 0, argc, argv);
	FETCH_INT_ARG(lParallelMode, 1, argc, argv);
	FETCH_INT_ARG(lSchedulingPolicy, 2, argc, argv);

#ifdef BUILD_CUDA
	if(lRunMode < 0 || lRunMode > 4)
		lRunMode = DEFAULT_RUN_MODE;
#else
	if(lRunMode < 0 || lRunMode > 2)
		lRunMode = DEFAULT_RUN_MODE;
#endif

	if(lParallelMode < 0 || lParallelMode > 7)
		lParallelMode = DEFAULT_PARALLEL_MODE;

	if(lSchedulingPolicy < 0 || lSchedulingPolicy > 4)
		lSchedulingPolicy = DEFAULT_SCHEDULING_POLICY;
    
    gSerialResultsCompared = (lRunMode == 1);

	SAFE_PM_EXEC( pmInitialize() );
    
    if(gPreSetupPostMpiInitFunc)
    {
        if(gPreSetupPostMpiInitFunc(argc, argv, COMMON_ARGS) != 0)
        {
			std::cout << "Presetup Error" << std::endl;
			exit(1);
        }
    }
    
    if(lRunMode == 0 || lRunMode == 1)
        RegisterLibraryCallbacks(pCallbackKey1, pCallbacksFunc1, pCallbackKey2, pCallbacksFunc2);

	if(pmGetHostId() == SUBMITTING_HOST_ID)
	{
		if(pInitFunc(argc, argv, COMMON_ARGS) != 0)
		{
			std::cout << "Initialization Error" << std::endl;
			exit(1);
		}

		if(lRunMode == 1 || lRunMode == 2 || lRunMode == 4)
		{
			lSerialExecTime = pSerialFunc(argc, argv, COMMON_ARGS);
			std::cout << "Serial Task Execution Time = " << lSerialExecTime << std::endl;
		}

        if(lRunMode == 3 || lRunMode == 4)
        {
            lSingleGpuExecTime = pSingleGpuFunc(argc, argv, COMMON_ARGS);
            if(lSingleGpuExecTime)
                std::cout << "Single GPU Task Execution Time = " << lSingleGpuExecTime << std::endl;
            else
                std::cout << "Single GPU Task Failed" << std::endl;
        }
        
        if(lRunMode == 4 && lSingleGpuExecTime > 0.0)
        {
            if(pCompareFunc(argc, argv, COMMON_ARGS) == 0)
                std::cout << "Single GPU Task's Sequential Comparison Test Passed" << std::endl;
            else
                std::cout << "Single GPU Task's Sequential Comparison Test Failed" << std::endl;
        }

		if(lRunMode == 0 || lRunMode == 1)
		{
            for(int policy = 0; policy <= 2; ++policy)
            {
                if(lSchedulingPolicy == policy || lSchedulingPolicy == 4)
                {
                    pmSchedulingPolicy lPolicy = SLOW_START;
                    if(policy == 1)
                        lPolicy = RANDOM_STEAL;
                    else if(policy == 2)
                        lPolicy = EQUAL_STATIC;
                    else if(policy == 3)
                        lPolicy = PROPORTIONAL_STATIC;

                    // Six Parallel Execution Modes
                    for(int i = 1; i <= 6; ++i)
                    {
                        if(lParallelMode == 0 || lParallelMode == i || (lParallelMode == 7 && (i == 4 || i == 5 || i == 6)))
                        {
                            Result lResult;
                            lResult.execTime = ExecuteParallelTask(argc, argv, i, pParallelFunc, lPolicy, (lRunMode == 1));

                            lResult.parallelMode = i;
                            lResult.schedulingPolicy = policy;
                            lResult.serialComparisonResult = false;
                            
                            if(lResult.execTime > 0.0 && lRunMode == 1)
                                lResult.serialComparisonResult = (pCompareFunc(argc, argv, COMMON_ARGS) ? false : true);
                            
                            gResultVector.push_back(lResult);
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

void commonStart2(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, singleGpuProcessFunc pSingleGpuFunc,
                    parallelProcessFunc2 pParallelFunc, callbacksFunc pCallbacksFunc1, compareFunc pCompareFunc, destroyFunc pDestroyFunc,
                    std::string pCallbackKey1, callbacksFunc pCallbacksFunc2, std::string pCallbackKey2)
{
    commonStartInternal(argc, argv, pInitFunc, pSerialFunc, pSingleGpuFunc, &pParallelFunc, pCallbacksFunc1, pCompareFunc, pDestroyFunc, pCallbackKey1, pCallbacksFunc2, pCallbackKey2);
}

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, singleGpuProcessFunc pSingleGpuFunc, parallelProcessFunc pParallelFunc,
                 callbacksFunc pCallbacksFunc, compareFunc pCompareFunc, destroyFunc pDestroyFunc, std::string pCallbackKey)
{
    commonStartInternal(argc, argv, pInitFunc, pSerialFunc, pSingleGpuFunc, &pParallelFunc, pCallbacksFunc, pCompareFunc, pDestroyFunc, pCallbackKey, NULL, std::string());
}

void commonFinish()
{
	SAFE_PM_EXEC( pmFinalize() );
    
    std::vector<Result>::iterator lIter = gResultVector.begin(), lEndIter = gResultVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if((*lIter).execTime < 0.0)
        {
            std::cout << "Parallel Task " << (*lIter).parallelMode << " Failed [Scheduling Policy: " << (*lIter).schedulingPolicy << "]" << std::endl;
        }
        else
        {
            std::cout << "Parallel Task " << (*lIter).parallelMode << " Execution Time = " << (*lIter).execTime << " [Scheduling Policy: " << (*lIter).schedulingPolicy << "]";
            
            if(gSerialResultsCompared)
                std::cout << " [Serial Comparison Test " << ((*lIter).serialComparisonResult ? "Passed" : "Failed") << "]";
            
            std::cout << std::endl;
        }
    }
}

void RequestPreSetupCallbackPostMpiInit(preSetupPostMpiInitFunc pFunc)
{
    gPreSetupPostMpiInitFunc = pFunc;
}


bool localDeviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo)
{
	if(pDeviceInfo.host != SUBMITTING_HOST_ID)
		return false;

	return true;
}

bool isMultiAssignEnabled()
{
    const char* lVal = getenv("PMLIB_DISABLE_MA");
    if(lVal && atoi(lVal) != 0)
        return false;
    
    return true;
}

bool isLazyMemEnabled()
{
    const char* lVal = getenv("PMLIB_ENABLE_LAZY_MEM");
    if(lVal && atoi(lVal) != 0)
        return true;
    
    return false;
}

bool isComputeCommunicationOverlapEnabled()
{
    const char* lVal = getenv("PMLIB_DISABLE_COMPUTE_COMMUNICATION_OVERLAP");
    if(lVal && atoi(lVal) != 0)
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





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

typedef enum memDistributionType
{
    LINEAR_DIST,
    BLOCK_DIST_1D_ROW,
    BLOCK_DIST_1D_COL,
    BLOCK_DIST_2D,
    BLOCK_DIST_2D_RANDOM,
    MAX_BLOCK_DISTRIBUTION_TYPES
} memDistributionType;

bool localDeviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo);
double getCurrentTimeInSecs();

#define CUDA_ERROR_CHECK(name, x) \
{ \
    cudaError_t dCudaError = x; \
    if(dCudaError != cudaSuccess) \
    { \
        std::cout << name << " failed with error " << dCudaError << std::endl; \
        exit(1); \
    } \
}

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

#define FETCH_INT_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = atoi(argArray[argIndex+1]); argName = argName;}
#define FETCH_BOOL_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = (bool)atoi(argArray[argIndex+1]); argName = argName;}
#define FETCH_STR_ARG(argName, argIndex, totalArgs, argArray) { if(argIndex+1 < totalArgs) argName = argArray[argIndex+1]; argName = argName;}

typedef double (*serialProcessFunc)(int argc, char** argv, int pCommonArgs);
typedef double (*singleGpuProcessFunc)(int argc, char** argv, int pCommonArgs);
typedef double (*parallelProcessFunc)(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack);
typedef pmCallbacks (*callbacksFunc)();
typedef int (*initFunc)(int argc, char** argv, int pCommonArgs);
typedef int (*destroyFunc)();
typedef int (*compareFunc)(int argc, char** argv, int pCommonArgs);
typedef int (*preSetupPostMpiInitFunc)(int argc, char** argv, int pCommonArgs);

int GetCommonArgsCount();

typedef struct callbackStruct
{
    callbacksFunc func;
    std::string key;
} callbackStruct;

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, singleGpuProcessFunc pSingleGpuFunc, parallelProcessFunc pParallelFunc,
	compareFunc pCompareFunc, destroyFunc pDestroyFunc, callbackStruct* pCallbacksStruct, size_t pCallbacksCount);

void commonFinish();

// Assumes that the address space has a 2D layout (pMatrixWidth * pMatrixHeight elems) logically divided into square tiles of size pBlockDim * pBlockDim
void DistributeMemory(pmMemHandle pMemHandle, memDistributionType pDistType, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, bool pRandomize);

void RequestPreSetupCallbackPostMpiInit(preSetupPostMpiInitFunc pFunc);

bool isMultiAssignEnabled();    /* by default, its enabled */
uint getAffinityCriterion();    /* by default, its 0 */
bool isComputeCommunicationOverlapEnabled();    /* by default, its enabled */
bool isLazyMemEnabled();    /* by default, its diabled */

#ifdef BUILD_CUDA
bool isCudaCacheEnabled();   /* by default, its enabled */
#endif

#define CREATE_MEM(memSize, memHandle) SAFE_PM_EXEC( pmCreateMemory(memSize, &memHandle) )
#define CREATE_MEM_2D(memRows, memCols, memHandle) SAFE_PM_EXEC( pmCreateMemory2D(memRows, memCols, &memHandle) )

#ifdef BUILD_CUDA

#define CREATE_TASK(totalSubtasks, cbHandle, schedPolicy) \
	pmTaskHandle lTaskHandle = NULL; \
    pmTaskDetails lTaskDetails; \
	lTaskDetails.callbackHandle = cbHandle; \
	lTaskDetails.subtaskCount = totalSubtasks; \
	lTaskDetails.policy = schedPolicy; \
	lTaskDetails.multiAssignEnabled = isMultiAssignEnabled(); \
    lTaskDetails.affinityCriterion = (pmAffinityCriterion)getAffinityCriterion(); \
    lTaskDetails.overlapComputeCommunication = isComputeCommunicationOverlapEnabled(); \
    lTaskDetails.cudaCacheEnabled = isCudaCacheEnabled();

#else

#define CREATE_TASK(totalSubtasks, cbHandle, schedPolicy) \
	pmTaskHandle lTaskHandle = NULL; \
    pmTaskDetails lTaskDetails; \
	lTaskDetails.callbackHandle = cbHandle; \
	lTaskDetails.subtaskCount = totalSubtasks; \
	lTaskDetails.policy = schedPolicy; \
	lTaskDetails.multiAssignEnabled = isMultiAssignEnabled(); \
    lTaskDetails.affinityCriterion = (pmAffinityCriterion)getAffinityCriterion(); \
    lTaskDetails.overlapComputeCommunication = isComputeCommunicationOverlapEnabled();

#endif

#define CREATE_SIMPLE_TASK(inputMemSize, outputMemSize, totalSubtasks, cbHandle, schedPolicy) \
    CREATE_TASK(totalSubtasks, cbHandle, schedPolicy) \
    pmTaskMem lTaskMem[2]; \
    { \
        uint lMemIndex = 0; \
        if(inputMemSize) \
        { \
            lTaskMem[lMemIndex].memType = READ_ONLY; \
            CREATE_MEM(inputMemSize, lTaskMem[lMemIndex].memHandle); \
            ++lMemIndex; \
        } \
        if(outputMemSize) \
        { \
            lTaskMem[lMemIndex].memType = WRITE_ONLY; \
            CREATE_MEM(outputMemSize, lTaskMem[lMemIndex].memHandle); \
            ++lMemIndex; \
        } \
        if(lMemIndex) \
            lTaskDetails.taskMem = (pmTaskMem*)(lTaskMem); \
        lTaskDetails.taskMemCount = lMemIndex; \
    }


#define FREE_TASK_AND_RESOURCES \
	SAFE_PM_EXEC( pmReleaseTask(lTaskHandle) ); \
    for(size_t dIndex = 0; dIndex < lTaskDetails.taskMemCount; ++dIndex) \
        SAFE_PM_EXEC( pmReleaseMemory(lTaskDetails.taskMem[dIndex].memHandle) ); \


#ifdef ENABLE_BLAS

#ifdef MACOS
#include <Accelerate/Accelerate.h>
#else
extern "C"
{
    #include <cblas.h>
}
#endif

#ifdef BUILD_CUDA
#ifdef __CUDACC__

#include <cublas_v2.h>
#include <map>

#define CUBLAS_ERROR_CHECK(name, x) \
{ \
    cublasStatus_t dStatus = x; \
    if(dStatus != CUBLAS_STATUS_SUCCESS) \
    { \
        std::cout << name << " failed with error " << dStatus << std::endl; \
        exit(1); \
    } \
}

typedef std::map<pmDeviceHandle, cublasHandle_t> cublasHandleMapType;

cublasHandle_t CreateCublasHandle();
void DestroyCublasHandle(cublasHandle_t pCublasHandle);
cublasHandleMapType& GetCublasHandleMap();

cublasHandle_t GetCublasHandle(pmDeviceHandle pDeviceHandle);
void FreeCublasHandles();

class cublasHandleManager
{
public:
    cublasHandleManager();
    ~cublasHandleManager();
    
    cublasHandle_t GetHandle();
    
private:
    cublasHandle_t mHandle;
};

#endif
#endif

#endif

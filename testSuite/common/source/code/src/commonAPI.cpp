
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <string>
#include <vector>
#include <set>
#include <random>
#include <algorithm>

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

std::vector<pmCallbackHandle> gCallbackHandles[6];

pmStatus memoryDistributionTask_dataDistributionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
bool memoryDistributionTask_deviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo);

pmCallbackHandle gMemoryDistributionCallbackHandle;
pmCallbacks gMemoryDistributionCallbacks(memoryDistributionTask_dataDistributionCallback, (pmSubtaskCallback_CPU)NULL, (pmSubtaskCallback_GPU_CUDA)NULL);

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

int GetCommonArgsCount()
{
    return COMMON_ARGS;
}

double ExecuteParallelTask(int argc, char** argv, int pParallelMode, parallelProcessFunc pParallelFunc, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
    return (pParallelFunc)(argc, argv, COMMON_ARGS, &(gCallbackHandles[pParallelMode - 1])[0], pSchedulingPolicy, pFetchBack);
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

    std::string lTempKey(pCallbackKey + lArray[pParallelMode - 1]);

    SAFE_PM_EXEC( pmRegisterCallbacks((char*)lTempKey.c_str(), pCallbacks, pCallbackHandle) );
}

void RegisterLibraryCallbacks(callbackStruct* pCallbacksStruct, size_t pCallbacksCount)
{
    gCallbackHandles[0].resize(pCallbacksCount);
    gCallbackHandles[1].resize(pCallbacksCount);
    gCallbackHandles[2].resize(pCallbacksCount);
    gCallbackHandles[3].resize(pCallbacksCount);
    gCallbackHandles[4].resize(pCallbacksCount);
    gCallbackHandles[5].resize(pCallbacksCount);

    for(size_t i = 0; i < pCallbacksCount; ++i)
    {
        pmCallbacks lCallbacks = pCallbacksStruct[i].func();

        RegisterLibraryCallback(1, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[0])[i]);
        RegisterLibraryCallback(2, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[1])[i]);
        RegisterLibraryCallback(3, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[2])[i]);
        RegisterLibraryCallback(4, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[3])[i]);
        RegisterLibraryCallback(5, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[4])[i]);
        RegisterLibraryCallback(6, pCallbacksStruct[i].key, lCallbacks, &(gCallbackHandles[5])[i]);
    }
}

void ReleaseLibraryCallbacks()
{
    for(size_t i = 0; i < 6; ++i)
    {
        std::vector<pmCallbackHandle>::iterator lIter = gCallbackHandles[i].begin(), lEndIter = gCallbackHandles[i].end();
        for(; lIter != lEndIter; ++lIter)
            SAFE_PM_EXEC( pmReleaseCallbacks(*lIter) );
    }
}

void commonStart(int argc, char** argv, initFunc pInitFunc, serialProcessFunc pSerialFunc, singleGpuProcessFunc pSingleGpuFunc, parallelProcessFunc pParallelFunc,
                 compareFunc pCompareFunc, destroyFunc pDestroyFunc, callbackStruct* pCallbacksStruct, size_t pCallbacksCount)
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
    
    gMemoryDistributionCallbacks.deviceSelection = memoryDistributionTask_deviceSelectionCallback;
    SAFE_PM_EXEC( pmRegisterCallbacks((char*)"MemoryDistributionCallback", gMemoryDistributionCallbacks, &gMemoryDistributionCallbackHandle) );
    
    if(lRunMode == 0 || lRunMode == 1)
        RegisterLibraryCallbacks(pCallbacksStruct, pCallbacksCount);

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

void commonFinish()
{
//    SAFE_PM_EXEC( pmReleaseCallbacks(gMemoryDistributionCallbackHandle) );

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
    
#ifdef BUILD_CUDA
#ifdef ENABLE_BLAS
    void FreeCublasHandles();
    FreeCublasHandles();
#endif
#endif
}

struct distributeMemoryTaskConf
{
    memDistributionType distType;
    unsigned int blockDim;
    unsigned int matrixWidth;
    unsigned int matrixHeight;
    unsigned int elemSize;
    // Has appended machines list post elemSize
};

// Selects one device per host
bool memoryDistributionTask_deviceSelectionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo)
{
    return (pDeviceInfo.deviceIdOnHost == 0);
}

void Do1DBlockRowDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, unsigned int* pMachinesList, unsigned int pMachineId, unsigned int pMachinesCount)
{
    unsigned int lBlockRows = pMatrixHeight / pBlockDim;
    unsigned int lBlockRowsPerMachine = lBlockRows / pMachinesCount;
    unsigned int lLeftoverBlockRows = lBlockRows - (lBlockRowsPerMachine * pMachinesCount);
    unsigned int lRowsPerMachine = lBlockRowsPerMachine * pBlockDim;
    size_t lLengthPerMachine = lRowsPerMachine * pMatrixWidth * pElemSize;

    unsigned int lMachineIndex = pMachineId;
    for(uint i = 0; i < pMachinesCount; ++i)
    {
        if(pMachinesList[i] == pMachineId)
        {
            lMachineIndex = i;
            break;
        }
    }
    
    if(pMachinesList[lMachineIndex] != pMachineId)
        exit(1);
    
    size_t lFirstRowOffset = lLengthPerMachine * lMachineIndex + pBlockDim * pMatrixWidth * pElemSize * std::min(lLeftoverBlockRows, lMachineIndex);
    size_t lTotalLength = ((lLeftoverBlockRows > lMachineIndex) ? (lLengthPerMachine + pBlockDim * pMatrixWidth * pElemSize) : lLengthPerMachine);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, 0, READ_WRITE_SUBSCRIPTION, pmSubscriptionInfo(lFirstRowOffset, lTotalLength));
}

void Do1DBlockColDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, unsigned int* pMachinesList, unsigned int pMachineId, unsigned int pMachinesCount)
{
    unsigned int lBlockCols = pMatrixWidth / pBlockDim;
    unsigned int lBlockColsPerMachine = lBlockCols / pMachinesCount;
    unsigned int lLeftoverBlockCols = lBlockCols - (lBlockColsPerMachine * pMachinesCount);
    unsigned int lColsPerMachine = lBlockColsPerMachine * pBlockDim;
    size_t lColLengthPerMachine = lColsPerMachine * pElemSize;

    unsigned int lMachineIndex = pMachineId;
    for(uint i = 0; i < pMachinesCount; ++i)
    {
        if(pMachinesList[i] == pMachineId)
        {
            lMachineIndex = i;
            break;
        }
    }
    
    if(pMachinesList[lMachineIndex] != pMachineId)
        exit(1);

    size_t lFirstColOffset = lColLengthPerMachine * lMachineIndex + pBlockDim * pElemSize * std::min(lLeftoverBlockCols, lMachineIndex);
    size_t lTotalColLength = ((lLeftoverBlockCols > lMachineIndex) ? (lColLengthPerMachine + pBlockDim * pElemSize) : lColLengthPerMachine);
    
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, 0, READ_WRITE_SUBSCRIPTION, pmScatteredSubscriptionInfo(lFirstColOffset, lTotalColLength, pMatrixWidth * pElemSize, pMatrixHeight));
}

void Do2DBlockDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, unsigned int* pMachinesList, unsigned int pMachineId, unsigned int pMachinesCount)
{
    unsigned int lBlockRows = pMatrixHeight / pBlockDim;    // Any left over partial blocks are kept on owner host
    unsigned int lBlockCols = pMatrixWidth / pBlockDim;
    unsigned int lTotalBlocks = lBlockRows * lBlockCols;
    unsigned int lBlocksPerMachine = lTotalBlocks / pMachinesCount;
    unsigned int lLeftoverBlocks = lTotalBlocks - (lBlocksPerMachine * pMachinesCount);
    size_t lBlockRowLength = pBlockDim * pElemSize;

    unsigned int lMachineIndex = pMachineId;
    for(uint i = 0; i < pMachinesCount; ++i)
    {
        if(pMachinesList[i] == pMachineId)
        {
            lMachineIndex = i;
            break;
        }
    }
    
    if(pMachinesList[lMachineIndex] != pMachineId)
        exit(1);

    unsigned int lFirstBlockId = lBlocksPerMachine * lMachineIndex + std::min(lLeftoverBlocks, lMachineIndex);
    unsigned int lBlocks = ((lLeftoverBlocks > lMachineIndex) ? (lBlocksPerMachine + 1) : lBlocksPerMachine);

    for(unsigned int j = 0; j < lBlocks; ++j)
    {
        unsigned int lBlockId = lFirstBlockId + j;
        unsigned int lBlockRow = lBlockId / lBlockCols;
        unsigned int lBlockCol = lBlockId % lBlockCols;
        
        size_t lBlockOffset = (lBlockRow * pMatrixWidth + lBlockCol) * pBlockDim * pElemSize;

        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, 0, READ_WRITE_SUBSCRIPTION, pmScatteredSubscriptionInfo(lBlockOffset, lBlockRowLength, pMatrixWidth * pElemSize, pBlockDim));
    }
}

void Do2DRandomBlockDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, unsigned int* pMachinesList, unsigned int pMachineId, unsigned int pMachinesCount)
{
    unsigned int lBlockRows = pMatrixHeight / pBlockDim;    // Any left over partial blocks are kept on owner host
    unsigned int lBlockCols = pMatrixWidth / pBlockDim;
    size_t lBlockRowLength = pBlockDim * pElemSize;
    
    for(unsigned int i = 0; i < lBlockRows; ++i)
    {
        for(unsigned int j = 0; j < lBlockCols; ++j)
        {
            if(pMachinesList[i] == pMachineId)
            {
                size_t lBlockOffset = (i * pMatrixWidth + j) * pBlockDim * pElemSize;

                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, 0, READ_WRITE_SUBSCRIPTION, pmScatteredSubscriptionInfo(lBlockOffset, lBlockRowLength, pMatrixWidth * pElemSize, pBlockDim));
            }
        }
    }
}

pmStatus memoryDistributionTask_dataDistributionCallback(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
    distributeMemoryTaskConf* lTaskConf = (distributeMemoryTaskConf*)(pTaskInfo.taskConf);
    unsigned int* lMachinesList = (unsigned int*)(((char*)lTaskConf) + sizeof(distributeMemoryTaskConf));

    switch(lTaskConf->distType)
    {
        case BLOCK_DIST_1D_ROW:
            Do1DBlockRowDistribution(pTaskInfo, pDeviceInfo, pSubtaskInfo, lTaskConf->blockDim, lTaskConf->matrixWidth, lTaskConf->matrixHeight, lTaskConf->elemSize, lMachinesList, (unsigned int)pSubtaskInfo.subtaskId, (unsigned int)pTaskInfo.subtaskCount);
            break;

        case BLOCK_DIST_1D_COL:
            Do1DBlockColDistribution(pTaskInfo, pDeviceInfo, pSubtaskInfo, lTaskConf->blockDim, lTaskConf->matrixWidth, lTaskConf->matrixHeight, lTaskConf->elemSize, lMachinesList, (unsigned int)pSubtaskInfo.subtaskId, (unsigned int)pTaskInfo.subtaskCount);
            break;
        
        case BLOCK_DIST_2D:
            Do2DBlockDistribution(pTaskInfo, pDeviceInfo, pSubtaskInfo, lTaskConf->blockDim, lTaskConf->matrixWidth, lTaskConf->matrixHeight, lTaskConf->elemSize, lMachinesList, (unsigned int)pSubtaskInfo.subtaskId, (unsigned int)pTaskInfo.subtaskCount);
            break;
            
        case BLOCK_DIST_2D_RANDOM:
            Do2DRandomBlockDistribution(pTaskInfo, pDeviceInfo, pSubtaskInfo, lTaskConf->blockDim, lTaskConf->matrixWidth, lTaskConf->matrixHeight, lTaskConf->elemSize, lMachinesList, (unsigned int)pSubtaskInfo.subtaskId, (unsigned int)pTaskInfo.subtaskCount);
            break;
            
        default:
            exit(1);
    }

    
    return pmSuccess;
}

// pRandomize is ignored when pDistType is BLOCK_DIST_RANDOM
void DistributeMemory(pmMemHandle pMemHandle, memDistributionType pDistType, unsigned int pBlockDim, unsigned int pMatrixWidth, unsigned int pMatrixHeight, unsigned int pElemSize, bool pRandomize)
{
    using namespace pm;

    if(pDistType >= MAX_BLOCK_DISTRIBUTION_TYPES || !pMemHandle)
        exit(1);

    unsigned int lMachines = pmGetHostCount();
    unsigned int lTaskConfDynamicEntries = lMachines;
    
    if(pDistType == BLOCK_DIST_2D_RANDOM)
    {
        unsigned int lBlockRows = pMatrixHeight / pBlockDim;    // Any left over partial blocks are kept on owner host
        unsigned int lBlockCols = pMatrixWidth / pBlockDim;
        unsigned int lTotalBlocks = lBlockRows * lBlockCols;
        
        lTaskConfDynamicEntries = lTotalBlocks;
    }
    
    unsigned int lTaskConfLength = sizeof(distributeMemoryTaskConf) + sizeof(unsigned int) * lTaskConfDynamicEntries;
    std::vector<char> lTaskConf(lTaskConfLength);
    
    distributeMemoryTaskConf* distTaskConf = (distributeMemoryTaskConf*)(&lTaskConf[0]);
    distTaskConf->distType = pDistType;
    distTaskConf->blockDim = pBlockDim;
    distTaskConf->matrixWidth = pMatrixWidth;
    distTaskConf->matrixHeight = pMatrixHeight;
    distTaskConf->elemSize = pElemSize;

    unsigned int* lMachinesList = (unsigned int*)((char*)(&lTaskConf[0]) + sizeof(distributeMemoryTaskConf));
    if(pDistType == BLOCK_DIST_2D_RANDOM)
    {
        std::random_device lRandomDevice;
        std::mt19937 lGenerator(lRandomDevice());

        std::generate_n(lMachinesList, lTaskConfDynamicEntries, [&] () {return (lGenerator() % lMachines);});
    }
    else
    {
        unsigned int lFirstMachineIndex = 0;
        std::generate_n(lMachinesList, lMachines, [lFirstMachineIndex] () mutable {return (lFirstMachineIndex++);});

        if(pRandomize)
        {
            std::random_device lRandomDevice;
            std::mt19937 lGenerator(lRandomDevice());
        
            std::shuffle(lMachinesList, lMachinesList + lMachines, lGenerator);
        }
    }

    pmTaskMem lTaskMem[1] = {{pMemHandle, READ_WRITE, SUBSCRIPTION_NATURAL, true}};
    pmTaskDetails lTaskDetails(&lTaskConf[0], lTaskConfLength, lTaskMem, 1, gMemoryDistributionCallbackHandle, lMachines);
    
	lTaskDetails.policy = EQUAL_STATIC;
	lTaskDetails.multiAssignEnabled = false;
    lTaskDetails.overlapComputeCommunication = false;
    lTaskDetails.suppressTaskLogs = true;

	pmTaskHandle lTaskHandle = NULL;

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
    SAFE_PM_EXEC( pmWaitForTaskCompletion(lTaskHandle) );
    SAFE_PM_EXEC( pmReleaseTask(lTaskHandle) );
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

#ifdef BUILD_CUDA
bool isCudaCacheEnabled()
{
    const char* lVal = getenv("PMLIB_DISABLE_CUDA_CACHE");
    if(lVal && atoi(lVal) != 0)
        return false;

    return true;
}
#endif

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




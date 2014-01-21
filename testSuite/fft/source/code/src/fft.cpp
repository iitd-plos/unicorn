
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "fft.h"

#include <fftw3.h>
#include <math.h>

namespace fft
{
    
size_t gDevicesPerSplitGroup = 6;   // Temporarily hardcoding
    
FFT_DATA_TYPE* gSampleInput;
FFT_DATA_TYPE* gSerialOutput;
FFT_DATA_TYPE* gParallelOutput;

struct fftwPlanner
{
    fftwPlanner(bool pRowPlanner)
    : mRowPlanner(pRowPlanner)
    , mPlan(NULL)
    {}
    
    void CreateDummyPlan(bool inplace, int dir, size_t N, size_t M, size_t pCount)
    {
        complex* lDummyInputData = new complex[N];
        complex* lDummyOutputData = (inplace ? NULL : (new complex[N]));

        fftwf_complex* inputData = (fftwf_complex*)lDummyInputData;
        fftwf_complex* outputData = (fftwf_complex*)lDummyOutputData;

        int lN[] = {(int)N};

    #ifdef NO_MATRIX_TRANSPOSE
        if(mRowPlanner)
        {
            mPlan = fftwf_plan_many_dft(1, lN, (int)pCount, inputData, lN, 1, (int)N, (inplace ? inputData : outputData), lN, 1, (int)N, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);
        }
        else
        {
            mPlan = fftwf_plan_many_dft(1, lN, (int)pCount, inputData, lN, (int)M, 1, (inplace ? inputData : outputData), lN, (int)M, 1, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);
        }
    #else
        mPlan = fftwf_plan_many_dft(1, lN, (int)pCount, inputData, lN, 1, (int)N, (inplace ? inputData : outputData), lN, 1, (int)N, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);
    #endif
        
        delete[] lDummyInputData;
        delete[] lDummyOutputData;
    }
    
    ~fftwPlanner()
    {
        if(mPlan)
            fftwf_destroy_plan(mPlan);
    }
    
    bool mRowPlanner;
    fftwf_plan mPlan;
};

fftwPlanner gRowPlanner(true);

#ifdef FFT_2D
    fftwPlanner gColPlanner(false);
#endif
    
#ifdef NO_MATRIX_TRANSPOSE
    fftwPlanner gSplitRowPlanner(true);
    fftwPlanner gSplitLastRowPlanner(true);

    #ifdef FFT_2D
        fftwPlanner gSplitColPlanner(false);
        fftwPlanner gSplitLastColPlanner(false);
    #endif
#endif

void fftSerial(complex* input, complex* output, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
{
#if 1   //ndef NO_MATRIX_TRANSPOSE

    bool lInplace = (input == output);

    fftwPlanner lRowPlan(true);
    lRowPlan.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, ny, nx, nx);
    
    fftwf_execute_dft(lRowPlan.mPlan, (fftwf_complex*)input, (fftwf_complex*)output);

    #ifdef FFT_2D
        matrixTranspose::serialMatrixTranspose(lInplace, output, input, nx, ny);
        
        fftwPlanner lColPlan(true); // passed true here because matrix has been transposed; so we actually need a row plan
        lColPlan.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, nx, ny, ny);
    
        fftwf_execute_dft(lColPlan.mPlan, (fftwf_complex*)input, (fftwf_complex*)output);

        matrixTranspose::serialMatrixTranspose(lInplace, output, input, ny, nx);
    #endif
    
#else
    
    #ifdef FFT_2D
        fftwf_complex* inputData = (fftwf_complex*)input;
        fftwf_complex* outputData = (fftwf_complex*)output;
    
        fftwf_plan lPlan = fftwf_plan_dft_2d((int)ny, (int)nx, inputData, outputData, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);

        fftwf_execute(lPlan);
        fftwf_destroy_plan(lPlan);
    #else
        fftwPlanner lRowPlan(true);
        lRowPlan.CreateDummyPlan((input == output), FORWARD_TRANSFORM_DIRECTION, ny, nx, nx);
    
        fftwf_execute_dft(lRowPlan.mPlan, (fftwf_complex*)input, (fftwf_complex*)output);
    #endif
    
#endif
}

pmStatus fftDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

#ifdef NO_MATRIX_TRANSPOSE
    if(pSubtaskInfo.splitInfo.splitCount)
    {
        
        if(pSubtaskInfo.splitInfo.splitCount != gDevicesPerSplitGroup)
        {
            std::cout << "FFT currently configured for " << gDevicesPerSplitGroup << " and not " << pSubtaskInfo.splitInfo.splitCount << " devices per split group" << std::endl;
            exit(1);
        }
        
        size_t lRowsPerSplit = ROWS_PER_FFT_SUBTASK / pSubtaskInfo.splitInfo.splitCount;
        if(lRowsPerSplit == 0)
            exit(1);

        size_t lStartingRow = lRowsPerSplit * pSubtaskInfo.splitInfo.splitId;
        size_t lRowsForCurrentSplit = (pSubtaskInfo.splitInfo.splitId == pSubtaskInfo.splitInfo.splitCount - 1) ? (ROWS_PER_FFT_SUBTASK - lStartingRow) : lRowsPerSplit;

        if(lTaskConf->rowPlanner)   // Row FFT
        {
            size_t lOffset = (ROWS_PER_FFT_SUBTASK * pSubtaskInfo.subtaskId + lStartingRow) * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
            pmSubscriptionInfo lSubscriptionInfo(lOffset, lRowsForCurrentSplit * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE));
            
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lSubscriptionInfo);
        }
        else    // Col FFT (for FFT_2D)
        {
            size_t lBlockDim = ROWS_PER_FFT_SUBTASK;
            size_t lOffset = (lBlockDim * pSubtaskInfo.subtaskId + lStartingRow) * sizeof(FFT_DATA_TYPE);
            size_t lBlockSpacing = ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
            size_t lBlockCount = lTaskConf->elemsX / lBlockDim; // Vertical blocks

            for(size_t i = 0; i < lBlockCount; ++i)
            {
                pmScatteredSubscriptionInfo lScatteredSubscriptionInfo(lOffset + i * lBlockSpacing, lRowsForCurrentSplit * sizeof(FFT_DATA_TYPE), lTaskConf->elemsY * sizeof(FFT_DATA_TYPE), lBlockDim);

                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lScatteredSubscriptionInfo);
                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lScatteredSubscriptionInfo);
            }
        }
    }
    else
    {
        if(lTaskConf->rowPlanner)   // Row FFT
        {
            size_t lOffset = ROWS_PER_FFT_SUBTASK * pSubtaskInfo.subtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
            pmSubscriptionInfo lSubscriptionInfo(lOffset, ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE));

            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lSubscriptionInfo);
        }
        else    // Col FFT (for FFT_2D)
        {
            size_t lBlockDim = ROWS_PER_FFT_SUBTASK;
            size_t lOffset = lBlockDim * pSubtaskInfo.subtaskId * sizeof(FFT_DATA_TYPE);
            size_t lBlockSpacing = ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
            size_t lBlockCount = lTaskConf->elemsX / lBlockDim; // Vertical blocks
            
            for(size_t i = 0; i < lBlockCount; ++i)
            {
                pmScatteredSubscriptionInfo lScatteredSubscriptionInfo(lOffset + i * lBlockSpacing, lBlockDim * sizeof(FFT_DATA_TYPE), lTaskConf->elemsY * sizeof(FFT_DATA_TYPE), lBlockDim);

                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lScatteredSubscriptionInfo);
                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lScatteredSubscriptionInfo);
            }
        }
    }
#else
    if(lTaskConf->rowPlanner)   // Row FFT
    {
        size_t lOffset = ROWS_PER_FFT_SUBTASK * pSubtaskInfo.subtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
        pmSubscriptionInfo lSubscriptionInfo(lOffset, ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE));

        if(lTaskConf->inplace)
        {
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPLACE_MEM_INDEX, READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);
        }
        else
        {
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lSubscriptionInfo);
            pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lSubscriptionInfo);
        }
    }
    else    // Col FFT (for FFT_2D)
    {
        // Instead of subscribing as a contiguous region, subscribe multiple scattered blocks (as they have been created by previous transpose)
        size_t lOffset = ROWS_PER_FFT_SUBTASK * pSubtaskInfo.subtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
        size_t lTransposeBlockDim = ROWS_PER_FFT_SUBTASK;  // For efficient execution, ROWS_PER_FFT_SUBTASK should be equal to block size for matrix transpose
        size_t lBlockCount = lTaskConf->elemsY / lTransposeBlockDim;
        
        for(size_t i = 0; i < lBlockCount; ++i)
        {
            pmScatteredSubscriptionInfo lScatteredSubscriptionInfo(lOffset + i * lTransposeBlockDim * sizeof(FFT_DATA_TYPE), lTransposeBlockDim * sizeof(FFT_DATA_TYPE), lTaskConf->elemsY * sizeof(FFT_DATA_TYPE), lTransposeBlockDim);

            if(lTaskConf->inplace)
            {
                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPLACE_MEM_INDEX, READ_WRITE_SUBSCRIPTION, lScatteredSubscriptionInfo);
            }
            else
            {
                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, INPUT_MEM_INDEX, READ_SUBSCRIPTION, lScatteredSubscriptionInfo);
                pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, OUTPUT_MEM_INDEX, WRITE_SUBSCRIPTION, lScatteredSubscriptionInfo);
            }
        }
    }
#endif

	return pmSuccess;
}

pmStatus fft_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

#ifdef NO_MATRIX_TRANSPOSE
    fftwf_plan lPlanner;

    if(lTaskConf->rowPlanner)
    {
        if(pSubtaskInfo.splitInfo.splitCount)
            lPlanner = (pSubtaskInfo.splitInfo.splitId == pSubtaskInfo.splitInfo.splitCount - 1) ? gSplitLastRowPlanner.mPlan : gSplitRowPlanner.mPlan;
        else
            lPlanner = gRowPlanner.mPlan;
    }
    else
    {
        if(pSubtaskInfo.splitInfo.splitCount)
            lPlanner = (pSubtaskInfo.splitInfo.splitId == pSubtaskInfo.splitInfo.splitCount - 1) ? gSplitLastColPlanner.mPlan : gSplitColPlanner.mPlan;
        else
            lPlanner = gColPlanner.mPlan;
    }
#else
    fftwf_plan lPlanner = (lTaskConf->rowPlanner ? gRowPlanner.mPlan : gColPlanner.mPlan);
#endif

    if(lTaskConf->inplace)
    {
        fftwf_complex* lOutputLoc = (fftwf_complex*)pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].ptr;
        fftwf_execute_dft(lPlanner, lOutputLoc, lOutputLoc);
    }
    else
    {
        fftwf_complex* lInputLoc = (fftwf_complex*)pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr;
        fftwf_complex* lOutputLoc = (fftwf_complex*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr;
        fftwf_execute_dft(lPlanner, lInputLoc, lOutputLoc);
    }

	return pmSuccess;
}

#ifdef USE_SQUARE_MATRIX
#define READ_NON_COMMON_ARGS \
    int lPowX = DEFAULT_POW_X; \
    bool lInplace = (bool)DEFAULT_INPLACE_VALUE; \
    FETCH_INT_ARG(lPowX, pCommonArgs, argc, argv); \
    FETCH_BOOL_ARG(lInplace, pCommonArgs + 1, argc, argv); \
    int lPowY = lPowX; \
    size_t lElemsX = 1 << lPowX; \
    size_t lElemsY = lElemsX;
#else
#define READ_NON_COMMON_ARGS \
    int lPowX = DEFAULT_POW_X; \
    int lPowY = DEFAULT_POW_Y; \
    bool lInplace = (bool)DEFAULT_INPLACE_VALUE; \
    FETCH_INT_ARG(lPowX, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lPowY, pCommonArgs + 1, argc, argv); \
    FETCH_BOOL_ARG(lInplace, pCommonArgs + 2, argc, argv); \
    size_t lElemsX = 1 << lPowX; \
    size_t lElemsY = 1 << lPowY;
#endif

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS
    
    size_t lSize = lElemsX * lElemsY * sizeof(FFT_DATA_TYPE);
    
#ifdef FFT_2D
    bool lInputCopyRequired = !lInplace;
#else
    bool lInputCopyRequired = false;
#endif
    
    FFT_DATA_TYPE* lInputCopy = (lInputCopyRequired ? (new FFT_DATA_TYPE[lElemsX * lElemsY]) : NULL);
    if(lInputCopyRequired)
        memcpy(lInputCopy, gSampleInput, lSize);

    double lStartTime = getCurrentTimeInSecs();

#ifdef FFT_2D
	fftSerial((lInplace ? gSerialOutput : lInputCopy), gSerialOutput, lPowX, lElemsX, lPowY, lElemsY, FORWARD_TRANSFORM_DIRECTION);
#else
	fftSerial((lInplace ? gSerialOutput : gSampleInput), gSerialOutput, lPowX, lElemsX, lPowY, lElemsY, FORWARD_TRANSFORM_DIRECTION);
#endif

	double lEndTime = getCurrentTimeInSecs();

#ifdef FFT_2D
    if(lInputCopyRequired)
        memcpy(gSerialOutput, lInputCopy, lSize);
#endif
    
    delete[] lInputCopy;

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

    double lStartTime = getCurrentTimeInSecs();
    
	if(fftSingleGpu2D(lInplace, gSampleInput, gParallelOutput, lPowX, lElemsX, lPowY, lElemsY, FORWARD_TRANSFORM_DIRECTION) != 0)
        return 0;
    
	double lEndTime = getCurrentTimeInSecs();
    
	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

bool Parallel_FFT_1D(pmMemHandle pInputMemHandle, pmMemType pInputMemType, pmMemHandle pOutputMemHandle, pmMemType pOutputMemType, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    bool lInplace = (pInputMemHandle == pOutputMemHandle);

    unsigned long lSubtaskCount = pTaskConf->elemsX / ROWS_PER_FFT_SUBTASK;
	CREATE_TASK(lSubtaskCount, pCallbackHandle, pSchedulingPolicy)

    pmTaskMem lTaskMem[MAX_MEM_INDICES];
    
    if(lInplace)
    {
        lTaskMem[INPLACE_MEM_INDEX] = {pOutputMemHandle, pOutputMemType, SUBSCRIPTION_NATURAL, true};

        lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
        lTaskDetails.taskMemCount = INPLACE_MAX_MEM_INDICES;
    }
    else
    {
        bool lOptimal = false;
        
    #ifdef NO_MATRIX_TRANSPOSE
        if(!pTaskConf->rowPlanner)
            lOptimal = true;
    #endif
        
        lTaskMem[INPUT_MEM_INDEX] = {pInputMemHandle, pInputMemType, lOptimal ? SUBSCRIPTION_OPTIMAL : SUBSCRIPTION_NATURAL};
        lTaskMem[OUTPUT_MEM_INDEX] = {pOutputMemHandle, pOutputMemType, lOptimal ? SUBSCRIPTION_OPTIMAL : SUBSCRIPTION_NATURAL};

        lTaskDetails.taskMem = (pmTaskMem*)lTaskMem;
        lTaskDetails.taskMemCount = MAX_MEM_INDICES;
    }

	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(fftTaskConf);
    
#ifdef NO_MATRIX_TRANSPOSE
    lTaskDetails.canSplitCpuSubtasks = true;
#endif

	SAFE_PM_EXEC( pmSubmitTask(lTaskDetails, &lTaskHandle) );
	if(pmWaitForTaskCompletion(lTaskHandle) != pmSuccess)
	{
		FREE_TASK_AND_RESOURCES
		return false;
	}

	pmReleaseTask(lTaskHandle);

	return true;
}

#ifdef FFT_2D
bool Parallel_Transpose(void* pInputMemHandle, pmMemType pInputMemType, void* pOutputMemHandle, pmMemType pOutputMemType, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    if(matrixTranspose::parallelMatrixTranspose(pTaskConf->powX, pTaskConf->powY, pTaskConf->elemsX, pTaskConf->elemsY, pInputMemHandle, pOutputMemHandle, pCallbackHandle, pSchedulingPolicy, pInputMemType, pOutputMemType) == -1.0)
        return false;

	return true;
}
#endif
    
// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle* pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, bool pFetchBack)
{
	READ_NON_COMMON_ARGS

	// Input Mem is null for inplace fft, contains input fft data otherwise
	// Output Mem contains the input fft data and also receives result for inplace operation, only receives result for non-inplace operation
	// Number of subtasks is equal to the number of rows/cols for row/col-major processing
	pmMemHandle lInputMemHandle, lOutputMemHandle;
    size_t lElems = lElemsX * lElemsY;
    size_t lMemSize = lElems * sizeof(FFT_DATA_TYPE);
    
	CREATE_MEM(lMemSize, lOutputMemHandle);
    
    if(lInplace)
        lInputMemHandle = lOutputMemHandle;
    else
        CREATE_MEM(lMemSize, lInputMemHandle);
    
	pmRawMemPtr lRawInputPtr;
	pmGetRawMemPtr(lInputMemHandle, &lRawInputPtr);
	memcpy(lRawInputPtr, (lInplace ? gParallelOutput : gSampleInput), lMemSize);

	fftTaskConf lTaskConf;
    lTaskConf.elemsX = lElemsX;
    lTaskConf.elemsY = lElemsY;
    lTaskConf.powX = lPowX;
    lTaskConf.powY = lPowY;
    lTaskConf.rowPlanner = true;
    lTaskConf.inplace = lInplace;

    pmMemType lOutputMemType = (lInplace ? READ_WRITE : WRITE_ONLY);
    pmMemType lInputMemType = READ_ONLY;
    
	double lStartTime = getCurrentTimeInSecs();

    if(!Parallel_FFT_1D(lInputMemHandle, lInputMemType, lOutputMemHandle, lOutputMemType, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;

#ifdef FFT_2D
    
#ifdef BUILD_CUDA
    ClearCufftWrapper();
#endif

#ifdef NO_MATRIX_TRANSPOSE
    if(lInplace)
        exit(1);    // In DoInit, this condition is checked
    
    lTaskConf.rowPlanner = false;
    
    if(!Parallel_FFT_1D(lOutputMemHandle, lInputMemType, lInputMemHandle, lOutputMemType, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;
#else
    if(!Parallel_Transpose(lOutputMemHandle, lInputMemType, lInputMemHandle, lOutputMemType, &lTaskConf, pCallbackHandle[1], pSchedulingPolicy))
        return (double)-1.0;
    
    lTaskConf.elemsX = lElemsY;
    lTaskConf.elemsY = lElemsX;
    lTaskConf.powX = lPowY;
    lTaskConf.powY = lPowX;
    lTaskConf.rowPlanner = false;
    
    if(!Parallel_FFT_1D(lInputMemHandle, lInputMemType, lOutputMemHandle, lOutputMemType, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;

    if(!Parallel_Transpose(lOutputMemHandle, lInputMemType, lInputMemHandle, lOutputMemType, &lTaskConf, pCallbackHandle[1], pSchedulingPolicy))
        return (double)-1.0;
#endif
    
    if(!lInplace)
    {
        pmMemHandle lTempMemHandle = lInputMemHandle;
        lInputMemHandle = lOutputMemHandle;
        lOutputMemHandle = lTempMemHandle;
    }

#endif

	double lEndTime = getCurrentTimeInSecs();
    
    if(pFetchBack)
    {
        SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );

        pmRawMemPtr lRawOutputPtr;
        pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);
        memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    }
    
	pmReleaseMemory(lOutputMemHandle);
    
    if(!lInplace)
        pmReleaseMemory(lInputMemHandle);
    
	return (lEndTime - lStartTime);
}

pmCallbacks DoSetDefaultCallbacks()
{
	pmCallbacks lCallbacks;

	lCallbacks.dataDistribution = fftDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = fft_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = fft_cudaLaunchFunc;
#endif

	return lCallbacks;
}

pmCallbacks DoSetDefaultCallbacks2()
{
	pmCallbacks lCallbacks;

#ifdef FFT_2D
	lCallbacks.dataDistribution = matrixTranspose::matrixTransposeDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixTranspose::matrixTranspose_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = matrixTranspose::matrixTranspose_cudaLaunchFunc;
#endif
#endif
	return lCallbacks;
}

// Returns 0 on success; non-zero on failure
int DoInit(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

    if(lElemsX < ROWS_PER_FFT_SUBTASK)
    {
        std::cout << "[ERROR]: No. of cols should be more than " << ROWS_PER_FFT_SUBTASK << std::endl;
        exit(1);
    }
    
#ifdef NO_MATRIX_TRANSPOSE
#ifdef FFT_2D
    if(lInplace)
        std::cout << "[ERROR]: Inplace 2D FFT not implemented with NO_MATRIX_TRANSPOSE" << std::endl;
#endif
#endif
    
	srand((unsigned int)time(NULL));

    size_t lElems = lElemsX * lElemsY;
    gSampleInput = (lInplace ? NULL : new FFT_DATA_TYPE[lElems]);
	gSerialOutput = new FFT_DATA_TYPE[lElems];
	gParallelOutput = new FFT_DATA_TYPE[lElems];

    if(lInplace)
    {
        for(size_t i = 0; i < lElems; ++i)
        {
            gSerialOutput[i].x = gParallelOutput[i].x = i;  //(float)rand() / (float)RAND_MAX;
            gSerialOutput[i].y = gParallelOutput[i].y = i + 1;  //(float)rand() / (float)RAND_MAX;
        }
    }
    else
    {
        for(size_t i = 0; i < lElems; ++i)
        {
            gSampleInput[i].x = i;  //(float)rand() / (float)RAND_MAX;
            gSampleInput[i].y = i + 1;  //(float)rand() / (float)RAND_MAX;
        }
    }

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
    delete[] gSampleInput;
	delete[] gSerialOutput;
	delete[] gParallelOutput;

	return 0;
}

// Returns 0 if serial and parallel executions have produced same result; non-zero otherwise
int DoCompare(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

#if 0
	for(size_t i = 0; i < lElemsY; ++i)
    {
        for(size_t j = 0; j < lElemsX; ++j)
        {
            std::cout << "[" << gSerialOutput[i * lElemsX + j].x << ", " << gSerialOutput[i * lElemsX + j].y << "] (" << gParallelOutput[i * lElemsX + j].x << " ," << gParallelOutput[i * lElemsX + j].y << ") ";
        }

        std::cout << std::endl;
    }
#endif

    float EPSILON = 0.1;
    
    size_t lElems = lElemsX * lElemsY;
	for(size_t i = 0; i < lElems; ++i)
	{
        if(fabs(gSerialOutput[i].x - gParallelOutput[i].x) > EPSILON || fabs(gSerialOutput[i].y - gParallelOutput[i].y) > EPSILON)
		{
            std::cout << "Mismatch index " << i << " Serial Value = (" << gSerialOutput[i].x << ", " << gSerialOutput[i].y << ") Parallel Value = (" << gParallelOutput[i].x << ", " << gParallelOutput[i].y << ")" << " Diff (" << gSerialOutput[i].x - gParallelOutput[i].x << ", " << gSerialOutput[i].y - gParallelOutput[i].y << ")" << std::endl;
            return 1;
		}
	}

	return 0;
}
    
int DoPreSetupPostMpiInit(int argc, char** argv, int pCommonArgs)
{
    READ_NON_COMMON_ARGS
    
#ifdef BUILD_CUDA
    size_t lMemReqd = (sizeof(FFT_DATA_TYPE) * std::max<size_t>(lElemsX, lElemsY) * ROWS_PER_FFT_SUBTASK);

    char lArray[64];
    sprintf(lArray, "%ld", lMemReqd);
    
    if(setenv("PMLIB_CUDA_MEM_PER_CARD_RESERVED_FOR_EXTERNAL_USE", lArray, 1) != 0)
    {
        std::cout << "Error in setting env variable PMLIB_CUDA_MEM_PER_CARD_RESERVED_FOR_EXTERNAL_USE" << std::endl;
        exit(1);
    }
#endif

    gRowPlanner.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, lElemsY, lElemsX, ROWS_PER_FFT_SUBTASK);

#ifdef FFT_2D
    gColPlanner.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, lElemsX, lElemsY, ROWS_PER_FFT_SUBTASK);
#endif
    
#ifdef NO_MATRIX_TRANSPOSE
    gSplitRowPlanner.CreateDummyPlan(false, FORWARD_TRANSFORM_DIRECTION, lElemsY, lElemsX, ROWS_PER_FFT_SUBTASK / gDevicesPerSplitGroup);
    gSplitLastRowPlanner.CreateDummyPlan(false, FORWARD_TRANSFORM_DIRECTION, lElemsY, lElemsX, ROWS_PER_FFT_SUBTASK - (gDevicesPerSplitGroup - 1) * (ROWS_PER_FFT_SUBTASK / gDevicesPerSplitGroup));

#ifdef FFT_2D
    gSplitColPlanner.CreateDummyPlan(false, FORWARD_TRANSFORM_DIRECTION, lElemsX, lElemsY, ROWS_PER_FFT_SUBTASK / gDevicesPerSplitGroup);
    gSplitLastColPlanner.CreateDummyPlan(false, FORWARD_TRANSFORM_DIRECTION, lElemsX, lElemsY, ROWS_PER_FFT_SUBTASK - (gDevicesPerSplitGroup - 1) * (ROWS_PER_FFT_SUBTASK / gDevicesPerSplitGroup));
#endif
#endif
    
    return 0;
}

/**	Non-common args
 *	1. log 2 nx
 *	2. log 2 ny
 */
int main(int argc, char** argv)
{
    RequestPreSetupCallbackPostMpiInit(DoPreSetupPostMpiInit);
    
    callbackStruct lStruct[2] = { {DoSetDefaultCallbacks, "FFT"}, {DoSetDefaultCallbacks2, "MATRIXTRANSPOSE"} };

	commonStart(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoCompare, DoDestroy, lStruct, 2);

	commonFinish();

	return 0;
}

}


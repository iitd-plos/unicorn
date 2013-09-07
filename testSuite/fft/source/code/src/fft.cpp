
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "fft.h"

#include <fftw3.h>
#include <math.h>

namespace fft
{
    
FFT_DATA_TYPE* gSampleInput;
FFT_DATA_TYPE* gSerialOutput;
FFT_DATA_TYPE* gParallelOutput;

struct fftwPlanner
{
    fftwPlanner()
    : mPlan(NULL)
    {}
    
    void CreateDummyPlan(bool inplace, int dir, size_t N)
    {
        complex* lDummyInputData = new complex[N];
        complex* lDummyOutputData = (inplace ? NULL : (new complex[N]));

        fftwf_complex* inputData = (fftwf_complex*)lDummyInputData;
        fftwf_complex* outputData = (fftwf_complex*)lDummyOutputData;

        mPlan = fftwf_plan_dft_1d((int)N, inputData, (inplace ? inputData : outputData), ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);
        
        delete[] lDummyInputData;
        delete[] lDummyOutputData;
    }
    
    ~fftwPlanner()
    {
        if(mPlan)
            fftwf_destroy_plan(mPlan);
    }
    
    fftwf_plan mPlan;
};

fftwPlanner gRowPlanner;

#ifdef FFT_2D
    fftwPlanner gColPlanner;
#endif

void fftSerial1D(int dir, size_t pown, size_t N, complex* input, complex* output, bool rowPlanner)
{
    fftwf_complex* inputData = (fftwf_complex*)input;
    fftwf_complex* outputData = (fftwf_complex*)output;

#ifdef FFT_2D
    fftwf_execute_dft(rowPlanner ? gRowPlanner.mPlan : gColPlanner.mPlan, inputData, outputData);
#else
    fftwf_execute_dft(gRowPlanner.mPlan, inputData, outputData);
#endif
}

void fftSerial(complex* input, complex* output, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
{
#if 1

    size_t i;
    for(i = 0; i < nx; ++i)
       fftSerial1D(dir, powy, ny, input + i * ny, output + i * ny, true);

    #ifdef FFT_2D
        bool lInplace = (input == output);

        matrixTranspose::serialMatrixTranspose(lInplace, output, input, nx, ny);
        
        for(i = 0; i < ny; ++i)
           fftSerial1D(dir, powx, nx, input + i * nx, output + i * nx, false);

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
        size_t i;
        for(i = 0; i < nx; ++i)
           fftSerial1D(dir, powy, ny, input + i * ny, output + i * ny, true);
    #endif
    
#endif
}

pmStatus fftDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    lSubscriptionInfo.offset = ROWS_PER_FFT_SUBTASK * pSubtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
    lSubscriptionInfo.length = ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);

    if(lTaskConf->inplace)
    {
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);
    }
    else
    {
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, INPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);
    }

	return pmSuccess;
}

pmStatus fft_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);
    
    if(lTaskConf->inplace)
    {
        for(unsigned int i = 0; i < ROWS_PER_FFT_SUBTASK; ++i)
        {
            FFT_DATA_TYPE* lOutputLoc = (FFT_DATA_TYPE*)pSubtaskInfo.outputMem + (i * lTaskConf->elemsY);
            fftSerial1D(FORWARD_TRANSFORM_DIRECTION, lTaskConf->powY, lTaskConf->elemsY, lOutputLoc, lOutputLoc, lTaskConf->rowPlanner);
        }
    }
    else
    {
        for(unsigned int i = 0; i < ROWS_PER_FFT_SUBTASK; ++i)
        {
            FFT_DATA_TYPE* lInputLoc = (FFT_DATA_TYPE*)pSubtaskInfo.inputMem + (i * lTaskConf->elemsY);
            FFT_DATA_TYPE* lOutputLoc = (FFT_DATA_TYPE*)pSubtaskInfo.outputMem + (i * lTaskConf->elemsY);
            fftSerial1D(FORWARD_TRANSFORM_DIRECTION, lTaskConf->powY, lTaskConf->elemsY, lInputLoc, lOutputLoc, lTaskConf->rowPlanner);
        }
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

bool Parallel_FFT_1D(pmMemHandle pInputMemHandle, pmMemInfo pInputMemInfo, pmMemHandle pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    bool lInplace = (pInputMemHandle == pOutputMemHandle);

    unsigned long lSubtaskCount = pTaskConf->elemsX / ROWS_PER_FFT_SUBTASK;
	CREATE_TASK(0, 0, lSubtaskCount, pCallbackHandle, pSchedulingPolicy)

    if(lInplace)
    {
        lTaskDetails.disjointReadWritesAcrossSubtasks = true;
    }
    else
    {
        lTaskDetails.inputMemHandle = pInputMemHandle;
        lTaskDetails.inputMemInfo = pInputMemInfo;
    }

    lTaskDetails.outputMemInfo = pOutputMemInfo;
	lTaskDetails.outputMemHandle = pOutputMemHandle;

	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(fftTaskConf);

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
bool Parallel_Transpose(void* pInputMemHandle, pmMemInfo pInputMemInfo, void* pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    if(matrixTranspose::parallelMatrixTranspose(pTaskConf->powX, pTaskConf->powY, pTaskConf->elemsX, pTaskConf->elemsY, pInputMemHandle, pOutputMemHandle, pCallbackHandle, pSchedulingPolicy, pInputMemInfo, pOutputMemInfo) == -1.0)
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

    pmMemInfo lOutputMemInfo = (lInplace ? OUTPUT_MEM_READ_WRITE : OUTPUT_MEM_WRITE_ONLY);
    pmMemInfo lInputMemInfo = INPUT_MEM_READ_ONLY;
    
	double lStartTime = getCurrentTimeInSecs();

    if(!Parallel_FFT_1D(lInputMemHandle, lInputMemInfo, lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;

#ifdef FFT_2D
    if(!Parallel_Transpose(lOutputMemHandle, lInputMemInfo, lInputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle[1], pSchedulingPolicy))
        return (double)-1.0;
    
    lTaskConf.elemsX = lElemsY;
    lTaskConf.elemsY = lElemsX;
    lTaskConf.powX = lPowY;
    lTaskConf.powY = lPowX;
    lTaskConf.rowPlanner = false;
    
    if(!Parallel_FFT_1D(lInputMemHandle, lInputMemInfo, lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle[0], pSchedulingPolicy))
        return (double)-1.0;

    if(!Parallel_Transpose(lOutputMemHandle, lInputMemInfo, lInputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle[1], pSchedulingPolicy))
        return (double)-1.0;
    
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
	for(i = 0; i < lElemsY; ++i)
    {
        for(size_t j = 0; j < lElemsX; ++j)
        {
            std::cout << gSerialOutput[i * lMatrixDimRows + j] << "(" << gParallelOutput[i * lMatrixDimRows + j] << ") ";
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
    
    gRowPlanner.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, lElemsY);

#ifdef FFT_2D
    gColPlanner.CreateDummyPlan(lInplace, FORWARD_TRANSFORM_DIRECTION, lElemsX);
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


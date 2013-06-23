
#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "fft.h"

#include <fftw3.h>
#include <math.h>

namespace fft
{
    
FFT_DATA_TYPE* gSerialOutput;
FFT_DATA_TYPE* gParallelOutput;

struct fftwPlanner
{
    fftwPlanner()
    : mPlan(NULL)
    {}
    
    void CreateDummyPlan(int dir, size_t N)
    {
        complex* lDummyData = new complex[N];
        fftwf_complex* data = (fftwf_complex*)lDummyData;

        mPlan = fftwf_plan_dft_1d((int)N, data, data, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);
        
        delete[] lDummyData;
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

void fftSerial1D(int dir, size_t pown, size_t N, complex* input, bool rowPlanner)
{
    fftwf_complex* data = (fftwf_complex*)input;

#ifdef FFT_2D
    fftwf_execute_dft(rowPlanner ? gRowPlanner.mPlan : gColPlanner.mPlan, data, data);
#else
    fftwf_execute_dft(gRowPlanner.mPlan, data, data);
#endif
}

void fftSerial2D(complex* input, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
{
#if 1

    size_t i;
    for(i = 0; i < nx; ++i)
       fftSerial1D(dir, powy, ny, input + i * ny, true);

    #ifdef FFT_2D
        matrixTranspose::serialmatrixTranspose(input, nx, ny);
        
        for(i = 0; i < ny; ++i)
           fftSerial1D(dir, powx, nx, input + i * nx, false);

        matrixTranspose::serialmatrixTranspose(input, ny, nx);
    #endif
    
#else
    
    #ifdef FFT_2D
        fftwf_complex* data = (fftwf_complex*)input;
        fftwf_plan lPlan = fftwf_plan_dft_2d((int)ny, (int)nx, data, data, ((dir == FORWARD_TRANSFORM_DIRECTION) ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE | FFTW_UNALIGNED);

        fftwf_execute(lPlan);
        fftwf_destroy_plan(lPlan);
    #else
        size_t i;
        for(i = 0; i < nx; ++i)
           fftSerial1D(dir, powy, ny, input + i * ny, true);    
    #endif
    
#endif
}

pmStatus fftDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    lSubscriptionInfo.offset = ROWS_PER_FFT_SUBTASK * pSubtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
    lSubscriptionInfo.length = ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_READ_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	return pmSuccess;
}

pmStatus fft_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    for(unsigned int i = 0; i < ROWS_PER_FFT_SUBTASK; ++i)
        fftSerial1D(FORWARD_TRANSFORM_DIRECTION, lTaskConf->powY, lTaskConf->elemsY, (FFT_DATA_TYPE*)pSubtaskInfo.outputMem + (i * lTaskConf->elemsY), lTaskConf->rowPlanner);

	return pmSuccess;
}

#define READ_NON_COMMON_ARGS \
    int lPowX = DEFAULT_POW_X; \
    int lPowY = DEFAULT_POW_Y; \
    FETCH_INT_ARG(lPowX, pCommonArgs, argc, argv); \
    FETCH_INT_ARG(lPowY, pCommonArgs + 1, argc, argv); \
    size_t lElemsX = 1 << lPowX; \
    size_t lElemsY = 1 << lPowY;

// Returns execution time on success; 0 on error
double DoSerialProcess(int argc, char** argv, int pCommonArgs)
{
	READ_NON_COMMON_ARGS

    double lStartTime = getCurrentTimeInSecs();

	fftSerial2D(gSerialOutput, lPowX, lElemsX, lPowY, lElemsY, FORWARD_TRANSFORM_DIRECTION);

	double lEndTime = getCurrentTimeInSecs();

	return (lEndTime - lStartTime);
}
    
// Returns execution time on success; 0 on error
double DoSingleGpuProcess(int argc, char** argv, int pCommonArgs)
{
#ifdef BUILD_CUDA
	READ_NON_COMMON_ARGS

    double lStartTime = getCurrentTimeInSecs();
    
	if(fftSingleGpu2D(gParallelOutput, lPowX, lElemsX, lPowY, lElemsY, FORWARD_TRANSFORM_DIRECTION) != 0)
        return 0;
    
	double lEndTime = getCurrentTimeInSecs();
    
	return (lEndTime - lStartTime);
#else
    return 0;
#endif
}

bool Parallel_FFT_1D(pmMemHandle pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    unsigned long lSubtaskCount = pTaskConf->elemsX / ROWS_PER_FFT_SUBTASK;
	CREATE_TASK(0, 0, lSubtaskCount, pCallbackHandle, pSchedulingPolicy)

    lTaskDetails.outputMemInfo = pOutputMemInfo;

	lTaskDetails.inputMemHandle = NULL;
	lTaskDetails.outputMemHandle = pOutputMemHandle;

	lTaskDetails.taskConf = (void*)(pTaskConf);
	lTaskDetails.taskConfLength = sizeof(fftTaskConf);
    
    lTaskDetails.sameReadWriteSubscriptions = true;

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
bool Parallel_Transpose(void* pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    if(matrixTranspose::parallelMatrixTranspose(pTaskConf->powX, pTaskConf->powY, pTaskConf->elemsX, pTaskConf->elemsY, pOutputMemHandle, pCallbackHandle, pSchedulingPolicy, pOutputMemInfo) == -1.0)
        return false;

	return true;
}
#endif
    
// Returns execution time on success; 0 on error
double DoParallelProcess(int argc, char** argv, int pCommonArgs, pmCallbackHandle pCallbackHandle1, pmCallbackHandle pCallbackHandle2, pmSchedulingPolicy pSchedulingPolicy)
{
	READ_NON_COMMON_ARGS

	// Input Mem is null
	// Output Mem contains the input fft data and will receive result inplace
	// Number of subtasks is equal to the number of rows/cols for row/col-major processing
	pmMemHandle lOutputMemHandle;
    size_t lElems = lElemsX * lElemsY;
    size_t lMemSize = lElems * sizeof(FFT_DATA_TYPE);
    
	CREATE_MEM(lMemSize, lOutputMemHandle);
    
	pmRawMemPtr lRawOutputPtr;
	pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);
	memcpy(lRawOutputPtr, gParallelOutput, lMemSize);

	fftTaskConf lTaskConf;
    lTaskConf.elemsX = lElemsX;
    lTaskConf.elemsY = lElemsY;
    lTaskConf.powX = lPowX;
    lTaskConf.powY = lPowY;
    lTaskConf.rowPlanner = true;

    pmMemInfo lOutputMemInfo = OUTPUT_MEM_READ_WRITE;
    
	double lStartTime = getCurrentTimeInSecs();

    if(!Parallel_FFT_1D(lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle1, pSchedulingPolicy))
        return (double)-1.0;

#ifdef FFT_2D
    if(!Parallel_Transpose(lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle2, pSchedulingPolicy))
        return (double)-1.0;
    
    lTaskConf.elemsX = lElemsY;
    lTaskConf.elemsY = lElemsX;
    lTaskConf.powX = lPowY;
    lTaskConf.powY = lPowX;
    lTaskConf.rowPlanner = false;
    
    if(!Parallel_FFT_1D(lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle1, pSchedulingPolicy))
        return (double)-1.0;

    if(!Parallel_Transpose(lOutputMemHandle, lOutputMemInfo, &lTaskConf, pCallbackHandle2, pSchedulingPolicy))
        return (double)-1.0;
#endif

	double lEndTime = getCurrentTimeInSecs();
    
	SAFE_PM_EXEC( pmFetchMemory(lOutputMemHandle) );
    
	pmGetRawMemPtr(lOutputMemHandle, &lRawOutputPtr);
	memcpy(gParallelOutput, lRawOutputPtr, lMemSize);
    
	pmReleaseMemory(lOutputMemHandle);
    
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
	gSerialOutput = new FFT_DATA_TYPE[lElems];
	gParallelOutput = new FFT_DATA_TYPE[lElems];

	for(size_t i=0; i<lElems; ++i)
    {
		gSerialOutput[i].x = gParallelOutput[i].x = i;  //(float)rand() / (float)RAND_MAX;
		gSerialOutput[i].y = gParallelOutput[i].y = i + 1;  //(float)rand() / (float)RAND_MAX;
    }

	return 0;
}

// Returns 0 on success; non-zero on failure
int DoDestroy()
{
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
    
    gRowPlanner.CreateDummyPlan(FORWARD_TRANSFORM_DIRECTION, lElemsY);

#ifdef FFT_2D
    gColPlanner.CreateDummyPlan(FORWARD_TRANSFORM_DIRECTION, lElemsX);
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
    
	// All the functions pointers passed here are executed only on the host submitting the task
	commonStart2(argc, argv, DoInit, DoSerialProcess, DoSingleGpuProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "FFT", DoSetDefaultCallbacks2, "MatrixTranspose");

	commonFinish();

	return 0;
}

}

#include <time.h>
#include <string.h>

#include "commonAPI.h"
#include "fft.h"

#include <math.h>

namespace fft
{
    
FFT_DATA_TYPE* gSerialOutput;
FFT_DATA_TYPE* gParallelOutput;

/*-------------------------------------------------------------------------
FFT Code taken from http://local.wasp.uwa.edu.au/~pbourke/miscellaneous/dft/

This computes an in-place complex-to-complex FFT
x and y are the real and imaginary arrays of 2^m points.
dir =  1 gives forward transform
dir = -1 gives reverse transform

Formula: forward
N-1
---
1   \          - j k 2 pi n / N
X(n) = ---   >   x(k) e                    = forward transform
N   /                                n=0..N-1
---
k=0

Formula: reverse
N-1
---
\          j k 2 pi n / N
X(n) =       >   x(k) e                    = forward transform
/                                n=0..N-1
---
k=0
*/
void fftSerial1D(int dir, unsigned long m, unsigned long nn, complex* data)
{
	unsigned long i, i1, j, k, i2, l, l1, l2;
	float c1, c2, tx, ty, t1, t2, u1, u2, z;

	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for(i=0; i<nn-1; i++)
	{
		if (i < j)
		{
			tx = data[i].x;
			ty = data[i].y;
			data[i].x = data[j].x;
			data[i].y = data[j].y;
			data[j].x = tx;
			data[j].y = ty;
		}

		k = i2;
		while (k <= j)
		{
			j -= k;
			k >>= 1;
		}

		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for(l=0; l<m; l++)
	{
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;

		for(j=0; j<l1; j++)
		{
			for(i=j;i<nn;i+=l2)
			{
				i1 = i + l1;
				t1 = u1 * data[i1].x - u2 * data[i1].y;
				t2 = u1 * data[i1].y + u2 * data[i1].x;
				data[i1].x = data[i].x - t1;
				data[i1].y = data[i].y - t2;
				data[i].x += t1;
				data[i].y += t2;
			}

			z =  u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}

		c2 = (float)sqrt((1.0 - c1) / 2.0);

		if(dir == 1)
			c2 = -c2;

		c1 = (float)sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
//	if(dir == 1)
//	{
//		for(i = 0; i < nn; ++i)
//		{
//			data[i].x /= (float)nn;
//			data[i].y /= (float)nn;
//		}
//	}
}

/*-------------------------------------------------------------------------
Perform a 2D FFT inplace given a complex 2D array
The direction dir, 1 for forward, -1 for reverse
The size of the array (nx, ny)
Return false if there are memory problems or
the dimensions are not powers of 2
*/
void fftSerial2D(complex* input, unsigned long powx, unsigned long nx, unsigned long powy, unsigned long ny, int dir)
{
	size_t i;
	for(i=0; i<nx; ++i)
		fftSerial1D(dir, powy, ny, input+i*ny);

#ifdef FFT_2D
    matrixTranspose::serialmatrixTranspose(input, nx, ny);
	
    for(i=0; i<ny; ++i)
		fftSerial1D(dir, powx, nx, input+i*nx);

    matrixTranspose::serialmatrixTranspose(input, ny, nx);
#endif
}

pmStatus fftDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId)
{
	pmSubscriptionInfo lSubscriptionInfo;
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    lSubscriptionInfo.offset = ROWS_PER_FFT_SUBTASK * pSubtaskId * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);
    lSubscriptionInfo.length = ROWS_PER_FFT_SUBTASK * lTaskConf->elemsY * sizeof(FFT_DATA_TYPE);

    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_READ_SUBSCRIPTION, lSubscriptionInfo);
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, pSubtaskId, OUTPUT_MEM_WRITE_SUBSCRIPTION, lSubscriptionInfo);

	return pmSuccess;
}

pmStatus fft_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    for(unsigned int i = 0; i < ROWS_PER_FFT_SUBTASK; ++i)
        fftSerial1D(FORWARD_TRANSFORM_DIRECTION, lTaskConf->powY, lTaskConf->elemsY, (FFT_DATA_TYPE*)pSubtaskInfo.outputMem + (i * lTaskConf->elemsY));

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

bool Parallel_FFT_1D(pmMemHandle pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    unsigned long lSubtaskCount = pTaskConf->elemsX / ROWS_PER_FFT_SUBTASK;
	CREATE_TASK(0, 0, lSubtaskCount, pCallbackHandle, pSchedulingPolicy)

    lTaskDetails.outputMemInfo = pOutputMemInfo;

	lTaskDetails.inputMemHandle = NULL;
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

bool Parallel_Transpose(void* pOutputMemHandle, pmMemInfo pOutputMemInfo, fftTaskConf* pTaskConf, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy)
{
    if(matrixTranspose::parallelMatrixTranspose(pTaskConf->powX, pTaskConf->powY, pTaskConf->elemsX, pTaskConf->elemsY, pOutputMemHandle, pCallbackHandle, pSchedulingPolicy, pOutputMemInfo) == -1.0)
        return false;

	return true;
}

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

	lCallbacks.dataDistribution = matrixTranspose::matrixTransposeDataDistribution;
	lCallbacks.deviceSelection = NULL;
	lCallbacks.subtask_cpu = matrixTranspose::matrixTranspose_cpu;

#ifdef BUILD_CUDA
	lCallbacks.subtask_gpu_custom = matrixTranspose::matrixTranspose_cudaLaunchFunc;
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

	std::cout << "Perfect match against serial execution" << std::endl;
	return 0;
}

/**	Non-common args
 *	1. log 2 nx
 *	2. log 2 ny
 */
int main(int argc, char** argv)
{
	// All the five functions pointers passed here are executed only on the host submitting the task
	commonStart2(argc, argv, DoInit, DoSerialProcess, DoParallelProcess, DoSetDefaultCallbacks, DoCompare, DoDestroy, "FFT", DoSetDefaultCallbacks2, "MatrixTranspose");

	commonFinish();

	return 0;
}

}
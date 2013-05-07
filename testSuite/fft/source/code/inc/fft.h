
namespace fft
{
    
#define DEFAULT_POW_X 10
#define DEFAULT_POW_Y 10

#define ROWS_PER_FFT_SUBTASK 128  // must be a power of 2
    
#ifndef FFT_DATA_TYPE
#error "FFT_DATA_TYPE not defined"
#endif

#ifndef MATRIX_DATA_TYPE
#error "MATRIX_DATA_TYPE not defined"
#endif

#define FORWARD_TRANSFORM_DIRECTION 1
#define REVERSE_TRANSFORM_DIRECTION 0

#define FFT_2D

using namespace pm;

typedef struct fftTaskConf
{
	size_t elemsX;  // rows
    size_t elemsY;  // cols
    size_t powX;
    size_t powY;
} fftTaskConf;

#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
#endif

}

#ifdef FFT_2D
namespace matrixTranspose
{
    using namespace pm;

    void serialmatrixTranspose(MATRIX_DATA_TYPE* pMatrix, size_t pInputDimRows, size_t pInputDimCols);

    pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId);

    pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    double parallelMatrixTranspose(size_t pPowRows, size_t pPowCols, size_t pMatrixDimRows, size_t pMatrixDimCols, pmMemHandle pMemHandle, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemInfo pMemInfo);

#ifdef BUILD_CUDA
    pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
#endif
}
#endif

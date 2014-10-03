
namespace fft
{

#define USE_SQUARE_MATRIX
#define NO_MATRIX_TRANSPOSE

#define DEFAULT_DIM_X 4096
    
#ifndef USE_SQUARE_MATRIX
#define DEFAULT_DIM_Y 2048
#endif

#define DEFAULT_INPLACE_VALUE 0

#define ROWS_PER_FFT_SUBTASK 2048  // must be a power of 2
    
#ifndef FFT_DATA_TYPE
#error "FFT_DATA_TYPE not defined"
#endif

#define FORWARD_TRANSFORM_DIRECTION 1
#define REVERSE_TRANSFORM_DIRECTION 0

#if defined(FFT_1D) && defined(FFT_2D)
#error "Both FFT_1D and FFT_2D defined !!!"
#endif
    
#if !defined(FFT_1D) && !defined(FFT_2D)
#define FFT_2D
#endif

#ifdef FFT_2D
    #ifndef MATRIX_DATA_TYPE
    #error "MATRIX_DATA_TYPE not defined"
    #endif
#endif

using namespace pm;

enum inplaceMemIndex
{
    INPLACE_MEM_INDEX = 0,
    INPLACE_MAX_MEM_INDICES
};
    
enum memIndex
{
    INPUT_MEM_INDEX = 0,
    OUTPUT_MEM_INDEX,
    MAX_MEM_INDICES
};

typedef struct fftTaskConf
{
	size_t elemsX;  // rows
    size_t elemsY;  // cols
    bool rowPlanner;
    bool inplace;
} fftTaskConf;

#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int fftSingleGpu2D(bool inplace, complex* input, complex* output, size_t powx, size_t nx, size_t powy, size_t ny, int dir);
#endif

}

#ifdef FFT_2D
namespace matrixTranspose
{
    using namespace pm;

    void serialMatrixTranspose(bool pInplace, MATRIX_DATA_TYPE* pInputMatrix, MATRIX_DATA_TYPE* pOutputMatrix, size_t pInputDimRows, size_t pInputDimCols);

    pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    double parallelMatrixTranspose(size_t pMatrixDimRows, size_t pMatrixDimCols, pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemType pInputMemType, pmMemType pOutputMemType);

#ifdef BUILD_CUDA
    pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
#endif
}
#endif

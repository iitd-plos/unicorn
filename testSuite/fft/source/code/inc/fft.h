
namespace fft
{
    
#define DEFAULT_POW_X 10
#define DEFAULT_POW_Y 10
    
#define FFT_DATA_TYPE complex

#ifndef MATRIX_DATA_TYPE
#error "MATRIX_DATA_TYPE not defined"
#endif

#define FORWARD_TRANSFORM_DIRECTION 1
#define REVERSE_TRANSFORM_DIRECTION 0

#define FFT_2D

using namespace pm;

void fft_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);

typedef struct fftTaskConf
{
	size_t elemsX;  // rows
    size_t elemsY;  // cols
    size_t powX;
    size_t powY;
} fftTaskConf;

}

#ifdef FFT_2D
namespace matrixTranspose
{
    void serialmatrixTranspose(MATRIX_DATA_TYPE* pMatrix, size_t pInputDimRows, size_t pInputDimCols);

    pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId);

    pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    double parallelMatrixTranspose(size_t pPowRows, size_t pPowCols, size_t pMatrixDimRows, size_t pMatrixDimCols, pmMemHandle pMemHandle, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemInfo pMemInfo);
}
#endif

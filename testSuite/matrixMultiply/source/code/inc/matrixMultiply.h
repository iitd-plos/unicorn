
#define DEFAULT_MATRIX_DIM 1000
#define MATRIX_DATA_TYPE int

using namespace pm;

#ifdef BUILD_CUDA
#include <cuda.h>
void matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct matrixMultiplyTaskConf
{
	int matrixDim;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} matrixMultiplyTaskConf;

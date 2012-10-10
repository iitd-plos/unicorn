
#define DEFAULT_MATRIX_DIM 1000
#define MATRIX_DATA_TYPE int

using namespace pm;

void matrixMultiply_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);

typedef struct matrixMultiplyTaskConf
{
	int matrixDim;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} matrixMultiplyTaskConf;

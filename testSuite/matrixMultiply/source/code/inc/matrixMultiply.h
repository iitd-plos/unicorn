
namespace matrixMultiply
{

#define DEFAULT_MATRIX_DIM 1000
#define MATRIX_DATA_TYPE int

using namespace pm;

#ifdef BUILD_CUDA
#include <cuda.h>
typedef void (*matrixMultiply_cudaFuncPtr)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
int singleGpuMatrixMultiply(MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, int pDim);
extern matrixMultiply_cudaFuncPtr matrixMultiply_cudaFunc;
#endif

typedef struct matrixMultiplyTaskConf
{
	int matrixDim;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} matrixMultiplyTaskConf;

}
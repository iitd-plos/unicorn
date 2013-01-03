
#define DEFAULT_POW_ROWS 10
#define DEFAULT_POW_COLS 11
#define MAX_BLOCK_SIZE 512   // Must be a power of 2
#define MATRIX_DATA_TYPE float

using namespace pm;

#ifdef BUILD_CUDA
#include <cuda.h>
void matrixTranspose_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct matrixTransposeTaskConf
{
	size_t matrixDimRows;
	size_t matrixDimCols;
    size_t blockSizeRows;
    size_t blockSizeCols;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} matrixTransposeTaskConf;


namespace matrixTranspose
{

#define DEFAULT_POW_ROWS 14
#define DEFAULT_POW_COLS 12
#define DEFAULT_INPLACE_VALUE 0

#define MAX_BLOCK_SIZE 4096   // Must be a power of 2

#ifndef MATRIX_DATA_TYPE
    #define MATRIX_DATA_TYPE int
    #define MATRIX_DATA_TYPE_INT 1
#else
    #define MATRIX_DATA_TYPE_INT 0
#endif

#define USE_SQUARE_BLOCKS     // Using square blocks ensures that there will be no overlap of input and output locations within a block

using namespace pm;

#ifdef BUILD_CUDA
#ifdef USE_SQUARE_BLOCKS

#define GPU_TILE_DIM 32
#define GPU_ELEMS_PER_THREAD 4

#include <cuda.h>
pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuMatrixTranspose(bool pInplace, MATRIX_DATA_TYPE* pInputMatrix, MATRIX_DATA_TYPE* pOutputMatrix, size_t pInputDimRows, size_t pInputDimCols);

#else
#endif
#endif

typedef struct matrixTransposeTaskConf
{
	size_t matrixDimRows;
	size_t matrixDimCols;
    size_t blockSizeRows;
    size_t blockSizeCols;
    bool inplace;
} matrixTransposeTaskConf;

}
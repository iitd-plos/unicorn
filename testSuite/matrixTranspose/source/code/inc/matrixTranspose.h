
namespace matrixTranspose
{

#define DEFAULT_POW_ROWS 10
#define DEFAULT_POW_COLS 11
#define MAX_BLOCK_SIZE 512   // Must be a power of 2
    
#ifndef MATRIX_DATA_TYPE
#define MATRIX_DATA_TYPE float
#endif

#define USE_SQUARE_BLOCKS     // Using square blocks ensures that there will be no overlap of input and output locations within a block

using namespace pm;

#ifdef BUILD_CUDA
#ifdef USE_SQUARE_BLOCKS

#define GPU_TILE_DIM 32
#define GPU_ELEMS_PER_THREAD 4

#include <cuda.h>
pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

#else
#endif
#endif

typedef struct matrixTransposeTaskConf
{
	size_t matrixDimRows;
	size_t matrixDimCols;
    size_t blockSizeRows;
    size_t blockSizeCols;
} matrixTransposeTaskConf;

}
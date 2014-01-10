
namespace luDecomposition
{

#define DEFAULT_POW_DIM 12
const size_t BLOCK_DIM = 2048;   // Must be a power of 2

#define MATRIX_DATA_TYPE_FLOAT
//#define MATRIX_DATA_TYPE_DOUBLE

// For double precision build, compile CUDA for correct architecture e.g. Add this line to Makefile for Kepler GK105 "CUDAFLAGS += -gencode arch=compute_30,code=sm_30"

#ifdef MATRIX_DATA_TYPE_FLOAT
#define MATRIX_DATA_TYPE float
#else
#ifdef MATRIX_DATA_TYPE_DOUBLE
#define MATRIX_DATA_TYPE double
#endif
#endif
    
#ifndef MATRIX_DATA_TYPE
#error "MATRIX_DATA_TYPE not defined"
#endif
    
#define BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, matrixDim) (((blockRow) + (blockCol) * (matrixDim)) * BLOCK_DIM)

#define SUBSCRIBE_SPLIT_BLOCK(blockRow, blockCol, startCol, endCol, startRow, endRow, matrixDim, subtaskId, splitInfo, subscriptionType) \
{ \
    size_t dBlockOffset = (startRow + BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, matrixDim)) * sizeof(MATRIX_DATA_TYPE); \
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, subtaskId, splitInfo, OUTPUT_MEM_INDEX, subscriptionType, pmScatteredSubscriptionInfo(dBlockOffset + startCol * matrixDim * sizeof(MATRIX_DATA_TYPE), (endRow - startRow) * sizeof(MATRIX_DATA_TYPE), matrixDim * sizeof(MATRIX_DATA_TYPE), (endCol - startCol))); \
}

#define SUBSCRIBE_BLOCK(blockRow, blockCol, matrixDim, subtaskId, splitInfo, subscriptionType) SUBSCRIBE_SPLIT_BLOCK(blockRow, blockCol, 0, BLOCK_DIM, 0, BLOCK_DIM, matrixDim, subtaskId, splitInfo, subscriptionType)
    
#define SUBSCRIBE_SPLIT_COL_BLOCK(blockRow, blockCol, startCol, endCol, matrixDim, subtaskId, splitInfo, subscriptionType) SUBSCRIBE_SPLIT_BLOCK(blockRow, blockCol, startCol, endCol, 0, BLOCK_DIM, matrixDim, subtaskId, splitInfo, subscriptionType)

#define SUBSCRIBE_SPLIT_ROW_BLOCK(blockRow, blockCol, startRow, endRow, matrixDim, subtaskId, splitInfo, subscriptionType) SUBSCRIBE_SPLIT_BLOCK(blockRow, blockCol, 0, BLOCK_DIM, startRow, endRow, matrixDim, subtaskId, splitInfo, subscriptionType)

using namespace pm;

#ifdef BUILD_CUDA

#define GPU_TILE_DIM 32
#define GPU_ELEMS_PER_THREAD 4

#include <cuda.h>
pmStatus luDecomposition_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
pmStatus horizVertComp_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
pmStatus diagComp_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuLUDecomposition(MATRIX_DATA_TYPE* pMatrix, size_t pDim);

#endif

/*
 ---------           ---------          ---------
|A00| A01 |         |L00| L01 |        |U00| U01 |
 ---------     =     ---------    *     ---------
|   |     |         |   |     |        |   |     |
|A10| A11 |         |L10| L11 |        |U10| U11 |
|   |     |         |   |     |        |   |     |
 ---------           ---------          ---------

Stage 1:   LU_DECOMPOSE     Decompose A00 into L00 and U00
Stage 2:   HORZ_VERT_COMP   Find L10 and U01 (L01 and U10 are zero); A01 = L00 * U01; A10 = L10 * U00
Stage 3:   DIAGONAL_COMP    Find A11' = L11 * U11 = A11 - L10 * U01
*/
    
enum taskStage
{
    LU_DECOMPOSE,
    HORZ_VERT_COMP,
    DIAGONAL_COMP
};
    
enum memIndex
{
    OUTPUT_MEM_INDEX,
    MAX_MEM_INDICES
};

typedef struct luTaskConf
{
	size_t matrixDim;
} luTaskConf;

}
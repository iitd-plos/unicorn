
namespace matrixMultiplyBlas
{

#define DEFAULT_POW_MATRIX_DIM 12

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

#define BLOCK_DIM 4096

#define BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, blockDim, matrixDim) (((blockRow) * (matrixDim) + (blockCol)) * (blockDim))

#define SUBSCRIBE_BLOCK(blockRow, blockCol, blockOffset, blockWidth, blockDim, matrixDim, subtaskId, splitInfo, memIndex, subscriptionType) \
{ \
    size_t dBlockOffset = (BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, blockDim, matrixDim) + (blockOffset)) * sizeof(MATRIX_DATA_TYPE); \
    for(size_t row = 0; row < (blockDim); ++row) \
    { \
        lSubscriptionInfo.offset = dBlockOffset + row * matrixDim * sizeof(MATRIX_DATA_TYPE); \
        lSubscriptionInfo.length = (blockWidth) * sizeof(MATRIX_DATA_TYPE); \
        pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, subtaskId, splitInfo, memIndex, subscriptionType, lSubscriptionInfo); \
    } \
}

using namespace pm;
    
#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus matrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuMatrixMultiply(MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, int pDim);
#endif

enum memIndex
{
    INPUT_MATRIX1_MEM_INDEX = 0,
    INPUT_MATRIX2_MEM_INDEX,
    OUTPUT_MATRIX_MEM_INDEX,
    MAX_MEM_INDICES
};
    
typedef struct matrixMultiplyTaskConf
{
	size_t matrixDim;
    size_t blockDim;
} matrixMultiplyTaskConf;

bool GetSplitData(size_t* pBlockOffset, size_t* pBlockWidth, matrixMultiplyTaskConf* pTaskConf, pmSplitInfo& pSplitInfo);
    
}
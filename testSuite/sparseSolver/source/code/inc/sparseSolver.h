
namespace sparseSolver
{

#define DEFAULT_MATRIX_DIM 8192
    
#define SPARSITY_FACTOR 0.05
#define NON_SPARSE_ELEMENT_COUNT(matrixElems) (matrixElems * (1 - SPARSITY_FACTOR))

#define MATRIX_DATA_TYPE_FLOAT
//#define MATRIX_DATA_TYPE_DOUBLE
    
#define INDICES_TYPE int
#define COUNT_TYPE unsigned int
    
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

#define MATRIX_ROWS_PER_SUBTASK 256

using namespace pm;
    
#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus sparseMatrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuSparseMatrixMultiply(INDICES_TYPE* pSampleInputRowIndices, INDICES_TYPE* pSampleInputColIndices, MATRIX_DATA_TYPE* pInputMatrices, MATRIX_DATA_TYPE* pOutputMatrix, COUNT_TYPE pCountNnz1, COUNT_TYPE pCountNnz2, int pDim);
bool GetRowIndicesFromSplitData(size_t pFirstRow, size_t pRowOffset, size_t pRowCount, int* pDistributionData, INDICES_TYPE* pRowIndices1, int& pStartIndex, int& pEndIndex, pmSplitInfo& pSplitInfo);
#endif

enum memIndex
{
    INPUT_MATRIX1_MEM_INDEX = 0,
    INPUT_MATRIX2_MEM_INDEX,
    INPUT_ROW_INDICES1_MEM_INDEX,
    INPUT_ROW_INDICES2_MEM_INDEX,
    INPUT_COL_INDICES1_MEM_INDEX,
    INPUT_COL_INDICES2_MEM_INDEX,
    INPUT_MEM_NNZ1_INDEX,
    OUTPUT_MATRIX_MEM_INDEX,
    MAX_MEM_INDICES
};
    
typedef struct sparseMatrixMultiplyTaskConf
{
	size_t matrixDim;
    COUNT_TYPE nnz2;
    // Has data distribution information appended
} sparseMatrixMultiplyTaskConf;

bool GetSplitData(size_t* pBlockOffset, size_t* pBlockHeight, sparseMatrixMultiplyTaskConf* pTaskConf, pmSplitInfo& pSplitInfo);
    
}
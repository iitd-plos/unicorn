
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

namespace sparseSolver
{

#define DEFAULT_MATRIX_DIM 1024
#define MAX_VAL(x, y) ((x > y) ? x : y)
    
#define SPARSITY_FACTOR 0.001
#define NON_SPARSE_ELEMENT_COUNT(matrixElems) MAX_VAL(1, (matrixElems * SPARSITY_FACTOR))

#define MATRIX_DATA_TYPE_FLOAT
//#define MATRIX_DATA_TYPE_DOUBLE
    
#define INDICES_TYPE int
#define COUNT_TYPE unsigned int
    
#define BLOCK_DIM 2048
    
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

#define BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, blockDim, matrixDim) (((blockRow) * (matrixDim) + (blockCol)) * (blockDim))

#define SUBSCRIBE_BLOCK(blockRow, blockCol, blockOffset, blockHeight, blockDim, matrixDim, subtaskId, splitInfo, memIndex, subscriptionType) \
{ \
    pmSubscribeToMemory(pTaskInfo.taskHandle, pDeviceInfo.deviceHandle, subtaskId, splitInfo, memIndex, subscriptionType, pmScatteredSubscriptionInfo(((blockOffset * matrixDim) + BLOCK_OFFSET_IN_ELEMS(blockRow, blockCol, blockDim, matrixDim)) * sizeof(MATRIX_DATA_TYPE), blockDim * sizeof(MATRIX_DATA_TYPE), matrixDim * sizeof(MATRIX_DATA_TYPE), (blockHeight))); \
}

using namespace pm;
    
#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus sparseMatrixMultiply_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuSparseMatrixMultiply(INDICES_TYPE* pSampleInputRowIndices, INDICES_TYPE* pSampleInputColIndices, MATRIX_DATA_TYPE* pSampleInputData1, MATRIX_DATA_TYPE* pSampleInputData2, MATRIX_DATA_TYPE* pOutputMatrix, COUNT_TYPE pCountNnz1, int pDim);
void FreeCusparseHandles();
#endif

enum memIndex
{
    INPUT_MATRIX1_MEM_INDEX = 0,
    INPUT_MATRIX2_MEM_INDEX,
    INPUT_ROW_INDICES1_MEM_INDEX,
    INPUT_COL_INDICES1_MEM_INDEX,
    INPUT_MEM_NNZ1_INDEX,
    OUTPUT_MATRIX_MEM_INDEX,
    MAX_MEM_INDICES
};
    
typedef struct sparseMatrixMultiplyTaskConf
{
	size_t matrixDim;
    size_t blockDim;
    // Has sparse matrix's data distribution information appended
} sparseMatrixMultiplyTaskConf;
    
}

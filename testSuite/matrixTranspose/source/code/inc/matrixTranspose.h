
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

namespace matrixTranspose
{

#define DEFAULT_POW_ROWS 14
#define DEFAULT_POW_COLS 12
#define DEFAULT_INPLACE_VALUE 0

#define MAX_BLOCK_SIZE 2048   // Must be a power of 2

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
    
enum inplaceMemIndex
{
    INPLACE_MEM_INDEX = 0,
    INPLACE_MAX_MEM_INDICES
};
    
enum memIndex
{
    INPUT_MEM_INDEX = 0,
    OUTPUT_MEM_INDEX,
    MAX_MEM_INDICES
};

typedef struct matrixTransposeTaskConf
{
	size_t matrixDimRows;
	size_t matrixDimCols;
    size_t blockSizeRows;
    size_t blockSizeCols;
    bool inplace;
} matrixTransposeTaskConf;

}

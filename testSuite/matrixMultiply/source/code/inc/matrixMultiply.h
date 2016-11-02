
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
    
enum memIndex
{
    INPUT_MATRIX1_MEM_INDEX = 0,
    INPUT_MATRIX2_MEM_INDEX,
    OUTPUT_MATRIX_MEM_INDEX,
    MAX_MEM_INDICES
};

typedef struct matrixMultiplyTaskConf
{
	int matrixDim;
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
} matrixMultiplyTaskConf;

}

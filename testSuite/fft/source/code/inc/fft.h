
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

namespace fft
{

#define USE_SQUARE_MATRIX
#define NO_MATRIX_TRANSPOSE

#define DEFAULT_DIM_X 4096
    
#ifndef USE_SQUARE_MATRIX
#define DEFAULT_DIM_Y 2048
#endif

#define DEFAULT_INPLACE_VALUE 0

#define ROWS_PER_FFT_SUBTASK 512  // must be a power of 2
    
#ifndef FFT_DATA_TYPE
#error "FFT_DATA_TYPE not defined"
#endif

#define FORWARD_TRANSFORM_DIRECTION 1
#define REVERSE_TRANSFORM_DIRECTION 0

#if defined(FFT_1D) && defined(FFT_2D)
#error "Both FFT_1D and FFT_2D defined !!!"
#endif
    
#if !defined(FFT_1D) && !defined(FFT_2D)
#define FFT_2D
#endif

#ifdef FFT_2D
    #ifndef MATRIX_DATA_TYPE
    #error "MATRIX_DATA_TYPE not defined"
    #endif
#endif

using namespace pm;

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

typedef struct fftTaskConf
{
	size_t elemsX;  // rows
    size_t elemsY;  // cols
    bool rowPlanner;
    bool inplace;
} fftTaskConf;

#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int fftSingleGpu2D(bool inplace, complex* input, complex* output, size_t nx, size_t ny, int dir);
#endif

}

#ifdef FFT_2D
namespace matrixTranspose
{
    using namespace pm;

    void serialMatrixTranspose(bool pInplace, MATRIX_DATA_TYPE* pInputMatrix, MATRIX_DATA_TYPE* pOutputMatrix, size_t pInputDimRows, size_t pInputDimCols);

    pmStatus matrixTransposeDataDistribution(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    pmStatus matrixTranspose_cpu(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);

    double parallelMatrixTranspose(size_t pMatrixDimRows, size_t pMatrixDimCols, pmMemHandle pInputMemHandle, pmMemHandle pOutputMemHandle, pmCallbackHandle pCallbackHandle, pmSchedulingPolicy pSchedulingPolicy, pmMemType pInputMemType, pmMemType pOutputMemType);

#ifdef BUILD_CUDA
    pmStatus matrixTranspose_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
#endif
}
#endif

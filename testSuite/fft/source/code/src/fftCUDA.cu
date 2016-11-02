
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

#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "commonAPI.h"
#include "fft.h"

#include <cufft.h>

#include <map>

namespace fft
{

#define CUFFT_ERROR_CHECK(name, x) \
{ \
    cufftResult dResult = x; \
    if(dResult != CUFFT_SUCCESS) \
    { \
        std::cout << name << " failed with error " << dResult << std::endl; \
        exit(1); \
    } \
}

struct cufftWrapper
{
    cufftWrapper()
    {}
    
    ~cufftWrapper()
    {
        Clear();
    }
    
    void Clear()
    {
        std::map<std::pair<int, size_t>, cufftHandle>::iterator lIter = cufftMap.begin(), lEndIter = cufftMap.end();
        
        for(; lIter != lEndIter; ++lIter)
            CUFFT_ERROR_CHECK("cufftDestroy", cufftDestroy(lIter->second));
        
        cufftMap.clear();
    }

    std::map<std::pair<int, size_t>, cufftHandle> cufftMap;  // pair<deviceId, transformSize> versus cufftPlan1d handle
};

struct cufftManyWrapper
{
    cufftManyWrapper()
    {}
    
    ~cufftManyWrapper()
    {
        Clear();
    }
    
    void Clear()
    {
        std::map<std::pair<int, std::pair<size_t, size_t> >, cufftHandle>::iterator lIter = cufftMap.begin(), lEndIter = cufftMap.end();
        
        for(; lIter != lEndIter; ++lIter)
            CUFFT_ERROR_CHECK("cufftDestroy", cufftDestroy(lIter->second));
        
        cufftMap.clear();
    }

    std::map<std::pair<int, std::pair<size_t, size_t> >, cufftHandle> cufftMap;  // pair<deviceId, pair<transformSize, rowStride>> versus cufftPlan1d handle
};

cufftWrapper& GetCufftWrapper()
{
    static cufftWrapper sWrapper;

    return sWrapper;
}

// A different plan is required for every GPU
cufftHandle GetCufftPlan1d(size_t pElemsY)
{
    cufftWrapper& lWrapper = GetCufftWrapper();

    int lDeviceId;
    CUDA_ERROR_CHECK("cudaGetDevice", cudaGetDevice(&lDeviceId));

    std::pair<int, size_t> lPair(lDeviceId, pElemsY);
    std::map<std::pair<int, size_t>, cufftHandle>::iterator lIter = lWrapper.cufftMap.find(lPair);

    if(lIter == lWrapper.cufftMap.end())
    {
        cufftHandle lPlan;

        // cufftEstimate1d(pElemsY, CUFFT_C2C, ROWS_PER_FFT_SUBTASK, &lMemReqd)
        CUFFT_ERROR_CHECK("cufftPlan1d", cufftPlan1d(&lPlan, pElemsY, CUFFT_C2C, ROWS_PER_FFT_SUBTASK));

        cudaDeviceProp lDeviceProp;
        CUDA_ERROR_CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&lDeviceProp, lDeviceId));
    
        lWrapper.cufftMap[lPair] = lPlan;
        
        return lPlan;
    }
    
    return lIter->second;
}

cufftManyWrapper& GetCufftManyWrapper()
{
    static cufftManyWrapper sWrapper;
    
    return sWrapper;
}

cufftHandle GetCufftPlanMany(size_t pN, size_t pM)
{
    cufftManyWrapper& lWrapper = GetCufftManyWrapper();

    int lDeviceId;
    CUDA_ERROR_CHECK("cudaGetDevice", cudaGetDevice(&lDeviceId));

    std::pair<int, std::pair<size_t, size_t> > lPair(lDeviceId, std::make_pair(pN, pM));
    std::map<std::pair<int, std::pair<size_t, size_t> >, cufftHandle>::iterator lIter = lWrapper.cufftMap.find(lPair);

    if(lIter == lWrapper.cufftMap.end())
    {
        cufftHandle lPlan;
        
        int lN[] = {pN};

        // cufftEstimateMany(1, lN, lN, (int)pM, 1, lN, (int)pM, 1, CUFFT_C2C, ROWS_PER_FFT_SUBTASK, &lMemReqd)
        CUFFT_ERROR_CHECK("cufftPlanMany", cufftPlanMany(&lPlan, 1, lN, lN, (int)pM, 1, lN, (int)pM, 1, CUFFT_C2C, ROWS_PER_FFT_SUBTASK));

        cudaDeviceProp lDeviceProp;
        CUDA_ERROR_CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&lDeviceProp, lDeviceId));
    
        lWrapper.cufftMap[lPair] = lPlan;
        
        return lPlan;
    }
    
    return lIter->second;
}
    
void ClearCufftWrapper()
{
    GetCufftWrapper().Clear();
}


pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

#ifdef NO_MATRIX_TRANSPOSE
    cufftHandle lPlan = lTaskConf->rowPlanner ? GetCufftPlan1d(lTaskConf->elemsY) : GetCufftPlanMany(lTaskConf->elemsX, ROWS_PER_FFT_SUBTASK);
#else
    cufftHandle lPlan = GetCufftPlan1d(lTaskConf->elemsY);
#endif

    CUFFT_ERROR_CHECK("cufftSetStream", cufftSetStream(lPlan, (cudaStream_t)pCudaStream));
    
    if(lTaskConf->inplace)
    {
        CUFFT_ERROR_CHECK("cufftExecC2C", cufftExecC2C(lPlan, (cufftComplex*)pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].ptr, (cufftComplex*)pSubtaskInfo.memInfo[INPLACE_MEM_INDEX].ptr, CUFFT_FORWARD));
    }
    else
    {
        CUFFT_ERROR_CHECK("cufftExecC2C", cufftExecC2C(lPlan, (cufftComplex*)pSubtaskInfo.memInfo[INPUT_MEM_INDEX].ptr, (cufftComplex*)pSubtaskInfo.memInfo[OUTPUT_MEM_INDEX].ptr, CUFFT_FORWARD));
    }

    return pmSuccess;
}

// Returns 0 on success; non-zero on failure
int fftSingleGpu2D(bool inplace, complex* inputData, complex* outputData, size_t nx, size_t ny, int dir)
{
    void* lInputData = (inplace ? outputData : inputData);
    
    void* lInputMemCudaPtr = NULL;
    size_t lSize = sizeof(FFT_DATA_TYPE) * nx * ny;
    CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lInputMemCudaPtr, lSize));
    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(lInputMemCudaPtr, lInputData, lSize, cudaMemcpyHostToDevice));

    void* lOutputMemCudaPtr = lInputMemCudaPtr;
    if(!inplace)
        CUDA_ERROR_CHECK("cudaMalloc", cudaMalloc((void**)&lOutputMemCudaPtr, lSize));
    
    cufftHandle lPlan;
    
#ifdef FFT_2D
    CUFFT_ERROR_CHECK("cufftPlan2d", cufftPlan2d(&lPlan, ny, nx, CUFFT_C2C));
#else
    CUFFT_ERROR_CHECK("cufftPlan1d", cufftPlan1d(&lPlan, ny, CUFFT_C2C, nx));
#endif
    
    CUFFT_ERROR_CHECK("cufftExecC2C", cufftExecC2C(lPlan, (cufftComplex*)lInputMemCudaPtr, (cufftComplex*)lOutputMemCudaPtr, CUFFT_FORWARD));
    CUFFT_ERROR_CHECK("cufftDestroy", cufftDestroy(lPlan));

    CUDA_ERROR_CHECK("cudaMemcpy", cudaMemcpy(outputData, lOutputMemCudaPtr, lSize, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK("cudaFree", cudaFree(lOutputMemCudaPtr));
    
    if(!inplace)
        CUDA_ERROR_CHECK("cudaFree", cudaFree(lInputMemCudaPtr));
    
    return 0;
}

}

#endif

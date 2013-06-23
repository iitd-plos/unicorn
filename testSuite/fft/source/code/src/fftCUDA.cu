
#ifdef BUILD_CUDA

#include "pmPublicDefinitions.h"
#include "commonAPI.h"
#include "fft.h"

#include <cufft.h>

namespace fft
{

pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    cufftHandle lPlan;
    cufftResult lResult;
    
    lResult = cufftPlan1d(&lPlan, lTaskConf->elemsY, CUFFT_C2C, ROWS_PER_FFT_SUBTASK);
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftPlan1d Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    lResult = cufftExecC2C(lPlan, (cufftComplex*)pSubtaskInfo.outputMem, (cufftComplex*)pSubtaskInfo.outputMem, CUFFT_FORWARD);
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftExecC2C Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    cufftDestroy(lPlan);
    
    return pmSuccess;
}

// Returns 0 on success; non-zero on failure
int fftSingleGpu2D(complex* data, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
{
    void* lMemCudaPtr = NULL;
    size_t lSize = sizeof(FFT_DATA_TYPE) * nx * ny;
    if(cudaMalloc((void**)&lMemCudaPtr, lSize) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Output Memory Allocation Failed" << std::endl;
        return 1;
    }

    if(cudaMemcpy(lMemCudaPtr, data, lSize, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    cufftHandle lPlan;
    cufftResult lResult;
    
    lResult = cufftPlan2d(&lPlan, ny, nx, CUFFT_C2C);
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftPlan2d Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    lResult = cufftExecC2C(lPlan, (cufftComplex*)lMemCudaPtr, (cufftComplex*)lMemCudaPtr, CUFFT_FORWARD);
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftExecC2C Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    cufftDestroy(lPlan);

    if(cudaMemcpy(data, lMemCudaPtr, lSize, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lMemCudaPtr) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memory Deallocation Failed" << std::endl;
        return 1;
    }
    
    return 0;
}

}

#endif

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
    
    if(lTaskConf->inplace)
        lResult = cufftExecC2C(lPlan, (cufftComplex*)pSubtaskInfo.outputMem, (cufftComplex*)pSubtaskInfo.outputMem, CUFFT_FORWARD);
    else
        lResult = cufftExecC2C(lPlan, (cufftComplex*)pSubtaskInfo.inputMem, (cufftComplex*)pSubtaskInfo.outputMem, CUFFT_FORWARD);
    
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftExecC2C Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    cufftDestroy(lPlan);
    
    return pmSuccess;
}

// Returns 0 on success; non-zero on failure
int fftSingleGpu2D(bool inplace, complex* inputData, complex* outputData, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
{
    void* lInputData = (inplace ? outputData : inputData);
    
    void* lInputMemCudaPtr = NULL;
    size_t lSize = sizeof(FFT_DATA_TYPE) * nx * ny;
    if(cudaMalloc((void**)&lInputMemCudaPtr, lSize) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Output Memory Allocation Failed" << std::endl;
        return 1;
    }

    if(cudaMemcpy(lInputMemCudaPtr, lInputData, lSize, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    void* lOutputMemCudaPtr = NULL;
    if(inplace)
    {
        lOutputMemCudaPtr = lInputMemCudaPtr;
    }
    else
    {
        if(cudaMalloc((void**)&lOutputMemCudaPtr, lSize) != cudaSuccess)
        {
            std::cout << "FFT: CUDA Output Memory Allocation Failed" << std::endl;
            return 1;
        }
    }
    
    cufftHandle lPlan;
    cufftResult lResult;
    
#ifdef FFT_2D
    lResult = cufftPlan2d(&lPlan, ny, nx, CUFFT_C2C);
#else
    lResult = cufftPlan1d(&lPlan, ny, CUFFT_C2C, nx);
#endif
    
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftPlan Error: " << lResult << std::endl;
        return pmUserError;
    }

    lResult = cufftExecC2C(lPlan, (cufftComplex*)lInputMemCudaPtr, (cufftComplex*)lOutputMemCudaPtr, CUFFT_FORWARD);
    if(lResult != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT cufftExecC2C Error: " << lResult << std::endl;
        return pmUserError;
    }
    
    cufftDestroy(lPlan);

    if(cudaMemcpy(outputData, lOutputMemCudaPtr, lSize, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memcpy Failed" << std::endl;
        return 1;
    }

    if(cudaFree(lOutputMemCudaPtr) != cudaSuccess)
    {
        std::cout << "FFT: CUDA Memory Deallocation Failed" << std::endl;
        return 1;
    }
    
    if(!inplace)
    {
        if(cudaFree(lInputMemCudaPtr) != cudaSuccess)
        {
            std::cout << "FFT: CUDA Memory Deallocation Failed" << std::endl;
            return 1;
        }
    }
    
    return 0;
}

}

#endif
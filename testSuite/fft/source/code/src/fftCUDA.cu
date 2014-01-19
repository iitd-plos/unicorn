
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
        std::map<int, cufftHandle>::iterator lIter = cufftMap.begin(), lEndIter = cufftMap.end();
        
        for(; lIter != lEndIter; ++lIter)
            CUFFT_ERROR_CHECK("cufftDestroy", cufftDestroy(lIter->second));
        
        cufftMap.clear();
    }

    std::map<int, cufftHandle> cufftMap;  // deviceId versus cufftPlan1d handle
};
    
// Assumes same pElemsY value is passed for every invocation
cufftHandle getCufftPlan1d(size_t pElemsY)
{
    static cufftWrapper lWrapper;

    int lDeviceId;
    CUDA_ERROR_CHECK("cudaGetDevice", cudaGetDevice(&lDeviceId));

    std::map<int, cufftHandle>::iterator lIter = lWrapper.cufftMap.find(lDeviceId);
    if(lIter == lWrapper.cufftMap.end())
    {
        cufftHandle lPlan;

        CUFFT_ERROR_CHECK("cufftPlan1d", cufftPlan1d(&lPlan, pElemsY, CUFFT_C2C, ROWS_PER_FFT_SUBTASK));

        cudaDeviceProp lDeviceProp;
        CUDA_ERROR_CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&lDeviceProp, lDeviceId));
    
        lWrapper.cufftMap[lDeviceId] = lPlan;
        
        return lPlan;
    }
    
    return lIter->second;
}

pmStatus fft_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream)
{
	fftTaskConf* lTaskConf = (fftTaskConf*)(pTaskInfo.taskConf);

    cufftHandle lPlan = getCufftPlan1d(lTaskConf->elemsY);
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
int fftSingleGpu2D(bool inplace, complex* inputData, complex* outputData, size_t powx, size_t nx, size_t powy, size_t ny, int dir)
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
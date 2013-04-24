
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

}

#endif
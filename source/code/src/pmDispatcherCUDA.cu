
#include "pmDispatcherGPU.h"

#ifdef SUPPORT_CUDA

#include "pmLogger.h"
#include "pmBase.h"
#include "cuda.h"

namespace pm
{

cudaError_t (*gFuncPtr_cudaGetDeviceCount)(int* count);
cudaError_t (*gFuncPtr_cudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);
cudaError_t (*gFuncPtr_cudaSetDevice)(int device);

#define EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, ...) \ 
	{ \ 
		void* lSymbolPtr = GetExportedSymbol(libPtr, symbol, prototype); \
		if(!lSymbolPtr)	\
		{ \
			pmLogger::GetLogger->Log(pmLogger::DEBUG, pmLogger::ERROR, cudaGetErrorString(lErrorCUDA)); \
			throw pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::UNDEFINED_SYMBOL); \
		} \
		((prototype)lSymbolPtr)(__VA_ARGS__); \
	}


#define SAFE_EXECUTE_CUDA(libPtr, symbol, prototype, ...) EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, __VA_ARGS__); \
	{ \
		cudaError_t lErrorCUDA = cudaGetLastError(); \
		if(lErrorCUDA != cudaSuccess) \
		{ \
			pmLogger::GetLogger->Log(pmLogger::MINIMAL, pmLogger::ERROR, cudaGetErrorString(lErrorCUDA)); \
			throw pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR); \
		} \
	}

pmStatus pmDispatcherCUDA::CountAndProbeProcessingElements()
{
	pmStatus lStatus;

	int lCountCUDA = 0;
	mCountCUDA = 0;

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaGetDeviceCount", gFuncPtr_cudaGetDeviceCount, &lCountCUDA );

	for(int i = 0; i<lCountCUDA; ++i)
	{
		cudaDeviceProp lDeviceProp;
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaGetDeviceProperties", gFuncPtr_cudaGetDeviceProperties, &lDeviceProp, i );

		if(!(lDeviceProp.major == 9999 && lDeviceProp.minor == 9999))
			mDeviceVector.push_back(std::pair<int, cudaDeviceProp>(i, lDeviceProp));			
	}

	mCountCUDA = mDeviceVector.size();

	return pmSuccess;
}

#else	// SUPPORT_CUDA

pmStatus pmDispatcherCUDA::CountAndProbeProcessingElements()
{
	mCountCUDA = 0;
	return pmSuccess;
}

#endif	// SUPPORT_CUDA

}

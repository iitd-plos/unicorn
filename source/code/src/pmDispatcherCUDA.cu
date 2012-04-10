
#include "pmBase.h"
#include "pmDispatcherGPU.h"

#ifdef SUPPORT_CUDA

#include "pmLogger.h"
#include <string>

#endif

namespace pm
{

using namespace dispatcherCUDA;

#ifdef SUPPORT_CUDA

cudaError_t (*gFuncPtr_cudaGetDeviceCount)(int* count);
cudaError_t (*gFuncPtr_cudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);
cudaError_t (*gFuncPtr_cudaSetDevice)(int device);
cudaError_t (*gFuncPtr_cudaMalloc)(void** pCudaPtr, int pLength);
cudaError_t (*gFuncPtr_cudaMemcpy)(void* pCudaPtr, void* pHostPtr, int pLength, int pDirection);
cudaError_t (*gFuncPtr_cudaFree)(void* pCudaPtr);


#define EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, ...) \
	{ \
		void* dSymbolPtr = GetExportedSymbol(libPtr, symbol); \
		if(!dSymbolPtr)	\
		{ \
			std::string dStr("Undefined CUDA Symbol "); \
			dStr += symbol; \
			pmLogger::GetLogger()->Log(pmLogger::DEBUG_INTERNAL, pmLogger::ERROR, dStr.c_str()); \
			PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::UNDEFINED_SYMBOL)); \
		} \
		*(void**)(&prototype) = dSymbolPtr; \
		(*prototype)(__VA_ARGS__); \
	}

#define SAFE_EXECUTE_CUDA(libPtr, symbol, prototype, ...) \
	{ \
		EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, __VA_ARGS__); \
		cudaError_t dErrorCUDA = cudaGetLastError(); \
		if(dErrorCUDA != cudaSuccess) \
		{ \
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, cudaGetErrorString(dErrorCUDA)); \
			PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR)); \
		} \
	}

pmStatus pmDispatcherCUDA::CountAndProbeProcessingElements()
{
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

pmStatus pmDispatcherCUDA::BindToDevice(size_t pDeviceIndex)
{
	int lHardwareId = mDeviceVector[pDeviceIndex].first;

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaSetDevice", gFuncPtr_cudaSetDevice, lHardwareId );

	return pmSuccess;
}

std::string pmDispatcherCUDA::GetDeviceName(size_t pDeviceIndex)
{
	cudaDeviceProp lProp = mDeviceVector[pDeviceIndex].second;
	return lProp.name;
}

std::string pmDispatcherCUDA::GetDeviceDescription(size_t pDeviceIndex)
{
	cudaDeviceProp lProp = mDeviceVector[pDeviceIndex].second;
	std::string lStr("Clock Rate=");
	lStr += lProp.clockRate;
	lStr += ";sharedMemPerBlock=";
	lStr += lProp.sharedMemPerBlock;

	return lStr;
}

pmStatus pmDispatcherCUDA::InvokeKernel(size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr)
{
    bool lFullInputMemSubscription = false;
    
    pmTask* lTask = (pmTask*)(pTaskInfo.taskHandle);
    uint lOriginatingMachineIndex = (uint)(*(lTask->GetOriginatingHost()));
    ulong lSequenceNumber = lTask->GetSequenceNumber();
    
    pmMemSection* lInputMemSection = lTask->GetMemSectionRO();
    if(lInputMemSection && lInputMemSection->GetLength() == pSubtaskInfo.inputMemLength)
        lFullInputMemSubscription = true;
        
    bool lMatchingLastExecutionRecord = false;
    lastExecutionRecord* lLastRecord = NULL;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());
        if(mLastExecutionMap.find(pBoundDeviceIndex) != mlastExecutionMap.end())
            lLastRecord = &(mLastExecutionMap[pBoundDeviceIndex]);
    }
    
    if(lLastRecord)
    {
        if(lLastRecord->taskOriginatingMachineIndex == lOriginatingMachineIndex && lLastRecord->taskSequenceNumber == lSequenceNumber)
            lMatchingLastExecutionRecord = true;

        if(!lMatchingLastExecutionRecord)
        {
            if(lLastRecord->fullInputMemSubscription && lLastRecord->inputMemCudaPtr)
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord->inputMemCudaPtr );
                
            if(lLastRecord->taskConfCudaPtr)
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord->taskConfCudaPtr );
        }
    }
    
	void* lTaskConfCudaPtr = NULL;
    if(pTaskInfo.taskConf && pTaskInfo.taskConfLength != 0)
    {
        if(lMatchingLastExecutionRecord && lLastRecord->taskConfCudaPtr)
        {
            lTaskConfCudaPtr = lLastRecord->taskConfCudaPtr;
        }
        else
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lTaskConfCudaPtr, pTaskInfo.taskConfLength );
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTaskConfCudaPtr, pTaskInfo.taskConf, pTaskInfo.taskConfLength, cudaMemcpyHostToDevice );
        }
    }

	void* lInputMemCudaPtr = NULL;
	void* lOutputMemCudaPtr = NULL;

	if(pSubtaskInfo.inputMem && pSubtaskInfo.inputMemLength != 0)
	{
        if(lMatchingLastExecutionRecord && lLastRecord->fullInputMemSubscription && lLastRecord->inputMemCudaPtr)
        {
            lInputMemCudaPtr = lLastRecord->inputMemCudaPtr;
        }
        else
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lInputMemCudaPtr, pSubtaskInfo.inputMemLength );
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lInputMemCudaPtr, pSubtaskInfo.inputMem, pSubtaskInfo.inputMemLength, cudaMemcpyHostToDevice );
        }
	}
    
    lastExecutionRecord lRecord;
    lRecord.taskOriginatingMachineIndex = lOriginatingMachineIndex;
    lRecord.taskSequenceNumber = lSequenceNumber;
    lRecord.fullInputMemSubscription = lFullInputMemSubscription;
    lRecord.inputMemCudaPtr = lFullInputMemSubscription?lInputMemCudaPtr:NULL;
    lRecord.taskConfCudaPtr = lTaskConfCudaPtr;
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());
        mLastExecutionMap[pBoundDeviceIndex] = lRecord;
    }

	if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
	{
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lOutputMemCudaPtr, pSubtaskInfo.outputMemLength );

        if(!pOutputMemWriteOnly)
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lOutputMemCudaPtr, pSubtaskInfo.outputMem, pSubtaskInfo.outputMemLength, cudaMemcpyHostToDevice );
	}

	pmStatus lStatus = pmStatusUnavailable;
	pmStatus* lStatusPtr = NULL;

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lStatusPtr, sizeof(pmStatus) );
	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lStatusPtr, &lStatus, sizeof(pmStatus), cudaMemcpyHostToDevice );

	pmTaskInfo lTaskInfo = pTaskInfo;
	lTaskInfo.taskConf = lTaskConfCudaPtr;

	pmSubtaskInfo lSubtaskInfo = pSubtaskInfo;
	lSubtaskInfo.inputMem = lInputMemCudaPtr;
	lSubtaskInfo.outputMem = lOutputMemCudaPtr;

	dim3 gridConf(pCudaLaunchConf.blocksX, pCudaLaunchConf.blocksY, pCudaLaunchConf.blocksZ);
	dim3 blockConf(pCudaLaunchConf.threadsX, pCudaLaunchConf.threadsY, pCudaLaunchConf.threadsZ);

	//pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem>>> (pTaskInfo, lSubtaskInfo, lStatusPtr);
	pKernelPtr <<<gridConf, blockConf>>> (lTaskInfo, lSubtaskInfo, lStatusPtr);

	if(cudaGetLastError() == cudaSuccess)
	{
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, pSubtaskInfo.outputMem, lOutputMemCudaPtr, pSubtaskInfo.outputMemLength, cudaMemcpyDeviceToHost );
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, &lStatus, lStatusPtr, sizeof(pmStatus), cudaMemcpyDeviceToHost );
	}

	if(lInputMemCudaPtr && !lFullInputMemSubscription)
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lInputMemCudaPtr );

	if(lOutputMemCudaPtr)
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lOutputMemCudaPtr );

//	if(lTaskConfCudaPtr)
//		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lTaskConfCudaPtr );
	
	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lStatusPtr );

	return lStatus;
}

pmStatus pmDispatcherCUDA::FreeLastExecutionResources(size_t pBoundDeviceIndex)
{
    FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());

    if(mLastExecutionMap.find(pBoundDeviceIndex) == mlastExecutionMap.end())
        return pmSuccess;
        
    pmLastExecutionRecord& lLastRecord = mLastExecutionMap[pBoundDeviceIndex];
    
    if(lLastRecord.fullInputMemSubscription && lLastRecord.inputMemCudaPtr)
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord.inputMemCudaPtr );
        
    if(lLastRecord.taskConfCudaPtr)
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord.taskConfCudaPtr );
    
    mLastExecutionMap.erase(pBoundDeviceIndex);
    
    return pmSuccess;
}

#else	// SUPPORT_CUDA
/* The below functions are there to satisfy compiler. These are never executed. */
pmStatus pmDispatcherCUDA::CountAndProbeProcessingElements()
{
	mCountCUDA = 0;
	return pmSuccess;
}

pmStatus pmDispatcherCUDA::BindToDevice(size_t pDeviceIndex)
{
	return pmSuccess;
}

std::string pmDispatcherCUDA::GetDeviceName(size_t pDeviceIndex)
{
	return std::string();
}

std::string pmDispatcherCUDA::GetDeviceDescription(size_t pDeviceIndex)
{
	return std::string();
}

pmStatus pmDispatcherCUDA::InvokeKernel(size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmSubtaskCallback_GPU_CUDA pKernelPtr)
{
	return pmSuccess;
}

#endif	// SUPPORT_CUDA

}


/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#include "pmBase.h"
#include "pmDispatcherGPU.h"
#include <string>

#ifdef SUPPORT_CUDA
#include "pmLogger.h"
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

pmStatus pmDispatcherCUDA::InvokeKernel(pmExecutionStub* pStub, size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmDeviceInfo& pDeviceInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, void* pTaskOutputMem)
{
    bool lMatchingLastExecutionRecord = false;
    lastExecutionRecord* lLastRecord = NULL;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());
        if(mLastExecutionMap.find(pBoundDeviceIndex) != mLastExecutionMap.end())
            lLastRecord = &(mLastExecutionMap[pBoundDeviceIndex]);
    }

    if(lLastRecord && lLastRecord->taskOriginatingMachineIndex == pOriginatingMachineIndex && lLastRecord->taskSequenceNumber == pSequenceNumber)
        lMatchingLastExecutionRecord = true;
    
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
        if(lMatchingLastExecutionRecord && SubtasksHaveMatchingSubscriptions(pStub, pOriginatingMachineIndex, pSequenceNumber, lLastRecord->lastSubtaskId, pSubtaskInfo.subtaskId, true))
        {
            lInputMemCudaPtr = lLastRecord->inputMemCudaPtr;
        }
        else
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lInputMemCudaPtr, pSubtaskInfo.inputMemLength );

            std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
            GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, true, pSubtaskInfo, lSubscriptionVector);
            
            std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
            for(; lIter != lEndIter; ++lIter)
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputMemCudaPtr) + (*lIter).first);
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.inputMem) + (*lIter).first);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTempDevicePtr, lTempHostPtr, (*lIter).second, cudaMemcpyHostToDevice );
            }
        }
    }
        
    if(lLastRecord && lLastRecord->inputMemCudaPtr && lLastRecord->inputMemCudaPtr != lInputMemCudaPtr)
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord->inputMemCudaPtr );
        
    if(lLastRecord && lLastRecord->taskConfCudaPtr && lLastRecord->taskConfCudaPtr != lTaskConfCudaPtr)
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lLastRecord->taskConfCudaPtr );

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());
        mLastExecutionMap[pBoundDeviceIndex].taskOriginatingMachineIndex = pOriginatingMachineIndex;
        mLastExecutionMap[pBoundDeviceIndex].taskSequenceNumber = pSequenceNumber;
        mLastExecutionMap[pBoundDeviceIndex].lastSubtaskId = pSubtaskInfo.subtaskId;
        mLastExecutionMap[pBoundDeviceIndex].inputMemCudaPtr = lInputMemCudaPtr;
        mLastExecutionMap[pBoundDeviceIndex].taskConfCudaPtr = lTaskConfCudaPtr;
    }

	if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
	{
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lOutputMemCudaPtr, pSubtaskInfo.outputMemLength );

        if(!pOutputMemWriteOnly)
        {
            pmSubscriptionInfo lSubscriptionInfo;
            GetOutputMemSubscriptionForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo, lSubscriptionInfo);
            
            std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
            GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, false, pSubtaskInfo, lSubscriptionVector);
            
            std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
            for(; lIter != lEndIter; ++lIter)
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + ((*lIter).first - lSubscriptionInfo.offset));
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pTaskOutputMem) + (*lIter).first);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTempDevicePtr, lTempHostPtr, (*lIter).second, cudaMemcpyHostToDevice );
            }
        }
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

    MarkInsideUserCode(pStub, pSubtaskInfo.subtaskId);
	//pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem>>> (pTaskInfo, pDeviceInfo, lSubtaskInfo, lStatusPtr);
	pKernelPtr <<<gridConf, blockConf>>> (lTaskInfo, pDeviceInfo, lSubtaskInfo, lStatusPtr);
    MarkInsideLibraryCode(pStub, pSubtaskInfo.subtaskId);

	if(cudaGetLastError() == cudaSuccess && !RequiresPrematureExit(pStub, pSubtaskInfo.subtaskId))
	{
        if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
        {
            pmSubscriptionInfo lSubscriptionInfo;
            GetOutputMemSubscriptionForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo, lSubscriptionInfo);

            std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
            GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, false, pSubtaskInfo, lSubscriptionVector);
            
            std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
            for(; lIter != lEndIter; ++lIter)
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + ((*lIter).first - lSubscriptionInfo.offset));
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.outputMem) + ((*lIter).first - lSubscriptionInfo.offset));
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTempHostPtr, lTempDevicePtr, (*lIter).second, cudaMemcpyDeviceToHost );
            }
        }
        
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, &lStatus, lStatusPtr, sizeof(pmStatus), cudaMemcpyDeviceToHost );
	}

	if(lOutputMemCudaPtr)
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lOutputMemCudaPtr );

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, lStatusPtr );

	return lStatus;
}

pmStatus pmDispatcherCUDA::FreeLastExecutionResources(size_t pBoundDeviceIndex)
{
    FINALIZE_RESOURCE_PTR(dLastExecutionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLastExecutionLock, Lock(), Unlock());

    if(mLastExecutionMap.find(pBoundDeviceIndex) == mLastExecutionMap.end())
        return pmSuccess;
        
    lastExecutionRecord& lLastRecord = mLastExecutionMap[pBoundDeviceIndex];
	
    if(lLastRecord.inputMemCudaPtr)
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

#endif	// SUPPORT_CUDA

}

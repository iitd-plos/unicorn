
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
#include "pmHardware.h"
#include "pmMemSection.h"

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

pmStatus pmDispatcherCUDA::InvokeKernel(size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, pmMemSection* pInputMemSection)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pSequenceNumber);

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
        if(lMatchingLastExecutionRecord && lTask->GetSubscriptionManager().SubtasksHaveMatchingSubscriptions(lLastRecord->lastSubtaskId, pSubtaskInfo.subtaskId, true))
        {
            lInputMemCudaPtr = lLastRecord->inputMemCudaPtr;
        }
        else
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lInputMemCudaPtr, pSubtaskInfo.inputMemLength );
            
            subscription::subscriptionRecordType::iterator lBegin, lEnd;
            lTask->GetSubscriptionManager().GetNonConsolidatedInputMemSubscriptionsForSubtask(pSubtaskInfo.subtaskId, lBegin, lEnd);
            if(lBegin == lEnd)
            {
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lInputMemCudaPtr, pSubtaskInfo.inputMem, pSubtaskInfo.inputMemLength, cudaMemcpyHostToDevice );
            }
            else
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputMemCudaPtr) + lBegin->first);
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.inputMem) + lBegin->first);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTempDevicePtr, lTempHostPtr, lBegin->second.first, cudaMemcpyHostToDevice );
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
            subscription::subscriptionRecordType::iterator lBegin, lEnd;
            lTask->GetSubscriptionManager().GetNonConsolidatedOutputMemSubscriptionsForSubtask(pSubtaskInfo.subtaskId, lBegin, lEnd);
            if(lBegin == lEnd)
            {
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lOutputMemCudaPtr, pSubtaskInfo.outputMem, pSubtaskInfo.outputMemLength, cudaMemcpyHostToDevice );
            }
            else
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + lBegin->first);
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.outputMem) + lBegin->first);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTempDevicePtr, lTempHostPtr, lBegin->second.first, cudaMemcpyHostToDevice );
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

	//pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem>>> (pTaskInfo, lSubtaskInfo, lStatusPtr);
	pKernelPtr <<<gridConf, blockConf>>> (lTaskInfo, lSubtaskInfo, lStatusPtr);

	if(cudaGetLastError() == cudaSuccess)
	{
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, pSubtaskInfo.outputMem, lOutputMemCudaPtr, pSubtaskInfo.outputMemLength, cudaMemcpyDeviceToHost );
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

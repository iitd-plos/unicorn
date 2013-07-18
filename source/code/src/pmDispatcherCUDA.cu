
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

#include "pmExecutionStub.h"

#ifdef SUPPORT_CUDA
#include "pmLogger.h"
#endif

#include <string>
#include <sstream>

namespace pm
{

#ifdef SUPPORT_CUDA

cudaError_t (*gFuncPtr_cudaGetDeviceCount)(int* count);
cudaError_t (*gFuncPtr_cudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);
cudaError_t (*gFuncPtr_cudaSetDevice)(int device);
cudaError_t (*gFuncPtr_cudaMalloc)(void** pCudaPtr, int pLength);
cudaError_t (*gFuncPtr_cudaMemcpy)(void* pCudaPtr, const void* pHostPtr, size_t pLength, enum cudaMemcpyKind pDirection);
cudaError_t (*gFuncPtr_cudaFree)(void* pCudaPtr);
cudaError_t (*gFuncPtr_cudaDeviceSynchronize)();
cudaError_t (*gFuncPtr_cudaStreamCreate)(cudaStream_t* pCudaStream);
cudaError_t (*gFuncPtr_cudaStreamDestroy)(cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaStreamSynchronize)(cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaMemcpyAsync)(void* pCudaPtr, const void* pHostPtr, size_t pLength, enum cudaMemcpyKind pDirection, cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaHostAlloc)(void** pHost, size_t pSize, unsigned int pFlags);
cudaError_t (*gFuncPtr_cudaFreeHost)(void* pPtr);



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
			PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR, dErrorCUDA)); \
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
	std::stringstream lStream;
    lStream << "Clock Rate=" << lProp.clockRate << ";sharedMemPerBlock=" << lProp.sharedMemPerBlock << ";computeCapability=" << lProp.major << "." << lProp.minor;

	return lStream.str();
}
    
void* pmDispatcherCUDA::CreateDeviceInfoCudaPtr(pmDeviceInfo& pDeviceInfo)
{
    void* lDeviceInfoCudaPtr = NULL;

    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lDeviceInfoCudaPtr, sizeof(pDeviceInfo) );
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lDeviceInfoCudaPtr, &pDeviceInfo, sizeof(pDeviceInfo), cudaMemcpyHostToDevice );
    
    return lDeviceInfoCudaPtr;
}
    
void pmDispatcherCUDA::DestroyDeviceInfoCudaPtr(void* pDeviceInfoCudaPtr)
{
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pDeviceInfoCudaPtr );
}
    
class pmCudaAutoPtr : public pmBase
{
public:
    pmCudaAutoPtr(void* pRuntimeHandle, size_t pAllocationSize = 0)
    : mRuntimeHandle(pRuntimeHandle)
    , mCudaPtr(NULL)
    {
        if(pAllocationSize)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&mCudaPtr, pAllocationSize );
        }
    }
    
    ~pmCudaAutoPtr()
    {
        if(mCudaPtr)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, mCudaPtr );
            mCudaPtr = NULL;
        }
    }
    
    void reset(size_t pAllocationSize)
    {
        if(mCudaPtr)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, mCudaPtr );
            mCudaPtr = NULL;
        }

        if(pAllocationSize)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&mCudaPtr, pAllocationSize );
        }
    }
    
    void release()
    {
        mCudaPtr = NULL;
    }
    
    void* getPtr()
    {
        return mCudaPtr;
    }
    
private:
    void* mRuntimeHandle;
    void* mCudaPtr;
};
    
struct pmCudaStreamAutoPtr : public pmBase
{
public:
    pmCudaStreamAutoPtr(void* pRuntimeHandle)
    : mRuntimeHandle(pRuntimeHandle)
    , mStream(NULL)
    {
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamCreate", gFuncPtr_cudaStreamCreate, &mStream );
    }
    
    ~pmCudaStreamAutoPtr()
    {
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamDestroy", gFuncPtr_cudaStreamDestroy, mStream );
    }
    
    cudaStream_t GetStream()
    {
        return mStream;
    }
    
private:
    void* mRuntimeHandle;
    cudaStream_t mStream;
};
    
struct pmCudaAutoPtrCollection
{
    pmCudaAutoPtr mInputMemAutoPtr, mOutputMemAutoPtr, mScratchBufferAutoPtr, mStatusAutoPtr;
    pmCudaStreamAutoPtr mCudaStreamAutoPtr;
    
    pmCudaAutoPtrCollection(void* pRuntimeHandle)
    : mInputMemAutoPtr(pRuntimeHandle)
    , mOutputMemAutoPtr(pRuntimeHandle)
    , mScratchBufferAutoPtr(pRuntimeHandle)
    , mStatusAutoPtr(pRuntimeHandle)
    , mCudaStreamAutoPtr(pRuntimeHandle)
    {}
};
    
std::pair<void*, void*> pmDispatcherCUDA::AllocatePinnedBuffer(size_t pSize)
{
    void* lMem = NULL;
    void* lCudaMem = NULL;

    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaHostAlloc", gFuncPtr_cudaHostAlloc, &lMem, pSize, cudaHostAllocDefault );
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)(&lCudaMem), pSize );
    
    return std::make_pair(lMem, lCudaMem);
}
    
void pmDispatcherCUDA::DeallocatePinnedBuffer(std::pair<void*, void*> pMemPair)
{
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFreeHost", gFuncPtr_cudaFreeHost, pMemPair.first );
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pMemPair.second );
}
    
void* pmDispatcherCUDA::CreateTaskConf(pmTaskInfo& pTaskInfo)
{
    void* lTaskConfCudaPtr = NULL;

    if(pTaskInfo.taskConfLength)
    {
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lTaskConfCudaPtr, pTaskInfo.taskConfLength );

        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTaskConfCudaPtr, pTaskInfo.taskConf, pTaskInfo.taskConfLength, cudaMemcpyHostToDevice );
    }
    
    return lTaskConfCudaPtr;
}

void pmDispatcherCUDA::DestroyTaskConf(void* pTaskConfCudaPtr)
{
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pTaskConfCudaPtr );
}
    
void pmDispatcherCUDA::ComputeMemoryRequiredForSubtask(pmExecutionStub* pStub, pmLastCudaExecutionRecord& pLastRecord, pmSubtaskInfo& pSubtaskInfo, pmSubtaskCallback_GPU_CUDA pKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, size_t& pInputMem, size_t& pOutputMem, size_t& pScratchMem)
{
    pInputMem = pOutputMem = pScratchMem = 0;

    bool lMatchingLastExecutionRecord = false;

    if(pLastRecord.valid && pLastRecord.taskOriginatingMachineIndex == pOriginatingMachineIndex && pLastRecord.taskSequenceNumber == pSequenceNumber)
        lMatchingLastExecutionRecord = true;
    
    if(pSubtaskInfo.inputMem && pSubtaskInfo.inputMemLength != 0)
    {
        if(!(lMatchingLastExecutionRecord && SubtasksHaveMatchingSubscriptions(pStub, pOriginatingMachineIndex, pSequenceNumber, pLastRecord.lastSubtaskId, pSubtaskInfo.subtaskId, INPUT_MEM_READ_SUBSCRIPTION)))
            pInputMem = pSubtaskInfo.inputMemLength;
    }
        
	if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
        pOutputMem = pSubtaskInfo.outputMemLength;

//    if(pKernelPtr)
//        lMemReqd += (sizeof(pmStatus));
    
    pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
    size_t lScratchBufferSize = 0;
    void* lCpuScratchBuffer = CheckAndGetScratchBuffer(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo.subtaskId, lScratchBufferSize, lScratchBufferInfo);
    if(lCpuScratchBuffer && lScratchBufferSize)
        pScratchMem = lScratchBufferSize;
}

pmStatus pmDispatcherCUDA::InvokeKernel(pmExecutionStub* pStub, pmLastCudaExecutionRecord& pLastRecord, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, void* pTaskOutputMem)
{
    size_t lLogAlignment = 8;   // 256 byte alignment
    pmCudaAutoPtrCollection lCudaAutoPtrCollection(mRuntimeHandle);
    
    size_t lInputMemReqd, lOutputMemReqd, lScratchMemReqd;
    pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->ComputeMemoryRequiredForSubtask(pStub, pLastRecord, pSubtaskInfo, pKernelPtr, pOriginatingMachineIndex, pSequenceNumber, lInputMemReqd, lOutputMemReqd, lScratchMemReqd);
    
    pmMemChunk* lPinnedMemChunk = ((pmStubCUDA*)pStub)->GetPinnedBufferChunk();

    void* lInputMemPinnedPtr = (lInputMemReqd ? lPinnedMemChunk->Allocate(lInputMemReqd, lLogAlignment) : NULL);
    void* lOutputMemPinnedPtr = (lOutputMemReqd ? lPinnedMemChunk->Allocate(lOutputMemReqd, lLogAlignment) : NULL);
    void* lScratchMemPinnedPtr = (lScratchMemReqd ? lPinnedMemChunk->Allocate(lScratchMemReqd, lLogAlignment) : NULL);
    
    if(lInputMemReqd && !lInputMemPinnedPtr)
        PMTHROW(pmFatalErrorException());

    if(lOutputMemReqd && !lOutputMemPinnedPtr)
        PMTHROW(pmFatalErrorException());
    
    if(lScratchMemReqd && !lScratchMemPinnedPtr)
        PMTHROW(pmFatalErrorException());
    
    pmSubtaskInfo lSubtaskInfoCuda = pSubtaskInfo;

    CopyMemoriesToGpu(pStub, pLastRecord, pSubtaskInfo, pOutputMemWriteOnly, pKernelPtr, pOriginatingMachineIndex, pSequenceNumber, pTaskOutputMem, lSubtaskInfoCuda, &lCudaAutoPtrCollection, lInputMemPinnedPtr, lScratchMemPinnedPtr);
    
    pmStatus lStatus = ExecuteKernel(pStub, pTaskInfo, pTaskInfoCuda, pDeviceInfo, pDeviceInfoCudaPtr, lSubtaskInfoCuda, pCudaLaunchConf, pKernelPtr, pCustomKernelPtr, &lCudaAutoPtrCollection);
    
    pmStatus lStatus2 = CopyMemoriesFromGpu(pStub, pSubtaskInfo, lSubtaskInfoCuda, pKernelPtr, pOriginatingMachineIndex, pSequenceNumber, &lCudaAutoPtrCollection);
    
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamSynchronize", gFuncPtr_cudaStreamSynchronize, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
    
    lPinnedMemChunk->Deallocate(lInputMemPinnedPtr);
    lPinnedMemChunk->Deallocate(lOutputMemPinnedPtr);
    lPinnedMemChunk->Deallocate(lScratchMemPinnedPtr);
    
    if(!pCustomKernelPtr)
        return lStatus2;
    
    return lStatus;
}

void pmDispatcherCUDA::CopyMemoriesToGpu(pmExecutionStub* pStub, pmLastCudaExecutionRecord& pLastRecord, pmSubtaskInfo& pSubtaskInfo, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, void* pTaskOutputMem, pmSubtaskInfo& pSubtaskInfoCuda, void* pAutoPtrCollection, void* pInputMemPinnedPtr, void* pScratchMemPinnedPtr)
{
    pmCudaAutoPtrCollection& lCudaAutoPtrCollection = *((pmCudaAutoPtrCollection*)pAutoPtrCollection);

    bool lMatchingLastExecutionRecord = false;

    if(pLastRecord.valid && pLastRecord.taskOriginatingMachineIndex == pOriginatingMachineIndex && pLastRecord.taskSequenceNumber == pSequenceNumber)
        lMatchingLastExecutionRecord = true;
    
    void* lInputMemCudaPtr = NULL;
    void* lOutputMemCudaPtr = NULL;

    if(pSubtaskInfo.inputMem && pSubtaskInfo.inputMemLength != 0)
    {
        if(lMatchingLastExecutionRecord && SubtasksHaveMatchingSubscriptions(pStub, pOriginatingMachineIndex, pSequenceNumber, pLastRecord.lastSubtaskId, pSubtaskInfo.subtaskId, INPUT_MEM_READ_SUBSCRIPTION))
        {
            lInputMemCudaPtr = pLastRecord.inputMemCudaPtr;
        }
        else
        {
            lCudaAutoPtrCollection.mInputMemAutoPtr.reset(pSubtaskInfo.inputMemLength);
            lInputMemCudaPtr = lCudaAutoPtrCollection.mInputMemAutoPtr.getPtr();

            pmSubscriptionInfo lInputMemSubscriptionInfo;
            GetInputMemSubscriptionForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo, lInputMemSubscriptionInfo);

            std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
            GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, INPUT_MEM_READ_SUBSCRIPTION, pSubtaskInfo, lSubscriptionVector);
            
            std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
            for(; lIter != lEndIter; ++lIter)
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lInputMemCudaPtr) + (*lIter).first - lInputMemSubscriptionInfo.offset);
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.inputMem) + (*lIter).first - lInputMemSubscriptionInfo.offset);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lTempDevicePtr, lTempHostPtr, (*lIter).second, cudaMemcpyHostToDevice, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
            }
        }
    }
        
    if(pLastRecord.valid && pLastRecord.inputMemCudaPtr && pLastRecord.inputMemCudaPtr != lInputMemCudaPtr)
    {
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pLastRecord.inputMemCudaPtr );
        pLastRecord.inputMemCudaPtr = NULL;
    }
        
    lCudaAutoPtrCollection.mInputMemAutoPtr.release();

    pLastRecord.taskOriginatingMachineIndex = pOriginatingMachineIndex;
    pLastRecord.taskSequenceNumber = pSequenceNumber;
    pLastRecord.lastSubtaskId = pSubtaskInfo.subtaskId;
    pLastRecord.inputMemCudaPtr = lInputMemCudaPtr;
    pLastRecord.valid = true;
    
    pmSubscriptionInfo lUnifiedSubscriptionInfo;
    GetUnifiedOutputMemSubscriptionForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo, lUnifiedSubscriptionInfo);
    
	if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
	{
        lCudaAutoPtrCollection.mOutputMemAutoPtr.reset(pSubtaskInfo.outputMemLength);
        lOutputMemCudaPtr = lCudaAutoPtrCollection.mOutputMemAutoPtr.getPtr();

        if(!pOutputMemWriteOnly)
        {
            std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
            GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, OUTPUT_MEM_READ_SUBSCRIPTION, pSubtaskInfo, lSubscriptionVector);

            std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
            for(; lIter != lEndIter; ++lIter)
            {
                void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + ((*lIter).first - lUnifiedSubscriptionInfo.offset));
                void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pTaskOutputMem) + (*lIter).first);
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lTempDevicePtr, lTempHostPtr, (*lIter).second, cudaMemcpyHostToDevice, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
            }
        }
	}

    if(pKernelPtr)
    {
        pmStatus lStatus = pmStatusUnavailable;

        lCudaAutoPtrCollection.mStatusAutoPtr.reset(sizeof(pmStatus));
        pmStatus* lStatusPtr = (pmStatus*)lCudaAutoPtrCollection.mStatusAutoPtr.getPtr();

        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lStatusPtr, &lStatus, sizeof(pmStatus), cudaMemcpyHostToDevice, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
    }

	pSubtaskInfoCuda.inputMem = lInputMemCudaPtr;
	pSubtaskInfoCuda.outputMem = lOutputMemCudaPtr;
    pSubtaskInfoCuda.outputMemRead = pSubtaskInfoCuda.outputMemWrite = NULL;
    pSubtaskInfoCuda.outputMemReadLength = pSubtaskInfoCuda.outputMemWriteLength = 0;
    if(lOutputMemCudaPtr)
    {
        if(!pOutputMemWriteOnly)
        {
            pSubtaskInfoCuda.outputMemRead = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + reinterpret_cast<size_t>(pSubtaskInfo.outputMemRead) - reinterpret_cast<size_t>(pSubtaskInfo.outputMem));
            pSubtaskInfoCuda.outputMemReadLength = pSubtaskInfo.outputMemReadLength;
        }

        pSubtaskInfoCuda.outputMemWrite = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + reinterpret_cast<size_t>(pSubtaskInfo.outputMemWrite) - reinterpret_cast<size_t>(pSubtaskInfo.outputMem));
        pSubtaskInfoCuda.outputMemWriteLength = pSubtaskInfo.outputMemWriteLength;
    }

    pSubtaskInfoCuda.inputMemLength = pSubtaskInfo.inputMemLength;
    
    pSubtaskInfoCuda.gpuContext.scratchBuffer = NULL;
    pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
    size_t lScratchBufferSize = 0;
    void* lCpuScratchBuffer = CheckAndGetScratchBuffer(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo.subtaskId, lScratchBufferSize, lScratchBufferInfo);
    if(lCpuScratchBuffer && lScratchBufferSize)
    {
        lCudaAutoPtrCollection.mScratchBufferAutoPtr.reset(lScratchBufferSize);
        pSubtaskInfoCuda.gpuContext.scratchBuffer = lCudaAutoPtrCollection.mScratchBufferAutoPtr.getPtr();

        if(lScratchBufferInfo == PRE_SUBTASK_TO_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, pSubtaskInfoCuda.gpuContext.scratchBuffer, lCpuScratchBuffer, lScratchBufferSize, cudaMemcpyHostToDevice, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
        }
    }
}
    
pmStatus pmDispatcherCUDA::ExecuteKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfoCuda, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, void* pAutoPtrCollection)
{
    pmCudaAutoPtrCollection& lCudaAutoPtrCollection = *((pmCudaAutoPtrCollection*)pAutoPtrCollection);

    pmStatus lStatus = pmStatusUnavailable;

    // Jmp Buffer Scope
    {
        pmJmpBufAutoPtr lJmpBufAutoPtr;
        
        sigjmp_buf lJmpBuf;
        int lJmpVal = sigsetjmp(lJmpBuf, 0);
        
        if(!lJmpVal)
        {
            lJmpBufAutoPtr.Reset(&lJmpBuf, pStub, pSubtaskInfoCuda.subtaskId);

            if(pKernelPtr)
            {
                pmStatus* lStatusPtr = (pmStatus*)lCudaAutoPtrCollection.mStatusAutoPtr.getPtr();

                dim3 gridConf(pCudaLaunchConf.blocksX, pCudaLaunchConf.blocksY, pCudaLaunchConf.blocksZ);
                dim3 blockConf(pCudaLaunchConf.threadsX, pCudaLaunchConf.threadsY, pCudaLaunchConf.threadsZ);

                if(pCudaLaunchConf.sharedMem)
                    pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() >>> (pTaskInfoCuda, (pmDeviceInfo*)pDeviceInfoCudaPtr, pSubtaskInfoCuda, lStatusPtr);
                else
                    pKernelPtr <<<gridConf, blockConf, 0, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() >>> (pTaskInfoCuda, (pmDeviceInfo*)pDeviceInfoCudaPtr, pSubtaskInfoCuda, lStatusPtr);
            }
            else
            {
                lStatus = pCustomKernelPtr(pTaskInfo, pDeviceInfo, pSubtaskInfoCuda);
            }
        }
        else
        {
            lJmpBufAutoPtr.SetHasJumped();
            PMTHROW_NODUMP(pmPrematureExitException(true));
        }
    }

	return lStatus;
}
    
pmStatus pmDispatcherCUDA::CopyMemoriesFromGpu(pmExecutionStub* pStub, pmSubtaskInfo& pSubtaskInfo, pmSubtaskInfo& pSubtaskInfoCuda, pmSubtaskCallback_GPU_CUDA pKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, void* pAutoPtrCollection)
{
    pmCudaAutoPtrCollection& lCudaAutoPtrCollection = *((pmCudaAutoPtrCollection*)pAutoPtrCollection);
    void* lOutputMemCudaPtr = lCudaAutoPtrCollection.mOutputMemAutoPtr.getPtr();

    pmStatus lStatus = pmStatusUnavailable;

    cudaError_t lLastError = cudaGetLastError();
    if(lLastError == cudaSuccess)
    {
        if(!RequiresPrematureExit(pStub, pSubtaskInfo.subtaskId))
        {
            if(pSubtaskInfo.outputMem && pSubtaskInfo.outputMemLength != 0)
            {
                pmSubscriptionInfo lUnifiedSubscriptionInfo;
                GetUnifiedOutputMemSubscriptionForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo, lUnifiedSubscriptionInfo);

                std::vector<std::pair<size_t, size_t> > lSubscriptionVector;
                GetNonConsolidatedSubscriptionsForSubtask(pStub, pOriginatingMachineIndex, pSequenceNumber, OUTPUT_MEM_WRITE_SUBSCRIPTION, pSubtaskInfo, lSubscriptionVector);

                std::vector<std::pair<size_t, size_t> >::iterator lIter = lSubscriptionVector.begin(), lEndIter = lSubscriptionVector.end();
                for(; lIter != lEndIter; ++lIter)
                {
                    void* lTempDevicePtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lOutputMemCudaPtr) + ((*lIter).first - lUnifiedSubscriptionInfo.offset));
                    void* lTempHostPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(pSubtaskInfo.outputMem) + ((*lIter).first - lUnifiedSubscriptionInfo.offset));
                    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lTempHostPtr, lTempDevicePtr, (*lIter).second, cudaMemcpyDeviceToHost, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
                }
            }
            
            if(pKernelPtr)
            {
                pmStatus* lStatusPtr = (pmStatus*)lCudaAutoPtrCollection.mStatusAutoPtr.getPtr();
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, &lStatus, lStatusPtr, sizeof(pmStatus), cudaMemcpyDeviceToHost, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
            }

            pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
            size_t lScratchBufferSize = 0;
            void* lCpuScratchBuffer = CheckAndGetScratchBuffer(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo.subtaskId, lScratchBufferSize, lScratchBufferInfo);
            if(lCpuScratchBuffer && lScratchBufferSize && (lScratchBufferInfo == SUBTASK_TO_POST_SUBTASK || lScratchBufferInfo == PRE_SUBTASK_TO_POST_SUBTASK))
            {
                SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lCpuScratchBuffer, pSubtaskInfoCuda.gpuContext.scratchBuffer, lScratchBufferSize, cudaMemcpyDeviceToHost, lCudaAutoPtrCollection.mCudaStreamAutoPtr.GetStream() );
            }
        }
    }
    else
    {
        // Check if the kernel is compiled for a different architecture and the GPU card has a different compute capability
        //std::cout << "CUDA Error: " << cudaGetLastError(lLastError) << std::endl;
        PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR, lLastError));
    }
    
    return lStatus;
}

pmStatus pmDispatcherCUDA::FreeLastExecutionResources(pmLastCudaExecutionRecord& pLastExecutionRecord)
{
    if(pLastExecutionRecord.valid)
    {
        if(pLastExecutionRecord.inputMemCudaPtr)
        {
            SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pLastExecutionRecord.inputMemCudaPtr );
            pLastExecutionRecord.inputMemCudaPtr = NULL;
        }
            
        pLastExecutionRecord.valid = false;
    }
    
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

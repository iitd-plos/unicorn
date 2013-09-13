
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

#ifdef SUPPORT_CUDA

#include "pmBase.h"
#include "pmDispatcherGPU.h"
#include "pmUtility.h"
#include "pmLogger.h"

#include <string>
#include <sstream>

namespace pm
{

cudaError_t (*gFuncPtr_cudaGetDeviceCount)(int* count);
cudaError_t (*gFuncPtr_cudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);
cudaError_t (*gFuncPtr_cudaSetDevice)(int device);
cudaError_t (*gFuncPtr_cudaMalloc)(void** pCudaPtr, int pLength);
cudaError_t (*gFuncPtr_cudaMemcpy)(void* pCudaPtr, const void* pHostPtr, size_t pLength, enum cudaMemcpyKind pDirection);
cudaError_t (*gFuncPtr_cudaFree)(void* pCudaPtr);
cudaError_t (*gFuncPtr_cudaDeviceSynchronize)();
cudaError_t (*gFuncPtr_cudaStreamCreate)(cudaStream_t* pCudaStream);
cudaError_t (*gFuncPtr_cudaStreamDestroy)(cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaStreamAddCallback)(cudaStream_t pCudaStream, cudaStreamCallback_t pCallback, void* pUserData, unsigned int pFlags);
cudaError_t (*gFuncPtr_cudaStreamSynchronize)(cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaMemcpyAsync)(void* pCudaPtr, const void* pHostPtr, size_t pLength, enum cudaMemcpyKind pDirection, cudaStream_t pCudaStream);
cudaError_t (*gFuncPtr_cudaHostAlloc)(void** pHost, size_t pSize, unsigned int pFlags);
cudaError_t (*gFuncPtr_cudaFreeHost)(void* pPtr);
cudaError_t (*gFuncPtr_cudaMemGetInfo)(size_t* pFree, size_t* pTotal);


#define THROW_CUDA_ERROR(errorCUDA) \
{ \
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, cudaGetErrorString(errorCUDA)); \
    PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR, errorCUDA)); \
}

#define EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, ...) \
{ \
    void* dSymbolPtr = pmUtility::GetExportedSymbol(libPtr, symbol); \
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
        THROW_CUDA_ERROR(dErrorCUDA); \
}

void cudaStreamCallbackFunc(cudaStream_t pCudaStream, cudaError_t pErrorCUDA, void* pUserData)
{
    if(pErrorCUDA != cudaSuccess)
        THROW_CUDA_ERROR(pErrorCUDA);
        
    pmDispatcherGPU::GetDispatcherGPU()->GetDispatcherCUDA()->StreamFinishCallback(pUserData);
}
    
void* AllocateCudaMem(void* pRuntimeHandle, size_t pSize)
{
    void* lAddr = NULL;

    if(pSize)
        SAFE_EXECUTE_CUDA( pRuntimeHandle, "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lAddr, pSize );
    
    return lAddr;
}
    
void DeallocateCudaMem(void* pRuntimeHandle, void* pPtr)
{
    if(pPtr)
        SAFE_EXECUTE_CUDA( pRuntimeHandle, "cudaFree", gFuncPtr_cudaFree, pPtr );
}


/* class pmCudaAutoPtr */
pmCudaAutoPtr::pmCudaAutoPtr(void* pRuntimeHandle, size_t pAllocationSize /* = 0 */)
: mRuntimeHandle(pRuntimeHandle)
, mCudaPtr(NULL)
{
    mCudaPtr = AllocateCudaMem(mRuntimeHandle, pAllocationSize);
}
    
pmCudaAutoPtr::~pmCudaAutoPtr()
{
    DeallocateCudaMem(mRuntimeHandle, mCudaPtr);
    mCudaPtr = NULL;
}
    
void pmCudaAutoPtr::reset(size_t pAllocationSize)
{
    DeallocateCudaMem(mRuntimeHandle, mCudaPtr);

    mCudaPtr = AllocateCudaMem(mRuntimeHandle, pAllocationSize);
}

void pmCudaAutoPtr::release()
{
    mCudaPtr = NULL;
}

void* pmCudaAutoPtr::getPtr()
{
    return mCudaPtr;
}


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

pmStatus pmDispatcherCUDA::CountAndProbeProcessingElements()
{
	int lCountCUDA = 0;
	mCountCUDA = 0;

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaGetDeviceCount", gFuncPtr_cudaGetDeviceCount, &lCountCUDA );

	for(int i = 0; i < lCountCUDA; ++i)
	{
		cudaDeviceProp lDeviceProp;
		SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaGetDeviceProperties", gFuncPtr_cudaGetDeviceProperties, &lDeviceProp, i );

        if(lDeviceProp.computeMode != cudaComputeModeProhibited)
        {
            if(!(lDeviceProp.major == 9999 && lDeviceProp.minor == 9999))
                mDeviceVector.push_back(std::pair<int, cudaDeviceProp>(i, lDeviceProp));
        }
	}

	mCountCUDA = mDeviceVector.size();

	return pmSuccess;
}

pmStatus pmDispatcherCUDA::BindToDevice(size_t pDeviceIndex)
{
	int lHardwareId = mDeviceVector[pDeviceIndex].first;

	SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaSetDevice", gFuncPtr_cudaSetDevice, lHardwareId );
    pmCudaAutoPtr lAutoPtr(mRuntimeHandle, 1024);    // Initialize context with a dummy allocation and deallocation
    
	return pmSuccess;
}

std::string pmDispatcherCUDA::GetDeviceName(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = mDeviceVector[pDeviceIndex].second;
	return lProp.name;
}

std::string pmDispatcherCUDA::GetDeviceDescription(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = mDeviceVector[pDeviceIndex].second;

	std::stringstream lStream;
    lStream << "Clock Rate=" << lProp.clockRate << ";sharedMemPerBlock=" << lProp.sharedMemPerBlock << ";computeCapability=" << lProp.major << "." << lProp.minor;

	return lStream.str();
}
    
void* pmDispatcherCUDA::CreateDeviceInfoCudaPtr(pmDeviceInfo& pDeviceInfo)
{
    void* lDeviceInfoCudaPtr = AllocateCudaMem(sizeof(pDeviceInfo));

    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lDeviceInfoCudaPtr, &pDeviceInfo, sizeof(pDeviceInfo), cudaMemcpyHostToDevice );
    
    return lDeviceInfoCudaPtr;
}

void pmDispatcherCUDA::DestroyDeviceInfoCudaPtr(void* pDeviceInfoCudaPtr)
{
    DeallocateCudaMem(pDeviceInfoCudaPtr);
}
    
size_t pmDispatcherCUDA::GetCudaAlignment(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = mDeviceVector[pDeviceIndex].second;
    
    return lProp.textureAlignment;
}

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
void* pmDispatcherCUDA::AllocatePinnedBuffer(size_t pSize)
{
    void* lMem = NULL;

    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaHostAlloc", gFuncPtr_cudaHostAlloc, &lMem, pSize, cudaHostAllocDefault );
    
    return lMem;
}
    
void pmDispatcherCUDA::DeallocatePinnedBuffer(void* pMem)
{
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaFreeHost", gFuncPtr_cudaFreeHost, pMem );
}
#endif
    
void* pmDispatcherCUDA::CreateTaskConf(pmTaskInfo& pTaskInfo)
{
    void* lTaskConfCudaPtr = NULL;

    if(pTaskInfo.taskConfLength)
    {
        lTaskConfCudaPtr = AllocateCudaMem(pTaskInfo.taskConfLength);

        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpy", gFuncPtr_cudaMemcpy, lTaskConfCudaPtr, pTaskInfo.taskConf, pTaskInfo.taskConfLength, cudaMemcpyHostToDevice );
    }
    
    return lTaskConfCudaPtr;
}

void pmDispatcherCUDA::DestroyTaskConf(void* pTaskConfCudaPtr)
{
    DeallocateCudaMem(pTaskConfCudaPtr);
}

size_t pmDispatcherCUDA::GetAvailableCudaMem()
{
    size_t lFreeMem, lTotalMem;
    
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemGetInfo", gFuncPtr_cudaMemGetInfo, &lFreeMem, &lTotalMem );
    
    return lFreeMem;
}
    
int pmDispatcherCUDA::GetCudaDriverVersion()
{
    int lDriverVersion;

    cudaError_t lErrorCUDA = cudaDriverGetVersion(&lDriverVersion);
    if(lErrorCUDA != cudaSuccess)
        THROW_CUDA_ERROR(lErrorCUDA);
    
    return lDriverVersion;
}
    
void* pmDispatcherCUDA::AllocateCudaMem(size_t pSize)
{
    return pm::AllocateCudaMem(mRuntimeHandle, pSize);
}
    
void pmDispatcherCUDA::DeallocateCudaMem(void* pPtr)
{
    pm::DeallocateCudaMem(mRuntimeHandle, pPtr);
}

size_t pmDispatcherCUDA::ComputeAlignedMemoryRequirement(size_t pAllocation1, size_t pAllocation2, size_t pDeviceIndex)
{
    if(!pAllocation2)
        return pAllocation1;
    
    size_t lCudaAlignment = GetCudaAlignment(pDeviceIndex);
    size_t lHighestVal = lCudaAlignment - 1;
    size_t lAllocation = ((pAllocation1 + lHighestVal) & ~lHighestVal);
    
    return lAllocation + pAllocation2;
}
    
void pmDispatcherCUDA::ComputeMemoryRequiredForSubtask(pmExecutionStub* pStub, pmSubtaskInfo& pSubtaskInfo, ulong* pLastSubtaskIdIfSameTask, uint pOriginatingMachineIndex, ulong pSequenceNumber, size_t& pInputMem, size_t& pOutputMem, size_t& pScratchMem, bool& pUseLastSubtaskInputMem)
{
    pInputMem = pOutputMem = pScratchMem = 0;
    pUseLastSubtaskInputMem = false;

#if 0
    if(pSubtaskInfo.inputMem && pSubtaskInfo.inputMemLength != 0)
    {
        if(!(pLastSubtaskIdIfSameTask && SubtasksHaveMatchingSubscriptions(pStub, pOriginatingMachineIndex, pSequenceNumber, *pLastSubtaskIdIfSameTask, pSubtaskInfo.subtaskId, INPUT_MEM_READ_SUBSCRIPTION)))
            pInputMem = pSubtaskInfo.inputMemLength;
        else
            pUseLastSubtaskInputMem = true;
    }
#else
    pInputMem = pSubtaskInfo.inputMemLength;
#endif

    // Shadow mem has not been created at this time; so pSubtaskInfo.outputMem may be NULL even when pSubtaskInfo.outputMemLength is non zero
	if(pSubtaskInfo.outputMemLength != 0)
        pOutputMem = pSubtaskInfo.outputMemLength;

    pmScratchBufferInfo lScratchBufferInfo = SUBTASK_TO_POST_SUBTASK;
    size_t lScratchBufferSize = 0;
    void* lCpuScratchBuffer = CheckAndGetScratchBuffer(pStub, pOriginatingMachineIndex, pSequenceNumber, pSubtaskInfo.subtaskId, pSubtaskInfo.splitInfo, lScratchBufferSize, lScratchBufferInfo);
    if(lCpuScratchBuffer && lScratchBufferSize)
        pScratchMem = lScratchBufferSize;
}

pmStatus pmDispatcherCUDA::InvokeKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, std::vector<void*>& pCudaPointers)
{
    pmCudaStreamAutoPtr lStreamAutoPtr(mRuntimeHandle);

    std::vector<pmCudaMemcpyCommand>::iterator lHostToDeviceIter = pHostToDeviceCommands.begin(), lHostToDeviceEndIter = pHostToDeviceCommands.end();
    for(; lHostToDeviceIter != lHostToDeviceEndIter; ++lHostToDeviceIter)
    {
        pmCudaMemcpyCommand& lMemcpyCommand = *lHostToDeviceIter;
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, lMemcpyCommand.srcPtr, lMemcpyCommand.size, cudaMemcpyHostToDevice, lStreamAutoPtr.GetStream() );
    }

    pmSubtaskInfo lSubtaskInfoCuda = pSubtaskInfo;
	lSubtaskInfoCuda.inputMem = pCudaPointers[0];
	lSubtaskInfoCuda.outputMem = pCudaPointers[1];
    lSubtaskInfoCuda.outputMemRead = lSubtaskInfoCuda.outputMemWrite = NULL;
    lSubtaskInfoCuda.outputMemReadLength = lSubtaskInfoCuda.outputMemWriteLength = 0;
    if(pCudaPointers[1])
    {
        if(!pOutputMemWriteOnly)
        {
            lSubtaskInfoCuda.outputMemRead = reinterpret_cast<void*>(reinterpret_cast<size_t>(pCudaPointers[1]) + reinterpret_cast<size_t>(pSubtaskInfo.outputMemRead) - reinterpret_cast<size_t>(pSubtaskInfo.outputMem));
            lSubtaskInfoCuda.outputMemReadLength = pSubtaskInfo.outputMemReadLength;
        }

        lSubtaskInfoCuda.outputMemWrite = reinterpret_cast<void*>(reinterpret_cast<size_t>(pCudaPointers[1]) + reinterpret_cast<size_t>(pSubtaskInfo.outputMemWrite) - reinterpret_cast<size_t>(pSubtaskInfo.outputMem));
        lSubtaskInfoCuda.outputMemWriteLength = pSubtaskInfo.outputMemWriteLength;
    }

    lSubtaskInfoCuda.inputMemLength = pSubtaskInfo.inputMemLength;
    lSubtaskInfoCuda.gpuContext.scratchBuffer = pCudaPointers[2];
    lSubtaskInfoCuda.gpuContext.reservedGlobalMem = pCudaPointers[3];

    ExecuteKernel(pStub, pTaskInfo, pTaskInfoCuda, pDeviceInfo, (pmDeviceInfo*)pDeviceInfoCudaPtr, lSubtaskInfoCuda, pCudaLaunchConf, pKernelPtr, pCustomKernelPtr, lStreamAutoPtr.GetStream(), ((pmStatus*)pCudaPointers[4]));

    cudaError_t lCudaError = cudaGetLastError();
    if(lCudaError != cudaSuccess)
        THROW_CUDA_ERROR(lCudaError);
    
    std::vector<pmCudaMemcpyCommand>::iterator lDeviceToHostIter = pDeviceToHostCommands.begin(), lDeviceToHostEndIter = pDeviceToHostCommands.end();
    for(; lDeviceToHostIter != lDeviceToHostEndIter; ++lDeviceToHostIter)
    {
        pmCudaMemcpyCommand& lMemcpyCommand = *lDeviceToHostIter;
        SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, lMemcpyCommand.srcPtr, lMemcpyCommand.size, cudaMemcpyDeviceToHost, lStreamAutoPtr.GetStream() );
    }
    
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamAddCallback", gFuncPtr_cudaStreamAddCallback, lStreamAutoPtr.GetStream(), cudaStreamCallbackFunc, pStub, 0 );

    return pmSuccess;
}

pmStatus pmDispatcherCUDA::ExecuteKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, pmDeviceInfo* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfoCuda, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, cudaStream_t pStream, pmStatus* pStatusPtr)
{
    pmStatus lStatus = pmStatusUnavailable;

    // Jmp Buffer Scope
    {
        pmJmpBufAutoPtr lJmpBufAutoPtr;
        
        sigjmp_buf lJmpBuf;
        int lJmpVal = sigsetjmp(lJmpBuf, 1);
        
        if(!lJmpVal)
        {
            lJmpBufAutoPtr.Reset(&lJmpBuf, pStub);

            if(pKernelPtr)
            {
                dim3 gridConf(pCudaLaunchConf.blocksX, pCudaLaunchConf.blocksY, pCudaLaunchConf.blocksZ);
                dim3 blockConf(pCudaLaunchConf.threadsX, pCudaLaunchConf.threadsY, pCudaLaunchConf.threadsZ);

                if(pCudaLaunchConf.sharedMem)
                    pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem, pStream>>> (pTaskInfoCuda, pDeviceInfoCudaPtr, pSubtaskInfoCuda, pStatusPtr);
                else
                    pKernelPtr <<<gridConf, blockConf, 0, pStream>>> (pTaskInfoCuda, pDeviceInfoCudaPtr, pSubtaskInfoCuda, pStatusPtr);
            }
            else
            {
                lStatus = pCustomKernelPtr(pTaskInfo, pDeviceInfo, pSubtaskInfoCuda, pStream);
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

#endif	// SUPPORT_CUDA

}

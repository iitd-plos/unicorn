
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

#include "pmCudaInterface.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <iostream>
#include <vector>
#include <map>
#include <sstream>

namespace pm
{
    
struct pmCudaDeviceData
{
    int mHardwareId;
    cudaDeviceProp mDeviceProp;
    
    pmCudaDeviceData(int pHardwareId, const cudaDeviceProp& pDeviceProp)
    : mHardwareId(pHardwareId)
    , mDeviceProp(pDeviceProp)
    {}
};
    
extern void* GetExportedSymbol(void* pLibHandle, char* pSymbol);

cudaError_t (*gFuncPtr_cudaGetLastError)(void);
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
cudaError_t (*gFuncPtr_cudaDriverGetVersion)(int* pDriverVersion);

void* GetCudaSymbol(void* pLibPtr, char* pSymbol)
{
    void* lSymbolPtr = GetExportedSymbol(pLibPtr, pSymbol);
    if(!lSymbolPtr)
    {
        std::string lStr("Undefined CUDA Symbol ");
        lStr += pSymbol;
        std::cout << lStr.c_str() << std::endl;
        PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::UNDEFINED_SYMBOL));
    }
    
    return lSymbolPtr;
}

std::vector<pmCudaDeviceData>& GetDeviceVector()
{
    static std::vector<pmCudaDeviceData> gDeviceVector;
    return gDeviceVector;
}

#ifdef CREATE_EXPLICIT_CUDA_CONTEXTS
std::map<int, CUcontext>& GetContextMap()
{
    static std::map<int, CUcontext> gContextMap;
    return gContextMap;
}
#endif

#define THROW_OUT_OF_MEMORY_CUDA_ERROR PMTHROW_NODUMP(pmOutOfMemoryException());

#define THROW_CUDA_ERROR(errorCUDA) \
{ \
    std::cout << cudaGetErrorString(errorCUDA) << std::endl; \
    PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::RUNTIME_ERROR, errorCUDA)); \
}

#define CHECK_LAST_ERROR(libPtr) \
{ \
    *(void**)(&gFuncPtr_cudaGetLastError) = GetCudaSymbol(libPtr, "cudaGetLastError"); \
    cudaError_t dErrorCUDA = (*gFuncPtr_cudaGetLastError)(); \
    if(dErrorCUDA != cudaSuccess) \
    { \
        if(dErrorCUDA == cudaErrorMemoryAllocation) \
            THROW_OUT_OF_MEMORY_CUDA_ERROR; \
        THROW_CUDA_ERROR(dErrorCUDA); \
    } \
}

#define EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, ...) \
{ \
    *(void**)(&prototype) = GetCudaSymbol(libPtr, symbol); \
    (*prototype)(__VA_ARGS__); \
}

#define SAFE_EXECUTE_CUDA(libPtr, symbol, prototype, ...) \
{ \
    EXECUTE_CUDA_SYMBOL(libPtr, symbol, prototype, __VA_ARGS__); \
    CHECK_LAST_ERROR(libPtr); \
}

    
/* class pmCudaStreamAutoPtr */
pmCudaStreamAutoPtr::pmCudaStreamAutoPtr()
: mRuntimeHandle(NULL)
, mStream(NULL)
{
}
    
void pmCudaStreamAutoPtr::Initialize(void* pRuntimeHandle)
{
    mRuntimeHandle = pRuntimeHandle;

    cudaStream_t lStream;
    
    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamCreate", gFuncPtr_cudaStreamCreate, &lStream );
    
    mStream = lStream;
}

pmCudaStreamAutoPtr::~pmCudaStreamAutoPtr()
{
    if(!mRuntimeHandle)
        PMTHROW(pmFatalErrorException());

    SAFE_EXECUTE_CUDA( mRuntimeHandle, "cudaStreamDestroy", gFuncPtr_cudaStreamDestroy, ((cudaStream_t)mStream) );
}

void* pmCudaStreamAutoPtr::GetStream()
{
    return mStream;
}


class pmCudaAutoPtr
{
public:
    pmCudaAutoPtr(size_t pAllocationSize = 0)
    : mCudaPtr(NULL)
    {
        mCudaPtr = pmCudaInterface::AllocateCudaMem(pAllocationSize);
    }
    
    ~pmCudaAutoPtr()
    {
        pmCudaInterface::DeallocateCudaMem(mCudaPtr);
        mCudaPtr = NULL;
    }
    
    void reset(size_t pAllocationSize)
    {
        pmCudaInterface::DeallocateCudaMem(mCudaPtr);
        mCudaPtr = pmCudaInterface::AllocateCudaMem(pAllocationSize);
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
    void* mCudaPtr;
};

void*& pmCudaInterface::GetRuntimeHandle()
{
    static void* sRuntimeHandle = NULL;
    return sRuntimeHandle;
}

void pmCudaInterface::SetRuntimeHandle(void* pRuntimeHandle)
{
    GetRuntimeHandle() = pRuntimeHandle;
}
    
void* pmCudaInterface::AllocateCudaMem(size_t pSize)
{
    void* lAddr = NULL;

    if(pSize)
    {
        size_t lAvailableCudaMem = GetAvailableCudaMem();

        if(lAvailableCudaMem < pSize + MIN_UNALLOCATED_CUDA_MEM_SIZE)
            THROW_OUT_OF_MEMORY_CUDA_ERROR;
            
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMalloc", gFuncPtr_cudaMalloc, (void**)&lAddr, pSize );
    }
    
    return lAddr;
}
    
void pmCudaInterface::DeallocateCudaMem(const void* pPtr)
{
    if(pPtr)
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaFree", gFuncPtr_cudaFree, const_cast<void*>(pPtr) );
}
    
void pmCudaInterface::CopyDataToCudaDevice(void* pCudaPtr, const void* pHostPtr, size_t pSize)
{
    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpy", gFuncPtr_cudaMemcpy, pCudaPtr, pHostPtr, pSize, cudaMemcpyHostToDevice );
}

void pmCudaInterface::CountAndProbeProcessingElements()
{
	int lCountCUDA = 0;

	SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaGetDeviceCount", gFuncPtr_cudaGetDeviceCount, &lCountCUDA );

	for(int i = 0; i < lCountCUDA; ++i)
	{
		cudaDeviceProp lDeviceProp;
		SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaGetDeviceProperties", gFuncPtr_cudaGetDeviceProperties, &lDeviceProp, i );

        if(lDeviceProp.computeMode != cudaComputeModeProhibited)
        {
            if(!(lDeviceProp.major == 9999 && lDeviceProp.minor == 9999))
                GetDeviceVector().push_back(pmCudaDeviceData(i, lDeviceProp));
        }
	}
}

void pmCudaInterface::BindToDevice(size_t pDeviceIndex)
{
	int lHardwareId = GetDeviceVector()[pDeviceIndex].mHardwareId;

#ifdef CREATE_EXPLICIT_CUDA_CONTEXTS
    CUcontext& lContext = GetContextMap()[pDeviceIndex];
    if(cuCtxCreate(&lContext, 0, lHardwareId) != CUDA_SUCCESS)
        PMTHROW(pmFatalErrorException());
#endif
    
	SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaSetDevice", gFuncPtr_cudaSetDevice, lHardwareId );
    pmCudaAutoPtr lAutoPtr(1024);    // Initialize context with a dummy allocation and deallocation
}

void pmCudaInterface::UnbindFromDevice(size_t pDeviceIndex)
{
#ifdef CREATE_EXPLICIT_CUDA_CONTEXTS
    CUcontext& lContext = GetContextMap()[pDeviceIndex];
    if(cuCtxDestroy(lContext) != CUDA_SUCCESS)
        PMTHROW(pmFatalErrorException());
#endif
}

std::string pmCudaInterface::GetDeviceName(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = GetDeviceVector()[pDeviceIndex].mDeviceProp;
	return lProp.name;
}

std::string pmCudaInterface::GetDeviceDescription(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = GetDeviceVector()[pDeviceIndex].mDeviceProp;

	std::stringstream lStream;
    lStream << "Clock Rate=" << lProp.clockRate << ";sharedMemPerBlock=" << lProp.sharedMemPerBlock << ";computeCapability=" << lProp.major << "." << lProp.minor;

	return lStream.str();
}

size_t pmCudaInterface::GetCudaAlignment(size_t pDeviceIndex)
{
	const cudaDeviceProp& lProp = GetDeviceVector()[pDeviceIndex].mDeviceProp;
    
    return lProp.textureAlignment;
}
    
void pmCudaInterface::WaitForStreamCompletion(pmCudaStreamAutoPtr& pStream)
{
    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaStreamSynchronize", gFuncPtr_cudaStreamSynchronize, (cudaStream_t)(pStream.GetStream()) );
}

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
void* pmCudaInterface::AllocatePinnedBuffer(size_t pSize)
{
    void* lMem = NULL;

    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaHostAlloc", gFuncPtr_cudaHostAlloc, &lMem, pSize, cudaHostAllocDefault );

    return lMem;
}
    
void pmCudaInterface::DeallocatePinnedBuffer(const void* pMem)
{
    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaFreeHost", gFuncPtr_cudaFreeHost, const_cast<void*>(pMem) );
}
#endif

size_t pmCudaInterface::GetAvailableCudaMem()
{
    size_t lFreeMem, lTotalMem;
    
    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemGetInfo", gFuncPtr_cudaMemGetInfo, &lFreeMem, &lTotalMem );
    
    return lFreeMem;
}
    
int pmCudaInterface::GetCudaDriverVersion()
{
    int lDriverVersion;

    SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaDriverGetVersion", gFuncPtr_cudaDriverGetVersion, &lDriverVersion );
    
    return lDriverVersion;
}
    
size_t pmCudaInterface::GetCudaDeviceCount()
{
    return GetDeviceVector().size();
}

pmStatus pmCudaInterface::InvokeKernel(pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, pmStatus* pStatusCudaPtr, pmCudaStreamAutoPtr& pStreamPtr)
{
    cudaStream_t lStream = (cudaStream_t)(pStreamPtr.GetStream());

    std::vector<pmCudaMemcpyCommand>::const_iterator lHostToDeviceIter = pHostToDeviceCommands.begin(), lHostToDeviceEndIter = pHostToDeviceCommands.end();
    for(; lHostToDeviceIter != lHostToDeviceEndIter; ++lHostToDeviceIter)
    {
        const pmCudaMemcpyCommand& lMemcpyCommand = *lHostToDeviceIter;
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, lMemcpyCommand.srcPtr, lMemcpyCommand.size, cudaMemcpyHostToDevice, lStream );
    }

    pmStatus lStatus = ExecuteKernel(pTaskInfo, pTaskInfoCuda, pDeviceInfo, (pmDeviceInfo*)pDeviceInfoCudaPtr, pSubtaskInfoCuda, pCudaLaunchConf, pKernelPtr, pCustomKernelPtr, (void*)lStream, pStatusCudaPtr);

    CHECK_LAST_ERROR(GetRuntimeHandle());
    
    std::vector<pmCudaMemcpyCommand>::const_iterator lDeviceToHostIter = pDeviceToHostCommands.begin(), lDeviceToHostEndIter = pDeviceToHostCommands.end();
    for(; lDeviceToHostIter != lDeviceToHostEndIter; ++lDeviceToHostIter)
    {
        const pmCudaMemcpyCommand& lMemcpyCommand = *lDeviceToHostIter;
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, lMemcpyCommand.srcPtr, lMemcpyCommand.size, cudaMemcpyDeviceToHost, lStream );
    }
    
    return lStatus;
}

pmStatus pmCudaInterface::ExecuteKernel(const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, pmDeviceInfo* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, void* pStream, pmStatus* pStatusCudaPtr)
{
    if(pKernelPtr)
    {
        dim3 gridConf(pCudaLaunchConf.blocksX, pCudaLaunchConf.blocksY, pCudaLaunchConf.blocksZ);
        dim3 blockConf(pCudaLaunchConf.threadsX, pCudaLaunchConf.threadsY, pCudaLaunchConf.threadsZ);
        
        if(pCudaLaunchConf.sharedMem)
            pKernelPtr <<<gridConf, blockConf, pCudaLaunchConf.sharedMem, (cudaStream_t)pStream>>> (pTaskInfoCuda, pDeviceInfoCudaPtr, pSubtaskInfoCuda, pStatusCudaPtr);
        else
            pKernelPtr <<<gridConf, blockConf, 0, (cudaStream_t)pStream>>> (pTaskInfoCuda, pDeviceInfoCudaPtr, pSubtaskInfoCuda, pStatusCudaPtr);
    }
    else
    {
        return pCustomKernelPtr(pTaskInfo, pDeviceInfo, pSubtaskInfoCuda, (cudaStream_t)pStream);
    }

	return pmSuccess;
}

#endif	// SUPPORT_CUDA

}


/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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
#include "pmDataTypesHelper.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <iostream>
#include <vector>
#include <map>
#include <sstream>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

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
cudaError_t (*gFuncPtr_cudaDeviceReset)(void);

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

size_t pmCudaInterface::GetUnallocatableCudaMemSize()
{
    static size_t sUnallocatableMem = 0;
    
    if(!sUnallocatableMem)
    {
        const char* lVal = getenv("PMLIB_CUDA_MEM_PER_CARD_RESERVED_FOR_EXTERNAL_USE");
        if(lVal)
            sUnallocatableMem = (size_t)atoi(lVal);
        
        sUnallocatableMem += MIN_UNALLOCATED_CUDA_MEM_SIZE;
    }
    
    return sUnallocatableMem;
}

void* pmCudaInterface::AllocateCudaMem(size_t pSize)
{
    void* lAddr = NULL;

    if(pSize)
    {
        size_t lAvailableCudaMem = GetAvailableCudaMem();

        if(lAvailableCudaMem < pSize + GetUnallocatableCudaMemSize())
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

template<typename T>
struct isNonSentinel : public thrust::unary_function<T, bool>
{
public:
    isNonSentinel(T pSentinel)
    : mSentinel(pSentinel)
    {
    }
    
    __host__ __device__ bool operator() (T x)
    {
        return x != mSentinel;
    }
    
private:
    T mSentinel;
};
    
template<typename T>
struct CompressForSentinelImpl
{
    bool operator() (void* pCudaPtr, size_t pSize, void* pCompressedPtr, size_t& pCompressedSize, uint& pNonSentinelCount)
    {
        const T lSentinel = 0;

        thrust::device_ptr<T> lCudaPtr = thrust::device_pointer_cast((T*)pCudaPtr);
        
        uint lElemCount = pSize / sizeof(T);
        uint lSentinelCount = thrust::count(lCudaPtr, lCudaPtr + lElemCount, lSentinel);
        pNonSentinelCount = lElemCount - lSentinelCount;

        // Compress if non-sentinels less than limit
        if(pNonSentinelCount < lElemCount * CUDA_SENTINEL_COMPRESSION_MAX_NON_SENTINELS)
        {
            pCompressedSize = sizeof(uint) + pNonSentinelCount * sizeof(T) + pNonSentinelCount * sizeof(uint);

            thrust::device_ptr<T> lCompressedCudaPtr = thrust::device_pointer_cast((T*)pCompressedPtr + 1);
            thrust::device_ptr<uint> lIndexLoc = thrust::device_pointer_cast((uint*)((T*)pCompressedPtr + 1 + pNonSentinelCount));

            thrust::copy_if(lCudaPtr, lCudaPtr + lElemCount, lCompressedCudaPtr, isNonSentinel<T>(lSentinel));
            thrust::copy_if(thrust::make_counting_iterator((uint)0), thrust::make_counting_iterator(lElemCount), lCudaPtr, lIndexLoc, isNonSentinel<T>(lSentinel));

            return true;
        }
        
        return false;
    }
};

bool pmCudaInterface::CompressForSentinel(pmReductionDataType pReductionDataType, void* pCudaPtr, size_t pSize, void* pCompressedPtr, size_t& pCompressedSize, uint& pNonSentinelCount)
{
    switch(pReductionDataType)
    {
        case REDUCE_INTS:
        {
            return CompressForSentinelImpl<int>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        }
            
        case REDUCE_UNSIGNED_INTS:
        {
            return CompressForSentinelImpl<uint>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        }
            
        case REDUCE_LONGS:
        {
            return CompressForSentinelImpl<long>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        }
            
        case REDUCE_UNSIGNED_LONGS:
        {
            return CompressForSentinelImpl<ulong>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        }
            
        case REDUCE_FLOATS:
        {
            return CompressForSentinelImpl<float>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        }
            
        case REDUCE_DOUBLES:
        {
        #if __CUDA_ARCH__ >= 130
            return CompressForSentinelImpl<double>() (pCudaPtr, pSize, pCompressedPtr, pCompressedSize, pNonSentinelCount);
        #else
            break;
        #endif
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
    
    return false;
}

pmStatus pmCudaInterface::InvokeKernel(pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, const pmSubtaskInfo& pSubtaskInfoCuda, const pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, pmStatus* pStatusCudaPtr, pmCudaStreamAutoPtr& pStreamPtr, pmReductionDataType pSentinelCompressionReductionDataType, void* pCompressedPtr
#ifdef ENABLE_TASK_PROFILING
   , pmTaskProfiler* pTaskProfiler
#endif
   )
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
    
    bool lCompressed = false;
    size_t lCompressedSize = 0;
    uint lNonSentinelCount = 0;

    if(pSentinelCompressionReductionDataType != MAX_REDUCTION_DATA_TYPES && pCompressedPtr)
    {
    #ifdef DUMP_DATA_COMPRESSION_STATISTICS
        #ifdef ENABLE_TASK_PROFILING
            pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTaskProfiler, taskProfiler::DATA_COMPRESSION);
        #endif
        
        lCompressed = CompressForSentinel(pSentinelCompressionReductionDataType, (*pDeviceToHostCommands.begin()).srcPtr, (*pDeviceToHostCommands.begin()).size, pCompressedPtr, lCompressedSize, lNonSentinelCount);

        pmCompressionDataRecorder::RecordCompressionData((*pDeviceToHostCommands.begin()).size, lCompressedSize, false);
    #else
        lCompressed = CompressForSentinel(pSentinelCompressionReductionDataType, (*pDeviceToHostCommands.begin()).srcPtr, (*pDeviceToHostCommands.begin()).size, pCompressedPtr, lCompressedSize, lNonSentinelCount);
    #endif
    }

    if(!lCompressed)
    {
        std::vector<pmCudaMemcpyCommand>::const_iterator lDeviceToHostIter = pDeviceToHostCommands.begin(), lDeviceToHostEndIter = pDeviceToHostCommands.end();
        for(; lDeviceToHostIter != lDeviceToHostEndIter; ++lDeviceToHostIter)
        {
            const pmCudaMemcpyCommand& lMemcpyCommand = *lDeviceToHostIter;
            SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, lMemcpyCommand.srcPtr, lMemcpyCommand.size, cudaMemcpyDeviceToHost, lStream );
        }
    }
    else
    {
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, (uint*)pCompressedPtr, &lNonSentinelCount, sizeof(uint), cudaMemcpyHostToDevice, lStream );

        const pmCudaMemcpyCommand& lMemcpyCommand = (*pDeviceToHostCommands.begin());
        SAFE_EXECUTE_CUDA( GetRuntimeHandle(), "cudaMemcpyAsync", gFuncPtr_cudaMemcpyAsync, lMemcpyCommand.destPtr, pCompressedPtr, lCompressedSize, cudaMemcpyDeviceToHost, lStream );
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
    
// This method is meant to be called at application exit. So, it does not throw and does not call SAFE_EXECUTE_CUDA
void pmCudaInterface::ForceResetAllCudaDevices()
{
    std::vector<pmCudaDeviceData>& lDeviceVector = GetDeviceVector();
    
    std::vector<pmCudaDeviceData>::iterator lIter = lDeviceVector.begin(), lEndIter = lDeviceVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        EXECUTE_CUDA_SYMBOL( GetRuntimeHandle(), "cudaSetDevice", gFuncPtr_cudaSetDevice, (*lIter).mHardwareId );
        EXECUTE_CUDA_SYMBOL( GetRuntimeHandle(), "cudaDeviceReset", gFuncPtr_cudaDeviceReset );
    }
}

}

#endif	// SUPPORT_CUDA

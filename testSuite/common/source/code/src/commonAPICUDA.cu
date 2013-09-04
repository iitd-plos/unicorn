
#ifdef BUILD_CUDA
#ifdef ENABLE_BLAS

#include "commonAPI.h"

cublasHandle_t CreateCublasHandle()
{
    cublasHandle_t lCublasHandle;
    
    CUBLAS_ERROR_CHECK("cublasCreate", cublasCreate(&lCublasHandle));
    
    return lCublasHandle;
}

void DestroyCublasHandle(cublasHandle_t pCublasHandle)
{
    CUBLAS_ERROR_CHECK("cublasDestroy", cublasDestroy(pCublasHandle));
}

cublasHandleMapType& GetCublasHandleMap()
{
    static cublasHandleMapType gMap;
    
    return gMap;
}

cublasHandle_t GetCublasHandle(pmDeviceHandle pDeviceHandle)
{
    cublasHandleMapType& lCublasHandleMap = GetCublasHandleMap();
    
    typename cublasHandleMapType::iterator lIter = lCublasHandleMap.find(pDeviceHandle);
    if(lIter == lCublasHandleMap.end())
        lIter = lCublasHandleMap.insert(std::make_pair(pDeviceHandle, CreateCublasHandle())).first;
    
    return lIter->second;
}

void FreeCublasHandles()
{
    cublasHandleMapType& lCublasHandleMap = GetCublasHandleMap();
    typename cublasHandleMapType::iterator lIter = lCublasHandleMap.begin(), lEndIter = lCublasHandleMap.end();
    
    for(; lIter != lEndIter; ++lIter)
        DestroyCublasHandle(lIter->second);
    
    lCublasHandleMap.clear();
}

/* class cublasHandleManager */
cublasHandleManager::cublasHandleManager()
: mHandle(CreateCublasHandle())
{
}
    
cublasHandleManager::~cublasHandleManager()
{
    DestroyCublasHandle(mHandle);
}
    
cublasHandle_t cublasHandleManager::GetHandle()
{
    return mHandle;
}

#endif
#endif


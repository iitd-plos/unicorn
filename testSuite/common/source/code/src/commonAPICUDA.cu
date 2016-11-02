
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
 */

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


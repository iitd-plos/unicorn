
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

#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmDispatcherGPU.h"

#ifdef MACOS
#include <sys/sysctl.h>
#else
#include <string>
#endif

#include SYSTEM_CONFIGURATION_HEADER	// For sysconf function

namespace pm
{

pmStubManager* pmStubManager::GetStubManager()
{
	static pmStubManager lStubManager;
    return &lStubManager;
}

pmStubManager::pmStubManager()
{
	CreateExecutionStubs();
    
#ifdef SUPPORT_SPLIT_SUBTASKS
    CreateCpuNumaDomains();
#endif
}

pmStubManager::~pmStubManager()
{
	DestroyExecutionStubs();
}

size_t pmStubManager::GetProcessingElementsCPU()
{
	return mProcessingElementsCPU;
}

size_t pmStubManager::GetProcessingElementsGPU()
{
	return mProcessingElementsGPU;
}

size_t pmStubManager::GetStubCount()
{
	return mStubCount;
}

pmExecutionStub* pmStubManager::GetStub(const pmProcessingElement* pDevice) const
{
	size_t pIndex = (size_t)(pDevice->GetDeviceIndexInMachine());

	return GetStub((uint)pIndex);
}

pmExecutionStub* pmStubManager::GetStub(uint pIndex) const
{
	if(pIndex >= mStubVector.size())
		PMTHROW(pmStubException(pmStubException::INVALID_STUB_INDEX));

	return mStubVector[pIndex];
}

pmExecutionStub* pmStubManager::GetCpuStub(uint pIndex) const
{
    return GetStub(pIndex);
}

pmExecutionStub* pmStubManager::GetGpuStub(uint pIndex) const
{
    return GetStub((uint)mProcessingElementsCPU + pIndex);
}
    
#ifdef SUPPORT_CUDA
size_t pmStubManager::GetMaxCpuDevicesPerHostForCpuPlusGpuTasks()
{
    const char* lVal = getenv("PMLIB_MAX_CPU_PER_HOST_FOR_CPU_PLUS_GPU_TASKS");
    if(lVal)
    {
        size_t lValue = (size_t)atoi(lVal);

        if(lValue != 0 && lValue < mProcessingElementsCPU)
            return lValue;
    }
    
    return mProcessingElementsCPU;
}
#endif
    
void pmStubManager::GetCpuIdInfo(uint pRegA, uint pRegC, uint& pEAX, uint& pEBX, uint& pECX, uint& pEDX)
{
    asm volatile ("cpuid" : "=a" (pEAX), "=b" (pEBX), "=c" (pECX), "=d" (pEDX) : "a" (pRegA), "c" (pRegC));
}
    
void pmStubManager::CreateExecutionStubs()
{
#if defined(MACOS)
    size_t lBufferLen = sizeof(mProcessingElementsCPU);
    if(sysctlbyname("hw.physicalcpu", &mProcessingElementsCPU, &lBufferLen, NULL, 0) != 0)
        mProcessingElementsCPU = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(LINUX)
    uint lA, lB, lC, lD;
    
    GetCpuIdInfo(1, 0, lA, lB, lC, lD);
    size_t lLogicalCores = ((lB >> 16) & 0xff);
    bool lHyperThreadingEnabled = (lD & (0x1 << 28));
    
    GetCpuIdInfo(0, 0, lA, lB, lC, lD);
    std::string lVendor((char*)(&lB), 4);
    lVendor.append(std::string((char*)(&lD), 4));
    lVendor.append(std::string((char*)(&lC), 4));
    
    size_t lPhysicalCores = 0, lPos = 0;
    if(lVendor.find(std::string("Intel"), lPos) != std::string::npos)
    {
        GetCpuIdInfo(4, 0, lA, lB, lC, lD);
        lPhysicalCores = ((lA >> 26) & 0x3f) + 1;
    }
    else if(lVendor.find(std::string("AMD"), lPos) != std::string::npos)
    {
        GetCpuIdInfo(0x80000008, 0, lA, lB, lC, lD);
        lPhysicalCores = ((size_t)(lC & 0xff)) + 1;
    }
    else
    {
        lPhysicalCores = sysconf(_SC_NPROCESSORS_ONLN);
    }
    
    size_t lAvailableCores = sysconf(_SC_NPROCESSORS_ONLN);
    
    if(lAvailableCores > lPhysicalCores)
    {
        lHyperThreadingEnabled &= (lPhysicalCores < lLogicalCores);
        mProcessingElementsCPU = sysconf(_SC_NPROCESSORS_ONLN) / (lHyperThreadingEnabled ? 2 : 1);
    }
    else
    {
        mProcessingElementsCPU = lAvailableCores;
    }
#else
	mProcessingElementsCPU = sysconf(_SC_NPROCESSORS_ONLN);
#endif

    const char* lVal = getenv("PMLIB_MAX_CPU_PER_HOST");
    if(lVal)
    {
        size_t lValue = (size_t)atoi(lVal);

        if(lValue != 0 && lValue < mProcessingElementsCPU)
            mProcessingElementsCPU = lValue;
    }

#if defined(MACOS)
    size_t lPhysicalMemory = 0;
    size_t lBufferLength = sizeof(lPhysicalMemory);

    if(sysctlbyname("hw.memsize", &lPhysicalMemory, &lBufferLength, NULL, 0) != 0)
        PMTHROW(pmFatalErrorException());
#else
    size_t lPhysicalMemory = sysconf(_SC_PHYS_PAGES) * ::getpagesize();
#endif
    
	for(size_t i = 0; i < mProcessingElementsCPU; ++i)
		mStubVector.push_back(new pmStubCPU(i, (uint)(mStubVector.size())));

	mProcessingElementsGPU = pmDispatcherGPU::GetDispatcherGPU()->ProbeProcessingElementsAndCreateStubs(mStubVector);

	mStubCount = mProcessingElementsCPU + mProcessingElementsGPU;
    
    std::vector<pmExecutionStub*>::iterator lIter = mStubVector.begin(), lEndIter = mStubVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        (*lIter)->ThreadBindEvent(lPhysicalMemory, mStubCount);
        (*lIter)->WaitForQueuedCommands();
    }
}

void pmStubManager::CreateCpuNumaDomains()
{
    // Put all CPU devices in the same NUMA domain for now
    mCpuNumaDomains.emplace_back(mStubVector.begin(), mStubVector.begin() + mProcessingElementsCPU);
    
    std::for_each(mStubVector.begin(), mStubVector.begin() + mProcessingElementsCPU, [&] (pmExecutionStub* pStub)
    {
        mCpuNumaDomainsMap.emplace(pStub, 0);
    });
}

const std::vector<std::vector<pmExecutionStub*>>& pmStubManager::GetCpuNumaDomains() const
{
    return mCpuNumaDomains;
}

ushort pmStubManager::GetCpuNumaDomainsCount() const
{
    return (ushort)mCpuNumaDomains.size();
}
    
const std::vector<pmExecutionStub*>& pmStubManager::GetCpuNumaDomain(ushort pDomainId) const
{
    DEBUG_EXCEPTION_ASSERT(pDomainId < mCpuNumaDomains.size());

    return mCpuNumaDomains[(size_t)pDomainId];
}

ushort pmStubManager::GetNumaDomainIdForCpuDevice(uint pIndex) const
{
    return mCpuNumaDomainsMap.find(GetCpuStub(pIndex))->second;
}

#ifdef SUPPORT_CUDA
void pmStubManager::FreeGpuResources()
{
    for(size_t i = 0; i < mStubCount; ++i)
    {
        if(dynamic_cast<pmStubGPU*>(mStubVector[i]))
            (static_cast<pmStubGPU*>(mStubVector[i]))->FreeResources();
	}

    for(size_t i = 0; i < mStubCount; ++i)
    {
        if(dynamic_cast<pmStubGPU*>(mStubVector[i]))
            (static_cast<pmStubGPU*>(mStubVector[i]))->WaitForQueuedCommands();
	}
}
    
void pmStubManager::PurgeAddressSpaceEntriesFromGpuCaches(const pmAddressSpace* pAddressSpace)
{
    for(size_t i = 0; i < mStubCount; ++i)
    {
        if(dynamic_cast<pmStubGPU*>(mStubVector[i]))
            (static_cast<pmStubGPU*>(mStubVector[i]))->PurgeAddressSpaceEntriesFromGpuCache(pAddressSpace);
	}
}
#endif

void pmStubManager::DestroyExecutionStubs()
{
#ifdef SUPPORT_CUDA
	FreeGpuResources();
#endif

	for(size_t i = 0; i < mStubCount; ++i)
		delete mStubVector[i];
}

#ifdef DUMP_EVENT_TIMELINE
void pmStubManager::InitializeEventTimelines()
{
    std::vector<pmExecutionStub*>::iterator lIter = mStubVector.begin(), lEndIter = mStubVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->InitializeEventTimeline();
}
#endif
    
void pmStubManager::WaitForAllStubsToFinish()
{
    std::vector<pmExecutionStub*>::iterator lIter = mStubVector.begin(), lEndIter = mStubVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->WaitForQueuedCommands();
}

};





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

#ifdef SUPPORT_CUDA
void pmStubManager::FreeGpuResources()
{
    for(size_t i=0; i<mStubCount; ++i)
    {
        if(dynamic_cast<pmStubGPU*>(mStubVector[i]))
            (static_cast<pmStubGPU*>(mStubVector[i]))->FreeResources();
	}

    for(size_t i=0; i<mStubCount; ++i)
    {
        if(dynamic_cast<pmStubGPU*>(mStubVector[i]))
            (static_cast<pmStubGPU*>(mStubVector[i]))->WaitForQueuedCommands();
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




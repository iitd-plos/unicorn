
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

pmExecutionStub* pmStubManager::GetStub(pmProcessingElement* pDevice)
{
	size_t pIndex = (size_t)(pDevice->GetDeviceIndexInMachine());

	return GetStub((uint)pIndex);
}

pmExecutionStub* pmStubManager::GetStub(uint pIndex)
{
	if(pIndex >= mStubVector.size())
		PMTHROW(pmStubException(pmStubException::INVALID_STUB_INDEX));

	return mStubVector[pIndex];
}

pmStatus pmStubManager::CreateExecutionStubs()
{
	/* WIN 32 Code
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	numCPU = sysinfo.dwNumberOfProcessors;
	*/
    
    // sysconf(_SC_HT_CAPABLE) && sysconf(_SC_HT_ENABLED);

	mProcessingElementsCPU = sysconf(_SC_NPROCESSORS_ONLN);
	for(size_t i=0; i<mProcessingElementsCPU; ++i)
		mStubVector.push_back(new pmStubCPU(i, (uint)(mStubVector.size())));

	mProcessingElementsGPU = pmDispatcherGPU::GetDispatcherGPU()->ProbeProcessingElementsAndCreateStubs(mStubVector);

	mStubCount = mProcessingElementsCPU + mProcessingElementsGPU;
    
    std::vector<pmExecutionStub*>::iterator lIter = mStubVector.begin(), lEndIter = mStubVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->ThreadBindEvent();

	return pmSuccess;
}

pmStatus pmStubManager::FreeGpuResources()
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
    
	return pmSuccess;
}

pmStatus pmStubManager::DestroyExecutionStubs()
{
	FreeGpuResources();

	for(size_t i=0; i<mStubCount; ++i)
		delete mStubVector[i];

	return pmSuccess;
}

#ifdef DUMP_EVENT_TIMELINE
void pmStubManager::InitializeEventTimelines()
{
    std::vector<pmExecutionStub*>::iterator lIter = mStubVector.begin(), lEndIter = mStubVector.end();
    for(; lIter != lEndIter; ++lIter)
        (*lIter)->InitializeEventTimeline();
}
#endif

};




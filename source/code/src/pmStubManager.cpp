
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmDispatcherGPU.h"

#include SYSTEM_CONFIGURATION_HEADER	// For sysconf function

namespace pm
{

pmStubManager* pmStubManager::mStubManager = NULL;

pmStubManager* pmStubManager::GetStubManager()
{
	if(!mStubManager)
		mStubManager = new pmStubManager();

	return mStubManager;
}

pmStatus pmStubManager::DestroyStubManager()
{
	delete mStubManager;
	mStubManager = NULL;

	return pmSuccess;
}

pmStubManager::pmStubManager()
{
	CountAndProbeProcessingElements();

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

pmExecutionStub* pmStubManager::GetStubAtIndex(ulong pIndex)
{
	if(pIndex >= mStubVector.size())
		throw pmStubException(pmStubException::INVALID_STUB_INDEX);

	return mStubVector[pIndex];
}

pmStatus pmStubManager::CreateExecutionStubs()
{
	mStubVector.resize(mStubCount);
	for(size_t i=0; i<mStubCount; ++i)
		mStubVector[i] = new pmExecutionStub();
}

pmStatus pmStubManager::DestroyExecutionStubs()
{
	for(size_t i=0; i<mStubCount; ++i)
		delete mStubVector[i];
}

pmStatus pmStubManager::CountAndProbeProcessingElements()
{
	/* WIN 32 Code
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	numCPU = sysinfo.dwNumberOfProcessors;
	*/

	mProcessingElementsCPU = sysconf(_SC_NPROCESSORS_ONLN);
	mProcessingElementsGPU = pmDispatcherGPU::GetDispatcherGPU()->GetCountGPU();

	mStubCount = mProcessingElementsCPU + mProcessingElementsGPU;

	return pmSuccess;
}

};

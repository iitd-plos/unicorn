
#include "pmDispatcherGPU.h"
#include "pmExecutionStub.h"
#include "pmLogger.h"

#define CUDA_LIBRARY_CUTIL (char*)"libcutil.so"
#define CUDA_LIBRARY_CUDART (char*)"libcudart.so"

namespace pm
{

pmDispatcherGPU* pmDispatcherGPU::mDispatcherGPU = NULL;

/* class pmDispatcherGPU */
pmDispatcherGPU* pmDispatcherGPU::GetDispatcherGPU()
{
	if(!mDispatcherGPU)
		mDispatcherGPU = new pmDispatcherGPU();

	return mDispatcherGPU;
}

pmStatus pmDispatcherGPU::DestroyDispatcherGPU()
{
	delete mDispatcherGPU;
	mDispatcherGPU = NULL;

	return pmSuccess;
}

pmDispatcherGPU::pmDispatcherGPU()
{
#ifdef SUPPORT_CUDA
	try
	{
		mDispatcherCUDA = new pmDispatcherCUDA();
	}
	catch(pmExceptionGPU e)
	{
		mDispatcherCUDA = NULL;
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be loaded");
	}
#else
	mDispatcherCUDA = NULL;
#endif
}

pmDispatcherGPU::~pmDispatcherGPU()
{
	delete mDispatcherCUDA;
}

pmDispatcherCUDA* pmDispatcherGPU::GetDispatcherCUDA()
{
	return mDispatcherCUDA;
}

size_t pmDispatcherGPU::GetCountGPU()
{
	return mCountGPU;
}

size_t pmDispatcherGPU::ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector)
{
	size_t lCountCUDA = 0;
	if(mDispatcherCUDA)
	{
		lCountCUDA = mDispatcherCUDA->GetCountCUDA();
		for(size_t i=0; i<lCountCUDA; ++i)
			pStubVector.push_back(new pmStubCUDA(i, pStubVector.size()));
	}

	mCountGPU = lCountCUDA;
	return mCountGPU;
}

/* class pmDispatcherCUDA */
pmDispatcherCUDA::pmDispatcherCUDA()
{
	mCutilHandle = OpenLibrary(CUDA_LIBRARY_CUTIL);
	mRuntimeHandle = OpenLibrary(CUDA_LIBRARY_CUDART);

	if(!mCutilHandle || !mRuntimeHandle)
	{
		CloseLibrary(mCutilHandle);
		CloseLibrary(mRuntimeHandle);
		
		throw pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::LIBRARY_OPEN_FAILURE);
	}

	CountAndProbeProcessingElements();
}

pmDispatcherCUDA::~pmDispatcherCUDA()
{
	try
	{
		CloseLibrary(mCutilHandle);
		CloseLibrary(mRuntimeHandle);
	}
	catch(pmIgnorableException e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be closed properly");
	}
}

size_t pmDispatcherCUDA::GetCountCUDA()
{
	return mCountCUDA;
}

}


#include "pmDispatcherGPU.h"
#include "pmLogger.h"

#define CUDA_LIBRARY_CUTIL "libcutil.so"
#define CUDA_LIBRARY_CUDART "libcudart.so"

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

	CountAndProbeProcessingElements();
}

pmDispatcherGPU::~pmDispatcherGPU()
{
	delete mDispatcherCUDA;
}

size_t pmDispatcherGPU::GetCountGPU()
{
	return mCountGPU;
}

pmStatus pmDispatcherGPU::CountAndProbeProcessingElements()
{
	mCountGPU = 0;
	if(mDispatcherCUDA)
		mCountGPU += mDispatcherCUDA->GetCountCUDA();

	return pmSuccess;
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

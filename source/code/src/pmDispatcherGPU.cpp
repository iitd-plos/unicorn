
#include "pmDispatcherGPU.h"
#include "pmExecutionStub.h"
#include "pmHardware.h"
#include "pmMemSection.h"
#include "pmLogger.h"
#include "pmTask.h"

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
			pStubVector.push_back(new pmStubCUDA(i, (uint)pStubVector.size()));
	}

	mCountGPU = lCountCUDA;
	return mCountGPU;
}

/* class pmDispatcherCUDA */
pmDispatcherCUDA::pmDispatcherCUDA()
{
#ifdef SUPPORT_CUDA
	//mCutilHandle = OpenLibrary(CUDA_LIBRARY_CUTIL);
	mRuntimeHandle = OpenLibrary(CUDA_LIBRARY_CUDART);

	//if(!mCutilHandle || !mRuntimeHandle)
	if(!mRuntimeHandle)
	{
		//CloseLibrary(mCutilHandle);
		CloseLibrary(mRuntimeHandle);
		
		PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::LIBRARY_OPEN_FAILURE));
	}

	CountAndProbeProcessingElements();
#endif
}

pmDispatcherCUDA::~pmDispatcherCUDA()
{
	try
	{
		//CloseLibrary(mCutilHandle);
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

pmStatus pmDispatcherCUDA::InvokeKernel(size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr)
{
#ifdef SUPPORT_CUDA
	pmTask* lTask = (pmTask*)(pTaskInfo.taskHandle);
	uint lOriginatingMachineIndex = (uint)(*(lTask->GetOriginatingHost()));
	ulong lSequenceNumber = lTask->GetSequenceNumber();

	pmMemSection* lInputMemSection = lTask->GetMemSectionRO();

	return InvokeKernel(pBoundDeviceIndex, pTaskInfo, pSubtaskInfo, pCudaLaunchConf, pOutputMemWriteOnly, pKernelPtr, lOriginatingMachineIndex, lSequenceNumber, lInputMemSection);
#else	
        return pmSuccess;
#endif
}

}

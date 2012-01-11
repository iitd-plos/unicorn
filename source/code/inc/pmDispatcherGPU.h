
#ifndef __PM_DISPATCHER_GPU__
#define __PM_DISPATCHER_GPU__

#include "pmInternalDefinitions.h"

#ifdef SUPPORT_CUDA
#include "cuda.h"
#include <vector>
#endif

namespace pm
{

class pmExecutionStub;

/**
 * \brief The class responsible for all GPU related operations on various graphics cards
 */

class pmGraphicsBase : public pmBase
{
};

class pmDispatcherCUDA : public pmGraphicsBase
{
	public:
		pmDispatcherCUDA();
		virtual ~pmDispatcherCUDA();

		pmStatus BindToDevice(size_t pDeviceIndex);

		size_t GetCountCUDA();

		std::string GetDeviceName(size_t pDeviceIndex);
		std::string GetDeviceDescription(size_t pDeviceIndex);
		pmStatus InvokeKernel(pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmSubtaskCallback_GPU_CUDA pKernelPtr);
	
	private:
		pmStatus CountAndProbeProcessingElements();

		size_t mCountCUDA;
		void* mCutilHandle;
		void* mRuntimeHandle;

#ifdef SUPPORT_CUDA
		std::vector<std::pair<int, cudaDeviceProp> > mDeviceVector;
#endif
};

class pmDispatcherGPU : public pmGraphicsBase
{
	public:
		static pmDispatcherGPU* GetDispatcherGPU();
		pmStatus DestroyDispatcherGPU();

		size_t ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector);
		pmDispatcherCUDA* GetDispatcherCUDA();

		size_t GetCountGPU();

	private:
		pmDispatcherGPU();
		virtual ~pmDispatcherGPU();
				
		static pmDispatcherGPU* mDispatcherGPU;

		size_t mCountGPU;
		pmDispatcherCUDA* mDispatcherCUDA;
};

} // end namespace pm

#endif

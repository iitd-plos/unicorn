
#ifndef __PM_DISPATCHER_GPU__
#define __PM_DISPATCHER_GPU__

#include "pmInternalDefinitions.h"

#ifdef SUPPORT_CUDA
#include "cuda.h"
#include <vector>
#endif

namespace pm
{

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
		~pmDispatcherCUDA();

		size_t GetCountCUDA();
	
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

		size_t GetCountGPU();

	private:
		pmDispatcherGPU();
		~pmDispatcherGPU();
		
		pmStatus CountAndProbeProcessingElements();
		
		static pmDispatcherGPU* mDispatcherGPU;

		size_t mCountGPU;
		pmDispatcherCUDA* mDispatcherCUDA;
};

} // end namespace pm

#endif

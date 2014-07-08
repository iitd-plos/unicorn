
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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

#ifndef __PM_DISPATCHER_GPU__
#define __PM_DISPATCHER_GPU__

#include "pmBase.h"
#include "pmCudaInterface.h"

namespace pm
{

class pmStubCUDA;
class pmAddressSpace;

/**
 * \brief The class responsible for all GPU related operations on various graphics cards
 */

#ifdef SUPPORT_CUDA
class pmDispatcherCUDA : public pmBase
{
	public:
		pmDispatcherCUDA();
		~pmDispatcherCUDA();

		pmStatus InvokeKernel(pmTask* pTask, pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, pmCudaStreamAutoPtr& pStreamPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands);
    
        void* GetRuntimeHandle();
    
	private:
		void* mRuntimeHandle;
};
#endif

class pmDispatcherGPU : public pmBase
{
	public:
		static pmDispatcherGPU* GetDispatcherGPU();

		size_t ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector);

    #ifdef SUPPORT_CUDA
		pmDispatcherCUDA* GetDispatcherCUDA();
    #endif

		size_t GetCountGPU();

	private:
		pmDispatcherGPU();
				
		size_t mCountGPU;
    
    #ifdef SUPPORT_CUDA
		finalize_ptr<pmDispatcherCUDA> mDispatcherCUDA;
    #endif
};

void* GetExportedSymbol(void* pLibHandle, char* pSymbol);
    
} // end namespace pm

#endif

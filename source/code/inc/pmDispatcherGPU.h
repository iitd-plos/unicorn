
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

		pmStatus InvokeKernel(pmTask* pTask, pmStubCUDA* pStub, const pmTaskInfo& pTaskInfo, const pmTaskInfo& pTaskInfoCuda, const pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, pmReductionDataType pSentinelCompressionReductionDataType, pmCudaStreamAutoPtr& pStreamPtr, const std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, const std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands);
    
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

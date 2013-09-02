
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

#ifndef __PM_DISPATCHER_GPU__
#define __PM_DISPATCHER_GPU__

#include "pmBase.h"

#ifdef SUPPORT_CUDA
#include "pmResourceLock.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include <map>
#endif

namespace pm
{

class pmExecutionStub;
class pmMemSection;

/**
 * \brief The class responsible for all GPU related operations on various graphics cards
 */

#ifdef SUPPORT_CUDA
class pmDispatcherCUDA : public pmBase
{
	public:
		pmDispatcherCUDA();
		virtual ~pmDispatcherCUDA();

		pmStatus BindToDevice(size_t pDeviceIndex);

		size_t GetCountCUDA();

		std::string GetDeviceName(size_t pDeviceIndex);
		std::string GetDeviceDescription(size_t pDeviceIndex);
		pmStatus InvokeKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr);

        void* CreateDeviceInfoCudaPtr(pmDeviceInfo& pDeviceInfo);
        void DestroyDeviceInfoCudaPtr(void* pDeviceInfoCudaPtr);

        size_t GetCudaAlignment(size_t pDeviceIndex);

    #ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
        void* AllocatePinnedBuffer(size_t pSize);
        void DeallocatePinnedBuffer(void* pMem);
    #endif

        void* CreateTaskConf(pmTaskInfo& pTaskInfo);
        void DestroyTaskConf(void* pTaskConfCudaPtr);
    
        ulong FindCollectivelyExecutableSubtaskRangeEnd(pmExecutionStub* pStub, const pmSubtaskRange& pSubtaskRange, bool pMultiAssign, std::vector<std::vector<std::pair<size_t, size_t> > >& pOffsets, size_t& pTotalMem);

        pmStatus InvokeKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, std::vector<pmCudaMemcpyCommand>& pHostToDeviceCommands, std::vector<pmCudaMemcpyCommand>& pDeviceToHostCommands, std::vector<void*>& pCudaPointers);
    
        pmStatus FreeLastExecutionResources(pmLastCudaExecutionRecord& pLastExecutionRecord);
    
        size_t GetAvailableCudaMem();
        void* GetRuntimeHandle();
    
        void* AllocateCudaMem(size_t pSize);
        void DeallocateCudaMem(void* pPtr);
    
        void StreamFinishCallback(void* pUserData);
    
	private:
		pmStatus CountAndProbeProcessingElements();

        size_t ComputeAlignedMemoryRequirement(size_t pAllocation1, size_t pAllocation2, size_t pDeviceIndex);
    
        void ComputeMemoryRequiredForSubtask(pmExecutionStub* pStub, pmSubtaskInfo& pSubtaskInfo, ulong* pLastSubtaskIdIfSameTask, uint pOriginatingMachineIndex, ulong pSequenceNumber, size_t& pInputMem, size_t& pOutputMem, size_t& pScratchMem, bool& pUseLastSubtaskInputMem);

        void* CheckAndGetScratchBuffer(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId, size_t& pScratchBufferSize, pmScratchBufferInfo& pScratchBufferInfo);

        void GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo);
        void GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo);
        void GetNonConsolidatedSubscriptionsForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubscriptionType pSubscriptionType, pmSubtaskInfo& pSubtaskInfo, std::vector<std::pair<size_t, size_t> >& pSubscriptionVector);
    
        void GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo);
    
        bool SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId1, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType);
    
        void MarkInsideUserCode(pmExecutionStub* pStub);
        void MarkInsideLibraryCode(pmExecutionStub* pStub);
        bool RequiresPrematureExit(pmExecutionStub* pStub);
    
        int GetCudaDriverVersion();

        pmStatus ExecuteKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmDeviceInfo& pDeviceInfo, pmDeviceInfo* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfoCuda, pmCudaLaunchConf& pCudaLaunchConf, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, cudaStream_t pStream, pmStatus* pStatusPtr);

		size_t mCountCUDA;
		void* mCutilHandle;
		void* mRuntimeHandle;

		std::vector<std::pair<int, cudaDeviceProp> > mDeviceVector;
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
		virtual ~pmDispatcherGPU();
				
		size_t mCountGPU;
    
    #ifdef SUPPORT_CUDA
		pmDispatcherCUDA* mDispatcherCUDA;
    #endif
};

} // end namespace pm

#endif

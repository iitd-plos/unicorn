
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

class pmGraphicsBase : public pmBase
{
};

#ifdef SUPPORT_CUDA
namespace dispatcherCUDA
{
    typedef struct lastExecutionRecord
    {
        uint taskOriginatingMachineIndex;
        ulong taskSequenceNumber;
        ulong lastSubtaskId;
        void* inputMemCudaPtr;
        void* taskConfCudaPtr;
    } lastExecutionRecord;
}
#endif
    
class pmDispatcherCUDA : public pmGraphicsBase
{
	public:
		pmDispatcherCUDA();
		virtual ~pmDispatcherCUDA();

		pmStatus BindToDevice(size_t pDeviceIndex);

		size_t GetCountCUDA();

		std::string GetDeviceName(size_t pDeviceIndex);
		std::string GetDeviceDescription(size_t pDeviceIndex);
		pmStatus InvokeKernel(pmExecutionStub* pStub, size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr);

#ifdef SUPPORT_CUDA
        void* GetDeviceInfoCudaPtr(pmDeviceInfo& pDeviceInfo);
        void FreeDeviceInfoCudaPtr(void* pDeviceInfoCudaPtr);

		pmStatus InvokeKernel(pmExecutionStub* pStub, size_t pBoundDeviceIndex, pmTaskInfo& pTaskInfo, pmDeviceInfo& pDeviceInfo, void* pDeviceInfoCudaPtr, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr, uint pOriginatingMachineIndex, ulong pSequenceNumber, void* pTaskOutputMem);

		pmStatus FreeLastExecutionResources(size_t pBoundDeviceIndex);
#endif
    
	private:
		pmStatus CountAndProbeProcessingElements();

        void GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo);
        void GetNonConsolidatedSubscriptionsForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubscriptionType pSubscriptionType, pmSubtaskInfo& pSubtaskInfo, std::vector<std::pair<size_t, size_t> >& pSubscriptionVector);
    
        void GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo);
    
        bool SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId1, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType);
    
        void MarkInsideUserCode(pmExecutionStub* pStub, ulong pSubtaskId);
        void MarkInsideLibraryCode(pmExecutionStub* pStub, ulong pSubtaskId);
        bool RequiresPrematureExit(pmExecutionStub* pStub, ulong pSubtaskId);

		size_t mCountCUDA;
		void* mCutilHandle;
		void* mRuntimeHandle;

#ifdef SUPPORT_CUDA
		std::vector<std::pair<int, cudaDeviceProp> > mDeviceVector;
        
		std::map<size_t, dispatcherCUDA::lastExecutionRecord> mLastExecutionMap;  // CUDA Device Index vs. last execution details
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mLastExecutionLock;
#endif
};

class pmDispatcherGPU : public pmGraphicsBase
{
    friend class pmController;
    
	public:
		static pmDispatcherGPU* GetDispatcherGPU();

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

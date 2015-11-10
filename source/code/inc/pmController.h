
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

#ifndef __PM_CONTROLLER__
#define __PM_CONTROLLER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmLogger.h"
#include "pmDispatcherGPU.h"
#include "pmStubManager.h"
#include "pmNetwork.h"
#include "pmCommunicator.h"
#include "pmDevicePool.h"
#include "pmMemoryManager.h"
#include "pmTaskManager.h"
#include "pmScheduler.h"
#include "pmTimedEventManager.h"
#include "pmTls.h"
#include "pmHeavyOperations.h"

namespace pm
{

class pmSignalWait;
class pmCluster;
extern pmCluster* PM_GLOBAL_CLUSTER;

/**
 * \brief The top level control object on each machine
 * This is a per machine singleton class i.e. exactly one instance of pmController exists per machine.
 * The instance of this class is created when application initializes the library. It is absolutely
 * necessary for the application to initialize the library on all machines participating in the MPI process.
 * The faulting machines will not be considered by PMLIB for task execution. Once pmController's are created
 * all services are setup and managed by it. The controllers shut down when application finalizes the library.
*/
class pmController : public pmBase
{
	public:
		static pmController* GetController();
		void FinalizeController();
	    
		void ProcessFinalization();
		void ProcessTermination();
    
		void SetLastErrorCode(uint pErrorCode) {mLastErrorCode = pErrorCode;}
		uint GetLastErrorCode() {return mLastErrorCode;}

		/* User API Functions */
		void RegisterCallbacks_Public(const char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle);
		void ReleaseCallbacks_Public(pmCallbackHandle pCallbackHandle);
        void CreateMemory_Public(size_t pLength, pmMemHandle* pMem);
        void CreateMemory2D_Public(size_t pRows, size_t pCols, pmMemHandle* pMem);
        void ReleaseMemory_Public(pmMemHandle pMem);
        void FetchMemory_Public(pmMemHandle pMem);
        void FetchMemoryRange_Public(pmMemHandle pMem, size_t pOffset, size_t pLength);
        void GetRawMemPtr_Public(pmMemHandle pMem, void** pPtr);
		void SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle, const std::set<const pmMachine*>& pRestrictToMachinesSet = std::set<const pmMachine*>());
		void ReleaseTask_Public(pmTaskHandle pTaskHandle);
		void WaitForTaskCompletion_Public(pmTaskHandle pTaskHandle);
		void GetTaskExecutionTimeInSecs_Public(pmTaskHandle pTaskHandle, double* pTime);
		void SubscribeToMemory_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo);
		void SubscribeToMemory_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);
        void RedistributeData_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pOffset, size_t pLength, uint pOrder);
		void SetCudaLaunchConf_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf);
        void ReserveCudaGlobalMem_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pSize);
        void* GetScratchBuffer_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize);
        void ReleaseScratchBuffer_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType);
        void* GetLastReductionScratchBuffer_Public(pmTaskHandle pTaskHandle);
        pmRedistributionMetadata* GetRedistributionMetadata_Public(pmTaskHandle pTaskHandle, uint pMemIndex, ulong* pCount);
    
		uint GetHostId_Public();
		uint GetHostCount_Public();
    
        void pmReduceInts_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
        void pmReduceUInts_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
        void pmReduceLongs_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
        void pmReduceULongs_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
        void pmReduceFloats_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
        void pmReduceDoubles_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionType pReductionType);
    
        void* GetMappedFile_Public(const char* pPath);
        void MapFile_Public(const char* pPath);
        void UnmapFile_Public(const char* pPath);
        void MapFiles_Public(const char* const* pPaths, uint pFileCount);
        void UnmapFiles_Public(const char* const* pPaths, uint pFileCount);

	private:
		pmController();
    
		void DestroyController();
	
        uint mLastErrorCode;
		uint mFinalizedHosts;
	    
		finalize_ptr<pmSignalWait> mSignalWait;

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

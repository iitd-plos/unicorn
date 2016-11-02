
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
    
        void pmReduceSubtasks_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, ulong pSubtask1Id, pmSplitInfo* pSplitInfo1, pmDeviceHandle pDevice2Handle, ulong pSubtask2Id, pmSplitInfo* pSplitInfo2, pmReductionOpType pReductionOperation, pmReductionDataType pReductionDataType);
    
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

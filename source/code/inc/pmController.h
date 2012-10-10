
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
		pmStatus FinalizeController();
	    
		pmStatus ProcessFinalization();
		pmStatus ProcessTermination();
    
		pmStatus SetLastErrorCode(uint pErrorCode) {mLastErrorCode = pErrorCode; return pmSuccess;}
		uint GetLastErrorCode() {return mLastErrorCode;}

		/* User API Functions */
		pmStatus RegisterCallbacks_Public(char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle);
		pmStatus ReleaseCallbacks_Public(pmCallbackHandle pCallbackHandle);
		pmStatus CreateMemory_Public(pmMemInfo pMemInfo, size_t pLength, pmMemHandle* pMem);
        pmStatus ReleaseMemory_Public(pmMemHandle pMem);
        pmStatus FetchMemory_Public(pmMemHandle pMem);
        pmStatus GetRawMemPtr_Public(pmMemHandle pMem, void** pPtr);
		pmStatus SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle);
		pmStatus ReleaseTask_Public(pmTaskHandle pTaskHandle);
		pmStatus WaitForTaskCompletion_Public(pmTaskHandle pTaskHandle);
		pmStatus GetTaskExecutionTimeInSecs_Public(pmTaskHandle pTaskHandle, double* pTime);
		pmStatus SubscribeToMemory_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pSubscriptionInfo);
        pmStatus RedistributeData_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, size_t pOffset, size_t pLength, unsigned int pOrder);
		pmStatus SetCudaLaunchConf_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf);
        void* GetScratchBuffer_Public(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, size_t pBufferSize);

		uint GetHostId_Public();
		uint GetHostCount_Public();

	private:
		pmController();
		virtual ~pmController();
    
		pmStatus DestroyController();
	
		static pmStatus CreateAndInitializeController();

		static pmController* mController;
		uint mLastErrorCode;
    
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		uint mFinalizedHosts;
	    
		pmSignalWait* mSignalWait;

        pmLogger mLogger;
        pmDispatcherGPU mDispatcherGPU;
        pmStubManager mStubManager;
        NETWORK_IMPLEMENTATION_CLASS mNetwork;
        pmCommunicator mCommunicator;
        pmMachinePool mMachinePool;
        MEMORY_MANAGER_IMPLEMENTATION_CLASS mMemoryManager;
        pmTaskManager mTaskManager;
        pmScheduler mScheduler;
        pmTimedEventManager mTimedEventManager;
};

} // end namespace pm

#endif

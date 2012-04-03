
#ifndef __PM_CONTROLLER__
#define __PM_CONTROLLER__

#include "pmBase.h"
#include "pmResourceLock.h"

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
		pmStatus SubmitTask_Public(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle);
		pmStatus ReleaseTask_Public(pmTaskHandle pTaskHandle);
		pmStatus WaitForTaskCompletion_Public(pmTaskHandle pTaskHandle);
		pmStatus GetTaskExecutionTimeInSecs_Public(pmTaskHandle pTaskHandle, double* pTime);
		pmStatus SubscribeToMemory_Public(pmTaskHandle pTaskHandle, ulong pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pScatterGatherInfo);
		pmStatus SetCudaLaunchConf_Public(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf);

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
};

} // end namespace pm

#endif


#ifndef __PM_PUBLIC_DEFINITIONS__
#define __PM_PUBLIC_DEFINITIONS__

/** 
 *						PARTITIONED MEMORY LIBRARY
 * PMLIB namespace. All PMLIB definitions are present in this namespace.
*/

#define MAX_NAME_STR_LEN 256
#define MAX_DESC_STR_LEN 1024
#define MAX_CB_KEY_LEN 128

#define SUPPORT_CUDA

namespace pm
{
	/** 
	 * This enumeration defines success and all error conditions for the PMLIB Application Programming Interface (API).
	 * Applications can depend upon these status flags to know the outcome of PMLIB functions.
	 * Applications may use pmGetLastError for a brief description of the error.
	*/
	typedef enum pmStatus
	{
		pmSuccess = 0,
		pmStatusUnavailable,
		pmFatalError,
		pmInitializationFailure,
		pmNetworkInitError,
		pmNetworkTerminationError,
		pmInvalidIndex,
		pmInvalidCommand,
		pmThreadingLibraryFailure,
		pmTimerFailure,
		pmMemoryError,
		pmNetworkError,
		pmIgnorableError,
		pmGraphicsCardError,
		pmBeyondComputationalLimits,
		pmUnrecognizedMemory,
		pmInvalidKey,
		pmMaxKeyLengthExceeded,
		pmDataProcessingFailure,
		pmMaxStatusValues
	} pmStatus;


	/** This function returns the PMLIB's version number as a char* in the format MajorVersion_MinorVersion_Update */
	const char* pmGetLibVersion();

	/** This function returns a brief description of the last error (if any) caused by execution of any PMLIB function */
	const char* pmGetLastError();

	/** This function initializes the PMLIB library. It must be the first PMLIB API called on all machines under MPI cluster. */
	pmStatus pmInitialize();

	/** This function marks the termination of use of PMLIB in an application. This must be the application's last call to PMLIB. */
	pmStatus pmFinalize();

	/** This function returns the id of the calling host */
	unsigned int pmGetHostId();

	/** This function returns the total number of hosts */
	unsigned int pmGetHostCount();

	
	/** Some basic type definitions */
	typedef void* pmMemHandle;
	typedef void* pmTaskHandle;
	typedef void* pmCallbackHandle;
	typedef void* pmClusterHandle;

	/** Scatter Gather structure for memory subscription */
	typedef struct pmSubscriptionInfo
	{
		size_t offset;				/* Offset from the start of the memory region */
		size_t length;				/* Number of bytes to be subscribed */
		//size_t blockLength;			/* The length of each block */
		//size_t jumpLength;			/* Number of bytes between start of two consecutive blocks */
		//unsigned long blockCount;	/* Total number of blocks */

		pmSubscriptionInfo();
	} pmSubscriptionInfo;

	/** Some utility typedefs */
	typedef struct pmSubtaskInfo
	{
		unsigned long subtaskId;
		pmMemHandle inputMem;
		pmMemHandle outputMem;
		size_t inputMemLength;
		size_t outputMemLength;
	} pmSubtaskInfo;

	typedef struct pmTaskInfo
	{
		pmTaskHandle taskHandle;
		void* taskConf;
		unsigned int taskConfLength;
		unsigned long taskId;
		unsigned long subtaskCount;
		unsigned short priority;
		unsigned int originatingHost;
	} pmTaskInfo;

	typedef enum pmMemInfo
	{
		INPUT_MEM_READ_ONLY,
		OUTPUT_MEM_WRITE_ONLY,
		OUTPUT_MEM_READ_WRITE
	} pmMemInfo;

	typedef struct pmDataTransferInfo
	{
		pmMemHandle mem;
		size_t memLength;
		size_t* operatedMemLength;	// Mem Length after programmer's compression/encryption
		pmMemInfo memInfo;
		unsigned int srcHost;
		unsigned int destHost;
	} pmDataTransferInfo;

	typedef enum pmDeviceTypes
	{
		CPU = 0,
	#ifdef SUPPORT_CUDA
		GPU_CUDA,
	#endif
		MAX_DEVICE_TYPES
	} pmDeviceTypes;

	typedef struct pmDeviceInfo
	{
		char name[MAX_NAME_STR_LEN];
		char description[MAX_DESC_STR_LEN];
		pmDeviceTypes deviceTypeInfo;
		unsigned int host;
	} pmDeviceInfo;


	/** The following type definitions stand for the callbacks implemented by the user programs.*/
	typedef pmStatus (*pmDataDistributionCallback)(pmTaskInfo pTaskInfo, unsigned long pSubtaskId);
	typedef pmStatus (*pmSubtaskCallback_CPU)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);
	typedef void (*pmSubtaskCallback_GPU_CUDA)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);	// pointer to CUDA kernel
	typedef pmStatus (*pmDataReductionCallback)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtask1Info, pmSubtaskInfo pSubtask2Info);
	typedef pmStatus (*pmDataScatterCallback)(pmTaskInfo pTaskInfo);
	typedef bool     (*pmDeviceSelectionCallback)(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo);
	typedef pmStatus (*pmPreDataTransferCallback)(pmTaskInfo pTaskInfo, pmDataTransferInfo pDataTransferInfo);
	typedef pmStatus (*pmPostDataTransferCallback)(pmTaskInfo pTaskInfo, pmDataTransferInfo pDataTransferInfo);


	/** Unified callback structure */
	typedef struct pmCallbacks
	{
		public:
			pmDataDistributionCallback dataDistribution;
			pmSubtaskCallback_CPU subtask_cpu;
			pmSubtaskCallback_GPU_CUDA subtask_gpu_cuda;
			pmDataReductionCallback dataReduction;
			pmDeviceSelectionCallback deviceSelection;
			pmPreDataTransferCallback preDataTransfer;
			pmPostDataTransferCallback postDataTransfer;

			pmCallbacks();
	} pmCallbacks;

	/** The callback registeration API. The callbacks must be registered on all machines using the same key.
	 *	The registered callbacks are returned in the pointer pCallbackHandle (if registeration is successful).
	 */
	pmStatus pmRegisterCallbacks(char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle);

	/* The registered callbacks must be released by the application using the following API */
	pmStatus pmReleaseCallbacks(pmCallbackHandle* pCallbackHandle);

	
	/** The memory creation API. The allocated memory is returned in the variable pMem. */
	pmStatus pmCreateMemory(pmMemInfo pMemInfo, size_t pLength, pmMemHandle* pMem);

	/* The memory destruction API. The same interface is used for both input and output memory. */
	pmStatus pmReleaseMemory(pmMemHandle* pMem);

	// The following two defines may be used in 3rd argument to pmSubscribeToMemory
	#define INPUT_MEM 1
	#define OUTPUT_MEM 0

	/** The memory subscription API. It establishes memory dependencies for a subtask.
	 *	Any subtask is also allowed to subscribe on behalf any other subtask.
	 *	This information must be provided as early as possible by the application to 
	 *	allow effective prefetching of the subscribed memory regions.
	 */
	pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pSubscriptionInfo);

	
	/** The task details structure used for task submission */
	typedef struct pmTaskDetails
	{
		void* taskConf;
		unsigned int taskConfLength;
		pmMemHandle* inputMem;
		pmMemHandle* outputMem;
		pmCallbackHandle* callbackHandle;
		unsigned long subtaskCount;
		unsigned long taskId;		/* Meant for application to assign and identify tasks */
		unsigned short priority;	/* By default, this is set to max priority level (0) */
		pmClusterHandle* cluster;	/* Unused */

		pmTaskDetails();
	} pmTaskDetails;

	/** The task submission API. Returns the task handle in variable pTaskHandle on success. */
	pmStatus pmSubmitTask(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle);

	/** The submitted tasks must be released by the application using the following API.
	 *	The API automatically blocks till task completion. Returns the task's exit status.
	 */
	pmStatus pmReleaseTask(pmTaskHandle* pTaskHandle);

	/** A task is by default non-blocking. The control comes back immediately.
	 *	Use the following API to wait for the task to finish.
	 *	The API returns the exit status of the task.
	 */
	pmStatus pmWaitForTaskCompletion(pmTaskHandle* pTaskHandle);

	/** Returns the task execution time (in seconds) in the variable pTime.
	 *	The API automatically blocks till task completion.
	 */
	pmStatus pmGetTaskExecutionTimeInSecs(pmTaskHandle* pTaskHandle, double* pTime);

	/** A unified API to release task and it's associated resources (callbacks, input and output memory).
	 *	Returns the task's exit status (if no error). The API automatically blocks till task finishes.
	 */
	pmStatus pmReleaseTaskAndResources(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle);

} // end namespace pm

#endif
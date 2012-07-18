
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

#ifndef __PM_PUBLIC_DEFINITIONS__
#define __PM_PUBLIC_DEFINITIONS__

/** 
 *						PARTITIONED MEMORY LIBRARY
 * PMLIB namespace. All PMLIB definitions are present in this namespace.
*/

#include <stdlib.h>

#define MAX_NAME_STR_LEN 256
#define MAX_DESC_STR_LEN 1024
#define MAX_CB_KEY_LEN 128

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
		pmOk,
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
        pmNoCompatibleDevice,
        pmConfFileNotFound,
        pmInvalidOffset,
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
	typedef void* pmRawMemPtr;
	typedef void* pmTaskHandle;
	typedef void* pmCallbackHandle;
	typedef void* pmClusterHandle;

	/** Structure for memory subscription */
	typedef struct pmSubscriptionInfo
	{
		size_t offset;				/* Offset from the start of the memory region */
		size_t length;				/* Number of bytes to be subscribed */

		pmSubscriptionInfo();
	} pmSubscriptionInfo;

	/** Some utility typedefs */
	typedef struct pmSubtaskInfo
	{
		unsigned long subtaskId;
		pmRawMemPtr inputMem;
		pmRawMemPtr outputMem;
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
		OUTPUT_MEM_READ_WRITE,
		INPUT_MEM_READ_ONLY_LAZY,
		OUTPUT_MEM_READ_WRITE_LAZY
	} pmMemInfo;

	typedef struct pmDataTransferInfo
	{
		pmMemHandle memHandle;
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
    
    typedef enum pmSchedulingPolicy
    {
        SLOW_START,
        RANDOM_STEAL,
        EQUAL_STATIC,
        PROPORTIONAL_STATIC
    } pmSchedulingPolicy;


	/** The following type definitions stand for the callbacks implemented by the user programs.*/
	typedef pmStatus (*pmDataDistributionCallback)(pmTaskInfo pTaskInfo, pmRawMemPtr pLazyInputMem, pmRawMemPtr pLazyOutputMem, unsigned long pSubtaskId, pmDeviceTypes pDeviceType);
	typedef pmStatus (*pmSubtaskCallback_CPU)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);
	typedef void (*pmSubtaskCallback_GPU_CUDA)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);	// pointer to CUDA kernel
	typedef pmStatus (*pmDataReductionCallback)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtask1Info, pmSubtaskInfo pSubtask2Info);
	typedef pmStatus (*pmDataRedistributionCallback)(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);
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
			pmDataRedistributionCallback dataRedistribution;
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
	pmStatus pmReleaseCallbacks(pmCallbackHandle pCallbackHandle);

	
	/** The memory creation API. The allocated memory is returned in the variable pMemHandle */
	pmStatus pmCreateMemory(pmMemInfo pMemInfo, size_t pLength, pmMemHandle* pMemHandle);

	/* The memory destruction API. The same interface is used for both input and output memory */
	pmStatus pmReleaseMemory(pmMemHandle pMemHandle);
    
    /** This routine reads the entire distributed memory pointed to by pMem from the entire cluster into the local buffer.
     *  This is a blocking call. 
     */
    pmStatus pmFetchMemory(pmMemHandle pMemHandle);
    
    /** This routine returns the naked memory pointer associated with pMem handle.
     *  This pointer may be used in memcpy and related functions.
     */
    pmStatus pmGetRawMemPtr(pmMemHandle pMemHandle, pmRawMemPtr* pPtr);

	// The following two defines may be used in 3rd argument to pmSubscribeToMemory
	#define INPUT_MEM 1
	#define OUTPUT_MEM 0

	/** The memory subscription API. It establishes memory dependencies for a subtask.
	 *	Any subtask is also allowed to subscribe on behalf any other subtask.
     *  This function can only be called from DataDistribution callback. The effect
     *  of calling this function otherwise is undefined.
     */
	pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, bool pIsInputMemory, pmSubscriptionInfo pSubscriptionInfo);
    
	/** The memory redistribution API. It establishes memory ordering for the
	 *	output section computed by a subtask in the final task memory. Order 0 is assumed
     *  to be the first order. Data for order 1 is placed after all data for order 0.
     *  There is no guaranteed ordering inside an order number if multiple subtasks
     *  produce data for that order. This function can only be called from DataRedistribution
     *  callback. The effect of calling this function otherwise is undefined.
     */
    pmStatus pmRedistributeData(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, size_t pOffset, size_t pLength, unsigned int pOrder);

    /** The CUDA launch configuration structure */
    typedef struct pmCudaLaunchConf
    {
        int blocksX;
        int blocksY;
        int blocksZ;
        int threadsX;
        int threadsY;
        int threadsZ;
        int sharedMem;
        
        pmCudaLaunchConf();
    } pmCudaLaunchConf;

    /** The CUDA launch configuration setting API. It sets kernel launch configuration for the subtask specified by
     *  pSubtaskId. The launch configuration is specified in the structure pCudaLaunchConf.
     *  This function can only be called from DataDistribution callback. The effect
     *  of calling this function otherwise is undefined.
     */
    pmStatus pmSetCudaLaunchConf(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, pmCudaLaunchConf pCudaLaunchConf);
	
	/** The task details structure used for task submission */
	typedef struct pmTaskDetails
	{
		void* taskConf;
		unsigned int taskConfLength;
		pmMemHandle inputMemHandle;
		pmMemHandle outputMemHandle;
		pmCallbackHandle callbackHandle;
		unsigned long subtaskCount;
		unsigned long taskId;		/* Meant for application to assign and identify tasks */
		unsigned short priority;	/* By default, this is set to max priority level (0) */
        pmSchedulingPolicy policy;  /* By default, this is SLOW_START */
		pmClusterHandle cluster;	/* Unused */

		pmTaskDetails();
	} pmTaskDetails;

	/** The task submission API. Returns the task handle in variable pTaskHandle on success. */
	pmStatus pmSubmitTask(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle);

	/** The submitted tasks must be released by the application using the following API.
	 *	The API automatically blocks till task completion. Returns the task's exit status.
	 */
	pmStatus pmReleaseTask(pmTaskHandle pTaskHandle);

	/** A task is by default non-blocking. The control comes back immediately.
	 *	Use the following API to wait for the task to finish.
	 *	The API returns the exit status of the task.
	 */
	pmStatus pmWaitForTaskCompletion(pmTaskHandle pTaskHandle);

	/** Returns the task execution time (in seconds) in the variable pTime.
	 *	The API automatically blocks till task completion.
	 */
	pmStatus pmGetTaskExecutionTimeInSecs(pmTaskHandle pTaskHandle, double* pTime);

	/** A unified API to release task and it's associated resources (callbacks, input and output memory).
	 *	Returns the task's exit status (if no error). The API automatically blocks till task finishes.
	 */
	pmStatus pmReleaseTaskAndResources(pmTaskDetails pTaskDetails, pmTaskHandle pTaskHandle);

    
    /** This function returns a writable buffer accessible to subtask, reduction and data redistribution callbacks. Size parameter
     is only honored for the first invocation of this function for a particular subtask. Successive invocations return the buffer
     allocated at initial request size. This buffer is only used to pass information generated in one callback to other callbacks */
    void* pmGetScratchBuffer(pmTaskHandle pTaskHandle, unsigned long pSubtaskId, size_t pBufferSize);
    
} // end namespace pm

#endif

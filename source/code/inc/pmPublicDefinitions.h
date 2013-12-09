
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
 * ***************  Public include file for PMLIB core  ***************
*/

#include <stdlib.h>

namespace pm
{
	typedef unsigned short int ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

    const size_t MAX_NAME_STR_LEN = 256;
    const size_t MAX_DESC_STR_LEN = 1024;
    const size_t MAX_CB_KEY_LEN = 128;
    const size_t MAX_MEM_SECTIONS_PER_TASK = 3;

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
        pmInvalidCallbacks,
        pmUserError,
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
	uint pmGetHostId();

	/** This function returns the total number of hosts */
	uint pmGetHostCount();
    
	
	/** Some basic type definitions */
	typedef void* pmMemHandle;
	typedef void* pmRawMemPtr;
	typedef void* pmTaskHandle;
    typedef void* pmDeviceHandle;
	typedef void* pmCallbackHandle;
	typedef void* pmClusterHandle;

	typedef enum pmMemType
	{
		READ_ONLY,
		WRITE_ONLY,
		READ_WRITE,
		READ_ONLY_LAZY,
        WRITE_ONLY_LAZY,
		READ_WRITE_LAZY,
        MAX_MEM_TYPE
	} pmMemType;
    
    typedef enum pmSubscriptionType
    {
        READ_SUBSCRIPTION,
        WRITE_SUBSCRIPTION,
        READ_WRITE_SUBSCRIPTION
    } pmSubscriptionType;
    
    typedef enum pmSubscriptionVisibilityType
    {
        SUBSCRIPTION_NATURAL,           // Disjoint subscriptions are mapped at same distances as in task memory (default option)
        SUBSCRIPTION_COMPACT,           // Disjoint subscriptions are mapped back to back
        SUBSCRIPTION_OPTIMAL,           // Either SUBSCRIPTION_NATURAL or SUBSCRIPTION_COMPACT whichever is optimal for the subtask
        MAX_SUBSCRIPTION_VISBILITY_TYPE
    } pmSubscriptionVisibilityType;

	/** Structures for memory subscription */
	typedef struct pmSubscriptionInfo
	{
		size_t offset;				/* Offset from the start of the memory region */
		size_t length;				/* Number of bytes to be subscribed */

		pmSubscriptionInfo();
        pmSubscriptionInfo(size_t, size_t);
	} pmSubscriptionInfo;
    
    typedef struct pmScatteredSubscriptionInfo
    {
        size_t offset;
        size_t size;
        size_t step;
        size_t count;
        
        pmScatteredSubscriptionInfo();
        pmScatteredSubscriptionInfo(size_t, size_t, size_t, size_t);
    } pmScatteredSubscriptionInfo;
    
    /** GPU context for subtask */
    typedef struct pmGpuContext
    {
        void* scratchBuffer;
        void* reservedGlobalMem;

        pmGpuContext();
    } pmGpuContext;

	/** User information typedefs */
    typedef struct pmSplitInfo
    {
        uint splitId;
        uint splitCount;
        
        pmSplitInfo();
        pmSplitInfo(uint, uint);
    } pmSplitInfo;
    
    typedef struct pmMemInfo
    {
        pmRawMemPtr ptr;
        pmRawMemPtr readPtr;
        pmRawMemPtr writePtr;
        size_t length;
        pmSubscriptionVisibilityType visibilityType;
        
        pmMemInfo();
        pmMemInfo(pmRawMemPtr, pmRawMemPtr, pmRawMemPtr, size_t);
        pmMemInfo(pmRawMemPtr, pmRawMemPtr, pmRawMemPtr, size_t, pmSubscriptionVisibilityType);
    } pmMemInfo;

	typedef struct pmSubtaskInfo
	{
		ulong subtaskId;
        pmMemInfo memInfo[MAX_MEM_SECTIONS_PER_TASK];
        uint memCount;
        pmGpuContext gpuContext;
        pmSplitInfo splitInfo;
        
        pmSubtaskInfo();
        pmSubtaskInfo(ulong, pmMemInfo*, uint);
	} pmSubtaskInfo;

	typedef struct pmTaskInfo
	{
		pmTaskHandle taskHandle;
		void* taskConf;
		uint taskConfLength;
		ulong taskId;
		ulong subtaskCount;
		ushort priority;
		uint originatingHost;
        
        pmTaskInfo();
	} pmTaskInfo;
    
	typedef struct pmDataTransferInfo
	{
		pmMemHandle memHandle;
		size_t memLength;
		size_t* operatedMemLength;	// Mem Length after programmer's compression/encryption
		pmMemType memType;
		uint srcHost;
		uint destHost;
        
        pmDataTransferInfo();
	} pmDataTransferInfo;

	typedef enum pmDeviceType
	{
		CPU = 0,
	#ifdef SUPPORT_CUDA
		GPU_CUDA,
	#endif
		MAX_DEVICE_TYPES
	} pmDeviceType;

	typedef struct pmDeviceInfo
	{
        pmDeviceHandle deviceHandle;
		char name[MAX_NAME_STR_LEN];
		char description[MAX_DESC_STR_LEN];
		pmDeviceType deviceType;
		uint host;
        
        pmDeviceInfo();
	} pmDeviceInfo;
    
    typedef enum pmSchedulingPolicy
    {
        SLOW_START,
        RANDOM_STEAL,
        EQUAL_STATIC,
        PROPORTIONAL_STATIC
    } pmSchedulingPolicy;

    /* The lifetime of scratch buffer for a subtask */
    typedef enum pmScratchBufferType
    {
        PRE_SUBTASK_TO_SUBTASK,         // Scratch buffer lives from data distribution callback to subtask callback
        SUBTASK_TO_POST_SUBTASK,        // Scractch buffer lives from subtask callback to data redistribution/reduction callback
        PRE_SUBTASK_TO_POST_SUBTASK     // Scratch buffer lives from data distribution callback to data redistribution/reduction callback
    } pmScratchBufferType;


	/** The following type definitions stand for the callbacks implemented by the user programs.*/
	typedef pmStatus (*pmDataDistributionCallback)(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
	typedef pmStatus (*pmSubtaskCallback_CPU)(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
	typedef void (*pmSubtaskCallback_GPU_CUDA)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);	// pointer to CUDA kernel
	typedef pmStatus (*pmSubtaskCallback_GPU_Custom)(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
	typedef pmStatus (*pmDataReductionCallback)(pmTaskInfo pTaskInfo, pmDeviceInfo pDevice1Info, pmSubtaskInfo pSubtask1Info, pmDeviceInfo pDevice2Info, pmSubtaskInfo pSubtask2Info);
	typedef pmStatus (*pmDataRedistributionCallback)(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
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
            pmSubtaskCallback_GPU_Custom subtask_gpu_custom;    // Atleast one of subtask_gpu_cuda and subtask_gpu_custom must be NULL
			pmDataReductionCallback dataReduction;
			pmDataRedistributionCallback dataRedistribution;
			pmDeviceSelectionCallback deviceSelection;
			pmPreDataTransferCallback preDataTransfer;
			pmPostDataTransferCallback postDataTransfer;

			pmCallbacks();
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_CUDA);
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_Custom);
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_CUDA, pmDataReductionCallback);
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_Custom, pmDataReductionCallback);
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_CUDA, pmDataRedistributionCallback);
            pmCallbacks(pmDataDistributionCallback, pmSubtaskCallback_CPU, pmSubtaskCallback_GPU_Custom, pmDataRedistributionCallback);
	} pmCallbacks;

	/** The callback registeration API. The callbacks must be registered on all machines using the same key.
	 *	The registered callbacks are returned in the pointer pCallbackHandle (if registeration is successful).
	 */
	pmStatus pmRegisterCallbacks(const char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle);

	/* The registered callbacks must be released by the application using the following API */
	pmStatus pmReleaseCallbacks(pmCallbackHandle pCallbackHandle);
	
	/** The memory creation API. The allocated memory is returned in the variable pMemHandle */
	pmStatus pmCreateMemory(size_t pLength, pmMemHandle* pMemHandle);

	/* The memory destruction API. The same interface is used for both input and output memory */
	pmStatus pmReleaseMemory(pmMemHandle pMemHandle);
    
    /** This routine reads the entire distributed memory pointed to by pMem from the entire cluster into the local buffer.
     *  This is a blocking call. 
     */
    pmStatus pmFetchMemory(pmMemHandle pMemHandle);
    
    /** This routine fetches pLength bytes of distributed memory pointed to by pMem from offset pOffset into the local buffer.
     *  This is a blocking call. 
     */
    pmStatus pmFetchMemoryRange(pmMemHandle pMemHandle, size_t pOffset, size_t pLength);

    /** This routine returns the naked memory pointer associated with pMem handle.
     *  This pointer may be used in memcpy and related functions.
     */
    pmStatus pmGetRawMemPtr(pmMemHandle pMemHandle, pmRawMemPtr* pPtr);

	/** The memory subscription APIs. These establish memory dependencies for a subtask.
	 *	Any subtask is not allowed to subscribe on behalf any other subtask.
     *  These function can only be called from DataDistribution callback. The effect
     *  of calling this function otherwise is undefined. The functions can be mixed in
     *  any order and may be called multiple times for a particular subtask.
     */
	pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo);
	pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);
    
	/** The memory redistribution API. It establishes memory ordering for the
	 *	output section computed by a subtask in the final task memory. Order 0 is assumed
     *  to be the first order. Data for order 1 is placed after all data for order 0.
     *  There is no guaranteed ordering inside an order number if multiple subtasks
     *  produce data for that order. This function can only be called from DataRedistribution
     *  callback. The effect of calling this function otherwise is undefined.
     */
    pmStatus pmRedistributeData(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, size_t pOffset, size_t pLength, uint pOrder);

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
        pmCudaLaunchConf(int, int, int, int, int, int);
        pmCudaLaunchConf(int, int, int, int, int, int, int);
    } pmCudaLaunchConf;

    /** The CUDA launch configuration setting API. It sets kernel launch configuration for the subtask specified by
     *  pSubtaskId. The launch configuration is specified in the structure pCudaLaunchConf.
     *  This function can only be called from DataDistribution callback. The effect
     *  of calling this function otherwise is undefined.
     */
    pmStatus pmSetCudaLaunchConf(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf);
    
    /** If subtask_gpu_custom is set, application may need to allocate a CUDA buffer in the custom callback.
     *  cudaMalloc and like functions are synchronous and they interrupt any possibility of asynchronous launches,
     *  resulting in limited occupancy on the device. By using this function, a subtask can upfront ask the library
     *  to reserve that buffer. This buffer can be accessed in kernels using pSubtaskInfo.gpuContext->reservedGlobalMem
     *  This function can only be called from DataDistribution callback. The effect of calling this function otherwise
     *  is undefined.
     */
    pmStatus pmReserveCudaGlobalMem(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, size_t pSize);
	
    /** The structure that associates tasks to address spaces */
    typedef struct pmTaskMem
    {
        pmMemHandle memHandle;
        pmMemType memType;
        pmSubscriptionVisibilityType subscriptionVisibilityType;    /* By default, this is SUBSCRIPTION_NATURAL */
        bool disjointReadWritesAcrossSubtasks;                      /* By default, this is false. Applies only to RW adress spaces. */
        
        pmTaskMem();
        pmTaskMem(pmMemHandle, pmMemType);
        pmTaskMem(pmMemHandle, pmMemType, pmSubscriptionVisibilityType);
        pmTaskMem(pmMemHandle, pmMemType, pmSubscriptionVisibilityType, bool);
    } pmTaskMem;

	/** The task details structure used for task submission */
	typedef struct pmTaskDetails
	{
		void* taskConf;
		uint taskConfLength;
        pmTaskMem* taskMem;
        uint taskMemCount;
		pmCallbackHandle callbackHandle;
		ulong subtaskCount;
		ulong taskId;                           /* Meant for application to assign and identify tasks */
		ushort priority;                        /* By default, this is set to max priority level (0) */
        pmSchedulingPolicy policy;              /* By default, this is SLOW_START */
        int timeOutInSecs;                      /* By default, this is max possible value in signed int, negative values mean no timeout */
        bool multiAssignEnabled;                /* By default, this is true */
        bool overlapComputeCommunication;       /* By default, this is true */
        bool canSplitCpuSubtasks;               /* By default, this is false */
        bool canSplitGpuSubtasks;               /* By default, this is false */
    #ifdef SUPPORT_CUDA
        bool cudaCacheEnabled;                  /* By default, this is true */
    #endif
		pmClusterHandle cluster;                /* Unused */

		pmTaskDetails();
        pmTaskDetails(void* taskConf, uint taskConfLength, pmTaskMem*, uint taskMemCount, pmCallbackHandle, ulong subtaskCount);
	} pmTaskDetails;
    
    /* The flag disjointReadWritesAcrossSubtasks should be true (for RW output memories) if the read subscriptions of a subtask do not overlap with write subscriptions of any subtasks other than itself. */

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

    
    /** This function returns a writable buffer accessible to data distribution, subtask, data reduction and data redistribution callbacks.
     ScratchBufferType and Size parameters are only honored for the first invocation of this function for a particular subtask. Successive
     invocations return the buffer allocated at initial request size. This buffer is only used to pass information generated in one callback
     to other callbacks */
    #ifdef __CUDACC__
    __host__ __device__
    #endif
    inline void* pmGetScratchBuffer(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize, pmGpuContext* pGpuContext)
    {
    #if defined(__CUDA_ARCH__)
        return (pGpuContext ? pGpuContext->scratchBuffer : NULL);
    #else
        void* pmGetScratchBufferHostFunc(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize);
        return pmGetScratchBufferHostFunc(pTaskHandle, pDeviceHandle, pSubtaskId, pSplitInfo, pScratchBufferType, pBufferSize);
    #endif
    }

} // end namespace pm

#endif


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

#include "pmBase.h"
#include "pmController.h"
#include "pmLogger.h"

#include "pmRangeAccessor.h"

namespace pm
{

/** 
 * This file defines all functions exported to applications.
 * All functions in this file must be wrapped inside try/catch blocks
 * and converted to pmStatus errors while reporting to the application.
 * No exception is ever sent to the applications (for C compatibility)
*/

#define SAFE_GET_CONTROLLER(x) { x = pmController::GetController(); if(!x) throw pmFatalError; }
#define SAFE_EXECUTE_ON_CONTROLLER(controllerFunc, ...) \
{ \
	pmStatus dStatus = pmSuccess; \
	try \
	{ \
		pmController* dController; \
		SAFE_GET_CONTROLLER(dController); \
		dStatus = dController->controllerFunc(__VA_ARGS__); \
	} \
	catch(pmException& dException) \
	{ \
		dStatus = dException.GetStatusCode(); \
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[dStatus]); \
		return dStatus; \
	} \
	return dStatus; \
}


/**
 * Error code to brief error description mappings
 * Error codes are defined in pmPublicDefinitions.h (inside pmStatus enum)
*/
static const char* pmErrorMessages[] =
{
	"No Error",
	"No Error",
	"Execution status unknown or can't be determined.",
	"Fatal error inside library. Can't continue.",
	"Error in PMLIB initialization",
	"Error in network initialization",
	"Error in shutting down network communications",
	"Index out of bounds",
	"PMLIB internal command object decoding failure",
	"Internal failure in threading library",
	"Failure in time measurement",
	"Memory allocation/management failure",
	"Error in network communication",
	"Minor problem. Exceution can continue.",
	"Problem with GPU card or driver software",
	"Computational Limits Exceeded",
	"Memory not allocated through PMLIB or wrong memory access type",
	"Invalid callback key or multiple invalid uses of same key",
	"Key length exceeds the maximum limit",
	"Internal failure in data packing/unpacking",
    "No compatible processing element found in the cluster",
    "Configuration file not found at expected location",
    "Memory offset out of bounds",
    "Custom error from application",
    "One or more callbacks are not valid"
};

const char* pmGetLibVersion()
{
	return (const char*)PMLIB_VERSION;
}

const char* pmGetLastError()
{
	uint lErrorCode = pmSuccess;

	try
	{
		pmController* lController = pmController::GetController();
		if(!lController)
			return pmErrorMessages[pmInitializationFailure];

		lErrorCode = lController->GetLastErrorCode();

		if(lErrorCode >= pmMaxStatusValues)
			return pmErrorMessages[pmSuccess];
	}
	catch(pmException&)
	{}

	return pmErrorMessages[lErrorCode];
}

pmStatus pmInitialize()
{
	pmStatus lStatus = pmSuccess;

	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);		// Initializes the variable lController; If no controller returns error

#if 0
        pm::pmRangeAccessor<int> lAccessor;
        lAccessor.Insert(0, 10, 4);
        lAccessor.Insert(12, 20, 6);
        
        int lVal = 0;
        if(lAccessor.Find(6, 1, lVal))
            std::cout << "Found " << lVal << std::endl;
        else
            std::cout << " Not Found" << std::endl;
#endif
	}
	catch(pmException& e)
	{
		lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
		return lStatus;
	}

	return lStatus;
}

pmStatus pmFinalize()
{
	pmStatus lStatus = pmSuccess;

	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);

		lStatus = lController->FinalizeController();
	}
	catch(pmException& e)
	{
		lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
		return lStatus;
	}

	return lStatus;
}

unsigned int pmGetHostId()
{
	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);

		return lController->GetHostId_Public();
	}
	catch(pmException& e)
	{
		pmStatus lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
	}

	return 0;
}

unsigned int pmGetHostCount()
{
	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);

		return lController->GetHostCount_Public();
	}
	catch(pmException& e)
	{
		pmStatus lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
	}

	return 0;
}

pmGpuContext::pmGpuContext()
    : scratchBuffer(NULL)
{
}
    
pmSubtaskInfo::pmSubtaskInfo()
    : subtaskId((unsigned long)-1)
    , inputMem(NULL)
    , outputMem(NULL)
    , outputMemRead(NULL)
    , outputMemWrite(NULL)
	, inputMemLength(0)
    , outputMemLength(0)
    , outputMemReadLength(0)
    , outputMemWriteLength(0)
    , gpuContext()
{
}
    
pmTaskInfo::pmTaskInfo()
    : taskHandle(NULL)
	, taskConf(NULL)
	, taskConfLength(0)
	, taskId(0)
	, subtaskCount(0)
	, priority(0)
	, originatingHost(0)
{
}

pmDataTransferInfo::pmDataTransferInfo()
    : memHandle(NULL)
	, memLength(0)
	, operatedMemLength(NULL)
	, memInfo(MAX_MEM_INFO)
	, srcHost((unsigned int)-1)
	, destHost((unsigned int)-1)
{
}
    
pmDeviceInfo::pmDeviceInfo()
    : deviceHandle(NULL)
    , deviceType(MAX_DEVICE_TYPES)
    , host((unsigned int)-1)
{
	name[0] = '\0';
	description[0] = '\0';
}

void* pmGetScratchBufferHostFunc(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmScratchBufferInfo pScratchBufferInfo, size_t pBufferSize);
void* pmGetScratchBufferHostFunc(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmScratchBufferInfo pScratchBufferInfo, size_t pBufferSize)
{
    try
    {
		pmController* lController = pmController::GetController();
		if(!lController)
			return NULL;
    
        if(pScratchBufferInfo != PRE_SUBTASK_TO_SUBTASK && pScratchBufferInfo != SUBTASK_TO_POST_SUBTASK && pScratchBufferInfo != PRE_SUBTASK_TO_POST_SUBTASK)
            PMTHROW(pmFatalErrorException());
        
        return lController->GetScratchBuffer_Public(pTaskHandle, pDeviceHandle, pSubtaskId, pScratchBufferInfo, pBufferSize);
    }
    catch(pmException& e)
    {
        pmStatus lStatus = e.GetStatusCode();
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
    }
    
    return NULL;
}

pmCallbacks::pmCallbacks()
    : dataDistribution(NULL)
	, subtask_cpu(NULL)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(NULL)
	, dataReduction(NULL)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
{
}

pmStatus pmRegisterCallbacks(char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle)
{
    if(pCallbacks.subtask_gpu_cuda != NULL && pCallbacks.subtask_gpu_custom != NULL)
        return pmInvalidCallbacks;

	SAFE_EXECUTE_ON_CONTROLLER(RegisterCallbacks_Public, pKey, pCallbacks, pCallbackHandle);
}

pmStatus pmReleaseCallbacks(pmCallbackHandle pCallbackHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(ReleaseCallbacks_Public, pCallbackHandle);
}

pmStatus pmCreateMemory(size_t pLength, pmMemHandle* pMem)
{
	SAFE_EXECUTE_ON_CONTROLLER(CreateMemory_Public, pLength, pMem);
}

pmStatus pmReleaseMemory(pmMemHandle pMem)
{
	SAFE_EXECUTE_ON_CONTROLLER(ReleaseMemory_Public, pMem);
}

pmStatus pmFetchMemory(pmMemHandle pMem)
{
    SAFE_EXECUTE_ON_CONTROLLER(FetchMemory_Public, pMem);
}
    
pmStatus pmFetchMemoryRange(pmMemHandle pMem, size_t pOffset, size_t pLength)
{
    SAFE_EXECUTE_ON_CONTROLLER(FetchMemoryRange_Public, pMem, pOffset, pLength);
}
    
pmStatus pmGetRawMemPtr(pmMemHandle pMem, void** pPtr)
{
    SAFE_EXECUTE_ON_CONTROLLER(GetRawMemPtr_Public, pMem, pPtr);
}
    
pmTaskDetails::pmTaskDetails()
    : taskConf(NULL)
	, taskConfLength(0)
	, inputMemHandle(NULL)
	, outputMemHandle(NULL)
    , inputMemInfo(INPUT_MEM_READ_ONLY_LAZY)
    , outputMemInfo(OUTPUT_MEM_WRITE_ONLY)
	, subtaskCount(0)
	, taskId(0)
	, priority(DEFAULT_PRIORITY_LEVEL)
    , policy(SLOW_START)
    , timeOutInSecs(__MAX(int))
    , multiAssignEnabled(true)
    , autoFetchOutputMem(true)
	, cluster(NULL)
{
}

pmStatus pmSubmitTask(pmTaskDetails pTaskDetails, pmTaskHandle* pTaskHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(SubmitTask_Public, pTaskDetails, pTaskHandle);
}

pmStatus pmReleaseTask(pmTaskHandle pTaskHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(ReleaseTask_Public, pTaskHandle);
}

pmStatus pmWaitForTaskCompletion(pmTaskHandle pTaskHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(WaitForTaskCompletion_Public, pTaskHandle);
}

pmStatus pmGetTaskExecutionTimeInSecs(pmTaskHandle pTaskHandle, double* pTime)
{
	SAFE_EXECUTE_ON_CONTROLLER(GetTaskExecutionTimeInSecs_Public, pTaskHandle, pTime);
}

pmStatus pmReleaseTaskAndResources(pmTaskDetails pTaskDetails, pmTaskHandle pTaskHandle)
{
	pmStatus lStatus1 = pmReleaseTask(pTaskHandle);
	pmStatus lStatus2 = pmReleaseCallbacks(pTaskDetails.callbackHandle);
	pmStatus lStatus3 = pmReleaseMemory(pTaskDetails.inputMemHandle);
	pmStatus lStatus4 = pmReleaseMemory(pTaskDetails.outputMemHandle);

	if(lStatus2 != pmSuccess) return lStatus1;
	if(lStatus3 != pmSuccess) return lStatus2;
	if(lStatus4 != pmSuccess) return lStatus3;
	
	return lStatus1;
}
    
pmCudaLaunchConf::pmCudaLaunchConf()
{
	blocksX = blocksY = blocksZ = threadsX = threadsY = threadsZ = 1;
	sharedMem = 0;
}
    
pmStatus pmSetCudaLaunchConf(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmCudaLaunchConf pCudaLaunchConf)
{
	SAFE_EXECUTE_ON_CONTROLLER(SetCudaLaunchConf_Public, pTaskHandle, pDeviceHandle, pSubtaskId, pCudaLaunchConf);
}

pmSubscriptionInfo::pmSubscriptionInfo()
	: offset(0)
	, length(0)
{
}

pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSubscriptionType pSubscriptionType, pmSubscriptionInfo pSubscriptionInfo)
{
	SAFE_EXECUTE_ON_CONTROLLER(SubscribeToMemory_Public, pTaskHandle, pDeviceHandle, pSubtaskId, pSubscriptionType, pSubscriptionInfo);
}
    
pmStatus pmRedistributeData(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, size_t pOffset, size_t pLength, unsigned int pOrder)
{
    SAFE_EXECUTE_ON_CONTROLLER(RedistributeData_Public, pTaskHandle, pDeviceHandle, pSubtaskId, pOffset, pLength, pOrder);
}
    
pmStatus pmReduceInts(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    SAFE_EXECUTE_ON_CONTROLLER(pmReduceInts_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, pDevice2Handle, pSubtask2Id, pReductionType);
}

pmStatus pmReduceUInts(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    SAFE_EXECUTE_ON_CONTROLLER(pmReduceUInts_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, pDevice2Handle, pSubtask2Id, pReductionType);
}

pmStatus pmReduceLongs(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    SAFE_EXECUTE_ON_CONTROLLER(pmReduceLongs_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, pDevice2Handle, pSubtask2Id, pReductionType);
}

pmStatus pmReduceULongs(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    SAFE_EXECUTE_ON_CONTROLLER(pmReduceULongs_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, pDevice2Handle, pSubtask2Id, pReductionType);
}

pmStatus pmReduceFloats(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType)
{
    SAFE_EXECUTE_ON_CONTROLLER(pmReduceFloats_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, pDevice2Handle, pSubtask2Id, pReductionType);
}

pmStatus pmMapFile(const char* pPath)
{
	SAFE_EXECUTE_ON_CONTROLLER(MapFile_Public, pPath);
}

void* pmGetMappedFile(const char* pPath)
{
	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);

		return lController->GetMappedFile_Public(pPath);
	}
	catch(pmException& e)
	{
		pmStatus lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
	}

	return NULL;
}

pmStatus pmUnmapFile(const char* pPath)
{
	SAFE_EXECUTE_ON_CONTROLLER(UnmapFile_Public, pPath);
}

} // end namespace pm

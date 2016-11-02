
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

#include "pmBase.h"
#include "pmController.h"
#include "pmLogger.h"

#include <limits>

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
		dController->controllerFunc(__VA_ARGS__); \
	} \
	catch(pmException& dException) \
	{ \
		dStatus = dException.GetStatusCode(); \
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[dStatus]); \
		return dStatus; \
	} \
	return pmSuccess; \
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
	"Minor problem. Execution can continue.",
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
	}
	catch(pmException& e)
	{
		lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
		return lStatus;
	}

	return pmSuccess;
}

pmStatus pmFinalize()
{
	pmStatus lStatus = pmSuccess;

	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController);

		lController->FinalizeController();
	}
	catch(pmException& e)
	{
		lStatus = e.GetStatusCode();
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
		return lStatus;
	}

	return pmSuccess;
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
    , reservedGlobalMem(NULL)
{
}
    
pmSubtaskInfo::pmSubtaskInfo()
    : subtaskId(std::numeric_limits<ulong>::max())
    , memCount(0)
{
}

pmSubtaskInfo::pmSubtaskInfo(unsigned long pSubtaskId, pmMemInfo* pMemInfo, unsigned int pMemCount)
    : subtaskId(pSubtaskId)
    , memCount(pMemCount)
{
    for(size_t i = 0; i < pMemCount; ++i)
        memInfo[i] = pMemInfo[i];
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

pmSplitInfo::pmSplitInfo()
    : splitId(0)
    , splitCount(0)
{
}
    
pmSplitInfo::pmSplitInfo(unsigned int pSplitId, unsigned int pSplitCount)
    : splitId(pSplitId)
    , splitCount(pSplitCount)
{
}

pmMemInfo::pmMemInfo()
    : ptr(NULL)
    , readPtr(NULL)
    , writePtr(NULL)
    , length(0)
    , visibilityType(SUBSCRIPTION_NATURAL)
{
}
    
pmMemInfo::pmMemInfo(pmRawMemPtr pPtr, pmRawMemPtr pReadPtr, pmRawMemPtr pWritePtr, size_t pLength)
    : ptr(pPtr)
    , readPtr(pReadPtr)
    , writePtr(pWritePtr)
    , length(pLength)
    , visibilityType(SUBSCRIPTION_NATURAL)
{
}

pmMemInfo::pmMemInfo(pmRawMemPtr pPtr, pmRawMemPtr pReadPtr, pmRawMemPtr pWritePtr, size_t pLength, pmSubscriptionVisibilityType pVisibilityType)
    : ptr(pPtr)
    , readPtr(pReadPtr)
    , writePtr(pWritePtr)
    , length(pLength)
    , visibilityType(pVisibilityType)
{
}

pmDataTransferInfo::pmDataTransferInfo()
    : memHandle(NULL)
	, memLength(0)
	, operatedMemLength(NULL)
	, memType(MAX_MEM_TYPE)
	, srcHost((unsigned int)-1)
	, destHost((unsigned int)-1)
{
}
    
pmDeviceInfo::pmDeviceInfo()
    : deviceHandle(NULL)
    , deviceType(MAX_DEVICE_TYPES)
    , host((unsigned int)-1)
    , deviceIdOnHost((unsigned int)-1)
    , deviceIdInCluster((unsigned int)-1)
{
	name[0] = '\0';
	description[0] = '\0';
}
    
pmRedistributionMetadata::pmRedistributionMetadata()
    : order(0)
    , count(0)
{}

pmRedistributionMetadata::pmRedistributionMetadata(uint pOrder, uint pCount)
    : order(pOrder)
    , count(pCount)
{}

void* pmGetScratchBufferHostFunc(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize);
void* pmGetScratchBufferHostFunc(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize)
{
    try
    {
		pmController* lController = pmController::GetController();
		if(!lController)
			return NULL;
    
        if(pScratchBufferType != PRE_SUBTASK_TO_SUBTASK && pScratchBufferType != SUBTASK_TO_POST_SUBTASK && pScratchBufferType != PRE_SUBTASK_TO_POST_SUBTASK && pScratchBufferType != REDUCTION_TO_REDUCTION)
            PMTHROW(pmFatalErrorException());

        pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);
        return lController->GetScratchBuffer_Public(pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pScratchBufferType, pBufferSize);
    }
    catch(pmException& e)
    {
        pmStatus lStatus = e.GetStatusCode();
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
    }
    
    return NULL;
}
    
pmStatus pmReleaseScratchBuffer(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, ulong pSubtaskId, pmSplitInfo& pSplitInfo, pmScratchBufferType pScratchBufferType)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

    SAFE_EXECUTE_ON_CONTROLLER(ReleaseScratchBuffer_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pScratchBufferType);
}

void* pmGetLastReductionScratchBuffer(pmTaskHandle pTaskHandle)
{
    try
    {
		pmController* lController = pmController::GetController();
		if(!lController)
			return NULL;
    
        return lController->GetLastReductionScratchBuffer_Public(pTaskHandle);
    }
    catch(pmException& e)
    {
        pmStatus lStatus = e.GetStatusCode();
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
    }
    
    return NULL;
}
    
pmRedistributionMetadata* pmGetRedistributionMetadata(pmTaskHandle pTaskHandle, uint pMemIndex, ulong* pCount)
{
    try
    {
		pmController* lController = pmController::GetController();
		if(!lController)
			return NULL;
    
        return lController->GetRedistributionMetadata_Public(pTaskHandle, pMemIndex, pCount);
    }
    catch(pmException& e)
    {
        pmStatus lStatus = e.GetStatusCode();
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::ERROR, pmErrorMessages[lStatus]);
    }
    
    *pCount = 0;
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
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{
}
    
pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_CUDA pGpuCudaCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(pGpuCudaCallback)
    , subtask_gpu_custom(NULL)
	, dataReduction(NULL)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_Custom pGpuCustomCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(pGpuCustomCallback)
	, dataReduction(NULL)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_CUDA pGpuCudaCallback, pmDataReductionCallback pReductionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(pGpuCudaCallback)
    , subtask_gpu_custom(NULL)
	, dataReduction(pReductionCallback)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_Custom pGpuCustomCallback, pmDataReductionCallback pReductionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(pGpuCustomCallback)
	, dataReduction(pReductionCallback)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_CUDA pGpuCudaCallback, pmDataRedistributionCallback pRedistributionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(pGpuCudaCallback)
    , subtask_gpu_custom(NULL)
	, dataReduction(NULL)
	, dataRedistribution(pRedistributionCallback)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, pmSubtaskCallback_CPU pCpuCallback, pmSubtaskCallback_GPU_Custom pGpuCustomCallback, pmDataRedistributionCallback pRedistributionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(pCpuCallback)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(pGpuCustomCallback)
	, dataReduction(NULL)
	, dataRedistribution(pRedistributionCallback)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(NULL)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, const char* pSubtask_opencl)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(NULL)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(NULL)
	, dataReduction(NULL)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(pSubtask_opencl)
{}

pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, const char* pSubtask_opencl, pmDataReductionCallback pReductionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(NULL)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(NULL)
	, dataReduction(pReductionCallback)
	, dataRedistribution(NULL)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(pSubtask_opencl)
{}
    
pmCallbacks::pmCallbacks(pmDataDistributionCallback pDataDistributionCallback, const char* pSubtask_opencl, pmDataRedistributionCallback pRedistributionCallback)
    : dataDistribution(pDataDistributionCallback)
	, subtask_cpu(NULL)
	, subtask_gpu_cuda(NULL)
    , subtask_gpu_custom(NULL)
	, dataReduction(NULL)
	, dataRedistribution(pRedistributionCallback)
	, deviceSelection(NULL)
	, preDataTransfer(NULL)
	, postDataTransfer(NULL)
    , taskCompletionCallback(NULL)
    , subtask_opencl(pSubtask_opencl)
{}


pmStatus pmRegisterCallbacks(const char* pKey, pmCallbacks pCallbacks, pmCallbackHandle* pCallbackHandle)
{
    if(pCallbacks.subtask_gpu_cuda != NULL && pCallbacks.subtask_gpu_custom != NULL)
        return pmInvalidCallbacks;
    
    if(pCallbacks.subtask_opencl && (pCallbacks.subtask_cpu || pCallbacks.subtask_gpu_cuda || pCallbacks.subtask_gpu_custom))
        return pmInvalidCallbacks;

	SAFE_EXECUTE_ON_CONTROLLER(RegisterCallbacks_Public, pKey, pCallbacks, pCallbackHandle);
}

pmStatus pmReleaseCallbacks(pmCallbackHandle pCallbackHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(ReleaseCallbacks_Public, pCallbackHandle);
}

pmStatus pmCreateMemory(size_t pLength, pmMemHandle* pMemHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(CreateMemory_Public, pLength, pMemHandle);
}

pmStatus pmCreateMemory2D(size_t pRows, size_t pCols, pmMemHandle* pMemHandle)
{
	SAFE_EXECUTE_ON_CONTROLLER(CreateMemory2D_Public, pRows, pCols, pMemHandle);
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

pmTaskMem::pmTaskMem()
    : memHandle(NULL)
    , memType(MAX_MEM_TYPE)
    , subscriptionVisibilityType(SUBSCRIPTION_NATURAL)
    , disjointReadWritesAcrossSubtasks(false)
{
}
    
pmTaskMem::pmTaskMem(pmMemHandle pMemHandle, pmMemType pMemType)
    : memHandle(pMemHandle)
    , memType(pMemType)
    , subscriptionVisibilityType(SUBSCRIPTION_NATURAL)
    , disjointReadWritesAcrossSubtasks(false)
{
}

pmTaskMem::pmTaskMem(pmMemHandle pMemHandle, pmMemType pMemType, pmSubscriptionVisibilityType pVisibility)
    : memHandle(pMemHandle)
    , memType(pMemType)
    , subscriptionVisibilityType(pVisibility)
    , disjointReadWritesAcrossSubtasks(false)
{
}

pmTaskMem::pmTaskMem(pmMemHandle pMemHandle, pmMemType pMemType, pmSubscriptionVisibilityType pVisibility, bool pDisjointReadWrites)
    : memHandle(pMemHandle)
    , memType(pMemType)
    , subscriptionVisibilityType(pVisibility)
    , disjointReadWritesAcrossSubtasks(pDisjointReadWrites)
{
}

pmTaskDetails::pmTaskDetails()
    : taskConf(NULL)
	, taskConfLength(0)
	, taskMem(NULL)
    , taskMemCount(0)
    , callbackHandle(NULL)
	, subtaskCount(0)
	, taskId(0)
	, priority(DEFAULT_PRIORITY_LEVEL)
    , policy(RANDOM_STEAL)
    , timeOutInSecs(__MAX(int))
    , multiAssignEnabled(true)
    , overlapComputeCommunication(true)
    , canSplitCpuSubtasks(false)
    , canSplitGpuSubtasks(false)
#ifdef SUPPORT_CUDA
    , cudaCacheEnabled(true)
#endif
    , suppressTaskLogs(false)
    , affinityCriterion(MAXIMIZE_LOCAL_DATA)
	, cluster(NULL)
{}

pmTaskDetails::pmTaskDetails(void* pTaskConf, uint pTaskConfLength, pmTaskMem* pTaskMem, uint pTaskMemCount, pmCallbackHandle pCallbackHandle, ulong pSubtaskCount)
    : taskConf(pTaskConf)
	, taskConfLength(pTaskConfLength)
	, taskMem(pTaskMem)
    , taskMemCount(pTaskMemCount)
    , callbackHandle(pCallbackHandle)
	, subtaskCount(pSubtaskCount)
	, taskId(0)
	, priority(DEFAULT_PRIORITY_LEVEL)
    , policy(RANDOM_STEAL)
    , timeOutInSecs(__MAX(int))
    , multiAssignEnabled(true)
    , overlapComputeCommunication(true)
    , canSplitCpuSubtasks(false)
    , canSplitGpuSubtasks(false)
#ifdef SUPPORT_CUDA
    , cudaCacheEnabled(true)
#endif
    , suppressTaskLogs(false)
    , affinityCriterion(MAXIMIZE_LOCAL_DATA)
	, cluster(NULL)
{}

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
    
    pmStatus lStatus3 = pmSuccess;
    for(uint i = 0; i < pTaskDetails.taskMemCount; ++i)
    {
        pmStatus lStatus4 = pmReleaseMemory(pTaskDetails.taskMem[i].memHandle);
        if(lStatus4 != pmSuccess)
            lStatus3 = lStatus4;
    }

	if(lStatus2 != pmSuccess) return lStatus1;
	if(lStatus3 != pmSuccess) return lStatus2;
	
	return lStatus1;
}
    
pmCudaLaunchConf::pmCudaLaunchConf()
	: blocksX(1)
    , blocksY(1)
    , blocksZ(1)
    , threadsX(1)
    , threadsY(1)
    , threadsZ(1)
	, sharedMem(0)
{}
    
pmCudaLaunchConf::pmCudaLaunchConf(int pBlocksX, int pBlocksY, int pBlocksZ, int pThreadsX, int pThreadsY, int pThreadsZ)
	: blocksX(pBlocksX)
    , blocksY(pBlocksY)
    , blocksZ(pBlocksZ)
    , threadsX(pThreadsX)
    , threadsY(pThreadsY)
    , threadsZ(pThreadsZ)
	, sharedMem(0)
{}

pmCudaLaunchConf::pmCudaLaunchConf(int pBlocksX, int pBlocksY, int pBlocksZ, int pThreadsX, int pThreadsY, int pThreadsZ, int pSharedMem)
	: blocksX(pBlocksX)
    , blocksY(pBlocksY)
    , blocksZ(pBlocksZ)
    , threadsX(pThreadsX)
    , threadsY(pThreadsY)
    , threadsZ(pThreadsZ)
	, sharedMem(pSharedMem)
{}
    
pmStatus pmSetCudaLaunchConf(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

	SAFE_EXECUTE_ON_CONTROLLER(SetCudaLaunchConf_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pCudaLaunchConf);
}
    
pmStatus pmReserveCudaGlobalMem(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, size_t pSize)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

	SAFE_EXECUTE_ON_CONTROLLER(ReserveCudaGlobalMem_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pSize);
}
    
pmSubscriptionInfo::pmSubscriptionInfo()
	: offset(0)
	, length(0)
{
}
    
pmSubscriptionInfo::pmSubscriptionInfo(size_t pOffset, size_t pLength)
    : offset(pOffset)
    , length(pLength)
{
}

pmScatteredSubscriptionInfo::pmScatteredSubscriptionInfo()
    : offset(0)
    , size(0)
    , step(0)
    , count(0)
{}

pmScatteredSubscriptionInfo::pmScatteredSubscriptionInfo(size_t pOffset, size_t pSize, size_t pStep, size_t pCount)
    : offset(pOffset)
    , size(pSize)
    , step(pStep)
    , count(pCount)
{}

template<pmReductionOpType pOperation>
pmDataReductionCallback pmGetSubtaskReductionCallbackInner(pmReductionDataType pDataType)
{
    switch(pDataType)
    {
        case REDUCE_INTS:
            return pmReduceSubtasks<pOperation, REDUCE_INTS>;

        case REDUCE_UNSIGNED_INTS:
            return pmReduceSubtasks<pOperation, REDUCE_UNSIGNED_INTS>;

        case REDUCE_LONGS:
            return pmReduceSubtasks<pOperation, REDUCE_LONGS>;

        case REDUCE_UNSIGNED_LONGS:
            return pmReduceSubtasks<pOperation, REDUCE_UNSIGNED_LONGS>;
            
        case REDUCE_FLOATS:
            return pmReduceSubtasks<pOperation, REDUCE_FLOATS>;

        case REDUCE_DOUBLES:
            return pmReduceSubtasks<pOperation, REDUCE_DOUBLES>;
            
        default:
            return NULL;
    }
};
    
pmDataReductionCallback pmGetSubtaskReductionCallbackImpl(pmReductionOpType pOperation, pmReductionDataType pDataType)
{
    switch(pOperation)
    {
        case REDUCE_ADD:
            return pmGetSubtaskReductionCallbackInner<REDUCE_ADD>(pDataType);
            
        case REDUCE_MIN:
            return pmGetSubtaskReductionCallbackInner<REDUCE_MIN>(pDataType);

        case REDUCE_MAX:
            return pmGetSubtaskReductionCallbackInner<REDUCE_MAX>(pDataType);

        case REDUCE_PRODUCT:
            return pmGetSubtaskReductionCallbackInner<REDUCE_PRODUCT>(pDataType);

        case REDUCE_LOGICAL_AND:
            return pmGetSubtaskReductionCallbackInner<REDUCE_LOGICAL_AND>(pDataType);

        case REDUCE_BITWISE_AND:
            return pmGetSubtaskReductionCallbackInner<REDUCE_BITWISE_AND>(pDataType);

        case REDUCE_LOGICAL_OR:
            return pmGetSubtaskReductionCallbackInner<REDUCE_LOGICAL_OR>(pDataType);

        case REDUCE_BITWISE_OR:
            return pmGetSubtaskReductionCallbackInner<REDUCE_BITWISE_OR>(pDataType);
    
        case REDUCE_LOGICAL_XOR:
            return pmGetSubtaskReductionCallbackInner<REDUCE_LOGICAL_XOR>(pDataType);

        case REDUCE_BITWISE_XOR:
            return pmGetSubtaskReductionCallbackInner<REDUCE_BITWISE_XOR>(pDataType);
            
        default:
            return NULL;
    }

    return NULL;
}

pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

	SAFE_EXECUTE_ON_CONTROLLER(SubscribeToMemory_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pMemIndex, pSubscriptionType, pSubscriptionInfo);
}

pmStatus pmSubscribeToMemory(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

	SAFE_EXECUTE_ON_CONTROLLER(SubscribeToMemory_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pMemIndex, pSubscriptionType, pScatteredSubscriptionInfo);
}
    
pmStatus pmRedistributeData(pmTaskHandle pTaskHandle, pmDeviceHandle pDeviceHandle, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, uint pMemIndex, size_t pOffset, size_t pLength, unsigned int pOrder)
{
    pmSplitInfo* lSplitInfo = ((pSplitInfo.splitCount == 0) ? NULL : &pSplitInfo);

    SAFE_EXECUTE_ON_CONTROLLER(RedistributeData_Public, pTaskHandle, pDeviceHandle, pSubtaskId, lSplitInfo, pMemIndex, pOffset, pLength, pOrder);
}
    
pmStatus pmReduceSubtasks(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmSplitInfo& pSplitInfo1, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmSplitInfo& pSplitInfo2, pmReductionOpType pOperation, pmReductionDataType pDataType)
{
    pmSplitInfo* lSplitInfo1 = ((pSplitInfo1.splitCount == 0) ? NULL : &pSplitInfo1);
    pmSplitInfo* lSplitInfo2 = ((pSplitInfo2.splitCount == 0) ? NULL : &pSplitInfo2);

    SAFE_EXECUTE_ON_CONTROLLER(pmReduceSubtasks_Public, pTaskHandle, pDevice1Handle, pSubtask1Id, lSplitInfo1, pDevice2Handle, pSubtask2Id, lSplitInfo2, pOperation, pDataType);
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

pmStatus pmMapFile(const char* pPath)
{
	SAFE_EXECUTE_ON_CONTROLLER(MapFile_Public, pPath);
}

pmStatus pmUnmapFile(const char* pPath)
{
	SAFE_EXECUTE_ON_CONTROLLER(UnmapFile_Public, pPath);
}

pmStatus pmMapFiles(const char* const* pPaths, uint pFileCount)
{
	SAFE_EXECUTE_ON_CONTROLLER(MapFiles_Public, pPaths, pFileCount);
}

pmStatus pmUnmapFiles(const char* const* pPaths, uint pFileCount)
{
	SAFE_EXECUTE_ON_CONTROLLER(UnmapFiles_Public, pPaths, pFileCount);
}

} // end namespace pm

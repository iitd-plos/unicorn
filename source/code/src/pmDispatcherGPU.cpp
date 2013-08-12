
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

#include "pmDispatcherGPU.h"
#include "pmExecutionStub.h"
#include "pmHardware.h"
#include "pmMemSection.h"
#include "pmDevicePool.h"
#include "pmTaskManager.h"
#include "pmSubscriptionManager.h"
#include "pmLogger.h"
#include "pmTask.h"
#include "pmUtility.h"

#ifdef MACOS
    #define CUDA_LIBRARY_CUTIL (char*)"libcutil.dylib"
    #define CUDA_LIBRARY_CUDART (char*)"libcudart.dylib"
#else
    #define CUDA_LIBRARY_CUTIL (char*)"libcutil.so"
    #define CUDA_LIBRARY_CUDART (char*)"libcudart.so"
#endif

const int MIN_SUPPORTED_CUDA_DRIVER_VERSION = 4000;

namespace pm
{

/* class pmDispatcherGPU */
pmDispatcherGPU* pmDispatcherGPU::GetDispatcherGPU()
{
    static pmDispatcherGPU lDisptacherGPU;
    return &lDisptacherGPU;
}

pmDispatcherGPU::pmDispatcherGPU()
    : mCountGPU(0)
#ifdef SUPPORT_CUDA
    , mDispatcherCUDA(NULL)
#endif
{
#ifdef SUPPORT_CUDA
	try
	{
		mDispatcherCUDA = new pmDispatcherCUDA();
	}
	catch(pmExceptionGPU& e)
	{
		mDispatcherCUDA = NULL;
        
        if(e.GetFailureId() == pmExceptionGPU::DRIVER_VERSION_UNSUPPORTED)
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Unsupported CUDA driver version");
        else
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be loaded");
	}
#endif
}

pmDispatcherGPU::~pmDispatcherGPU()
{
#ifdef SUPPORT_CUDA
	delete mDispatcherCUDA;
#endif
}

#ifdef SUPPORT_CUDA
pmDispatcherCUDA* pmDispatcherGPU::GetDispatcherCUDA()
{
	return mDispatcherCUDA;
}
#endif

size_t pmDispatcherGPU::GetCountGPU()
{
	return mCountGPU;
}

size_t pmDispatcherGPU::ProbeProcessingElementsAndCreateStubs(std::vector<pmExecutionStub*>& pStubVector)
{
#ifdef SUPPORT_CUDA
	size_t lCountCUDA = 0;
	if(mDispatcherCUDA)
	{
		lCountCUDA = mDispatcherCUDA->GetCountCUDA();
		for(size_t i=0; i<lCountCUDA; ++i)
			pStubVector.push_back(new pmStubCUDA(i, (uint)pStubVector.size()));
	}

	mCountGPU += lCountCUDA;
#endif

	return mCountGPU;
}

#ifdef SUPPORT_CUDA
/* class pmDispatcherCUDA */
pmDispatcherCUDA::pmDispatcherCUDA()
{
    if(GetCudaDriverVersion() < MIN_SUPPORTED_CUDA_DRIVER_VERSION)
        PMTHROW_NODUMP(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::DRIVER_VERSION_UNSUPPORTED));
    
	//mCutilHandle = pmUtility::OpenLibrary(CUDA_LIBRARY_CUTIL);
	mRuntimeHandle = pmUtility::OpenLibrary(CUDA_LIBRARY_CUDART);

	//if(!mCutilHandle || !mRuntimeHandle)
	if(!mRuntimeHandle)
	{
		//CloseLibrary(mCutilHandle);
		pmUtility::CloseLibrary(mRuntimeHandle);
		
		PMTHROW(pmExceptionGPU(pmExceptionGPU::NVIDIA_CUDA, pmExceptionGPU::LIBRARY_OPEN_FAILURE));
	}

	CountAndProbeProcessingElements();
}

pmDispatcherCUDA::~pmDispatcherCUDA()
{
	try
	{
		//pmUtility::CloseLibrary(mCutilHandle);
        pmUtility::CloseLibrary(mRuntimeHandle);
	}
	catch(pmIgnorableException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "One or more CUDA libraries could not be closed properly");
	}
}

size_t pmDispatcherCUDA::GetCountCUDA()
{
	return mCountCUDA;
}
    
void* pmDispatcherCUDA::GetRuntimeHandle()
{
    return mRuntimeHandle;
}

pmStatus pmDispatcherCUDA::InvokeKernel(pmExecutionStub* pStub, pmTaskInfo& pTaskInfo, pmTaskInfo& pTaskInfoCuda, pmSubtaskInfo& pSubtaskInfo, pmCudaLaunchConf& pCudaLaunchConf, bool pOutputMemWriteOnly, pmSubtaskCallback_GPU_CUDA pKernelPtr, pmSubtaskCallback_GPU_Custom pCustomKernelPtr)
{
    void* lDeviceInfoCudaPtr = dynamic_cast<pmStubCUDA*>(pStub)->GetDeviceInfoCudaPtr();

	return InvokeKernel(pStub, pTaskInfo, pTaskInfoCuda, pStub->GetProcessingElement()->GetDeviceInfo(), lDeviceInfoCudaPtr, pSubtaskInfo, pCudaLaunchConf, pOutputMemWriteOnly, pKernelPtr, pCustomKernelPtr, ((pmStubCUDA*)pStub)->mHostToDeviceCommands, ((pmStubCUDA*)pStub)->mDeviceToHostCommands, ((pmStubCUDA*)pStub)->mCudaPointersMap[pSubtaskInfo.subtaskId]);
}
    
void* pmDispatcherCUDA::CheckAndGetScratchBuffer(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId, size_t& pScratchBufferSize, pmScratchBufferInfo& pScratchBufferInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    return lTask->GetSubscriptionManager().CheckAndGetScratchBuffer(pStub, pSubtaskId, pScratchBufferSize, pScratchBufferInfo);
}
    
void pmDispatcherCUDA::GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetInputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pSubscriptionInfo);
}

void pmDispatcherCUDA::GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pReadSubscription, pSubscriptionInfo);
}

void pmDispatcherCUDA::GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubtaskInfo& pSubtaskInfo, pmSubscriptionInfo& pSubscriptionInfo)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    lTask->GetSubscriptionManager().GetUnifiedOutputMemSubscriptionForSubtask(pStub, pSubtaskInfo.subtaskId, pSubscriptionInfo);
}
    
void pmDispatcherCUDA::GetNonConsolidatedSubscriptionsForSubtask(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, pmSubscriptionType pSubscriptionType, pmSubtaskInfo& pSubtaskInfo, std::vector<std::pair<size_t, size_t> >& pSubscriptionVector)
{
    subscription::subscriptionRecordType::const_iterator lBegin, lEnd, lIter;

    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);

    bool lRetVal = true;
    bool lIsInputMem = (pSubscriptionType == INPUT_MEM_READ_SUBSCRIPTION);
    if(lIsInputMem)
        lRetVal = lTask->GetSubscriptionManager().GetNonConsolidatedInputMemSubscriptionsForSubtask(pStub, pSubtaskInfo.subtaskId, lBegin, lEnd);
    else
        lRetVal = lTask->GetSubscriptionManager().GetNonConsolidatedOutputMemSubscriptionsForSubtask(pStub, pSubtaskInfo.subtaskId, (pSubscriptionType == OUTPUT_MEM_READ_SUBSCRIPTION), lBegin, lEnd);
    
    if(lRetVal)
    {
        for(lIter = lBegin; lIter != lEnd; ++lIter)
            pSubscriptionVector.push_back(std::make_pair(lIter->first, lIter->second.first));
    }
}

bool pmDispatcherCUDA::SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub, uint pTaskOriginatingMachineIndex, ulong pTaskSequenceNumber, ulong pSubtaskId1, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType)
{
    pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(pTaskOriginatingMachineIndex);
    pmTask* lTask = pmTaskManager::GetTaskManager()->FindTask(lOriginatingHost, pTaskSequenceNumber);
    
    return lTask->GetSubscriptionManager().SubtasksHaveMatchingSubscriptions(pStub, pSubtaskId1, pStub, pSubtaskId2, pSubscriptionType);
}
    
void pmDispatcherCUDA::MarkInsideUserCode(pmExecutionStub* pStub)
{
    pStub->MarkInsideUserCode();
}

void pmDispatcherCUDA::MarkInsideLibraryCode(pmExecutionStub* pStub)
{
    pStub->MarkInsideLibraryCode();
}
    
bool pmDispatcherCUDA::RequiresPrematureExit(pmExecutionStub* pStub)
{
    return pStub->RequiresPrematureExit();
}

ulong pmDispatcherCUDA::FindCollectivelyExecutableSubtaskRangeEnd(pmExecutionStub* pStub, const pmSubtaskRange& pSubtaskRange, bool pMultiAssign, std::vector<std::vector<std::pair<size_t, size_t> > >& pOffsets, size_t& pTotalMem)
{
    size_t lStatusSize = sizeof(pmStatus);

    pmTask* lTask = pSubtaskRange.task;
    uint lOriginatingHost = (uint)(*(lTask->GetOriginatingHost()));
    ulong lSequenceNumber = lTask->GetSequenceNumber();

    ulong lLastSubtaskId = 0;
    ulong* lLastSubtaskIdPtr = NULL;

    pTotalMem = 0;

#if 0
    pmLastCudaExecutionRecord& lLastRecord = dynamic_cast<pmStubCUDA*>(pStub)->GetLastExecutionRecord();
    if(lLastRecord.valid && lLastRecord.taskOriginatingMachineIndex == lOriginatingHost && lLastRecord.taskSequenceNumber == lSequenceNumber)
    {
        lLastSubtaskIdPtr = &lLastSubtaskId;
        lLastSubtaskId = lLastRecord.lastSubtaskId;
    }
#endif

    size_t lAvailableCudaMem = GetAvailableCudaMem();
    
#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    size_t lAvailablePinnedMem = ((pmStubCUDA*)pStub)->GetPinnedBufferChunk()->GetBiggestAvaialbleContiguousAllocation();
#else
    size_t lAvailablePinnedMem = std::numeric_limits<size_t>::max();
#endif
    
#ifdef DUMP_SUBTASK_COLLECTIONS
    std::cout << "Collection ... ";
#endif
    
    size_t lSubtaskCount = 0;
    size_t lLastInputMemOffset = 0;
    size_t lLastInputMemSize = 0;

    for(ulong lSubtaskId = pSubtaskRange.startSubtask; lSubtaskId <= pSubtaskRange.endSubtask; ++lSubtaskId, ++lSubtaskCount)
    {
        size_t lInitialTotalMem = pTotalMem;

        std::vector<std::pair<size_t, size_t> > lVector;
        bool lOutputMemWriteOnly = false;

        pmSubtaskInfo lSubtaskInfo;
        pStub->FindSubtaskMemDependencies(lTask, lSubtaskId);
        lTask->GetSubtaskInfo(pStub, lSubtaskId, pMultiAssign, lSubtaskInfo, lOutputMemWriteOnly);
        
        size_t lInputMem, lOutputMem, lScratchMem;
        bool lUseLastSubtaskInputMem = false;
        ComputeMemoryRequiredForSubtask(pStub, lSubtaskInfo, lLastSubtaskIdPtr, lOriginatingHost, lSequenceNumber, lInputMem, lOutputMem, lScratchMem, lUseLastSubtaskInputMem);
        
        size_t lReservedMem = lTask->GetSubscriptionManager().GetReservedCudaGlobalMemSize(pStub, lSubtaskId);

        if(lUseLastSubtaskInputMem)
        {
            lVector.push_back(std::make_pair(lLastInputMemOffset, lLastInputMemSize));
        }
        else
        {
            pTotalMem = ComputeAlignedMemoryRequirement(pTotalMem, lInputMem);
            lVector.push_back(std::make_pair(pTotalMem - lInputMem, lInputMem));
            
            lLastInputMemOffset = pTotalMem - lInputMem;
            lLastInputMemSize = lInputMem;
        }

        pTotalMem = ComputeAlignedMemoryRequirement(pTotalMem, lOutputMem);
        lVector.push_back(std::make_pair(pTotalMem - lOutputMem, lOutputMem));
        
        pTotalMem = ComputeAlignedMemoryRequirement(pTotalMem, lScratchMem);
        lVector.push_back(std::make_pair(pTotalMem - lScratchMem, lScratchMem));
        
        pTotalMem = ComputeAlignedMemoryRequirement(pTotalMem, lReservedMem);
        lVector.push_back(std::make_pair(pTotalMem - lReservedMem, lReservedMem));

        pTotalMem = ComputeAlignedMemoryRequirement(pTotalMem, lStatusSize);
        lVector.push_back(std::make_pair(pTotalMem - lStatusSize, lStatusSize));

        if(pTotalMem > lAvailableCudaMem || pTotalMem > lAvailablePinnedMem)
        {
            pTotalMem = lInitialTotalMem;
            break;
        }
        
    #ifdef DUMP_SUBTASK_COLLECTIONS
        std::cout << lSubtaskId << "(" << lInputMem << " " << lOutputMem << " " << lScratchMem << ") ";
    #endif
        
        lLastSubtaskIdPtr = &lLastSubtaskId;
        lLastSubtaskId = lSubtaskId;
        
        pOffsets.push_back(lVector);
    }
    
#ifdef DUMP_SUBTASK_COLLECTIONS
    std::cout << std::endl;
#endif
    
    if(!lSubtaskCount)
        PMTHROW(pmFatalErrorException());
    
    return (pSubtaskRange.startSubtask + lSubtaskCount - 1);
}

void pmDispatcherCUDA::StreamFinishCallback(void* pUserData)
{
    ((pmStubCUDA*)pUserData)->StreamFinishCallback();
}
    
#endif

}


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

#include "pmCommunicator.h"
#include "pmNetwork.h"
#include "pmHeavyOperations.h"
#include "pmTask.h"
#include "pmHardware.h"
#include "pmCallbackUnit.h"
#include "pmAddressSpace.h"
#include "pmMemoryManager.h"

namespace pm
{

using namespace communicator;
    
#define SAFE_GET_NETWORK(x) \
pmNetwork* x = NETWORK_IMPLEMENTATION_CLASS::GetNetwork(); \
if(!x) \
    PMTHROW(pmFatalErrorException());

pmCommunicator::pmCommunicator()
{
}

pmCommunicator* pmCommunicator::GetCommunicator()
{
	static pmCommunicator lCommunicator;
    return &lCommunicator;
}

void pmCommunicator::Send(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);

    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->QueueNetworkRequest(pCommand, heavyOperations::NETWORK_SEND_REQUEST);

	if(pBlocking)
		pCommand->WaitForFinish();
}

// This method is called by heavy operations thread
void pmCommunicator::SendPacked(pmCommunicatorCommandPtr&& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);

    auto lCommand = lNetwork->PackData(pCommand);
	lNetwork->SendNonBlocking(lCommand);

	if(pBlocking)
		lCommand->WaitForFinish();
}

// This method is called by heavy operations thread
void pmCommunicator::SendMemory(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);

	lNetwork->SendMemory(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}

void pmCommunicator::Broadcast(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);
	
	lNetwork->BroadcastNonBlocking(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}

// All receive operations are persistent and initiated at library intialization time, so these are not done on heavy operations thread
void pmCommunicator::Receive(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);
	
	lNetwork->ReceiveNonBlocking(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}

void pmCommunicator::ReceiveMemory(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);
	
	lNetwork->ReceiveMemory(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}

void pmCommunicator::All2All(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);
	
	lNetwork->All2AllNonBlocking(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}


/* struct noReductionReqdStruct */
noReductionReqdStruct::noReductionReqdStruct(pmTask* pTask)
	: originatingHost(*(pTask->GetOriginatingHost()))
    , sequenceNumber(pTask->GetSequenceNumber())
{}

    
/* struct remoteTaskAssignStruct */
remoteTaskAssignStruct::remoteTaskAssignStruct(pmLocalTask* pLocalTask)
    : taskConfLength(pLocalTask->GetTaskConfigurationLength())
    , taskMemCount((uint)pLocalTask->GetAddressSpaceCount())
	, taskId(pLocalTask->GetTaskId())
	, subtaskCount(pLocalTask->GetSubtaskCount())
	, assignedDeviceCount(pLocalTask->GetAssignedDeviceCount())
	, originatingHost(*(pLocalTask->GetOriginatingHost()))
    , sequenceNumber(pLocalTask->GetSequenceNumber())
	, priority(pLocalTask->GetPriority())
	, schedModel((ushort)(pLocalTask->GetSchedulingModel()))
    , affinityCriterion(pLocalTask->GetAffinityCriterion())
    , flags(0)
{
    if(pLocalTask->IsMultiAssignEnabled())
        flags |= TASK_MULTI_ASSIGN_FLAG_VAL;
    
    if(pLocalTask->ShouldOverlapComputeCommunication())
        flags |= TASK_SHOULD_OVERLAP_COMPUTE_COMMUNICATION_FLAG_VAL;
    
    if(pLocalTask->CanForciblyCancelSubtasks())
        flags |= TASK_CAN_FORCIBLY_CANCEL_SUBTASKS_FLAG_VAL;
    
    if(pLocalTask->CanSplitCpuSubtasks())
        flags |= TASK_CAN_SPLIT_CPU_SUBTASKS_FLAG_VAL;

    if(pLocalTask->CanSplitGpuSubtasks())
        flags |= TASK_CAN_SPLIT_GPU_SUBTASKS_FLAG_VAL;

#ifdef SUPPORT_CUDA
    if(pLocalTask->IsCudaCacheEnabled())
        flags |= TASK_HAS_CUDA_CACHE_ENABLED_FLAG_VAL;
#endif
    
    if(pLocalTask->ShouldSuppressTaskLogs())
        flags |= TASK_SUPPRESS_LOGS_FLAG_VAL;

	strncpy(callbackKey, pLocalTask->GetCallbackUnit()->GetKey(), MAX_CB_KEY_LEN-1);
	callbackKey[MAX_CB_KEY_LEN-1] = '\0';
}
    
remoteTaskAssignPacked::remoteTaskAssignPacked(pmLocalTask* pLocalTask)
    : taskStruct(pLocalTask)
{
    if(taskStruct.taskConfLength)
        taskConf.reset(static_cast<char*>(pLocalTask->GetTaskConfiguration()), false);
    
    if(taskStruct.taskMemCount)
    {
        taskMem.reserve(taskStruct.taskMemCount);

        for_each(pLocalTask->GetTaskMemVector(), [&] (const pmTaskMemory& pTaskMemory)
        {
            const pmAddressSpace* lAddressSpace = pTaskMemory.addressSpace;
            taskMem.emplace_back(memoryIdentifierStruct(*(lAddressSpace->GetMemOwnerHost()), lAddressSpace->GetGenerationNumber()), lAddressSpace->GetLength(), pLocalTask->GetMemType(lAddressSpace), pTaskMemory.subscriptionVisibilityType, (ushort)pTaskMemory.disjointReadWritesAcrossSubtasks);
        });
    }

	if(taskStruct.assignedDeviceCount != 0)
	{
		devices.reset(new uint[taskStruct.assignedDeviceCount]);

		const std::vector<const pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();

        std::vector<const pmProcessingElement*>::const_iterator lIter = lDevices.begin(), lEndIter = lDevices.end();
        for(size_t i = 0; lIter != lEndIter; ++lIter, ++i)
            (devices.get_ptr())[i] = (*lIter)->GetGlobalDeviceIndex();
	}
}

/* struct subtaskReducePacked */
subtaskReducePacked::subtaskReducePacked(pmExecutionStub* pReducingStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
    : reduceStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), pSubtaskId, 0, 0, 0, 0)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

    size_t lScratchBuffer1Size = 0;
    char* lScratchBuffer1 = (char*)lSubscriptionManager.CheckAndGetScratchBuffer(pReducingStub, pSubtaskId, pSplitInfo, PRE_SUBTASK_TO_POST_SUBTASK, lScratchBuffer1Size);
    
    if(lScratchBuffer1Size)
    {
        reduceStruct.scratchBuffer3Length = (uint)lScratchBuffer1Size;
        scratchBuffer1.reset(lScratchBuffer1, false);
    }
    
    size_t lScratchBuffer2Size = 0;
    char* lScratchBuffer2 = (char*)lSubscriptionManager.CheckAndGetScratchBuffer(pReducingStub, pSubtaskId, pSplitInfo, SUBTASK_TO_POST_SUBTASK, lScratchBuffer2Size);
    
    if(lScratchBuffer2Size)
    {
        reduceStruct.scratchBuffer2Length = (uint)lScratchBuffer2Size;
        scratchBuffer2.reset(lScratchBuffer2, false);
    }

    size_t lScratchBuffer3Size = 0;
    char* lScratchBuffer3 = (char*)lSubscriptionManager.CheckAndGetScratchBuffer(pReducingStub, pSubtaskId, pSplitInfo, REDUCTION_TO_REDUCTION, lScratchBuffer3Size);
    
    if(lScratchBuffer3Size)
    {
        reduceStruct.scratchBuffer3Length = (uint)lScratchBuffer3Size;
        scratchBuffer3.reset(lScratchBuffer3, false);
    }
    
    filtered_for_each_with_index(pTask->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace) {return (pTask->IsWritable(pAddressSpace) && pTask->IsReducible(pAddressSpace));},
    [&] (const pmAddressSpace* pAddressSpace, size_t pAddressSpaceIndex, size_t pOutputAddressSpaceIndex)
    {
        ++reduceStruct.shadowMemsCount;

        uint lMemIndex = (uint)pAddressSpaceIndex;
        
        void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);
        
        if(pTask->GetAddressSpaceSubscriptionVisibility(pAddressSpace, pReducingStub) == SUBSCRIPTION_NATURAL)
        {
            pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);

        #ifdef SUPPORT_LAZY_MEMORY
            if(pTask->IsLazyWriteOnly(pAddressSpace))
            {
                size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
                const std::map<size_t, size_t>& lMap = lSubscriptionManager.GetWriteOnlyLazyUnprotectedPageRanges(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);
                size_t lRangesSize = lMap.size() * 2 * sizeof(uint);
                size_t lUnprotectedLength = lRangesSize + std::min(lSubscriptionManager.GetWriteOnlyLazyUnprotectedPagesCount(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex) * lPageSize, lUnifiedSubscriptionInfo.length);

                shadowMems.push_back(shadowMemTransferPacked((uint)lMap.size(), (uint)lUnprotectedLength));
                shadowMemTransferPacked& shadowMemTransfer = shadowMems.back();

                uint* lPageRanges = (uint*)shadowMemTransfer.shadowMem.get_ptr();
                char* lMem = (char*)(lPageRanges + lRangesSize);
            
                std::map<size_t, size_t>::const_iterator lIter = lMap.begin(), lEndIter = lMap.end();
                for(; lIter != lEndIter; ++lIter)
                {
                    *lPageRanges++ = (uint)lIter->first;
                    *lPageRanges++ = (uint)lIter->second;
                    
                    uint lMemSize = std::min((uint)(lIter->second * lPageSize), (uint)(lUnifiedSubscriptionInfo.length - lIter->first * lPageSize));
                    PMLIB_MEMCPY(lMem, ((char*)lShadowMem) + (lIter->first * lPageSize), lMemSize);
                    lMem += lMemSize;
                }
            }
            else
        #endif
            {
                subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex, lBeginIter, lEndIter);
                
                shadowMems.push_back(shadowMemTransferPacked(0, 0));
                shadowMemTransferPacked& shadowMemTransfer = shadowMems.back();
                
                if(std::distance(lBeginIter, lEndIter) == 1)    // Only one write subscription
                {
                    shadowMemTransfer.shadowMemData.subtaskMemLength = (uint)lBeginIter->second.first;
                    shadowMemTransfer.shadowMem.reset(reinterpret_cast<char*>(reinterpret_cast<size_t>(lShadowMem) + lBeginIter->first - lUnifiedSubscriptionInfo.offset), false);
                }
                else
                {
                    size_t lTotalWriteSubscriptionLength = 0;
                    for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
                        lTotalWriteSubscriptionLength += lIter->second.first;
                    
                    shadowMemTransfer.shadowMemData.subtaskMemLength = (uint)lTotalWriteSubscriptionLength;
                    shadowMemTransfer.shadowMem.reset(new char[lTotalWriteSubscriptionLength]);
                
                    uint lLocation = 0;
                    for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
                    {
                        void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lShadowMem) + lBeginIter->first - lUnifiedSubscriptionInfo.offset);
                        PMLIB_MEMCPY((char*)(shadowMemTransfer.shadowMem.get_ptr()) + lLocation, lSrcPtr, lBeginIter->second.first);
                    
                        lLocation += (uint)lBeginIter->second.first;
                    }
                }
            }
        }
        else    // SUBSCRIPTION_COMPACT
        {
            const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);

        #ifdef SUPPORT_LAZY_MEMORY
            if(pTask->IsLazyWriteOnly(pAddressSpace))
            {
                DEBUG_EXCEPTION_ASSERT(lCompactViewData.nonConsolidatedReadSubscriptionOffsets.empty());

                size_t lPageSize = MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
                const std::map<size_t, size_t>& lMap = lSubscriptionManager.GetWriteOnlyLazyUnprotectedPageRanges(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);
                size_t lRangesSize = lMap.size() * 2 * sizeof(uint);
                size_t lUnprotectedLength = lRangesSize + std::min(lSubscriptionManager.GetWriteOnlyLazyUnprotectedPagesCount(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex) * lPageSize, lCompactViewData.subscriptionInfo.length);

                shadowMems.push_back(shadowMemTransferPacked((uint)lMap.size(), (uint)lUnprotectedLength));
                shadowMemTransferPacked& shadowMemTransfer = shadowMems.back();

                uint* lPageRanges = (uint*)shadowMemTransfer.shadowMem.get_ptr();
                char* lMem = (char*)(lPageRanges + lRangesSize);
            
                std::map<size_t, size_t>::const_iterator lIter = lMap.begin(), lEndIter = lMap.end();
                for(; lIter != lEndIter; ++lIter)
                {
                    *lPageRanges++ = (uint)lIter->first;
                    *lPageRanges++ = (uint)lIter->second;
                    
                    uint lMemSize = std::min((uint)(lIter->second * lPageSize), (uint)(lCompactViewData.subscriptionInfo.length - lIter->first * lPageSize));
                    PMLIB_MEMCPY(lMem, ((char*)lShadowMem) + (lIter->first * lPageSize), lMemSize);
                    lMem += lMemSize;
                }
            }
            else
        #endif
            {
                subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex, lBeginIter, lEndIter);
                
                shadowMems.push_back(shadowMemTransferPacked(0, 0));
                shadowMemTransferPacked& shadowMemTransfer = shadowMems.back();
                
                auto lCompactWriteIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
                
                if(std::distance(lBeginIter, lEndIter) == 1)    // Only one write subscription
                {
                    shadowMemTransfer.shadowMemData.subtaskMemLength = (uint)lBeginIter->second.first;
                    shadowMemTransfer.shadowMem.reset(reinterpret_cast<char*>(reinterpret_cast<size_t>(lShadowMem) + *lCompactWriteIter), false);
                }
                else
                {
                    size_t lTotalWriteSubscriptionLength = 0;
                    for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter)
                        lTotalWriteSubscriptionLength += lIter->second.first;
                    
                    shadowMemTransfer.shadowMemData.subtaskMemLength = (uint)lTotalWriteSubscriptionLength;
                    shadowMemTransfer.shadowMem.reset(new char[lTotalWriteSubscriptionLength]);
                
                    uint lLocation = 0;
                    for(auto lIter = lBeginIter; lIter != lEndIter; ++lIter, ++lCompactWriteIter)
                    {
                        void* lSrcPtr = reinterpret_cast<void*>(reinterpret_cast<size_t>(lShadowMem) + *lCompactWriteIter);
                        PMLIB_MEMCPY((char*)(shadowMemTransfer.shadowMem.get_ptr()) + lLocation, lSrcPtr, lBeginIter->second.first);
                    
                        lLocation += (uint)lBeginIter->second.first;
                    }
                }
            }
        }
    });
}


/* struct ownershipTransferPacked */
ownershipTransferPacked::ownershipTransferPacked(pmAddressSpace* pAddressSpace, std::shared_ptr<std::vector<ownershipChangeStruct> >& pChangeData)
    : memIdentifier(*pAddressSpace->GetMemOwnerHost(), pAddressSpace->GetGenerationNumber())
    , transferDataElements((uint)(pChangeData->size()))
    , transferData(pChangeData)
{}
 

/* struct sendAcknowledgementPacked */
sendAcknowledgementPacked::sendAcknowledgementPacked(const pmProcessingElement* pSourceDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector)
    : ackStruct(pSourceDevice->GetGlobalDeviceIndex(), *pRange.task->GetOriginatingHost(), pRange.task->GetSequenceNumber(), pRange.startSubtask, pRange.endSubtask, pExecStatus, (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pSourceDevice->GetGlobalDeviceIndex()), (uint)pOwnershipVector.size(), (uint)pAddressSpaceIndexVector.size())
    , ownershipVector(std::move(pOwnershipVector))
    , addressSpaceIndexVector(std::move(pAddressSpaceIndexVector))
{
}

    
/* struct dataRedistributionPacked */
dataRedistributionPacked::dataRedistributionPacked(pmTask* pTask, uint pAddressSpaceIndex, finalize_ptr<std::vector<redistributionOrderStruct>>& pRedistributionAutoPtr)
    : redistributionStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), *PM_LOCAL_MACHINE, pTask->GetSubtasksExecuted(), (uint)pRedistributionAutoPtr->size(), pAddressSpaceIndex)
    , redistributionData(std::move(pRedistributionAutoPtr))
{
}
    

/* struct redistributionOffsetsPacked */
redistributionOffsetsPacked::redistributionOffsetsPacked(pmTask* pTask, uint pAddressSpaceIndex, finalize_ptr<std::vector<ulong>>& pOffsetsDataAutoPtr, pmAddressSpace* pRedistributedAddressSpace)
    : redistributionStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), pRedistributedAddressSpace->GetGenerationNumber(), (uint)pOffsetsDataAutoPtr->size(), pAddressSpaceIndex)
    , offsetsData(std::move(pOffsetsDataAutoPtr))
{
}
    

} // end namespace pm




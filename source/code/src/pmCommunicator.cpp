
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

#include "pmCommunicator.h"
#include "pmNetwork.h"
#include "pmHeavyOperations.h"
#include "pmTask.h"
#include "pmHardware.h"
#include "pmCallbackUnit.h"
#include "pmMemSection.h"
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

	lNetwork->SendNonBlocking(pCommand);

	if(pBlocking)
		pCommand->WaitForFinish();
}

void pmCommunicator::SendPacked(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);

    lNetwork->PackData(pCommand);
	lNetwork->SendNonBlocking(pCommand);

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

void pmCommunicator::Receive(pmCommunicatorCommandPtr& pCommand, bool pBlocking /* = false */)
{
	SAFE_GET_NETWORK(lNetwork);
	
	lNetwork->ReceiveNonBlocking(pCommand);

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

    
/* struct remoteTaskAssignStruct */
remoteTaskAssignStruct::remoteTaskAssignStruct(pmLocalTask* pLocalTask)
    : taskConfLength(pLocalTask->GetTaskConfigurationLength())
    , taskMemCount((uint)pLocalTask->GetMemSectionCount())
	, taskId(pLocalTask->GetTaskId())
	, subtaskCount(pLocalTask->GetSubtaskCount())
	, assignedDeviceCount(pLocalTask->GetAssignedDeviceCount())
	, originatingHost(*(pLocalTask->GetOriginatingHost()))
    , sequenceNumber(pLocalTask->GetSequenceNumber())
	, priority(pLocalTask->GetPriority())
	, schedModel((ushort)(pLocalTask->GetSchedulingModel()))
    , flags(0)
{
    if(pLocalTask->IsMultiAssignEnabled())
        flags |= TASK_MULTI_ASSIGN_FLAG_VAL;
    
    if(pLocalTask->HasDisjointReadWritesAcrossSubtasks())
        flags |= TASK_DISJOINT_READ_WRITES_ACROSS_SUBTASKS_FLAG_VAL;

    if(pLocalTask->ShouldOverlapComputeCommunication())
        flags |= TASK_SHOULD_OVERLAP_COMPUTE_COMMUNICATION_FLAG_VAL;
    
    if(pLocalTask->CanForciblyCancelSubtasks())
        flags |= TASK_CAN_FORCIBLY_CANCEL_SUBTASKS_FLAG_VAL;
    
    if(pLocalTask->CanSplitCpuSubtasks())
        flags |= TASK_CAN_SPLIT_CPU_SUBTASKS_FLAG_VAL;

    if(pLocalTask->CanSplitGpuSubtasks())
        flags |= TASK_CAN_SPLIT_GPU_SUBTASKS_FLAG_VAL;

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

        const std::vector<pmMemSection*>& lMemSectionVector = pLocalTask->GetMemSections();

        std::vector<pmMemSection*>::const_iterator lIter = lMemSectionVector.begin(), lEndIter = lMemSectionVector.end();
        for(; lIter != lEndIter; ++lIter)
        {
            taskMem.push_back(taskMemoryStruct(memoryIdentifierStruct(*((*lIter)->GetMemOwnerHost()), (*lIter)->GetGenerationNumber()), (*lIter)->GetLength(), (ushort)((*lIter)->GetMemType())));
        }
    }

	// Transfer device list if the task scehduling model is pull or if reduction callback is defined
	if(taskStruct.assignedDeviceCount != 0 && (pLocalTask->GetSchedulingModel() == scheduler::PULL || pLocalTask->GetCallbackUnit()->GetDataReductionCB()))
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
    : reduceStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), pSubtaskId, 0)
{
    pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

    filtered_for_each_with_index(pTask->GetMemSections(), [&] (const pmMemSection* pMemSection) {return (pMemSection->IsOutput() && pTask->IsReducible(pMemSection));},
    [&] (const pmMemSection* pMemSection, size_t pMemSectionIndex, size_t pOutputMemSectionIndex)
    {
        uint lMemIndex = (uint)pMemSectionIndex;

        pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);
        
        void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(pReducingStub, pSubtaskId, pSplitInfo, lMemIndex);
        
    #ifdef SUPPORT_LAZY_MEMORY
        if(pMemSection->IsLazyWriteOnly())
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
                memcpy(lMem, ((char*)lShadowMem) + (lIter->first * lPageSize), lMemSize);
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
                    memcpy((char*)(shadowMemTransfer.shadowMem.get_ptr()) + lLocation, lSrcPtr, lBeginIter->second.first);
                
                    lLocation += (uint)lBeginIter->second.first;
                }
            }
        }
    });
}


/* struct ownershipTransferPacked */
    ownershipTransferPacked::ownershipTransferPacked(pmMemSection* pMemSection, std::shared_ptr<std::vector<ownershipChangeStruct> >& pChangeData)
    : memIdentifier(*pMemSection->GetMemOwnerHost(), pMemSection->GetGenerationNumber())
    , transferDataElements((uint)(pChangeData->size()))
    , transferData(pChangeData)
{}
 
    
/* struct memoryReceivePacked */
memoryReceivePacked::memoryReceivePacked(uint pMemOwnerHost, ulong pGenerationNumber, ulong pOffset, ulong pLength, void* pMemPtr, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber)
    : receiveStruct(pMemOwnerHost, pGenerationNumber, pOffset, pLength, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber)
{
    mem.reset(static_cast<char*>(pMemPtr), false);
}

    
/* struct sendAcknowledgementPacked */
sendAcknowledgementPacked::sendAcknowledgementPacked(const pmProcessingElement* pSourceDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pMemSectionIndexVector)
    : ackStruct(pSourceDevice->GetGlobalDeviceIndex(), *pRange.task->GetOriginatingHost(), pRange.task->GetSequenceNumber(), pRange.startSubtask, pRange.endSubtask, pExecStatus, (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pSourceDevice->GetGlobalDeviceIndex()), (uint)pOwnershipVector.size(), (uint)pMemSectionIndexVector.size())
    , ownershipVector(pOwnershipVector)
    , memSectionIndexVector(std::move(pMemSectionIndexVector))
{
}

    
/* struct dataRedistributionPacked */
dataRedistributionPacked::dataRedistributionPacked(pmTask* pTask, uint pMemSectionIndex, finalize_ptr<std::vector<redistributionOrderStruct>>& pRedistributionAutoPtr)
    : redistributionStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), *PM_LOCAL_MACHINE, pTask->GetSubtasksExecuted(), (uint)pRedistributionAutoPtr->size(), pMemSectionIndex)
    , redistributionData(std::move(pRedistributionAutoPtr))
{
}
    

/* struct redistributionOffsetsPacked */
redistributionOffsetsPacked::redistributionOffsetsPacked(pmTask* pTask, uint pMemSectionIndex, finalize_ptr<std::vector<ulong>>& pOffsetsDataAutoPtr, pmMemSection* pRedistributedMemSection)
    : redistributionStruct(*pTask->GetOriginatingHost(), pTask->GetSequenceNumber(), pRedistributedMemSection->GetGenerationNumber(), (uint)pOffsetsDataAutoPtr->size(), pMemSectionIndex)
    , offsetsData(std::move(pOffsetsDataAutoPtr))
{
}
    

} // end namespace pm




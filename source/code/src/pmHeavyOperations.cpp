
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

#include "pmHeavyOperations.h"
#include "pmMemSection.h"
#include "pmDevicePool.h"
#include "pmHardware.h"
#include "pmCommunicator.h"
#include "pmCommand.h"
#include "pmMemoryManager.h"
#include "pmScheduler.h"
#include "pmNetwork.h"
#include "pmLogger.h"
#include "pmUtility.h"
#include "pmTask.h"
#include "pmTaskManager.h"

#include <memory>

namespace pm
{
    
using namespace heavyOperations;
using namespace communicator;

#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_forward(const pmMemSection* memSection, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, memoryIdentifierStruct& newIdentifier, ulong newOffset);
void __dump_mem_transfer(const pmMemSection* memSection, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_forward(const pmMemSection* memSection, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, memoryIdentifierStruct&  newIdentifier, ulong newOffset)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Forwarding input mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    else
        sprintf(lStr, "Forwarding out mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_transfer(const pmMemSection* memSection, memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Transferring input mem section %p (Remote mem (%d, %ld)) from offset %ld (Remote offset %ld) for length %ld to host %d", memSection,identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, host);
    else
        sprintf(lStr, "Transferring out mem section %p (Remote mem (%d, %ld)) from offset %ld (Remote Offset %ld) for length %ld to host %d", memSection, identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_TRANSFER_DUMP(memSection, identifier, receiverOffset, offset, length, host) __dump_mem_transfer(memSection, identifier, receiverOffset, offset, length, host);
#define MEM_FORWARD_DUMP(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset) __dump_mem_forward(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset);
#else
#define MEM_TRANSFER_DUMP(memSection, identifier, receiverOffset, offset, length, host)
#define MEM_FORWARD_DUMP(memSection, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset)
#endif

void HeavyOperationsCommandCompletionCallback(const pmCommandPtr& pCommand)
{
	pmHeavyOperationsThreadPool* lHeavyOperationsThreadPool = pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool();
	lHeavyOperationsThreadPool->CommandCompletionEvent(pCommand);
}

pmHeavyOperationsThreadPool* pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()
{
	static pmHeavyOperationsThreadPool lHeavyOperationsThreadPool(1);   //std::max<size_t>(1, pmStubManager::GetStubManager()->GetProcessingElementsCPU() / 2))
    return &lHeavyOperationsThreadPool;
}

pmHeavyOperationsThreadPool::pmHeavyOperationsThreadPool(size_t pThreadCount)
    : mCurrentThread(0)
    , mResourceLock __LOCK_NAME__("pmHeavyOperationsThreadPool::mResourceLock")
{
    if(pThreadCount == 0)
        PMTHROW(pmFatalErrorException());

    for(size_t i = 0; i < pThreadCount; ++i)
        mThreadVector.push_back(new pmHeavyOperationsThread(i));
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(FILE_OPERATIONS_STRUCT);
	SetupPersistentCommunicationCommands();
}

pmHeavyOperationsThreadPool::~pmHeavyOperationsThreadPool()
{
    size_t lThreadCount = mThreadVector.size();
    for(size_t i = 0; i < lThreadCount; ++i)
        delete mThreadVector[i];

    mThreadVector.clear();
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(FILE_OPERATIONS_STRUCT);
	DestroyPersistentCommunicationCommands();
}

void pmHeavyOperationsThreadPool::SetupPersistentCommunicationCommands()
{
    finalize_ptr<communicator::fileOperationsStruct> lFileOperationsRecvData(new fileOperationsStruct());

#define PERSISTENT_RECV_COMMAND(tag, structType, structEnumType, recvDataPtr) pmCommunicatorCommand<structEnumType>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, tag, NULL, structType, recvDataPtr, 1, HeavyOperationsCommandCompletionCallback)

	mFileOperationsRecvCommand = PERSISTENT_RECV_COMMAND(FILE_OPERATIONS_TAG, FILE_OPERATIONS_STRUCT, fileOperationsStruct, lFileOperationsRecvData);
    
    mFileOperationsRecvCommand->SetPersistent();
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mFileOperationsRecvCommand);

	SetupNewFileOperationsReception();
}
    
void pmHeavyOperationsThreadPool::DestroyPersistentCommunicationCommands()
{
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->TerminatePersistentCommand(mFileOperationsRecvCommand);
}

void pmHeavyOperationsThreadPool::SetupNewFileOperationsReception()
{
	pmCommunicator::GetCommunicator()->Receive(mFileOperationsRecvCommand, false);
}

void pmHeavyOperationsThreadPool::PackAndSendData(const pmCommunicatorCommandPtr& pCommand)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new packEvent(PACK_DATA, pCommand)), pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::UnpackDataEvent(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pPackedLength, ushort pPriority)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new unpackEvent(UNPACK_DATA, std::move(pPackedData), pPackedLength)), pPriority);
}
    
void pmHeavyOperationsThreadPool::MemTransferEvent(memoryIdentifierStruct& pSrcMemIdentifier, memoryIdentifierStruct& pDestMemIdentifier, ulong pOffset, ulong pLength, const pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new memTransferEvent(MEM_TRANSFER, pSrcMemIdentifier, pDestMemIdentifier, pOffset, pLength, pDestMachine, pReceiverOffset, pPriority, pIsForwarded, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber)), pPriority);
}

void pmHeavyOperationsThreadPool::CommandCompletionEvent(pmCommandPtr pCommand)
{
	SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new commandCompletionEvent(COMMAND_COMPLETION, pCommand)), pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::CancelMemoryTransferEvents(pmMemSection* pMemSection)
{
    size_t lPoolSize = mThreadVector.size();

	FINALIZE_PTR_ARRAY(dSignalWaitArray, SIGNAL_WAIT_IMPLEMENTATION_CLASS, new SIGNAL_WAIT_IMPLEMENTATION_CLASS[lPoolSize]);
    
	SubmitToAllThreadsInPool(std::shared_ptr<heavyOperationsEvent>(new memTransferCancelEvent(MEM_TRANSFER_CANCEL, pMemSection, dSignalWaitArray)), MAX_CONTROL_PRIORITY);
    
    for(size_t i = 0; i < lPoolSize; ++i)
        dSignalWaitArray[i].Wait();
}
    
void pmHeavyOperationsThreadPool::CancelTaskSpecificMemoryTransferEvents(pmTask* pTask)
{
	SubmitToAllThreadsInPool(std::shared_ptr<heavyOperationsEvent>(new taskMemTransferCancelEvent(TASK_MEM_TRANSFER_CANCEL, pTask)), MAX_CONTROL_PRIORITY);
}

void pmHeavyOperationsThreadPool::SubmitToThreadPool(const std::shared_ptr<heavyOperationsEvent>& pEvent, ushort pPriority)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mThreadVector[mCurrentThread]->SwitchThread(pEvent, pPriority);
    
    ++mCurrentThread;
    if(mCurrentThread == mThreadVector.size())
        mCurrentThread = 0;
}

void pmHeavyOperationsThreadPool::SubmitToAllThreadsInPool(const std::shared_ptr<heavyOperationsEvent>& pEvent, ushort pPriority) const
{
    std::vector<pmHeavyOperationsThread*>::const_iterator lIter = mThreadVector.begin(), lEndIter = mThreadVector.end();

    for(; lIter != lEndIter; ++lIter)
        (*lIter)->SwitchThread(pEvent, pPriority);
}


/* class pmHeavyOperationsThread */
pmHeavyOperationsThread::pmHeavyOperationsThread(size_t pThreadIndex)
    : mThreadIndex(pThreadIndex)
{
}
    
pmHeavyOperationsThread::~pmHeavyOperationsThread()
{
}

void pmHeavyOperationsThread::ThreadSwitchCallback(std::shared_ptr<heavyOperationsEvent>& pEvent)
{
    try
	{
		ProcessEvent(*pEvent);
	}
	catch(pmException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from heavy operations thread");
	}
}

void pmHeavyOperationsThread::ProcessEvent(heavyOperationsEvent& pEvent)
{
    switch(pEvent.eventId)
    {
        case PACK_DATA:
        {
            packEvent& lEventDetails = static_cast<packEvent&>(pEvent);

			pmCommunicator::GetCommunicator()->SendPacked(std::move(lEventDetails.command), false);

            break;
        }
        
        case UNPACK_DATA:
        {
            unpackEvent& lEventDetails = static_cast<unpackEvent&>(pEvent);
        
            pmCommunicatorCommandPtr lCommand = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnpackData(lEventDetails.packedData.get_ptr(), lEventDetails.packedLength);

            lCommand->MarkExecutionStart();
            NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->ReceiveComplete(lCommand, pmSuccess);

            break;
        }
        
        case MEM_TRANSFER:
        {
            memTransferEvent& lEventDetails = static_cast<memTransferEvent&>(pEvent);

            if(lEventDetails.machine == PM_LOCAL_MACHINE && !lEventDetails.isForwarded)
                PMTHROW(pmFatalErrorException());   // Cyclic reference

            pmMemSection* lSrcMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.srcMemIdentifier.memOwnerHost), lEventDetails.srcMemIdentifier.generationNumber);
            if(!lSrcMemSection)
                return;
            
            pmTask* lRequestingTask = NULL;
            if(lEventDetails.isTaskOriginated)
            {
                const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.taskOriginatingHost);
                lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lEventDetails.taskSequenceNumber);
                
//                if(lOriginatingHost == PM_LOCAL_MACHINE && !lRequestingTask)
//                    return pmSuccess;
            }
            
            // Check if the memory is residing locally or forward the request to the owner machine
            pmMemSection::pmMemOwnership lOwnerships;
            lSrcMemSection->GetOwners(lEventDetails.offset, lEventDetails.length, lOwnerships);
            
            pmMemSection* lDestMemSection = NULL;

            pmMemSection::pmMemOwnership::iterator lStartIter = lOwnerships.begin(), lEndIter = lOwnerships.end(), lIter;
            for(lIter = lStartIter; lIter != lEndIter; ++lIter)
            {
                ulong lInternalOffset = lIter->first;
                ulong lInternalLength = lIter->second.first;
                pmMemSection::vmRangeOwner& lRangeOwner = lIter->second.second;
                
                if(lRangeOwner.host == PM_LOCAL_MACHINE)
                {
                    pmMemSection* lOwnerMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lRangeOwner.memIdentifier.memOwnerHost), lRangeOwner.memIdentifier.generationNumber);
                
                    if(!lOwnerMemSection)
                        PMTHROW(pmFatalErrorException());
                
                #ifdef ENABLE_MEM_PROFILING
                    lSrcMemSection->RecordMemTransfer(lInternalLength);
                #endif
                
                    if(lEventDetails.machine == PM_LOCAL_MACHINE)
                    {
                        lDestMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.destMemIdentifier.memOwnerHost), lEventDetails.destMemIdentifier.generationNumber);
                    
                        if(!lDestMemSection)
                            PMTHROW(pmFatalErrorException());

                        if(!lEventDetails.isTaskOriginated || lRequestingTask)
                            MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lDestMemSection, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset), lRequestingTask);
                    }
                    else
                    {
                        finalize_ptr<memoryReceivePacked> lPackedData(new memoryReceivePacked(lEventDetails.destMemIdentifier.memOwnerHost, lEventDetails.destMemIdentifier.generationNumber, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset), lEventDetails.isTaskOriginated, lEventDetails.taskOriginatingHost, lEventDetails.taskSequenceNumber));
                    
                        MEM_TRANSFER_DUMP(lSrcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)))

                        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryReceivePacked>::CreateSharedPtr(lEventDetails.priority, SEND, MEMORY_RECEIVE_TAG, lEventDetails.machine, MEMORY_RECEIVE_PACKED, lPackedData, 1, pmScheduler::GetScheduler()->GetSchedulerCommandCompletionCallback());

                        pmCommunicator::GetCommunicator()->SendPacked(std::move(lCommand), false);
                    }
                }
                else
                {
                    if(lEventDetails.isForwarded)
                        PMTHROW(pmFatalErrorException());
                    
                    finalize_ptr<memoryTransferRequest> lData(new memoryTransferRequest(memoryIdentifierStruct(lRangeOwner.memIdentifier.memOwnerHost, lRangeOwner.memIdentifier.generationNumber), memoryIdentifierStruct(lEventDetails.destMemIdentifier.memOwnerHost, lEventDetails.destMemIdentifier.generationNumber), lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lRangeOwner.hostOffset, lInternalLength, *lEventDetails.machine, 1, lEventDetails.isTaskOriginated, lEventDetails.taskOriginatingHost, lEventDetails.taskSequenceNumber, lEventDetails.priority));
                    
                    MEM_FORWARD_DUMP(lSrcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)), *lRangeOwner.host, lRangeOwner.memIdentifier, lRangeOwner.hostOffset)

                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryTransferRequest>::CreateSharedPtr(MAX_CONTROL_PRIORITY, SEND, MEMORY_TRANSFER_REQUEST_TAG, lRangeOwner.host, MEMORY_TRANSFER_REQUEST_STRUCT, lData, 1, pmScheduler::GetScheduler()->GetSchedulerCommandCompletionCallback());
                    
                    pmCommunicator::GetCommunicator()->Send(lCommand);
                }
            }

            break;
        }
        
        case COMMAND_COMPLETION:
        {
            commandCompletionEvent& lEventDetails = static_cast<commandCompletionEvent&>(pEvent);

            HandleCommandCompletion(lEventDetails.command);

            break;
        }
            
        case MEM_TRANSFER_CANCEL:
        {
            memTransferCancelEvent& lEventDetails = static_cast<memTransferCancelEvent&>(pEvent);
            
            /* There is no need to actually cancel any mem transfer event becuase even after cancelling the ones in queue,
             another may still come (because of MA). These need to be handled separately anyway. This handling is done in
             MEM_TRANSFER event of this function, where a memory request is only processed if that memory is still alive.
             The only requirement here is that when a pmMemSection is being deleted, it should not be currently being processed.
             This is ensured by issuing a dummy MEM_TRANSFER_CANCEL event. */

            lEventDetails.signalWaitArray[mThreadIndex].Signal();
            
            break;
        }
            
        case TASK_MEM_TRANSFER_CANCEL:
        {
            taskMemTransferCancelEvent& lEventDetails = static_cast<taskMemTransferCancelEvent&>(pEvent);
            DeleteMatchingCommands(lEventDetails.task->GetPriority(), taskMemTransferEventsMatchFunc, lEventDetails.task);

            break;
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
}
    
void pmHeavyOperationsThread::HandleCommandCompletion(pmCommandPtr& pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);

	switch(lCommunicatorCommand->GetType())
	{
        case RECEIVE:
        {
            switch(lCommunicatorCommand->GetTag())
			{
                case FILE_OPERATIONS_TAG:
                {
                    fileOperationsStruct* lData = (fileOperationsStruct*)(lCommunicatorCommand->GetData());
                    switch((fileOperations)(lData->fileOp))
                    {
                        case MMAP_FILE:
                        {
                            pmUtility::MapFile((char*)(lData->fileName));
                            pmUtility::SendFileMappingAcknowledgement((char*)(lData->fileName), pmMachinePool::GetMachinePool()->GetMachine(lData->sourceHost));

                            break;
                        }

                        case MUNMAP_FILE:
                        {
                            pmUtility::UnmapFile((char*)(lData->fileName));
                            pmUtility::SendFileUnmappingAcknowledgement((char*)(lData->fileName), pmMachinePool::GetMachinePool()->GetMachine(lData->sourceHost));
                            
                            break;
                        }
                            
                        case MMAP_ACK:
                        {
                            pmUtility::RegisterFileMappingResponse((char*)(lData->fileName));
                            break;
                        }

                        case MUNMAP_ACK:
                        {
                            pmUtility::RegisterFileUnmappingResponse((char*)(lData->fileName));
                            break;
                        }

                        default:
                            PMTHROW(pmFatalErrorException());
                    }

                    pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->SetupNewFileOperationsReception();
                
                    break;
                }
                
                default:
                    PMTHROW(pmFatalErrorException());
            }

            break;
        }
        
        default:
			PMTHROW(pmFatalErrorException());
    }
}
    
bool taskMemTransferEventsMatchFunc(const heavyOperationsEvent& pEvent, void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case MEM_TRANSFER:
        {
            const memTransferEvent& lEventDetails = static_cast<const memTransferEvent&>(pEvent);

            if(lEventDetails.isTaskOriginated && lEventDetails.taskOriginatingHost == (uint)(*((pmTask*)pCriterion)->GetOriginatingHost()) && lEventDetails.taskSequenceNumber == ((pmTask*)pCriterion)->GetSequenceNumber())
                return true;
        
            break;
        }
        
        default:
            return false;
    }
    
    return false;
}

}




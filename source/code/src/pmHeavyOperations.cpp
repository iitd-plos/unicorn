
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

namespace pm
{
    
using namespace heavyOperations;

#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_forward(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, pmCommunicatorCommand::memoryIdentifierStruct& newIdentifier, ulong newOffset);
void __dump_mem_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_forward(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, pmCommunicatorCommand::memoryIdentifierStruct&  newIdentifier, ulong newOffset)
{
    char lStr[512];
    
    if(memSection->IsInput())
        sprintf(lStr, "Forwarding input mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    else
        sprintf(lStr, "Forwarding out mem section %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", memSection, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_transfer(const pmMemSection* memSection, pmCommunicatorCommand::memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
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

pmStatus HeavyOperationsCommandCompletionCallback(pmCommandPtr pCommand)
{
	pmHeavyOperationsThreadPool* lHeavyOperationsThreadPool = pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool();
	lHeavyOperationsThreadPool->CommandCompletionEvent(pCommand);
    
    return pmSuccess;
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
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::FILE_OPERATIONS_STRUCT);
	SetupPersistentCommunicationCommands();
}

pmHeavyOperationsThreadPool::~pmHeavyOperationsThreadPool()
{
    size_t lThreadCount = mThreadVector.size();
    for(size_t i = 0; i < lThreadCount; ++i)
        delete mThreadVector[i];

    mThreadVector.clear();
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::FILE_OPERATIONS_STRUCT);
	DestroyPersistentCommunicationCommands();
}

void pmHeavyOperationsThreadPool::SetupPersistentCommunicationCommands()
{
#define PERSISTENT_RECV_COMMAND(tag, structType, recvDataPtr) pmPersistentCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::RECEIVE, \
	pmCommunicatorCommand::tag, NULL, pmCommunicatorCommand::structType, recvDataPtr, 1, NULL, 0, HeavyOperationsCommandCompletionCallback)

	mFileOperationsRecvCommand = PERSISTENT_RECV_COMMAND(FILE_OPERATIONS_TAG, FILE_OPERATIONS_STRUCT, &mFileOperationsRecvData);
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mFileOperationsRecvCommand.get());

	SetupNewFileOperationsReception();
}
    
void pmHeavyOperationsThreadPool::DestroyPersistentCommunicationCommands()
{
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->TerminatePersistentCommand(mFileOperationsRecvCommand.get());
}

void pmHeavyOperationsThreadPool::SetupNewFileOperationsReception()
{
	pmCommunicator::GetCommunicator()->Receive(mFileOperationsRecvCommand, false);
}

void pmHeavyOperationsThreadPool::PackAndSendData(pmCommunicatorCommand::communicatorCommandTags pCommandTag, pmCommunicatorCommand::communicatorDataTypes pDataType, pmHardware* pDestination, void* pData, ushort pPriority)
{
    heavyOperationsEvent lEvent;
	lEvent.eventId = PACK_DATA;

    lEvent.packDetails.commandTag = pCommandTag;
    lEvent.packDetails.dataType = pDataType;
    lEvent.packDetails.destination = pDestination;
    lEvent.packDetails.data = pData;
    lEvent.packDetails.priority = pPriority;
    
    SubmitToThreadPool(lEvent, pPriority);
}
    
void pmHeavyOperationsThreadPool::UnpackDataEvent(char* pPackedData, int pPackedLength, ushort pPriority)
{
    heavyOperationsEvent lEvent;
	lEvent.eventId = UNPACK_DATA;

    lEvent.unpackDetails.packedData = pPackedData;
    lEvent.unpackDetails.packedLength = pPackedLength;
    
    SubmitToThreadPool(lEvent, pPriority);
}
    
void pmHeavyOperationsThreadPool::MemTransferEvent(pmCommunicatorCommand::memoryIdentifierStruct& pSrcMemIdentifier, pmCommunicatorCommand::memoryIdentifierStruct& pDestMemIdentifier, ulong pOffset, ulong pLength, pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority)
{
    heavyOperationsEvent lEvent;
	lEvent.eventId = MEM_TRANSFER;

	lEvent.memTransferDetails.srcMemIdentifier = pSrcMemIdentifier;
    lEvent.memTransferDetails.destMemIdentifier = pDestMemIdentifier;
	lEvent.memTransferDetails.offset = pOffset;
	lEvent.memTransferDetails.length = pLength;
	lEvent.memTransferDetails.machine = pDestMachine;
	lEvent.memTransferDetails.receiverOffset = pReceiverOffset;
	lEvent.memTransferDetails.priority = pPriority;
    lEvent.memTransferDetails.isForwarded = pIsForwarded;
    
    SubmitToThreadPool(lEvent, pPriority);
}

void pmHeavyOperationsThreadPool::CommandCompletionEvent(pmCommandPtr pCommand)
{
	heavyOperationsEvent lEvent;
	lEvent.eventId = COMMAND_COMPLETION;
	lEvent.commandCompletionDetails.command = pCommand;

	SubmitToThreadPool(lEvent, pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::CancelMemoryTransferEvents(pmMemSection* pMemSection)
{
    size_t lPoolSize = mThreadVector.size();

	FINALIZE_PTR_ARRAY(dSignalWaitArray, SIGNAL_WAIT_IMPLEMENTATION_CLASS, new SIGNAL_WAIT_IMPLEMENTATION_CLASS[lPoolSize]);

	heavyOperationsEvent lEvent;
	lEvent.eventId = MEM_TRANSFER_CANCEL;
	lEvent.memTransferCancelDetails.memSection = pMemSection;
    lEvent.memTransferCancelDetails.signalWaitArray = dSignalWaitArray;
    
	SubmitToAllThreadsInPool(lEvent, MAX_CONTROL_PRIORITY);
    
    for(size_t i = 0; i < lPoolSize; ++i)
        dSignalWaitArray[i].Wait();
}

void pmHeavyOperationsThreadPool::SubmitToThreadPool(heavyOperationsEvent& pEvent, ushort pPriority)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mThreadVector[mCurrentThread]->SwitchThread(pEvent, pPriority);
    
    ++mCurrentThread;
    if(mCurrentThread == mThreadVector.size())
        mCurrentThread = 0;
}

void pmHeavyOperationsThreadPool::SubmitToAllThreadsInPool(heavyOperationsEvent& pEvent, ushort pPriority)
{
    std::vector<pmHeavyOperationsThread*>::iterator lIter = mThreadVector.begin(), lEndIter = mThreadVector.end();

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

pmStatus pmHeavyOperationsThread::ThreadSwitchCallback(heavyOperationsEvent& pEvent)
{
    try
	{
		return ProcessEvent(pEvent);
	}
	catch(pmException& e)
	{
		pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from heavy operations thread");
	}
    
    return pmSuccess;
}

pmStatus pmHeavyOperationsThread::ProcessEvent(heavyOperationsEvent& pEvent)
{
    switch(pEvent.eventId)
    {
        case PACK_DATA:
        {
            packEvent& lEventDetails = pEvent.packDetails;

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.priority, pmCommunicatorCommand::SEND, lEventDetails.commandTag, lEventDetails.destination, lEventDetails.dataType, lEventDetails.data, 1, NULL, 0, pmScheduler::GetScheduler()->GetUnknownLengthCommandCompletionCallback());

			pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);

            break;
        }
        
        case UNPACK_DATA:
        {
            unpackEvent& lEventDetails = pEvent.unpackDetails;
        
            pmCommunicatorCommandPtr lCommand = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnpackData(lEventDetails.packedData, lEventDetails.packedLength);
            lCommand->MarkExecutionStart();
            NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->ReceiveComplete(lCommand, pmSuccess);

            delete[] lEventDetails.packedData;

            break;
        }
        
        case MEM_TRANSFER:
        {
            memTransferEvent& lEventDetails = pEvent.memTransferDetails;

            if(lEventDetails.machine == PM_LOCAL_MACHINE && !lEventDetails.isForwarded)
                PMTHROW(pmFatalErrorException());   // Cyclic reference
            
            pmCommunicatorCommand::memoryReceivePacked* lPackedData = NULL;
            
            pmMemSection* lSrcMemSection = pmMemSection::FindMemSection(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.srcMemIdentifier.memOwnerHost), lEventDetails.srcMemIdentifier.generationNumber);
            if(!lSrcMemSection)
                return pmSuccess;
            
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
                    
                        MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lDestMemSection, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset));
                    }
                    else
                    {
                        lPackedData = new pmCommunicatorCommand::memoryReceivePacked(lEventDetails.destMemIdentifier.memOwnerHost, lEventDetails.destMemIdentifier.generationNumber, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalLength, (void*)((char*)(lOwnerMemSection->GetMem()) + lInternalOffset));
                    
                        MEM_TRANSFER_DUMP(lSrcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)))

                        pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(lEventDetails.priority, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_RECEIVE_TAG, lEventDetails.machine, pmCommunicatorCommand::MEMORY_RECEIVE_PACKED, lPackedData, 1, NULL, 0, pmScheduler::GetScheduler()->GetUnknownLengthCommandCompletionCallback());

                        pmCommunicator::GetCommunicator()->SendPacked(lCommand, false);
                    }
                }
                else
                {
                    if(lEventDetails.isForwarded)
                        PMTHROW(pmFatalErrorException());
                    
                    pmCommunicatorCommand::memoryTransferRequest* lData = new pmCommunicatorCommand::memoryTransferRequest();
                    lData->sourceMemIdentifier.memOwnerHost = lRangeOwner.memIdentifier.memOwnerHost;
                    lData->sourceMemIdentifier.generationNumber = lRangeOwner.memIdentifier.generationNumber;
                    lData->destMemIdentifier.memOwnerHost = lEventDetails.destMemIdentifier.memOwnerHost;
                    lData->destMemIdentifier.generationNumber = lEventDetails.destMemIdentifier.generationNumber;
                    lData->receiverOffset = lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset;
                    lData->offset = lRangeOwner.hostOffset;
                    lData->length = lInternalLength;
                    lData->destHost = *(lEventDetails.machine);
                    lData->isForwarded = 1;
                    
                    MEM_FORWARD_DUMP(lSrcMemSection, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset + lInternalOffset - lEventDetails.offset, lInternalOffset, lInternalLength, (uint)(*(lEventDetails.machine)), *lRangeOwner.host, lRangeOwner.memIdentifier, lRangeOwner.hostOffset)

                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::SEND, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_TAG, lRangeOwner.host, pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT, (void*)lData, 1, NULL, 0, pmScheduler::GetScheduler()->GetUnknownLengthCommandCompletionCallback());
                    
                    pmCommunicator::GetCommunicator()->Send(lCommand);
                }
            }

            break;
        }
        
        case COMMAND_COMPLETION:
        {
            commandCompletion& lEventDetails = pEvent.commandCompletionDetails;

            HandleCommandCompletion(lEventDetails.command);

            break;
        }
            
        case MEM_TRANSFER_CANCEL:
        {
            memTransferCancelEvent& lEventDetails = pEvent.memTransferCancelDetails;
            
            /* There is no need to actually cancel any mem transfer event becuase even after cancelling the ones in queue,
             another may still come (because of MA). These need to be handled separately anyway. This handling is done in
             MEM_TRANSFER event of this function, where a memory request is only processed if that memory is still alive.
             The only requirement here is that when a pmMemSection is being deleted, it should not be currently being processed.
             This is ensured by issuing a dummy MEM_TRANSFER_CANCEL event. */

            lEventDetails.signalWaitArray[mThreadIndex].Signal();
            
            break;
        }
    }
    
    return pmSuccess;
}
    
void pmHeavyOperationsThread::HandleCommandCompletion(pmCommandPtr pCommand)
{
	pmCommunicatorCommandPtr lCommunicatorCommand = std::tr1::dynamic_pointer_cast<pmCommunicatorCommand>(pCommand);

	switch(lCommunicatorCommand->GetType())
	{
        case pmCommunicatorCommand::RECEIVE:
        {
            switch(lCommunicatorCommand->GetTag())
			{
                case pmCommunicatorCommand::FILE_OPERATIONS_TAG:
                {
                    pmCommunicatorCommand::fileOperationsStruct* lData = (pmCommunicatorCommand::fileOperationsStruct*)(lCommunicatorCommand->GetData());
                    switch((pmCommunicatorCommand::fileOperations)(lData->fileOp))
                    {
                        case pmCommunicatorCommand::MMAP_FILE:
                        {
                            pmUtility::MapFile((char*)(lData->fileName));
                            pmUtility::SendFileMappingAcknowledgement((char*)(lData->fileName), pmMachinePool::GetMachinePool()->GetMachine(lData->sourceHost));

                            break;
                        }

                        case pmCommunicatorCommand::MUNMAP_FILE:
                        {
                            pmUtility::UnmapFile((char*)(lData->fileName));
                            pmUtility::SendFileUnmappingAcknowledgement((char*)(lData->fileName), pmMachinePool::GetMachinePool()->GetMachine(lData->sourceHost));
                            
                            break;
                        }
                            
                        case pmCommunicatorCommand::MMAP_ACK:
                        {
                            pmUtility::RegisterFileMappingResponse((char*)(lData->fileName));
                            break;
                        }

                        case pmCommunicatorCommand::MUNMAP_ACK:
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

}




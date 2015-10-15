
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

#include "pmHeavyOperations.h"
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
void __dump_mem_forward(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, const memoryIdentifierStruct& newIdentifier, ulong newOffset);
void __dump_mem_transfer(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host);
    
void __dump_mem_forward(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host, uint newHost, const memoryIdentifierStruct&  newIdentifier, ulong newOffset)
{
    char lStr[512];
    
    sprintf(lStr, "Forwarding address space %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld to host %d (Dest host %d)", addressSpace, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, newHost, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_transfer(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, uint host)
{
    char lStr[512];
    
    sprintf(lStr, "Transferring address space %p (Remote mem (%d, %ld)) from offset %ld (Remote offset %ld) for length %ld to host %d", addressSpace,identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_TRANSFER_DUMP(addressSpace, identifier, receiverOffset, offset, length, host) __dump_mem_transfer(addressSpace, identifier, receiverOffset, offset, length, host);
#define MEM_FORWARD_DUMP(addressSpace, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset) __dump_mem_forward(addressSpace, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset);
#else
#define MEM_TRANSFER_DUMP(addressSpace, identifier, receiverOffset, offset, length, host)
#define MEM_FORWARD_DUMP(addressSpace, identifier, receiverOffset, offset, length, host, newHost, newIdentifier, newOffset)
#endif

void HeavyOperationsCommandCompletionCallback(const pmCommandPtr& pCommand)
{
	pmHeavyOperationsThreadPool* lHeavyOperationsThreadPool = pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool();
	lHeavyOperationsThreadPool->CommandCompletionEvent(pCommand);
}
    
pmCommandCompletionCallbackType pmHeavyOperationsThreadPool::GetHeavyOperationsCommandCompletionCallback()
{
    return HeavyOperationsCommandCompletionCallback;
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
    EXCEPTION_ASSERT(pThreadCount != 0);

    for(size_t i = 0; i < pThreadCount; ++i)
        mThreadVector.emplace_back(new pmHeavyOperationsThread(i));

	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(MEMORY_IDENTIFIER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(MEMORY_TRANSFER_REQUEST_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(FILE_OPERATIONS_STRUCT);

	SetupPersistentCommunicationCommands();
}

pmHeavyOperationsThreadPool::~pmHeavyOperationsThreadPool()
{
    mThreadVector.clear();

	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MEMORY_IDENTIFIER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MEMORY_TRANSFER_REQUEST_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(FILE_OPERATIONS_STRUCT);

	DestroyPersistentCommunicationCommands();
}

void pmHeavyOperationsThreadPool::SetupPersistentCommunicationCommands()
{
    finalize_ptr<communicator::memoryTransferRequest> lMemTransferRequestData(new memoryTransferRequest());
    finalize_ptr<communicator::fileOperationsStruct> lFileOperationsRecvData(new fileOperationsStruct());

#define PERSISTENT_RECV_COMMAND(tag, structType, structEnumType, recvDataPtr) pmCommunicatorCommand<structEnumType>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, tag, NULL, structType, recvDataPtr, 1, HeavyOperationsCommandCompletionCallback)

	mMemTransferRequestCommand = PERSISTENT_RECV_COMMAND(MEMORY_TRANSFER_REQUEST_TAG, MEMORY_TRANSFER_REQUEST_STRUCT, memoryTransferRequest, lMemTransferRequestData);
	mFileOperationsRecvCommand = PERSISTENT_RECV_COMMAND(FILE_OPERATIONS_TAG, FILE_OPERATIONS_STRUCT, fileOperationsStruct, lFileOperationsRecvData);
    
    mMemTransferRequestCommand->SetPersistent();
    mFileOperationsRecvCommand->SetPersistent();
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mMemTransferRequestCommand);
    lNetwork->InitializePersistentCommand(mFileOperationsRecvCommand);

	pmCommunicator::GetCommunicator()->Receive(mMemTransferRequestCommand, false);
	pmCommunicator::GetCommunicator()->Receive(mFileOperationsRecvCommand, false);
}
    
void pmHeavyOperationsThreadPool::DestroyPersistentCommunicationCommands()
{
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();

    lNetwork->TerminatePersistentCommand(mMemTransferRequestCommand);
    lNetwork->TerminatePersistentCommand(mFileOperationsRecvCommand);
}

void pmHeavyOperationsThreadPool::QueueNetworkRequest(pmCommunicatorCommandPtr& pCommand, heavyOperations::networkRequestType pType)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new networkRequestEvent(NETWORK_REQUEST_EVENT, pCommand, pType)), pCommand->GetPriority());
}

void pmHeavyOperationsThreadPool::PackAndSendData(const pmCommunicatorCommandPtr& pCommand)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new packEvent(PACK_DATA, pCommand)), pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::UnpackDataEvent(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pPackedLength, ushort pPriority)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new unpackEvent(UNPACK_DATA, std::move(pPackedData), pPackedLength)), pPriority);
}
    
void pmHeavyOperationsThreadPool::MemTransferEvent(memoryIdentifierStruct& pSrcMemIdentifier, memoryIdentifierStruct& pDestMemIdentifier, memoryTransferType pTransferType, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new memTransferEvent(MEM_TRANSFER, pSrcMemIdentifier, pDestMemIdentifier, pTransferType, pOffset, pLength, pStep, pCount, pDestMachine, pReceiverOffset, pPriority, pIsForwarded, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber)), pPriority);
}

void pmHeavyOperationsThreadPool::CommandCompletionEvent(pmCommandPtr pCommand)
{
	SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new commandCompletionEvent(COMMAND_COMPLETION, pCommand)), pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::CancelMemoryTransferEvents(pmAddressSpace* pAddressSpace)
{
    size_t lPoolSize = mThreadVector.size();

    std::vector<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitArray(lPoolSize, SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
	SubmitToAllThreadsInPool(std::shared_ptr<heavyOperationsEvent>(new memTransferCancelEvent(MEM_TRANSFER_CANCEL, pAddressSpace, &lSignalWaitArray[0])), MAX_CONTROL_PRIORITY);
    
    for(size_t i = 0; i < lPoolSize; ++i)
        lSignalWaitArray[i].Wait();
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
    for_each(mThreadVector, [&] (const std::unique_ptr<pmHeavyOperationsThread>& pThread)
    {
        pThread->SwitchThread(pEvent, pPriority);
    });
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
        case NETWORK_REQUEST_EVENT:
        {
            networkRequestEvent& lEvent = static_cast<networkRequestEvent&>(pEvent);
            
            switch(lEvent.type)
            {
                case NETWORK_SEND_REQUEST:
                    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->SendNonBlocking(lEvent.command);
                    break;
                    
                case NETWORK_RECEIVE_REQUEST:
                    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->ReceiveNonBlocking(lEvent.command);
                    break;
            }
            
            break;
        }

        case PACK_DATA:
        {
            packEvent& lEventDetails = static_cast<packEvent&>(pEvent);

			pmCommunicator::GetCommunicator()->SendPacked(std::move(lEventDetails.command), false);

            break;
        }
        
        case UNPACK_DATA:
        {
            unpackEvent& lEventDetails = static_cast<unpackEvent&>(pEvent);
        
            pmCommunicatorCommandPtr lCommand = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnpackData(std::move(lEventDetails.packedData), lEventDetails.packedLength);

            lCommand->MarkExecutionStart();

            pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lCommand);
            lCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

            break;
        }
        
        case MEM_TRANSFER:
        {
            memTransferEvent& lEventDetails = static_cast<memTransferEvent&>(pEvent);

            EXCEPTION_ASSERT(lEventDetails.machine != PM_LOCAL_MACHINE || lEventDetails.isForwarded);   // Cyclic reference

            pmAddressSpace* lSrcAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.srcMemIdentifier.memOwnerHost), lEventDetails.srcMemIdentifier.generationNumber);
            if(!lSrcAddressSpace)
                return;
            
            pmTask* lRequestingTask = NULL;
            if(lEventDetails.isTaskOriginated)
            {
                const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.taskOriginatingHost);
                lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lEventDetails.taskSequenceNumber);
                
                // Turn down all requests for tasks that have already finished
                if(lOriginatingHost == PM_LOCAL_MACHINE)
                {
                    if(!lRequestingTask)
                        return;
                }
                else
                {
                    if(pmTaskManager::GetTaskManager()->IsRemoteTaskFinished(*lOriginatingHost, lEventDetails.taskSequenceNumber))
                        return;
                }
            }
            
            if(lEventDetails.transferType == TRANSFER_GENERAL)
                ServeGeneralMemoryRequest(lSrcAddressSpace, lRequestingTask, lEventDetails.machine, lEventDetails.offset, lEventDetails.length, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset, lEventDetails.isTaskOriginated, lEventDetails.taskOriginatingHost, lEventDetails.taskSequenceNumber, lEventDetails.priority, lEventDetails.isForwarded);
            else
                ServeScatteredMemoryRequest(lSrcAddressSpace, lRequestingTask, lEventDetails.machine, lEventDetails.offset, lEventDetails.length, lEventDetails.step, lEventDetails.count, lEventDetails.destMemIdentifier, lEventDetails.receiverOffset, lEventDetails.isTaskOriginated, lEventDetails.taskOriginatingHost, lEventDetails.taskSequenceNumber, lEventDetails.priority, lEventDetails.isForwarded);

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
             The only requirement here is that when a pmAddressSpace is being deleted, it should not be currently being processed.
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
    
void pmHeavyOperationsThread::ServeGeneralMemoryRequest(pmAddressSpace* pSrcAddressSpace, pmTask* pRequestingTask, const pmMachine* pRequestingMachine, ulong pOffset, ulong pLength, const communicator::memoryIdentifierStruct& pDestMemIdentifier, ulong pReceiverOffset, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority, bool pIsForwarded)
{
    // Check if the memory is residing locally or forward the request to the owner machine
    pmAddressSpace::pmMemOwnership lOwnerships;
    pSrcAddressSpace->GetOwners(pOffset, pLength, lOwnerships);
    
    for_each(lOwnerships, [&] (const pmAddressSpace::pmMemOwnership::value_type& pPair)
    {
        ulong lInternalOffset = pPair.first;
        ulong lInternalLength = pPair.second.first;
        const pmAddressSpace::vmRangeOwner& lRangeOwner = pPair.second.second;
        
        if(lRangeOwner.host == PM_LOCAL_MACHINE)
        {
            pmAddressSpace* lOwnerAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lRangeOwner.memIdentifier.memOwnerHost), lRangeOwner.memIdentifier.generationNumber);
        
            EXCEPTION_ASSERT(lOwnerAddressSpace);
            
            if(pRequestingMachine == PM_LOCAL_MACHINE)
            {
                pmAddressSpace* lDestAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(pDestMemIdentifier.memOwnerHost), pDestMemIdentifier.generationNumber);
            
                EXCEPTION_ASSERT(lDestAddressSpace);
                
                std::function<void (char*, ulong)> lFunc([&] (char* pMem, ulong pCopyLength)
                {
                    DEBUG_EXCEPTION_ASSERT(pCopyLength == pLength);
                    memcpy(pMem, (void*)((char*)(lOwnerAddressSpace->GetMem()) + lInternalOffset), pCopyLength);
                });

                if(!pIsTaskOriginated || pRequestingTask)
                    MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->CopyReceivedMemory(lDestAddressSpace, pReceiverOffset + lInternalOffset - pOffset, pLength, lFunc, pRequestingTask);
            }
            else
            {
            #ifdef ENABLE_MEM_PROFILING
                if(!pRequestingTask || !pRequestingTask->ShouldSuppressTaskLogs())
                {
                    pSrcAddressSpace->RecordMemTransfer(lInternalLength);
                    
                    if(pRequestingTask)
                        pRequestingTask->GetTaskExecStats().RecordMemTransferEvent(lInternalLength, false);
                }
            #endif

                // If the memory request is too large than what MPI can handle in a single transport, then break the request into multiple ones
                ulong lMaxTransport = __MAX_SIGNED(int) - MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
                ulong lSteps = (lInternalLength / lMaxTransport) + ((lInternalLength % lMaxTransport) ? 1 : 0);

                for(ulong step = 0; step < lSteps; ++step)
                {
                    ulong lStepLength = ((step == lSteps - 1) ? (lInternalLength - lMaxTransport * step) : lMaxTransport);

                    finalize_ptr<memoryReceiveStruct> lHelperData(new memoryReceiveStruct(pDestMemIdentifier.memOwnerHost, pDestMemIdentifier.generationNumber, pReceiverOffset + lInternalOffset - pOffset, lStepLength, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, std::numeric_limits<int>::max(), pmGetHostId()));

                    MEM_TRANSFER_DUMP(pSrcAddressSpace, pDestMemIdentifier, pReceiverOffset + lInternalOffset - pOffset, lInternalOffset, lStepLength, (uint)(*pRequestingMachine))

                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryReceiveStruct>::CreateSharedPtr(pPriority, SEND, MEMORY_RECEIVE_TAG, pRequestingMachine, MEMORY_RECEIVE_STRUCT, lHelperData, 1, NULL, static_cast<void*>(static_cast<char*>(lOwnerAddressSpace->GetMem()) + lInternalOffset));

                    pmCommunicator::GetCommunicator()->SendMemory(lCommand, false);

                    lInternalOffset += lStepLength;
                }
            }
        }
        else
        {
            DEBUG_EXCEPTION_ASSERT(!pIsForwarded);

            ForwardMemoryRequest(pSrcAddressSpace, lRangeOwner, memoryIdentifierStruct(lRangeOwner.memIdentifier.memOwnerHost, lRangeOwner.memIdentifier.generationNumber), pDestMemIdentifier, TRANSFER_GENERAL, pReceiverOffset + lInternalOffset - pOffset, lRangeOwner.hostOffset, lInternalLength, std::numeric_limits<ulong>::max(), std::numeric_limits<ulong>::max(), pRequestingMachine, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, pPriority);
        }
    });
}

void pmHeavyOperationsThread::ServeScatteredMemoryRequest(pmAddressSpace* pSrcAddressSpace, pmTask* pRequestingTask, const pmMachine* pRequestingMachine, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const communicator::memoryIdentifierStruct& pDestMemIdentifier, ulong pReceiverOffset, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority, bool pIsForwarded)
{
    pmScatteredSubscriptionFilter lBlocksFilter(pmScatteredSubscriptionInfo(pOffset, pLength, pStep, pCount));
    auto lBlocks = lBlocksFilter.FilterBlocks([&] (size_t pRow)
    {
        pmAddressSpace::pmMemOwnership lOwnerships;
        pSrcAddressSpace->GetOwners(pOffset + pRow * pStep, pLength, lOwnerships);
        
        for_each(lOwnerships, [&] (pmAddressSpace::pmMemOwnership::value_type& pPair)
        {
            lBlocksFilter.AddNextSubRow(pPair.first, pPair.second.first, pPair.second.second);
        });
    });

    for_each(lBlocks, [&] (const typename decltype(lBlocks)::value_type& pMapKeyValue)
    {
        for_each(pMapKeyValue.second, [&] (const std::pair<pmScatteredSubscriptionInfo, pmAddressSpace::vmRangeOwner>& pPair)
        {
            const pmScatteredSubscriptionInfo& lScatteredInfo = pPair.first;
            const pmAddressSpace::vmRangeOwner& lRangeOwner = pPair.second;
            ulong lReceiverOffset = pReceiverOffset + lRangeOwner.hostOffset - pOffset;

            EXCEPTION_ASSERT(lScatteredInfo.size && lScatteredInfo.step && lScatteredInfo.count);

            if(lRangeOwner.host == PM_LOCAL_MACHINE)
            {
            #ifdef ENABLE_MEM_PROFILING
                if(!pRequestingTask || !pRequestingTask->ShouldSuppressTaskLogs())
                {
                    pSrcAddressSpace->RecordMemTransfer(lScatteredInfo.size * lScatteredInfo.count);

                    if(pRequestingTask)
                        pRequestingTask->GetTaskExecStats().RecordMemTransferEvent(lScatteredInfo.size * lScatteredInfo.count, true);
                }
            #endif

                pmAddressSpace* lOwnerAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lRangeOwner.memIdentifier.memOwnerHost), lRangeOwner.memIdentifier.generationNumber);
                char* lBeginAddr = (char*)(lOwnerAddressSpace->GetMem());

                // If the memory request is too large than what MPI can handle in a single transport, then break the request into multiple ones
                // To maintain scattered transfers, we keep the lScatteredInfo.size unchanged but adjust lScatteredInfo.count
                ulong lMaxTransport = __MAX_SIGNED(int) - MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()->GetVirtualMemoryPageSize();
                ulong lMaxCounts = lMaxTransport / lScatteredInfo.size;    // How many scattered counts can be transported once?
                
                EXCEPTION_ASSERT(lMaxCounts);   // Atleast one count must be transferrable (otherwise, the request must be broken down into general request --- not implemented yet)
                ulong lSteps = (lScatteredInfo.count / lMaxCounts) + ((lScatteredInfo.count % lMaxCounts) ? 1 : 0);

                ulong lInternalStepOffset = 0;

                for(ulong step = 0; step < lSteps; ++step)
                {
                    ulong lStepCounts = ((step == lSteps - 1) ? (lScatteredInfo.count - lMaxCounts * step) : lMaxCounts);
                    
                    pmScatteredSubscriptionInfo lStepScatteredInfo(lScatteredInfo.offset + lInternalStepOffset, lScatteredInfo.size, lScatteredInfo.step, lStepCounts);

                    finalize_ptr<memoryReceiveStruct> lHelperData(new memoryReceiveStruct(pDestMemIdentifier.memOwnerHost, pDestMemIdentifier.generationNumber, lReceiverOffset + lInternalStepOffset, lStepScatteredInfo.size, lStepScatteredInfo.step, lStepCounts, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, std::numeric_limits<int>::max(), pmGetHostId()));
                
                    MEM_TRANSFER_DUMP(pSrcAddressSpace, pDestMemIdentifier, lReceiverOffset + lInternalStepOffset, lStepScatteredInfo.offset, lStepScatteredInfo.size * lStepScatteredInfo.count, (uint)(*pRequestingMachine))

                    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryReceiveStruct>::CreateSharedPtr(pPriority, SEND, MEMORY_RECEIVE_TAG, pRequestingMachine, MEMORY_RECEIVE_STRUCT, lHelperData, 1, NULL, static_cast<void*>(lBeginAddr + lRangeOwner.hostOffset + lInternalStepOffset));

                    pmCommunicator::GetCommunicator()->SendMemory(lCommand, false);
                    
                    lInternalStepOffset += lStepCounts * lScatteredInfo.step;
                }
            }
            else
            {
                DEBUG_EXCEPTION_ASSERT(!pIsForwarded);

                ForwardMemoryRequest(pSrcAddressSpace, lRangeOwner, lRangeOwner.memIdentifier, pDestMemIdentifier, TRANSFER_SCATTERED, lReceiverOffset, lScatteredInfo.offset, lScatteredInfo.size, lScatteredInfo.step, lScatteredInfo.count, pRequestingMachine, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, pPriority);
            }
        });
    });
}
    
void pmHeavyOperationsThread::ForwardMemoryRequest(pmAddressSpace* pSrcAddressSpace, const pmAddressSpace::vmRangeOwner& pRangeOwner, const memoryIdentifierStruct& pSrcMemIdentifier, const memoryIdentifierStruct& pDestMemIdentifier, memoryTransferType pTransferType, ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pRequestingMachine, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority)
{
    finalize_ptr<memoryTransferRequest> lData(new memoryTransferRequest(pSrcMemIdentifier, pDestMemIdentifier, pTransferType, pReceiverOffset, pOffset, pLength, pStep, pCount, *pRequestingMachine, 1, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, pPriority));
    
    MEM_FORWARD_DUMP(pSrcAddressSpace, pDestMemIdentifier, pReceiverOffset, pOffset, pLength, (uint)(*pRequestingMachine), *pRangeOwner.host, pRangeOwner.memIdentifier, pRangeOwner.hostOffset)

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryTransferRequest>::CreateSharedPtr(MAX_CONTROL_PRIORITY, SEND, MEMORY_TRANSFER_REQUEST_TAG, pRangeOwner.host, MEMORY_TRANSFER_REQUEST_STRUCT, lData, 1);
    
    pmCommunicator::GetCommunicator()->Send(lCommand);
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
				case MEMORY_TRANSFER_REQUEST_TAG:
				{
					memoryTransferRequest* lData = (memoryTransferRequest*)(lCommunicatorCommand->GetData());

					pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lData->sourceMemIdentifier.memOwnerHost), lData->sourceMemIdentifier.generationNumber);

					if(lAddressSpace)
                    {
                        pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->MemTransferEvent(lData->sourceMemIdentifier, lData->destMemIdentifier, (communicator::memoryTransferType)lData->transferType, lData->offset, lData->length, lData->step, lData->count, pmMachinePool::GetMachinePool()->GetMachine(lData->destHost), lData->receiverOffset, lData->isForwarded, lData->priority, lData->isTaskOriginated, lData->originatingHost, lData->sequenceNumber);
                    }

					break;
				}
                    
                case SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_TAG:
                {
					scatteredMemoryTransferRequestCombinedPacked* lData = (scatteredMemoryTransferRequestCombinedPacked*)(lCommunicatorCommand->GetData());

					pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lData->sourceMemIdentifier.memOwnerHost), lData->sourceMemIdentifier.generationNumber);

					if(lAddressSpace)
                    {
                        const std::vector<scatteredMemoryTransferRequestCombinedStruct>& lVector = *lData->requestData.get_ptr();
                        
                        for_each(lVector, [&] (const scatteredMemoryTransferRequestCombinedStruct& pStruct)
                        {
                            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->MemTransferEvent(lData->sourceMemIdentifier, lData->destMemIdentifier, communicator::TRANSFER_SCATTERED, pStruct.offset, pStruct.length, pStruct.step, pStruct.count, pmMachinePool::GetMachinePool()->GetMachine(lData->destHost), pStruct.receiverOffset, false, lData->priority, lData->isTaskOriginated, lData->originatingHost, lData->sequenceNumber);
                        });
                    }

                    break;
                }

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
    
bool taskMemTransferEventsMatchFunc(const heavyOperationsEvent& pEvent, const void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case MEM_TRANSFER:
        {
            const memTransferEvent& lEventDetails = static_cast<const memTransferEvent&>(pEvent);

            if(lEventDetails.isTaskOriginated
               && lEventDetails.taskOriginatingHost == (uint)(*(static_cast<const pmTask*>(pCriterion))->GetOriginatingHost())
               && lEventDetails.taskSequenceNumber == (static_cast<const pmTask*>(pCriterion))->GetSequenceNumber())
                return true;
        
            break;
        }
        
        default:
            return false;
    }
    
    return false;
}

}




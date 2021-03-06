
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
#include "pmCallbackUnit.h"

#include <memory>

namespace pm
{
    
using namespace heavyOperations;
using namespace communicator;

#ifdef TRACK_MEMORY_REQUESTS
void __dump_mem_forward(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host, uint newHost, const memoryIdentifierStruct& newIdentifier, ulong newOffset);
void __dump_mem_transfer(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host);
    
void __dump_mem_forward(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host, uint newHost, const memoryIdentifierStruct&  newIdentifier, ulong newOffset)
{
    char lStr[512];
    
    sprintf(lStr, "Forwarding address space %p (Dest mem (%d, %ld); Remote mem (%d, %ld)) from offset %ld (Dest offset %ld; Remote Offset %ld) for length %ld (step %ld, count %ld) to host %d (Dest host %d)", addressSpace, identifier.memOwnerHost, identifier.generationNumber, newIdentifier.memOwnerHost, newIdentifier.generationNumber, offset, receiverOffset, newOffset, length, step, count, newHost, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

void __dump_mem_transfer(const pmAddressSpace* addressSpace, const memoryIdentifierStruct& identifier, size_t receiverOffset, size_t offset, size_t length, size_t step, size_t count, uint host)
{
    char lStr[512];
    
    sprintf(lStr, "Transferring address space %p (Remote mem (%d, %ld)) from offset %ld (Remote offset %ld) for length %ld (step %ld, count %ld) to host %d", addressSpace,identifier.memOwnerHost, identifier.generationNumber, offset, receiverOffset, length, step, count, host);
    
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define MEM_TRANSFER_DUMP(addressSpace, identifier, receiverOffset, offset, length, step, count, host) __dump_mem_transfer(addressSpace, identifier, receiverOffset, offset, length, step, count, host);
#define MEM_FORWARD_DUMP(addressSpace, identifier, receiverOffset, offset, length, step, count, host, newHost, newIdentifier, newOffset) __dump_mem_forward(addressSpace, identifier, receiverOffset, offset, length, step, count, host, newHost, newIdentifier, newOffset);
#else
#define MEM_TRANSFER_DUMP(addressSpace, identifier, receiverOffset, offset, length, step, count, host)
#define MEM_FORWARD_DUMP(addressSpace, identifier, receiverOffset, offset, length, step, count, host, newHost, newIdentifier, newOffset)
#endif

void HeavyOperationsCommandCompletionCallback(const pmCommandPtr& pCommand)
{
	pmHeavyOperationsThreadPool* lHeavyOperationsThreadPool = pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool();
	lHeavyOperationsThreadPool->CommandCompletionEvent(pCommand);
}
    
// If the macro PROCESS_METADATA_RECEIVE_IN_NETWORK_THREAD is defined, then this function is
// executed directly in network thread's context. This is because memory is received in two MPI messages
// where the first message contains the metadata while the second contains actual memory. By executing this callback on
// network thread, we omit the time this message would have taken through the scheduler queue and post an MPI_recv for
// the actual data quickly. Also, note that since this method is executed on network thread, it has its lock acquired at
// that time. Calling a network layer's method (from this function) that acquires the lock again will cause a deadlock.
void MemoryMetaDataReceiveCommandCompletionCallback(const pmCommandPtr& pCommand)
{
    pmCommunicatorCommandPtr lCommunicatorCommand = std::dynamic_pointer_cast<pmCommunicatorCommandBase>(pCommand);

    memoryReceiveStruct* lReceiveStruct = (memoryReceiveStruct*)(lCommunicatorCommand->GetData());
    pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->memOwnerHost), lReceiveStruct->generationNumber);

    if(lAddressSpace)		// If memory still exists
    {
        communicator::communicatorCommandTags lTag = (communicator::communicatorCommandTags)lReceiveStruct->mpiTag;
        const pmMachine* lSendingMachine = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->senderHost);
        
        pmTask* lRequestingTask = NULL;
        if(lReceiveStruct->isTaskOriginated)
        {
            const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->originatingHost);
            lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lReceiveStruct->sequenceNumber);
        }

        if(!lReceiveStruct->isTaskOriginated || (lRequestingTask && lAddressSpace->GetLockingTask() == lRequestingTask))
        {
            char* lBaseAddr = (char*)(lAddressSpace->GetMem()) + lReceiveStruct->offset;

            finalize_ptr<memoryReceiveStruct> lMemoryReceiveData(new memoryReceiveStruct(*lReceiveStruct));

            pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<memoryReceiveStruct>::CreateSharedPtr(lCommunicatorCommand->GetPriority(), RECEIVE, lTag, lSendingMachine, BYTE, lMemoryReceiveData, 1, HeavyOperationsCommandCompletionCallback, static_cast<void*>(lBaseAddr));

            pmCommunicator::GetCommunicator()->ReceiveMemory(lCommand, false);
        }
    }
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
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(MEMORY_RECEIVE_STRUCT);

	SetupPersistentCommunicationCommands();
}

pmHeavyOperationsThreadPool::~pmHeavyOperationsThreadPool()
{
    mThreadVector.clear();

	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MEMORY_IDENTIFIER_STRUCT);
	NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MEMORY_TRANSFER_REQUEST_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(FILE_OPERATIONS_STRUCT);
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MEMORY_RECEIVE_STRUCT);

	DestroyPersistentCommunicationCommands();
}

void pmHeavyOperationsThreadPool::SetupPersistentCommunicationCommands()
{
    finalize_ptr<communicator::memoryTransferRequest> lMemTransferRequestData(new memoryTransferRequest());
    finalize_ptr<communicator::fileOperationsStruct> lFileOperationsRecvData(new fileOperationsStruct());
    finalize_ptr<communicator::memoryReceiveStruct> lMemoryReceiveRecvData(new memoryReceiveStruct());

#define PERSISTENT_RECV_COMMAND(tag, structType, structEnumType, recvDataPtr) pmCommunicatorCommand<structEnumType>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, tag, NULL, structType, recvDataPtr, 1, HeavyOperationsCommandCompletionCallback)

#ifdef PROCESS_METADATA_RECEIVE_IN_NETWORK_THREAD
#define PERSISTENT_MEMORY_RECV_COMMAND(tag, structEnumType, structType, recvDataPtr) pmCommunicatorCommand<structType>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, tag, NULL, structEnumType, recvDataPtr, 1, MemoryMetaDataReceiveCommandCompletionCallback)
#endif

	mMemTransferRequestCommand = PERSISTENT_RECV_COMMAND(MEMORY_TRANSFER_REQUEST_TAG, MEMORY_TRANSFER_REQUEST_STRUCT, memoryTransferRequest, lMemTransferRequestData);
	mFileOperationsRecvCommand = PERSISTENT_RECV_COMMAND(FILE_OPERATIONS_TAG, FILE_OPERATIONS_STRUCT, fileOperationsStruct, lFileOperationsRecvData);
#ifdef PROCESS_METADATA_RECEIVE_IN_NETWORK_THREAD
    mMemoryReceiveRecvCommand = PERSISTENT_MEMORY_RECV_COMMAND(MEMORY_RECEIVE_TAG, MEMORY_RECEIVE_STRUCT, memoryReceiveStruct, lMemoryReceiveRecvData);
#else
    mMemoryReceiveRecvCommand = PERSISTENT_RECV_COMMAND(MEMORY_RECEIVE_TAG, MEMORY_RECEIVE_STRUCT, memoryReceiveStruct, lMemoryReceiveRecvData);
#endif
    
    mMemTransferRequestCommand->SetPersistent();
    mFileOperationsRecvCommand->SetPersistent();
    mMemoryReceiveRecvCommand->SetPersistent();
    
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();
    lNetwork->InitializePersistentCommand(mMemTransferRequestCommand);
    lNetwork->InitializePersistentCommand(mFileOperationsRecvCommand);
    lNetwork->InitializePersistentCommand(mMemoryReceiveRecvCommand);

	pmCommunicator::GetCommunicator()->Receive(mMemTransferRequestCommand, false);
	pmCommunicator::GetCommunicator()->Receive(mFileOperationsRecvCommand, false);
    pmCommunicator::GetCommunicator()->Receive(mMemoryReceiveRecvCommand, false);
}
    
void pmHeavyOperationsThreadPool::DestroyPersistentCommunicationCommands()
{
    pmNetwork* lNetwork = NETWORK_IMPLEMENTATION_CLASS::GetNetwork();

    lNetwork->TerminatePersistentCommand(mMemTransferRequestCommand);
    lNetwork->TerminatePersistentCommand(mFileOperationsRecvCommand);
    lNetwork->TerminatePersistentCommand(mMemoryReceiveRecvCommand);
}

void pmHeavyOperationsThreadPool::QueueNetworkRequest(pmCommunicatorCommandPtr& pCommand, heavyOperations::networkRequestType pType)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new networkRequestEvent(NETWORK_REQUEST_EVENT, pCommand, pType)), pCommand->GetPriority());
}

void pmHeavyOperationsThreadPool::PackAndSendData(const pmCommunicatorCommandPtr& pCommand)
{
    SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new packEvent(PACK_DATA, pCommand)), pCommand->GetPriority());
}
    
void pmHeavyOperationsThreadPool::ReduceRequestEvent(pmExecutionStub* pReducingStub, pmTask* pTask, const pmMachine* pDestMachine, ulong pSubtaskId, pmSplitInfo* pSplitInfo)
{
    pmSplitData lSplitData(pSplitInfo);

	SubmitToThreadPool(std::shared_ptr<heavyOperationsEvent>(new subtaskReduceEvent(SUBTASK_REDUCE, pTask, pDestMachine, pReducingStub, pSubtaskId, lSplitData)), pTask->GetPriority());
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

    std::vector<std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS>> lSignalWaitArray(lPoolSize, std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS>(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true)));
	SubmitToAllThreadsInPool(std::shared_ptr<heavyOperationsEvent>(new memTransferCancelEvent(MEM_TRANSFER_CANCEL, pAddressSpace, lSignalWaitArray[0].get())), MAX_CONTROL_PRIORITY);
    
    for(size_t i = 0; i < lPoolSize; ++i)
        lSignalWaitArray[i]->Wait();
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
        
		case SUBTASK_REDUCE:
        {
            subtaskReduceEvent& lEventDetails = static_cast<subtaskReduceEvent&>(pEvent);
            
            std::unique_ptr<pmSplitInfo> lSplitInfoAutoPtr(lEventDetails.splitData.operator std::unique_ptr<pmSplitInfo>());
            
            pmSubscriptionManager& lSubscriptionManager = lEventDetails.task->GetSubscriptionManager();
            bool lHasScratchBuffers = lSubscriptionManager.HasScratchBuffers(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get());
            
            const pmAddressSpace* lAddressSpace = 0;
            size_t lReducibleAddressSpaces = 0;
            size_t lAddressSpaceIndex = 0;
            for_each_with_index(lEventDetails.task->GetAddressSpaces(), [&] (const pmAddressSpace* pAddressSpace, size_t pIndex)
            {
                if(lEventDetails.task->IsWritable(pAddressSpace) && lEventDetails.task->IsReducible(pAddressSpace))
                {
                    lAddressSpace = pAddressSpace;
                    lAddressSpaceIndex = pIndex;
                    ++lReducibleAddressSpaces;
                }
            });

            bool lOptimalSendDone = false;

        #ifdef SUPPORT_LAZY_MEMORY
        #else
            // Check if reduction data can be transferred without mpi packing
            if(!lHasScratchBuffers && lReducibleAddressSpaces == 1 && NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->IsImplicitlyReducible(lEventDetails.task))
            {
                void* lShadowMem = lSubscriptionManager.GetSubtaskShadowMem(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);
                ulong lOffset = 0, lLength = 0;
                
                if(lEventDetails.task->GetAddressSpaceSubscriptionVisibility(lAddressSpace, lEventDetails.reducingStub) == SUBSCRIPTION_NATURAL)
                {
                    pmSubscriptionInfo lUnifiedSubscriptionInfo = lSubscriptionManager.GetUnifiedReadWriteSubscription(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);

                    subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                    lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
                    
                    EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

                    lOffset =  lBeginIter->first - lUnifiedSubscriptionInfo.offset;
                    lLength = (uint)lBeginIter->second.first;
                }
                else    // SUBSCRIPTION_COMPACT
                {
                    const subscription::pmCompactViewData& lCompactViewData = lSubscriptionManager.GetCompactedSubscription(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex);

                    subscription::subscriptionRecordType::const_iterator lBeginIter, lEndIter;
                    lSubscriptionManager.GetNonConsolidatedWriteSubscriptions(lEventDetails.reducingStub, lEventDetails.subtaskId, lSplitInfoAutoPtr.get(), (uint)lAddressSpaceIndex, lBeginIter, lEndIter);
                    
                    auto lCompactWriteIter = lCompactViewData.nonConsolidatedWriteSubscriptionOffsets.begin();
                    
                    EXCEPTION_ASSERT(std::distance(lBeginIter, lEndIter) == 1);    // Only one write subscription

                    lOffset = *lCompactWriteIter;
                    lLength = (uint)lBeginIter->second.first;
                }
                
                pmReductionOpType lOpType;
                pmReductionDataType lDataType;
                
                findReductionOpAndDataType(lEventDetails.task->GetCallbackUnit()->GetDataReductionCB()->GetCallback(), lOpType, lDataType);

                char* lTargetMem = (static_cast<char*>(lShadowMem) + lOffset);
                
                ulong lCompressedLength = 0;
                std::shared_ptr<void> lCompressedMem;

            #ifndef TURN_OFF_NETWORK_SENTINEL_COMPRESSION
                // Scope for task profiling
                {
                #ifdef ENABLE_TASK_PROFILING
                    #ifdef DUMP_DATA_COMPRESSION_STATISTICS
                        pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(lEventDetails.task->GetTaskProfiler(), taskProfiler::NETWORK_DATA_COMPRESSION);
                    #endif
                #endif

                    switch(lDataType)
                    {
                        case REDUCE_INTS:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<int>((int*)lTargetMem, 0, lLength / sizeof(int), lCompressedLength);
                            break;
                        }
                            
                        case REDUCE_UNSIGNED_INTS:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<uint>((uint*)lTargetMem, 0, lLength / sizeof(uint), lCompressedLength);
                            break;
                        }
                            
                        case REDUCE_LONGS:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<long>((long*)lTargetMem, 0, lLength / sizeof(long), lCompressedLength);
                            break;
                        }
                            
                        case REDUCE_UNSIGNED_LONGS:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<ulong>((ulong*)lTargetMem, 0, lLength / sizeof(ulong), lCompressedLength);
                            break;
                        }
                            
                        case REDUCE_FLOATS:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<float>((float*)lTargetMem, 0, lLength / sizeof(float), lCompressedLength);
                            break;
                        }
                            
                        case REDUCE_DOUBLES:
                        {
                            lCompressedMem = pmUtility::CompressForSentinel<double>((double*)lTargetMem, 0, lLength / sizeof(double), lCompressedLength);
                            break;
                        }
                            
                        default:
                            PMTHROW(pmFatalErrorException());
                    }
                }
            #endif

                finalize_ptr<subtaskMemoryReduceStruct> lData(new subtaskMemoryReduceStruct(*lEventDetails.task->GetOriginatingHost(), lEventDetails.task->GetSequenceNumber(), lEventDetails.subtaskId, lOffset, (lCompressedMem.get() ? lCompressedLength : lLength), std::numeric_limits<int>::max(), *PM_LOCAL_MACHINE, (lCompressedMem.get() != NULL)));
                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<subtaskMemoryReduceStruct>::CreateSharedPtr(lEventDetails.task->GetPriority(), SEND, SUBTASK_MEMORY_REDUCE_TAG, lEventDetails.machine, SUBTASK_MEMORY_REDUCE_STRUCT, lData, 1, NULL, (lCompressedMem.get() ? lCompressedMem.get() : lTargetMem));
                
                if(lCompressedMem.get())
                    lCommand->HoldExternalDataForLifetimeOfCommand(lCompressedMem);
                
                pmCommunicator::GetCommunicator()->SendReduce(lCommand, false);

                lOptimalSendDone = true;
            }
        #endif
            
            // Send reduction data as MPI_PACKED
            if(!lOptimalSendDone)
            {
                finalize_ptr<subtaskReducePacked> lPackedData(new subtaskReducePacked(lEventDetails.reducingStub, lEventDetails.task, lEventDetails.subtaskId, lSplitInfoAutoPtr.get()));
            
                pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<subtaskReducePacked>::CreateSharedPtr(lEventDetails.task->GetPriority(), SEND, SUBTASK_REDUCE_TAG, lEventDetails.machine, SUBTASK_REDUCE_PACKED, lPackedData, 1, NULL, lEventDetails.task);

                pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->PackAndSendData(lCommand);
            }

            break;
        }

        case MEM_TRANSFER:
        {
            memTransferEvent& lEventDetails = static_cast<memTransferEvent&>(pEvent);

            EXCEPTION_ASSERT(lEventDetails.machine != PM_LOCAL_MACHINE);   // Cyclic reference

            pmAddressSpace* lSrcAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.srcMemIdentifier.memOwnerHost), lEventDetails.srcMemIdentifier.generationNumber);
            if(!lSrcAddressSpace)
                return;
            
            pmTask* lRequestingTask = NULL;
            if(lEventDetails.isTaskOriginated)
            {
                const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lEventDetails.taskOriginatingHost);
                lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lEventDetails.taskSequenceNumber);
                
                if(lOriginatingHost == PM_LOCAL_MACHINE)
                {
                    if(!lRequestingTask)
                        return;
                }
                else
                {
                    // Turn down all requests for tasks that have already finished
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
    pmMemOwnership lOwnerships;
    pSrcAddressSpace->GetOwners(pOffset, pLength, lOwnerships);
    
    for_each(lOwnerships, [&] (const pmMemOwnership::value_type& pPair)
    {
        ulong lInternalOffset = pPair.first;
        ulong lInternalLength = pPair.second.first;
        const vmRangeOwner& lRangeOwner = pPair.second.second;
        
        if(lRangeOwner.host == PM_LOCAL_MACHINE)
        {
            pmAddressSpace* lOwnerAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lRangeOwner.memIdentifier.memOwnerHost), lRangeOwner.memIdentifier.generationNumber);
        
            EXCEPTION_ASSERT(lOwnerAddressSpace);
            
            if(pRequestingMachine == PM_LOCAL_MACHINE)
            {
                EXCEPTION_ASSERT(0);    // This should not happen

            #if 0
                pmAddressSpace* lDestAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(pDestMemIdentifier.memOwnerHost), pDestMemIdentifier.generationNumber);
            
                EXCEPTION_ASSERT(lDestAddressSpace);
                
                std::function<void (char*, ulong)> lFunc([&] (char* pMem, ulong pCopyLength)
                {
                    DEBUG_EXCEPTION_ASSERT(pCopyLength == pLength);
                    PMLIB_MEMCPY(pMem, (void*)((char*)(lOwnerAddressSpace->GetMem()) + lInternalOffset), pCopyLength, std::string("pmHeavyOperationsThread::ServeGeneralMemoryRequest"));
                });

                if(!pIsTaskOriginated || pRequestingTask)
                    lDestAddressSpace->CopyOrUpdateReceivedMemory(pRequestingTask, pReceiverOffset + lInternalOffset - pOffset, pLength, &lFunc);
            #endif
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

                    MEM_TRANSFER_DUMP(pSrcAddressSpace, pDestMemIdentifier, pReceiverOffset + lInternalOffset - pOffset, lInternalOffset, lStepLength, 0, 0, (uint)(*pRequestingMachine))

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
    auto lLambda = [&] (const pmScatteredMemOwnership& pScatteredMemOwnership)
    {
        for_each(pScatteredMemOwnership, [&] (const std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>& pPair)
        {
            const pmScatteredSubscriptionInfo& lScatteredInfo = pPair.first;
            const vmRangeOwner& lRangeOwner = pPair.second;
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

            #ifdef DUMP_DATA_TRANSFER_FREQUENCY
                if(!pRequestingTask->ShouldSuppressTaskLogs())
                    pSrcAddressSpace->RecordDataTransferFrequency(lScatteredInfo.offset, lScatteredInfo.size, lScatteredInfo.step, lScatteredInfo.count);
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
                
                    MEM_TRANSFER_DUMP(pSrcAddressSpace, pDestMemIdentifier, lReceiverOffset + lInternalStepOffset, lStepScatteredInfo.offset, lStepScatteredInfo.size, lStepScatteredInfo.step, lStepScatteredInfo.count, (uint)(*pRequestingMachine))

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
    };

    if(pSrcAddressSpace->GetAddressSpaceType() == ADDRESS_SPACE_LINEAR)
    {
        pmScatteredSubscriptionFilter lBlocksFilter(pmScatteredSubscriptionInfo(pOffset, pLength, pStep, pCount));
        const auto& lBlocks = lBlocksFilter.FilterBlocks([&] (size_t pRow)
        {
            pmMemOwnership lOwnerships;
            pSrcAddressSpace->GetOwners(pOffset + pRow * pStep, pLength, lOwnerships);
            
            for_each(lOwnerships, [&] (pmMemOwnership::value_type& pPair)
            {
                lBlocksFilter.AddNextSubRow(pPair.first, pPair.second.first, pPair.second.second);
            });
        });

        for_each(lBlocks, [&] (const std::pair<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& pMapKeyValue)
        {
            lLambda(pMapKeyValue.second);
        });
    }
    else
    {
        pmScatteredMemOwnership lScatteredMemOwnership;
        pSrcAddressSpace->GetOwners(pOffset, pLength, pStep, pCount, lScatteredMemOwnership);

        lLambda(lScatteredMemOwnership);
    }
}
    
void pmHeavyOperationsThread::ForwardMemoryRequest(pmAddressSpace* pSrcAddressSpace, const vmRangeOwner& pRangeOwner, const memoryIdentifierStruct& pSrcMemIdentifier, const memoryIdentifierStruct& pDestMemIdentifier, memoryTransferType pTransferType, ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pRequestingMachine, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority)
{
    finalize_ptr<memoryTransferRequest> lData(new memoryTransferRequest(pSrcMemIdentifier, pDestMemIdentifier, pTransferType, pReceiverOffset, pOffset, pLength, pStep, pCount, *pRequestingMachine, 1, pIsTaskOriginated, pTaskOriginatingHost, pTaskSequenceNumber, pPriority));
    
    MEM_FORWARD_DUMP(pSrcAddressSpace, pDestMemIdentifier, pReceiverOffset, pOffset, pLength, pStep, pCount, (uint)(*pRequestingMachine), *pRangeOwner.host, pRangeOwner.memIdentifier, pRangeOwner.hostOffset)

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
                
				case MEMORY_RECEIVE_TAG:
				{
                #ifdef PROCESS_METADATA_RECEIVE_IN_NETWORK_THREAD
                    // Since memory transfers are received in two MPI messages (the first one contains size of the upcoming memory in the second one),
                    // the first message is not handled in this callback as it is executed on scheduler thread. Rather, in order to reduce the turnaround
                    // time, the callback is executed in network thread's context and an MPI_Irecv for the upcoming second message is immediately posted
                    PMTHROW(pmFatalErrorException());
                #else
                    MemoryMetaDataReceiveCommandCompletionCallback(pCommand);
                #endif

					break;
				}
                    
                default:
                {
                    memoryReceiveStruct* lReceiveStruct = static_cast<memoryReceiveStruct*>(lCommunicatorCommand->GetData());
                    if(lReceiveStruct)
                    {
                        pmAddressSpace* lAddressSpace = pmAddressSpace::FindAddressSpace(pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->memOwnerHost), lReceiveStruct->generationNumber);
                    
                        if(lAddressSpace)		// If memory still exists
                        {
                            pmTask* lRequestingTask = NULL;
                            if(lReceiveStruct->isTaskOriginated)
                            {
                                const pmMachine* lOriginatingHost = pmMachinePool::GetMachinePool()->GetMachine(lReceiveStruct->originatingHost);
                                lRequestingTask = pmTaskManager::GetTaskManager()->FindTaskNoThrow(lOriginatingHost, lReceiveStruct->sequenceNumber);
                            }

                            if(lReceiveStruct->transferType == TRANSFER_GENERAL || (lReceiveStruct->step == 0 && lReceiveStruct->count == 1))
                            {
                            #ifdef ENABLE_MEM_PROFILING
                                if(!lRequestingTask || !lRequestingTask->ShouldSuppressTaskLogs())
                                {
                                    lAddressSpace->RecordMemReceive(lReceiveStruct->length);

                                    if(lRequestingTask)
                                        lRequestingTask->GetTaskExecStats().RecordMemReceiveEvent(lReceiveStruct->length, false);
                                }
                            #endif

                                lAddressSpace->CopyOrUpdateReceivedMemory(lRequestingTask, lReceiveStruct->offset, lReceiveStruct->length, NULL);
                            }
                            else    // TRANSFER_SCATTERED
                            {
                                DEBUG_EXCEPTION_ASSERT(lReceiveStruct->transferType == TRANSFER_SCATTERED);

                            #ifdef ENABLE_MEM_PROFILING
                                if(!lRequestingTask || !lRequestingTask->ShouldSuppressTaskLogs())
                                {
                                    lAddressSpace->RecordMemReceive(lReceiveStruct->length * lReceiveStruct->count);

                                    if(lRequestingTask)
                                        lRequestingTask->GetTaskExecStats().RecordMemReceiveEvent(lReceiveStruct->length * lReceiveStruct->count, true);
                                }
                            #endif

                                lAddressSpace->UpdateReceivedMemory(lRequestingTask, lReceiveStruct->offset, lReceiveStruct->length, lReceiveStruct->step, lReceiveStruct->count);
                            }
                        }

                        break;
                    }
                    
                    PMTHROW(pmFatalErrorException());
                }
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




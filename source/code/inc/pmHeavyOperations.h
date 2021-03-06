
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

#ifndef __PM_HEAVY_OPERATIONS__
#define __PM_HEAVY_OPERATIONS__

#include "pmBase.h"
#include "pmThread.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"
#include "pmHardware.h"
#include "pmCommunicator.h"
#include "pmAddressSpace.h"

#include <vector>

namespace pm
{

class pmAddressSpace;

namespace heavyOperations
{

enum eventIdentifier
{
    NETWORK_REQUEST_EVENT,
	PACK_DATA,
    UNPACK_DATA,
    SUBTASK_REDUCE,
    MEM_TRANSFER,
    MEM_TRANSFER_CANCEL,
    TASK_MEM_TRANSFER_CANCEL,
    COMMAND_COMPLETION,
    MAX_HEAVY_OPERATIONS_EVENT
};
    
enum networkRequestType
{
    NETWORK_SEND_REQUEST,
    NETWORK_RECEIVE_REQUEST
};

struct heavyOperationsEvent : public pmBasicThreadEvent
{
	eventIdentifier eventId;
    
    heavyOperationsEvent(eventIdentifier pEventId = MAX_HEAVY_OPERATIONS_EVENT)
    : eventId(pEventId)
    {}
};
    
struct networkRequestEvent : public heavyOperationsEvent
{
    pmCommunicatorCommandPtr command;
    networkRequestType type;
    
    networkRequestEvent(eventIdentifier pEventId, pmCommunicatorCommandPtr& pCommand, networkRequestType pType)
    : heavyOperationsEvent(pEventId)
    , command(pCommand)
    , type(pType)
    {}
};

struct packEvent : public heavyOperationsEvent
{
    pmCommunicatorCommandPtr command;
    
    packEvent(eventIdentifier pEventId, const pmCommunicatorCommandPtr& pCommand)
    : heavyOperationsEvent(pEventId)
    , command(pCommand)
    {}
};

struct unpackEvent : public heavyOperationsEvent
{
    finalize_ptr<char, deleteArrayDeallocator<char>> packedData;
    int packedLength;
    
    unpackEvent(eventIdentifier pEventId, finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pPackedLength)
    : heavyOperationsEvent(pEventId)
    , packedData(std::move(pPackedData))
    , packedLength(pPackedLength)
    {}
};

struct subtaskReduceEvent : public heavyOperationsEvent
{
	pmTask* task;
	const pmMachine* machine;
    pmExecutionStub* reducingStub;
	ulong subtaskId;
    pmSplitData splitData;
    
    subtaskReduceEvent(eventIdentifier pEventId, pmTask* pTask, const pmMachine* pMachine, pmExecutionStub* pReducingStub, ulong pSubtaskId, pmSplitData& pSplitData)
    : heavyOperationsEvent(pEventId)
    , task(pTask)
    , machine(pMachine)
    , reducingStub(pReducingStub)
    , subtaskId(pSubtaskId)
    , splitData(pSplitData)
    {}
};

struct memTransferEvent : public heavyOperationsEvent
{
	communicator::memoryIdentifierStruct srcMemIdentifier;
    communicator::memoryIdentifierStruct destMemIdentifier;
    communicator::memoryTransferType transferType;
	ulong offset;
	ulong length;
    ulong step;
    ulong count;
	const pmMachine* machine;
    ulong receiverOffset;
	ushort priority;
    bool isForwarded;
    bool isTaskOriginated;
    uint taskOriginatingHost;
    ulong taskSequenceNumber;
    
    memTransferEvent(eventIdentifier pEventId, communicator::memoryIdentifierStruct& pSrcMemIdentifier,
                     communicator::memoryIdentifierStruct& pDestMemIdentifier, communicator::memoryTransferType pTransferType,
                     ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pMachine, ulong pReceiverOffset,
                     ushort pPriority, bool pIsForwarded, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber)
    : heavyOperationsEvent(pEventId)
    , srcMemIdentifier(pSrcMemIdentifier)
    , destMemIdentifier(pDestMemIdentifier)
    , transferType(pTransferType)
    , offset(pOffset)
    , length(pLength)
    , step(pStep)
    , count(pCount)
    , machine(pMachine)
    , receiverOffset(pReceiverOffset)
    , priority(pPriority)
    , isForwarded(pIsForwarded)
    , isTaskOriginated(pIsTaskOriginated)
    , taskOriginatingHost(pTaskOriginatingHost)
    , taskSequenceNumber(pTaskSequenceNumber)
    {}
};
    
struct memTransferCancelEvent : public heavyOperationsEvent
{
    pmAddressSpace* addressSpace;
    SIGNAL_WAIT_IMPLEMENTATION_CLASS* signalWaitArray;

    memTransferCancelEvent(eventIdentifier pEventId, pmAddressSpace* pAddressSpace, SIGNAL_WAIT_IMPLEMENTATION_CLASS* pSignalWaitArray)
    : heavyOperationsEvent(pEventId)
    , addressSpace(pAddressSpace)
    , signalWaitArray(pSignalWaitArray)
    {}
};

struct taskMemTransferCancelEvent : public heavyOperationsEvent
{
    pmTask* task;
    
    taskMemTransferCancelEvent(eventIdentifier pEventId, pmTask* pTask)
    : heavyOperationsEvent(pEventId)
    , task(pTask)
    {}
};

struct commandCompletionEvent : public heavyOperationsEvent
{
	pmCommandPtr command;
    
    commandCompletionEvent(eventIdentifier pEventId, pmCommandPtr& pCommand)
    : heavyOperationsEvent(pEventId)
    , command(pCommand)
    {}
};

}
    
class pmHeavyOperationsThread : public THREADING_IMPLEMENTATION_CLASS<heavyOperations::heavyOperationsEvent>
{
public:
    pmHeavyOperationsThread(size_t pThreadIndex);
    virtual ~pmHeavyOperationsThread();
    
private:
    virtual void ThreadSwitchCallback(std::shared_ptr<heavyOperations::heavyOperationsEvent>& pEvent);
    void ProcessEvent(heavyOperations::heavyOperationsEvent& pEvent);

    void ServeGeneralMemoryRequest(pmAddressSpace* pSrcAddressSpace, pmTask* pRequestingTask, const pmMachine* pRequestingMachine, ulong pOffset, ulong pLength, const communicator::memoryIdentifierStruct& pDestMemIdentifier, ulong pReceiverOffset, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority, bool pIsForwarded);
    
    void ServeScatteredMemoryRequest(pmAddressSpace* pSrcAddressSpace, pmTask* pRequestingTask, const pmMachine* pRequestingMachine, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const communicator::memoryIdentifierStruct& pDestMemIdentifier, ulong pReceiverOffset, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority, bool pIsForwarded);
    
    void ForwardMemoryRequest(pmAddressSpace* pSrcAddressSpace, const vmRangeOwner& pRangeOwner, const communicator::memoryIdentifierStruct& pSrcMemIdentifier, const communicator::memoryIdentifierStruct& pDestMemIdentifier, communicator::memoryTransferType pTransferType, ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pRequestingMachine, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority);

    void HandleCommandCompletion(pmCommandPtr& pCommand);
    
    size_t mThreadIndex;
};

class pmHeavyOperationsThreadPool
{
    friend class pmHeavyOperationsThread;
	friend void HeavyOperationsCommandCompletionCallback(const pmCommandPtr& pCommand);
    friend void MemoryMetaDataReceiveCommandCompletionCallback(const pmCommandPtr& pCommand);
    
public:
    virtual ~pmHeavyOperationsThreadPool();

    void QueueNetworkRequest(pmCommunicatorCommandPtr& pCommand, heavyOperations::networkRequestType pType);
    void PackAndSendData(const pmCommunicatorCommandPtr& pCommand);
    void UnpackDataEvent(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pPackedLength, ushort pPriority);
    void ReduceRequestEvent(pmExecutionStub* pReducingStub, pmTask* pTask, const pmMachine* pDestMachine, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    void MemTransferEvent(communicator::memoryIdentifierStruct& pSrcMemIdentifier, communicator::memoryIdentifierStruct& pDestMemIdentifier, communicator::memoryTransferType pTransferType, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber);
    void CancelMemoryTransferEvents(pmAddressSpace* pAddressSpace);
    void CancelTaskSpecificMemoryTransferEvents(pmTask* pTask);
    
    pmCommandCompletionCallbackType GetHeavyOperationsCommandCompletionCallback();
    
    static pmHeavyOperationsThreadPool* GetHeavyOperationsThreadPool();

private:
    pmHeavyOperationsThreadPool(size_t pThreadCount);

    void SubmitToThreadPool(const std::shared_ptr<heavyOperations::heavyOperationsEvent>& pEvent, ushort pPriority);
    void SubmitToAllThreadsInPool(const std::shared_ptr<heavyOperations::heavyOperationsEvent>& pEvent, ushort pPriority) const;
	void SetupPersistentCommunicationCommands();
	void DestroyPersistentCommunicationCommands();

    void CommandCompletionEvent(pmCommandPtr pCommand);
    
    std::vector<std::unique_ptr<pmHeavyOperationsThread>> mThreadVector;
    size_t mCurrentThread;
    
    pmCommunicatorCommandPtr mFileOperationsRecvCommand;
    pmCommunicatorCommandPtr mMemTransferRequestCommand;
    pmCommunicatorCommandPtr mMemoryReceiveRecvCommand;
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

bool taskMemTransferEventsMatchFunc(const heavyOperations::heavyOperationsEvent& pEvent, const void* pCriterion);
    
}

#endif


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
    
    void ForwardMemoryRequest(pmAddressSpace* pSrcAddressSpace, const pmAddressSpace::vmRangeOwner& pRangeOwner, const communicator::memoryIdentifierStruct& pSrcMemIdentifier, const communicator::memoryIdentifierStruct& pDestMemIdentifier, communicator::memoryTransferType pTransferType, ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pRequestingMachine, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber, ushort pPriority);

    void HandleCommandCompletion(pmCommandPtr& pCommand);
    
    size_t mThreadIndex;
};

class pmHeavyOperationsThreadPool
{
    friend class pmHeavyOperationsThread;
	friend void HeavyOperationsCommandCompletionCallback(const pmCommandPtr& pCommand);
    
public:
    virtual ~pmHeavyOperationsThreadPool();

    void QueueNetworkRequest(pmCommunicatorCommandPtr& pCommand, heavyOperations::networkRequestType pType);
    void PackAndSendData(const pmCommunicatorCommandPtr& pCommand);
    void UnpackDataEvent(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pPackedLength, ushort pPriority);
    void MemTransferEvent(communicator::memoryIdentifierStruct& pSrcMemIdentifier, communicator::memoryIdentifierStruct& pDestMemIdentifier, communicator::memoryTransferType pTransferType, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, const pmMachine* pDestMachine, ulong pReceiverOffset, bool pIsForwarded, ushort pPriority, bool pIsTaskOriginated, uint pTaskOriginatingHost, ulong pTaskSequenceNumber);
    void CancelMemoryTransferEvents(pmAddressSpace* pAddressSpace);
    void CancelTaskSpecificMemoryTransferEvents(pmTask* pTask);
    
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
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

bool taskMemTransferEventsMatchFunc(const heavyOperations::heavyOperationsEvent& pEvent, void* pCriterion);
    
}

#endif

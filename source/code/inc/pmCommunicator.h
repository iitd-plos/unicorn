
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

#ifndef __PM_COMMUNICATOR__
#define __PM_COMMUNICATOR__

#include "pmBase.h"
#include "pmCommand.h"

#include <string.h>

#include <limits>
#include <memory>

namespace pm
{

class pmTask;
class pmSignalWait;
class pmLocalTask;
class pmHardware;
class pmMachine;
class pmExecutionStub;
class pmAddressSpace;
    
namespace communicator
{
    
template<typename T>
struct all2AllWrapper
{
    T localData;
    std::vector<T> all2AllData;
    
    all2AllWrapper(const T& pLocalData, size_t pParticipants)
    : localData(pLocalData)
    {
        all2AllData.resize(pParticipants);
    }
    
    all2AllWrapper(const all2AllWrapper<T>&) = delete;
    all2AllWrapper<T> operator=(const all2AllWrapper<T>&) = delete;
};

struct machinePool
{
    uint cpuCores;
    uint gpuCards;
    uint cpuNumaDomains;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 3
    } fieldCount;

    machinePool()
    : cpuCores(0)
    , gpuCards(0)
    , cpuNumaDomains(0)
    {}
    
    machinePool(uint pCpuCores, uint pGpuCards, uint pCpuNumaDomains)
    : cpuCores(pCpuCores)
    , gpuCards(pGpuCards)
    , cpuNumaDomains(pCpuNumaDomains)
    {}
};

struct devicePool
{
    char name[MAX_NAME_STR_LEN];
    char description[MAX_DESC_STR_LEN];
    uint numaDomain;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 3
    } fieldCount;

    devicePool()
    {
        memset(this, 0, sizeof(*this));
    }
};

struct memoryIdentifierStruct
{
    uint memOwnerHost;
    ulong generationNumber;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 2
    } fieldCount;
    
    memoryIdentifierStruct()
    : memOwnerHost(std::numeric_limits<uint>::max())
    , generationNumber(std::numeric_limits<ulong>::max())
    {}
    
    memoryIdentifierStruct(uint pMemOwnerHost, ulong pGenerationNumber)
    : memOwnerHost(pMemOwnerHost)
    , generationNumber(pGenerationNumber)
    {}
    
    memoryIdentifierStruct(const memoryIdentifierStruct& pMemStruct)
    : memOwnerHost(pMemStruct.memOwnerHost)
    , generationNumber(pMemStruct.generationNumber)
    {}

    bool operator==(const memoryIdentifierStruct& pIdentifier) const
    {
        return (memOwnerHost == pIdentifier.memOwnerHost && generationNumber == pIdentifier.generationNumber);
    }
};

struct taskMemoryStruct
{
    memoryIdentifierStruct memIdentifier;
    ulong memLength;    // For 2D address spaces, this contains the number of rows
    ulong cols;
    ushort memType;     // enum pmMemType
    ushort subscriptionVisibility;  // enum pmSubscriptionVisibilityType
    ushort flags;       // LSB 1 - disjointReadWriteSubscriptionsAcrossSubtasks
    ushort addressSpaceType;    // enum pmAddressSpaceType

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 7
    } fieldCount;
    
    taskMemoryStruct()
    : memIdentifier()
    , memLength(0)
    , cols(std::numeric_limits<ulong>::max())
    , memType(std::numeric_limits<ushort>::max())
    , subscriptionVisibility(SUBSCRIPTION_NATURAL)
    , flags(0)
    , addressSpaceType(MAX_ADDRESS_SPACE_TYPES)
    {}

    taskMemoryStruct(const memoryIdentifierStruct& pMemStruct, ulong pMemLength, pmMemType pMemType, pmSubscriptionVisibilityType pSubscriptionVisibility, ushort pFlags)
    : memIdentifier(pMemStruct)
    , memLength(pMemLength)
    , cols(std::numeric_limits<ulong>::max())
    , memType((ushort)pMemType)
    , subscriptionVisibility(pSubscriptionVisibility)
    , flags(pFlags)
    , addressSpaceType(ADDRESS_SPACE_LINEAR)
    {}

    taskMemoryStruct(const memoryIdentifierStruct& pMemStruct, ulong pRows, ulong pCols, pmMemType pMemType, pmSubscriptionVisibilityType pSubscriptionVisibility, ushort pFlags)
    : memIdentifier(pMemStruct)
    , memLength(pRows)
    , cols(pCols)
    , memType((ushort)pMemType)
    , subscriptionVisibility(pSubscriptionVisibility)
    , flags(pFlags)
    , addressSpaceType(ADDRESS_SPACE_2D)
    {}
};

struct remoteTaskAssignStruct
{
    uint taskConfLength;
    uint taskMemCount;
    ulong taskId;
    ulong subtaskCount;
    char callbackKey[MAX_CB_KEY_LEN];
    uint assignedDeviceCount;
    uint originatingHost;
    ulong sequenceNumber;   // Sequence number of task on originating host
    ushort priority;
    ushort schedModel;
    ushort affinityCriterion;   // enum pmAffinityCriterion
    ushort flags;

    remoteTaskAssignStruct()
    : taskConfLength(0)
    , taskMemCount(0)
    , taskId(std::numeric_limits<ulong>::max())
    , subtaskCount(0)
    , assignedDeviceCount(0)
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , priority(MIN_PRIORITY_LEVEL)
    , schedModel(std::numeric_limits<ushort>::max())
    , affinityCriterion((ushort)MAX_AFFINITY_CRITERION)
    , flags(0)
    {
        callbackKey[0] = '\0';
    }
    
    remoteTaskAssignStruct(pmLocalTask* pLocalTask);

    enum fieldCount
    {
        FIELD_COUNT_VALUE = 12
    };

};

struct remoteTaskAssignPacked
{
    remoteTaskAssignStruct taskStruct;
    finalize_ptr<char, deleteArrayDeallocator<char>> taskConf;
    std::vector<taskMemoryStruct> taskMem;
    finalize_ptr<uint, deleteArrayDeallocator<uint> > devices;

    remoteTaskAssignPacked()
    : taskStruct()
    {}
    
    remoteTaskAssignPacked(pmLocalTask* pLocalTask);
};

enum subtaskAssignmentType
{
    SUBTASK_ASSIGNMENT_REGULAR,
    RANGE_NEGOTIATION,
    SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED
};

struct remoteSubtaskAssignStruct
{
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ulong startSubtask;
    ulong endSubtask;
    uint originatingHost;
    uint targetDeviceGlobalIndex;
    uint originalAllotteeGlobalIndex;
    ushort assignmentType;  // enum subtaskAssignmentType

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 7
    } fieldCount;

    remoteSubtaskAssignStruct()
    : sequenceNumber(std::numeric_limits<ulong>::max())
    , startSubtask(std::numeric_limits<ulong>::max())
    , endSubtask(std::numeric_limits<ulong>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , targetDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originalAllotteeGlobalIndex(std::numeric_limits<uint>::max())
    , assignmentType(std::numeric_limits<ushort>::max())
    {}
    
    remoteSubtaskAssignStruct(ulong pSequenceNumber, ulong pStartSubtask, ulong pEndSubtask, uint pOriginatingHost, uint pTargetDeviceGlobalIndex, uint pOriginalAllotteeGlobalIndex, subtaskAssignmentType pAssignmentType)
    : sequenceNumber(pSequenceNumber)
    , startSubtask(pStartSubtask)
    , endSubtask(pEndSubtask)
    , originatingHost(pOriginatingHost)
    , targetDeviceGlobalIndex(pTargetDeviceGlobalIndex)
    , originalAllotteeGlobalIndex(pOriginalAllotteeGlobalIndex)
    , assignmentType((ushort)pAssignmentType)
    {}
};

struct ownershipDataStruct
{
    ulong offset;
    ulong length;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 2
    } fieldCount;
    
    ownershipDataStruct()
    : offset(std::numeric_limits<ulong>::max())
    , length(0)
    {}

    ownershipDataStruct(ulong pOffset, ulong pLength)
    : offset(pOffset)
    , length(pLength)
    {}
};

struct scatteredOwnershipDataStruct
{
    ulong offset;
    ulong size;
    ulong step;
    ulong count;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 4
    } fieldCount;
    
    scatteredOwnershipDataStruct()
    : offset(std::numeric_limits<ulong>::max())
    , size(0)
    , step(0)
    , count(0)
    {}

    scatteredOwnershipDataStruct(ulong pOffset, ulong pSize, ulong pStep, ulong pCount)
    : offset(pOffset)
    , size(pSize)
    , step(pStep)
    , count(pCount)
    {}

    scatteredOwnershipDataStruct(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
    : offset(pScatteredSubscriptionInfo.offset)
    , size(pScatteredSubscriptionInfo.size)
    , step(pScatteredSubscriptionInfo.step)
    , count(pScatteredSubscriptionInfo.count)
    {}
};
    
struct sendAcknowledgementStruct
{
    uint sourceDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ulong startSubtask;
    ulong endSubtask;
    uint execStatus;
    uint originalAllotteeGlobalIndex;
    uint ownershipDataElements;
    uint addressSpaceIndices;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 9
    } fieldCount;
    
    sendAcknowledgementStruct()
    : sourceDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , startSubtask(std::numeric_limits<ulong>::max())
    , endSubtask(std::numeric_limits<ulong>::max())
    , execStatus(pmStatusUnavailable)
    , originalAllotteeGlobalIndex(std::numeric_limits<uint>::max())
    , ownershipDataElements(0)
    , addressSpaceIndices(0)
    {}

    sendAcknowledgementStruct(uint pSourceDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, ulong pStartSubtask, ulong pEndSubtask, uint pExecStatus, uint pOriginalAllotteeGlobalIndex, uint pOwnershipDataElements, uint pAddressSpaceIndices)
    : sourceDeviceGlobalIndex(pSourceDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , startSubtask(pStartSubtask)
    , endSubtask(pEndSubtask)
    , execStatus(pExecStatus)
    , originalAllotteeGlobalIndex(pOriginalAllotteeGlobalIndex)
    , ownershipDataElements(pOwnershipDataElements)
    , addressSpaceIndices(pAddressSpaceIndices)
    {}
};

struct sendAcknowledgementPacked
{
    sendAcknowledgementStruct ackStruct;
    std::vector<ownershipDataStruct> ownershipVector;
    std::vector<uint> addressSpaceIndexVector;

    sendAcknowledgementPacked()
    : ackStruct()
    {}

    sendAcknowledgementPacked(const pmProcessingElement* pSourceDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<ownershipDataStruct>&& pOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);
};

struct sendAcknowledgementScatteredPacked
{
    sendAcknowledgementStruct ackStruct;
    std::vector<scatteredOwnershipDataStruct> scatteredOwnershipVector;
    std::vector<uint> addressSpaceIndexVector;

    sendAcknowledgementScatteredPacked()
    : ackStruct()
    {}

    sendAcknowledgementScatteredPacked(const pmProcessingElement* pSourceDevice, const pmSubtaskRange& pRange, pmStatus pExecStatus, std::vector<scatteredOwnershipDataStruct>&& pScatteredOwnershipVector, std::vector<uint>&& pAddressSpaceIndexVector);
};
    
enum taskEvents
{
    TASK_FINISH_EVENT,
    TASK_COMPLETE_EVENT,
    REDUCTION_TERMINATE_EVENT,
    TASK_CANCEL_EVENT
};

struct taskEventStruct
{
    uint taskEvent;			// Map to enum taskEvents
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 3
    } fieldCount;

    taskEventStruct()
    : taskEvent(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    {}
    
    taskEventStruct(taskEvents pTaskEvent, uint pOriginatingHost, ulong pSequenceNumber)
    : taskEvent((uint)pTaskEvent)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    {}
};

#ifdef ENABLE_TWO_LEVEL_STEALING
struct stealRequestStruct
{
    uint stealingDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    double stealingDeviceExecutionRate;
    ushort shouldMultiAssign;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 5
    } fieldCount;

    stealRequestStruct()
    : stealingDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , stealingDeviceExecutionRate(0)
    , shouldMultiAssign(1)
    {}
    
    stealRequestStruct(uint pStealingDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, double pStealingDeviceExecutionRate, bool pShouldMultiAssign)
    : stealingDeviceGlobalIndex(pStealingDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , stealingDeviceExecutionRate(pStealingDeviceExecutionRate)
    , shouldMultiAssign(pShouldMultiAssign)
    {}
};
#else
struct stealRequestStruct
{
    uint stealingDeviceGlobalIndex;
    uint targetDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    double stealingDeviceExecutionRate;
    ushort shouldMultiAssign;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 6
    } fieldCount;

    stealRequestStruct()
    : stealingDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , targetDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , stealingDeviceExecutionRate(0)
    , shouldMultiAssign(1)
    {}
    
    stealRequestStruct(uint pStealingDeviceGlobalIndex, uint pTargetDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, double pStealingDeviceExecutionRate, bool pShouldMultiAssign)
    : stealingDeviceGlobalIndex(pStealingDeviceGlobalIndex)
    , targetDeviceGlobalIndex(pTargetDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , stealingDeviceExecutionRate(pStealingDeviceExecutionRate)
    , shouldMultiAssign(pShouldMultiAssign)
    {}
};
#endif

enum stealResponseType
{
    STEAL_SUCCESS_RESPONSE,
    STEAL_FAILURE_RESPONSE
};

struct stealResponseStruct
{
    uint stealingDeviceGlobalIndex;
    uint targetDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ushort success;			// enum stealResponseType
    ulong startSubtask;
    ulong endSubtask;
    uint originalAllotteeGlobalIndex;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 8
    } fieldCount;
    
    stealResponseStruct()
    : stealingDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , targetDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , success(std::numeric_limits<ushort>::max())
    , startSubtask(std::numeric_limits<ulong>::max())
    , endSubtask(std::numeric_limits<ulong>::max())
    , originalAllotteeGlobalIndex(std::numeric_limits<uint>::max())
    {}

    stealResponseStruct(uint pStealingDeviceGlobalIndex, uint pTargetDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, stealResponseType pSuccess, ulong pStartSubtask, ulong pEndSubtask, uint pOriginalAllotteeGlobalIndex)
    : stealingDeviceGlobalIndex(pStealingDeviceGlobalIndex)
    , targetDeviceGlobalIndex(pTargetDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , success((ushort)pSuccess)
    , startSubtask(pStartSubtask)
    , endSubtask(pEndSubtask)
    , originalAllotteeGlobalIndex(pOriginalAllotteeGlobalIndex)
    {}
};

struct ownershipChangeStruct
{
    ulong offset;
    ulong length;
    uint newOwnerHost;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 3
    } fieldCount;
    
    ownershipChangeStruct()
    : offset(std::numeric_limits<ulong>::max())
    , length(std::numeric_limits<ulong>::max())
    , newOwnerHost(std::numeric_limits<uint>::max())
    {}

    ownershipChangeStruct(ulong pOffset, ulong pLength, uint pNewOwnerHost)
    : offset(pOffset)
    , length(pLength)
    , newOwnerHost(pNewOwnerHost)
    {}
};

struct scatteredOwnershipChangeStruct
{
    ulong offset;
    ulong size;
    ulong step;
    ulong count;
    uint newOwnerHost;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 5
    } fieldCount;
    
    scatteredOwnershipChangeStruct()
    : offset(std::numeric_limits<ulong>::max())
    , size(std::numeric_limits<ulong>::max())
    , step(std::numeric_limits<ulong>::max())
    , count(std::numeric_limits<ulong>::max())
    , newOwnerHost(std::numeric_limits<uint>::max())
    {}

    scatteredOwnershipChangeStruct(ulong pOffset, ulong pSize, ulong pStep, ulong pCount, uint pNewOwnerHost)
    : offset(pOffset)
    , size(pSize)
    , step(pStep)
    , count(pCount)
    , newOwnerHost(pNewOwnerHost)
    {}
};

struct ownershipTransferPacked
{
    memoryIdentifierStruct memIdentifier;
    uint transferDataElements;
    std::shared_ptr<std::vector<ownershipChangeStruct> > transferData;

    ownershipTransferPacked()
    : memIdentifier()
    , transferDataElements(0)
    {}
    
    ownershipTransferPacked(pmAddressSpace* pAddressSpace, std::shared_ptr<std::vector<ownershipChangeStruct> >& pChangeData);
};

struct scatteredOwnershipTransferPacked
{
    memoryIdentifierStruct memIdentifier;
    uint transferDataElements;
    std::shared_ptr<std::vector<scatteredOwnershipChangeStruct> > transferData;

    scatteredOwnershipTransferPacked()
    : memIdentifier()
    , transferDataElements(0)
    {}
    
    scatteredOwnershipTransferPacked(pmAddressSpace* pAddressSpace, std::shared_ptr<std::vector<scatteredOwnershipChangeStruct> >& pChangeData);
};
    
enum memoryTransferType
{
    TRANSFER_SCATTERED,
    TRANSFER_GENERAL
};

struct memoryTransferRequest
{
    memoryIdentifierStruct sourceMemIdentifier;
    memoryIdentifierStruct destMemIdentifier;
    ushort transferType;    // enum memoryTransferType
    ulong receiverOffset;
    ulong offset;
    ulong length;
    ulong step;             // used for scattered transfer
    ulong count;            // Used for scattered transfer
    uint destHost;			// Host that will receive the memory (generally same as the requesting host)
    ushort isForwarded;     // Signifies a forwarded memory request. Transfer is made directly from owner host to requesting host.
    ushort isTaskOriginated;    // Tells whether a task has demanded this memory or user has explicitly requested it
    uint originatingHost;   // Valid only if isTaskOriginated is true
    ulong sequenceNumber;	// Valid only if isTaskOriginated is true; sequence number of local task object (on originating host)
    ushort priority;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 14
    } fieldCount;

    memoryTransferRequest()
    : sourceMemIdentifier()
    , destMemIdentifier()
    , transferType(TRANSFER_GENERAL)
    , receiverOffset(std::numeric_limits<ulong>::max())
    , offset(std::numeric_limits<ulong>::max())
    , length(std::numeric_limits<ulong>::max())
    , step(std::numeric_limits<ulong>::max())
    , count(std::numeric_limits<ulong>::max())
    , destHost(std::numeric_limits<uint>::max())
    , isForwarded(std::numeric_limits<ushort>::max())
    , isTaskOriginated(std::numeric_limits<ushort>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , priority(std::numeric_limits<ushort>::max())
    {}
    
    memoryTransferRequest(const memoryIdentifierStruct& pSourceStruct, const memoryIdentifierStruct& pDestStruct, memoryTransferType pTransferType, ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, uint pDestHost, ushort pIsForwarded, ushort pIsTaskOriginated, uint pOriginatingHost, ulong pSequenceNumber, ushort pPriority)
    : sourceMemIdentifier(pSourceStruct)
    , destMemIdentifier(pDestStruct)
    , transferType(pTransferType)
    , receiverOffset(pReceiverOffset)
    , offset(pOffset)
    , length(pLength)
    , step(pStep)
    , count(pCount)
    , destHost(pDestHost)
    , isForwarded(pIsForwarded)
    , isTaskOriginated(pIsTaskOriginated)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , priority(pPriority)
    {
        DEBUG_EXCEPTION_ASSERT(transferType == TRANSFER_GENERAL || (length != 0 && step != 0 && count != 0));
    }
};

struct scatteredMemoryTransferRequestCombinedStruct
{
    ulong receiverOffset;
    ulong offset;
    ulong length;
    ulong step;
    ulong count;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 5
    } fieldCount;

    scatteredMemoryTransferRequestCombinedStruct()
    : receiverOffset(std::numeric_limits<ulong>::max())
    , offset(std::numeric_limits<ulong>::max())
    , length(std::numeric_limits<ulong>::max())
    , step(std::numeric_limits<ulong>::max())
    , count(std::numeric_limits<ulong>::max())
    {}
    
    scatteredMemoryTransferRequestCombinedStruct(ulong pReceiverOffset, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
    : receiverOffset(pReceiverOffset)
    , offset(pOffset)
    , length(pLength)
    , step(pStep)
    , count(pCount)
    {
        DEBUG_EXCEPTION_ASSERT(length != 0 && step != 0 && count != 0);
    }
};

struct scatteredMemoryTransferRequestCombinedPacked
{
    memoryIdentifierStruct sourceMemIdentifier;
    memoryIdentifierStruct destMemIdentifier;
    uint destHost;			// Host that will receive the memory (generally same as the requesting host)
    ushort isTaskOriginated;    // Tells whether a task has demanded this memory or user has explicitly requested it
    uint originatingHost;   // Valid only if isTaskOriginated is true
    ulong sequenceNumber;	// Valid only if isTaskOriginated is true; sequence number of local task object (on originating host)
    ushort priority;
    ushort count;       // Count of elements in vector of scatteredMemoryTransferRequestCombinedStruct
    finalize_ptr<std::vector<scatteredMemoryTransferRequestCombinedStruct>> requestData;
    
    scatteredMemoryTransferRequestCombinedPacked()
    : sourceMemIdentifier()
    , destMemIdentifier()
    , destHost(std::numeric_limits<uint>::max())
    , isTaskOriginated(std::numeric_limits<ushort>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , priority(std::numeric_limits<ushort>::max())
    , count(0)
    {}

    scatteredMemoryTransferRequestCombinedPacked(const memoryIdentifierStruct& pSourceStruct, const memoryIdentifierStruct& pDestStruct, uint pDestHost, ushort pIsTaskOriginated, uint pOriginatingHost, ulong pSequenceNumber, ushort pPriority, finalize_ptr<std::vector<scatteredMemoryTransferRequestCombinedStruct>>& pRequestAutoPtr)
    : sourceMemIdentifier(pSourceStruct)
    , destMemIdentifier(pDestStruct)
    , destHost(pDestHost)
    , isTaskOriginated(pIsTaskOriginated)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , priority(pPriority)
    , count(pRequestAutoPtr->size())
    , requestData(std::move(pRequestAutoPtr))
    {}
};
    
struct shadowMemTransferStruct
{
    uint writeOnlyUnprotectedPageRangesCount;
    uint subtaskMemLength;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 2
    } fieldCount;

    shadowMemTransferStruct()
    : writeOnlyUnprotectedPageRangesCount(0)
    , subtaskMemLength(0)
    {}
    
    shadowMemTransferStruct(uint pWriteOnlyUnprotectedPageRangesCount, uint pSubtaskMemLength)
    : writeOnlyUnprotectedPageRangesCount(pWriteOnlyUnprotectedPageRangesCount)
    , subtaskMemLength(pSubtaskMemLength)
    {}
};
    
struct shadowMemTransferPacked
{
    shadowMemTransferStruct shadowMemData;
    finalize_ptr<char, deleteArrayDeallocator<char> > shadowMem; // writeOnlyMemUnprotectedPageRanges followed by output mem write subscription only
    
    shadowMemTransferPacked()
    : shadowMemData()
    {}
    
    shadowMemTransferPacked(uint pWriteOnlyUnprotectedPageRangesCount, uint pSubtaskMemLength)
    : shadowMemData(pWriteOnlyUnprotectedPageRangesCount, pSubtaskMemLength)
    , shadowMem((pWriteOnlyUnprotectedPageRangesCount + pSubtaskMemLength) ? new char[pWriteOnlyUnprotectedPageRangesCount + pSubtaskMemLength] : NULL)
    {}
};
    
struct noReductionReqdStruct
{
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 2
    } fieldCount;
    
    noReductionReqdStruct()
    : originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    {}

    noReductionReqdStruct(pmTask* pTask);
};

struct subtaskReduceStruct
{
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ulong subtaskId;
    uint shadowMemsCount;
    uint scratchBuffer1Length;  // PRE_SUBTASK_TO_POST_SUBTASK scratch buffer
    uint scratchBuffer2Length;  // SUBTASK_TO_POST_SUBTASK scratch buffer
    uint scratchBuffer3Length;  // REDUCTION_TO_REDUCTION scratch buffer

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 7
    } fieldCount;
    
    subtaskReduceStruct()
    : originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , subtaskId(std::numeric_limits<ulong>::max())
    , shadowMemsCount(0)
    , scratchBuffer1Length(0)
    , scratchBuffer2Length(0)
    , scratchBuffer3Length(0)
    {}

    subtaskReduceStruct(uint pOriginatingHost, ulong pSequenceNumber, ulong pSubtaskId, uint pShadowMemsCount, uint pScratchBuffer1Length, uint pScratchBuffer2Length, uint pScratchBuffer3Length)
    : originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , subtaskId(pSubtaskId)
    , shadowMemsCount(pShadowMemsCount)
    , scratchBuffer1Length(pScratchBuffer1Length)
    , scratchBuffer2Length(pScratchBuffer2Length)
    , scratchBuffer3Length(pScratchBuffer3Length)
    {}
};

struct subtaskReducePacked
{
    subtaskReduceStruct reduceStruct;
    std::vector<shadowMemTransferPacked> shadowMems;
    finalize_ptr<char, deleteArrayDeallocator<char>> scratchBuffer1;
    std::function<void (char*)> scratchBuffer1Receiver;   // Takes a mem and unpacks PRE_SUBTASK_TO_POST_SUBTASK scratch buffer into it.
    finalize_ptr<char, deleteArrayDeallocator<char>> scratchBuffer2;
    std::function<void (char*)> scratchBuffer2Receiver;   // Takes a mem and unpacks SUBTASK_TO_POST_SUBTASK scratch buffer into it.
    finalize_ptr<char, deleteArrayDeallocator<char>> scratchBuffer3;
    std::function<void (char*)> scratchBuffer3Receiver;   // Takes a mem and unpacks REDUCTION_TO_REDUCTION scratch buffer into it.

    subtaskReducePacked()
    : reduceStruct()
    {}

    subtaskReducePacked(pmExecutionStub* pReducingStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
};

struct memoryReceiveStruct
{
    uint memOwnerHost;
    ulong generationNumber;
    ushort transferType;        // enum memoryTransferType
    ulong offset;
    ulong length;
    ulong step;
    ulong count;
    ushort isTaskOriginated;    // Tells whether a task has demanded this memory or user has explicitly requested it
    uint originatingHost;       // Valid only if isTaskOriginated is true
    ulong sequenceNumber;       // Valid only if isTaskOriginated is true; sequence number of local task object (on originating host)
    int mpiTag;                 // MPI tag of the upcoming message that contains actual memory
    uint senderHost;            // Id of the host sending this message (memory can come from forwarded messages, this is the host that is actually transmitting memory)

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 12
    } fieldCount;

    memoryReceiveStruct()
    : memOwnerHost(std::numeric_limits<uint>::max())
    , generationNumber(0)
    , transferType(TRANSFER_GENERAL)
    , offset(0)
    , length(0)
    , step(std::numeric_limits<ulong>::max())
    , count(std::numeric_limits<ulong>::max())
    , isTaskOriginated(true)
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , mpiTag(0)
    , senderHost(std::numeric_limits<uint>::max())
    {}

    memoryReceiveStruct(uint pMemOwnerHost, ulong pGenerationNumber, ulong pOffset, ulong pLength, ushort pIsTaskOriginated, uint pOriginatingHost, ulong pSequenceNumber, int pMpiTag, uint pSenderHost)
    : memOwnerHost(pMemOwnerHost)
    , generationNumber(pGenerationNumber)
    , transferType(TRANSFER_GENERAL)
    , offset(pOffset)
    , length(pLength)
    , step(std::numeric_limits<ulong>::max())
    , count(std::numeric_limits<ulong>::max())
    , isTaskOriginated(pIsTaskOriginated)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , mpiTag(pMpiTag)
    , senderHost(pSenderHost)
    {}

    memoryReceiveStruct(uint pMemOwnerHost, ulong pGenerationNumber, ulong pOffset, ulong pLength, ulong pStep, ulong pCount, ushort pIsTaskOriginated, uint pOriginatingHost, ulong pSequenceNumber, int pMpiTag, uint pSenderHost)
    : memOwnerHost(pMemOwnerHost)
    , generationNumber(pGenerationNumber)
    , transferType(TRANSFER_SCATTERED)
    , offset(pOffset)
    , length(pLength)
    , step(pStep)
    , count(pCount)
    , isTaskOriginated(pIsTaskOriginated)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , mpiTag(pMpiTag)
    , senderHost(pSenderHost)
    {}
    
    memoryReceiveStruct(const memoryReceiveStruct& pStruct)
    : memOwnerHost(pStruct.memOwnerHost)
    , generationNumber(pStruct.generationNumber)
    , transferType(pStruct.transferType)
    , offset(pStruct.offset)
    , length(pStruct.length)
    , step(pStruct.step)
    , count(pStruct.count)
    , isTaskOriginated(pStruct.isTaskOriginated)
    , originatingHost(pStruct.originatingHost)
    , sequenceNumber(pStruct.sequenceNumber)
    , mpiTag(pStruct.mpiTag)
    , senderHost(pStruct.senderHost)
    {}
};

struct hostFinalizationStruct
{
    ushort terminate;   // firstly all machines send to master with terminate false; then master sends to all machines with terminate true

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 1
    } fieldCount;
    
    hostFinalizationStruct()
    : terminate(std::numeric_limits<ushort>::max())
    {}

    hostFinalizationStruct(bool pTerminate)
    : terminate((ushort)pTerminate)
    {}
};

struct redistributionOrderStruct
{
    uint order;
    ulong length;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 2
    } fieldCount;
    
    redistributionOrderStruct()
    : order(std::numeric_limits<uint>::max())
    , length(0)
    {}

    redistributionOrderStruct(uint pOrder, ulong pLength)
    : order(pOrder)
    , length(pLength)
    {}
};

struct dataRedistributionStruct
{
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    uint remoteHost;
    ulong subtasksAccounted;
    uint orderDataCount;
    uint addressSpaceIndex;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 6
    } fieldCount;
    
    dataRedistributionStruct()
    : originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , remoteHost(std::numeric_limits<uint>::max())
    , subtasksAccounted(0)
    , orderDataCount(0)
    , addressSpaceIndex(std::numeric_limits<uint>::max())
    {}

    dataRedistributionStruct(uint pOriginatingHost, ulong pSequenceNumber, uint pRemoteHost, ulong pSubtasksAccounted, uint pOrderDataCount, uint pAddressSpaceIndex)
    : originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , remoteHost(pRemoteHost)
    , subtasksAccounted(pSubtasksAccounted)
    , orderDataCount(pOrderDataCount)
    , addressSpaceIndex(pAddressSpaceIndex)
    {}
};

struct dataRedistributionPacked
{
    dataRedistributionStruct redistributionStruct;
    finalize_ptr<std::vector<redistributionOrderStruct>> redistributionData;

    dataRedistributionPacked()
    : redistributionStruct()
    {}
    
    dataRedistributionPacked(pmTask* pTask, uint pAddressSpaceIndex, finalize_ptr<std::vector<redistributionOrderStruct>>& pRedistributionAutoPtr);
};

struct redistributionOffsetsStruct
{
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ulong redistributedMemGenerationNumber;
    uint offsetsDataCount;
    uint addressSpaceIndex;
    
    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 5
    } fieldCount;
    
    redistributionOffsetsStruct()
    : originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(0)
    , redistributedMemGenerationNumber(std::numeric_limits<ulong>::max())
    , offsetsDataCount(0)
    , addressSpaceIndex(0)
    {}

    redistributionOffsetsStruct(uint pOriginatingHost, ulong pSequenceNumber, ulong pRedistributedMemGenerationNumber, uint pOffsetsDataCount, uint pAddressSpaceIndex)
    : originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , redistributedMemGenerationNumber(pRedistributedMemGenerationNumber)
    , offsetsDataCount(pOffsetsDataCount)
    , addressSpaceIndex(pAddressSpaceIndex)
    {}
};

struct redistributionOffsetsPacked
{
    redistributionOffsetsStruct redistributionStruct;
    finalize_ptr<std::vector<ulong>> offsetsData;

    redistributionOffsetsPacked()
    {}

    redistributionOffsetsPacked(pmTask* pTask, uint pAddressSpaceIndex, finalize_ptr<std::vector<ulong>>& pOffsetsDataAutoPtr, pmAddressSpace* pRedistributedAddressSpace);
};

struct subtaskRangeCancelStruct
{
    uint targetDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    ulong startSubtask;
    ulong endSubtask;
    uint originalAllotteeGlobalIndex;

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 6
    } fieldCount;
    
    subtaskRangeCancelStruct()
    : targetDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , startSubtask(std::numeric_limits<ulong>::max())
    , endSubtask(std::numeric_limits<ulong>::max())
    , originalAllotteeGlobalIndex(std::numeric_limits<uint>::max())
    {}

    subtaskRangeCancelStruct(uint pTargetDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, ulong pStartSubtask, ulong pEndSubtask, uint pOriginalAllotteeGlobalIndex)
    : targetDeviceGlobalIndex(pTargetDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , startSubtask(pStartSubtask)
    , endSubtask(pEndSubtask)
    , originalAllotteeGlobalIndex(pOriginalAllotteeGlobalIndex)
    {}
};

enum fileOperations
{
    MMAP_FILE,
    MUNMAP_FILE,
    MMAP_ACK,
    MUNMAP_ACK
};

struct fileOperationsStruct
{
    char fileName[MAX_FILE_SIZE_LEN];
    ushort fileOp;  // enum fileOperations
    uint sourceHost;    // host to which ack needs to be sent

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 3
    } fieldCount;

    fileOperationsStruct()
    : fileOp(std::numeric_limits<ushort>::max())
    , sourceHost(std::numeric_limits<uint>::max())
    {
        fileName[0] = '\0';
    }
    
    fileOperationsStruct(const char* pFileName, fileOperations pFileOp, uint pSourceHost)
    : fileOp((ushort)pFileOp)
    , sourceHost(pSourceHost)
    {
        if(strlen(pFileName) >= MAX_FILE_SIZE_LEN)
            PMTHROW(pmFatalErrorException());
        
        strcpy(fileName, pFileName);
    }
};

struct multiFileOperationsStruct
{
    ushort fileOp;  // enum fileOperations
    uint sourceHost;    // host to which ack needs to be sent
    ulong userId;    // identifier used to bind received ack to the correct request
    uint fileCount;
    uint totalLength;   // length in bytes of all file names

    typedef enum fieldCount
    {
        FIELD_COUNT_VALUE = 5
    } fieldCount;

    multiFileOperationsStruct()
    : fileOp(std::numeric_limits<ushort>::max())
    , sourceHost(std::numeric_limits<uint>::max())
    , userId(std::numeric_limits<ulong>::max())
    , fileCount(std::numeric_limits<uint>::max())
    , totalLength(std::numeric_limits<uint>::max())
    {
    }
    
    multiFileOperationsStruct(fileOperations pFileOp, uint pSourceHost, ulong pUserId, uint pFileCount, uint pTotalLength)
    : fileOp((ushort)pFileOp)
    , sourceHost(pSourceHost)
    , userId(pUserId)
    , fileCount(pFileCount)
    , totalLength(pTotalLength)
    {
    }
};
    
struct multiFileOperationsPacked
{
    multiFileOperationsStruct multiFileOpsStruct;
    
    finalize_ptr<ushort, deleteArrayDeallocator<ushort>> fileNameLengthsArray;  // MAX_FILE_SIZE_LEN
    finalize_ptr<char, deleteArrayDeallocator<char>> fileNames;
    
    multiFileOperationsPacked()
    {}
    
    multiFileOperationsPacked(fileOperations pFileOp, uint pSourceHost, ulong pUserId)
    : multiFileOpsStruct(pFileOp, pSourceHost, pUserId, 0, 0)
    {}

    multiFileOperationsPacked(fileOperations pFileOp, uint pSourceHost, ulong pUserId, uint pFileCount, uint pTotalLength, finalize_ptr<ushort, deleteArrayDeallocator<ushort>>&& pFileNameLengthsArray, finalize_ptr<char, deleteArrayDeallocator<char>>&& pFileNames)
    : multiFileOpsStruct(pFileOp, pSourceHost, pUserId, pFileCount, pTotalLength)
    , fileNameLengthsArray(std::move(pFileNameLengthsArray))
    , fileNames(std::move(pFileNames))
    {}
};
    
struct affinityDataTransferPacked
{
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    memoryIdentifierStruct affinityAddressSpace;
    ulong affinityAddressSpaceLength;
    uint transferDataElements;
    finalize_ptr<ulong, deleteArrayDeallocator<ulong>> logicalToPhysicalSubtaskMapping;

    affinityDataTransferPacked()
    : originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , affinityAddressSpaceLength(std::numeric_limits<ulong>::max())
    , transferDataElements(0)
    {}
    
    affinityDataTransferPacked(uint pOriginatingHost, ulong pSequenceNumber, uint pMemOwnerHost, ulong pGenerationNumber, ulong pAffinityAddressSpaceLength, uint pElementCount, finalize_ptr<ulong, deleteArrayDeallocator<ulong>>&& pLogicalToPhysicalSubtaskMapping)
    : originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , affinityAddressSpace(pMemOwnerHost, pGenerationNumber)
    , affinityAddressSpaceLength(pAffinityAddressSpaceLength)
    , transferDataElements(pElementCount)
    , logicalToPhysicalSubtaskMapping(std::move(pLogicalToPhysicalSubtaskMapping))
    {}
};

#ifdef USE_AFFINITY_IN_STEAL
struct stealSuccessDiscontiguousPacked
{
    uint stealingDeviceGlobalIndex;
    uint targetDeviceGlobalIndex;
    uint originatingHost;
    ulong sequenceNumber;	// sequence number of local task object (on originating host)
    uint stealDataElements;
    std::vector<ulong> discontiguousStealData;
    
    stealSuccessDiscontiguousPacked()
    : stealingDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , targetDeviceGlobalIndex(std::numeric_limits<uint>::max())
    , originatingHost(std::numeric_limits<uint>::max())
    , sequenceNumber(std::numeric_limits<ulong>::max())
    , stealDataElements(0)
    {}
    
    stealSuccessDiscontiguousPacked(uint pStealingDeviceGlobalIndex, uint pTargetDeviceGlobalIndex, uint pOriginatingHost, ulong pSequenceNumber, uint pElementCount, std::vector<ulong>&& pDiscontiguousStealData)
    : stealingDeviceGlobalIndex(pStealingDeviceGlobalIndex)
    , targetDeviceGlobalIndex(pTargetDeviceGlobalIndex)
    , originatingHost(pOriginatingHost)
    , sequenceNumber(pSequenceNumber)
    , stealDataElements(pElementCount)
    , discontiguousStealData(std::move(pDiscontiguousStealData))
    {}
};
#endif
    
}

/**
 * \brief The communicator class of PMLIB. Controllers on different machines talk through communicator.
 * This class is implemented over MPI and is the only class in PMLIB that provides communication between
 * different machines. All PMLIB components (like scheduler) talk to pmController which sends pmCommands
 * to other pmControllers using pmCommunicator's API. pmCommunicator only allows pmCommand objects to be
 * sent or received by various pmControllers. This is a per machine singleton class i.e. only one instance
 * of pmCommunicator exists on each machine.
*/

class pmCommunicator : public pmBase
{
    friend class pmHeavyOperationsThread;

    public:
		static pmCommunicator* GetCommunicator();

        void Send(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);
		void Receive(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);	// If no source is provided, any machine is assumed (MPI_ANY)
        void Broadcast(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);
		void All2All(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);
		
        void SendMemory(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);
        void ReceiveMemory(pmCommunicatorCommandPtr& pCommand, bool pBlocking = false);

    private:
		pmCommunicator();

		void SendPacked(pmCommunicatorCommandPtr&& pCommand, bool pBlocking = false);
};

} // end namespace pm

#endif


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

#include "pmCommand.h"
#include "pmTask.h"
#include "pmSignalWait.h"
#include "pmMemSection.h"
#include "pmCallbackUnit.h"
#include "pmHardware.h"
#include "pmNetwork.h"

namespace pm
{

/* class pmCommand */
pmCommand::pmCommand(ushort pPriority, ushort pCommandType, void* pCommandData /* = NULL */, ulong pDataLength /* = 0 */, pmCommandCompletionCallback pCallback /* = NULL */)
{
	mPriority = pPriority;
	mCommandType = pCommandType;
	mCommandData = pCommandData;
	mDataLength = pDataLength;
	mCallback = pCallback;
	mStatus = pmStatusUnavailable;
}

pmCommand::~pmCommand()
{
}

ushort pmCommand::GetType()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mCommandType;
}

void* pmCommand::GetData()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mCommandData;
}

ulong pmCommand::GetDataLength()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mDataLength;
}

pmStatus pmCommand::GetStatus()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mStatus;
}

ushort pmCommand::GetPriority()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mStatus;
}

pmCommandCompletionCallback pmCommand::GetCommandCompletionCallback()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mCallback;
}

pmStatus pmCommand::SetData(void* pCommandData, ulong pDataLength)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mCommandData = pCommandData;
	mDataLength = pDataLength;
	
	return pmSuccess;
}

pmStatus pmCommand::SetStatus(pmStatus pStatus)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mStatus = pStatus;

	return pmSuccess;
}

pmStatus pmCommand::SetCommandCompletionCallback(pmCommandCompletionCallback pCallback)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mCallback = pCallback;

	return pmSuccess;
}

pmStatus pmCommand::WaitForFinish()
{
    pmSignalWait* lSignalWait = NULL;
    pmStatus lStatus = pmStatusUnavailable;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if((lStatus = mStatus) == pmStatusUnavailable)
        {
            if((lSignalWait = mSignalWait.get_ptr()) == NULL)
            {
                lSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();
                mSignalWait.reset(lSignalWait);
            }
        }
    }

    if(lSignalWait)
    {
        lSignalWait->Wait();
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        return mStatus;
    }
    
	return lStatus;
}

bool pmCommand::WaitWithTimeOut(ulong pTriggerTime)
{
    pmSignalWait* lSignalWait = NULL;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if(mStatus == pmStatusUnavailable)
        {
            if((lSignalWait = mSignalWait.get_ptr()) == NULL)
            {
                lSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();
                mSignalWait.reset(lSignalWait);
            }
        }
    }

    if(lSignalWait)
        return lSignalWait->WaitWithTimeOut(pTriggerTime);
    
	return false;
}

pmStatus pmCommand::MarkExecutionStart()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mTimer.Start();

	return pmSuccess;
}

pmStatus pmCommand::MarkExecutionEnd(pmStatus pStatus, pmCommandPtr pSharedPtr)
{
	assert(pSharedPtr.get() == this);

	// Auto Lock/Unlock Scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

		mTimer.Stop();

		mStatus = pStatus;

		if(mSignalWait.get_ptr())
			mSignalWait->Signal();
	}

	if(mCallback)
		mCallback(pSharedPtr);

	return pmSuccess;
}

double pmCommand::GetExecutionTimeInSecs()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	double lTime = mTimer.GetElapsedTimeInSecs();

	return lTime;
}


/* class pmCommunicatorCommand */
pmCommunicatorCommand::pmCommunicatorCommand(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
	void* pCommandData, ulong pDataLength, void* pSecondaryData /* = NULL */, ulong pSecondaryDataLength /* = 0 */, pmCommandCompletionCallback pCallback /* = NULL */)
	: pmCommand(pPriority, (ushort)pCommandType, pCommandData, pDataLength, pCallback)
{
	mCommandTag = pCommandTag;
	mDestination = pDestination;
	mDataType = pDataType;
	mSecondaryData = pSecondaryData;
	mSecondaryDataLength = pSecondaryDataLength;
}

pmCommunicatorCommandPtr pmCommunicatorCommand::CreateSharedPtr(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
	void* pCommandData, ulong pDataUnits, void* pSecondaryData /* = NULL */, ulong pSecondaryDataUnits /* = 0 */, pmCommandCompletionCallback pCallback /* = NULL */)
{
	pmCommunicatorCommandPtr lSharedPtr(new pmCommunicatorCommand(pPriority, pCommandType, pCommandTag, pDestination, pDataType, pCommandData, pDataUnits, pSecondaryData, pSecondaryDataUnits, pCallback));
	return lSharedPtr;
}


pmStatus pmCommunicatorCommand::SetTag(communicatorCommandTags pTag)
{
	mCommandTag = pTag;

	return pmSuccess;
}

pmStatus pmCommunicatorCommand::SetSecondaryData(void* pSecondaryData, ulong pSecondaryLength)
{
	mSecondaryData = pSecondaryData;
	mSecondaryDataLength = pSecondaryLength;

	return pmSuccess;
}

pmCommunicatorCommand::communicatorCommandTags pmCommunicatorCommand::GetTag()
{
	return mCommandTag;
}

pmHardware* pmCommunicatorCommand::GetDestination()
{
	return mDestination;
}

pmCommunicatorCommand::communicatorDataTypes pmCommunicatorCommand::GetDataType()
{
	return mDataType;
}

bool pmCommunicatorCommand::IsValid()
{
	if(mCommandType >= MAX_COMMUNICATOR_COMMAND_TYPES)
		return false;

	return true;
}

void* pmCommunicatorCommand::GetSecondaryData()
{
	return mSecondaryData;
}

ulong pmCommunicatorCommand::GetSecondaryDataLength()
{
	return mSecondaryDataLength;
}


/* class pmPersistentCommunicatorCommand */
pmPersistentCommunicatorCommand::pmPersistentCommunicatorCommand(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType,
	void* pCommandData, ulong pDataUnits, void* pSecondaryData /* = NULL */, ulong pSecondaryDataUnits /* = 0 */, pmCommandCompletionCallback pCallback /* = NULL */)
	: pmCommunicatorCommand(pPriority, pCommandType, pCommandTag, pDestination, pDataType, pCommandData, pDataUnits, pSecondaryData, pSecondaryDataUnits, pCallback)
{
}

pmPersistentCommunicatorCommand::~pmPersistentCommunicatorCommand()
{
}

pmPersistentCommunicatorCommandPtr pmPersistentCommunicatorCommand::CreateSharedPtr(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
	void* pCommandData, ulong pDataUnits, void* pSecondaryData /* = NULL */, ulong pSecondaryDataUnits /* = 0 */, pmCommandCompletionCallback pCallback /* = NULL */)
{
	pmPersistentCommunicatorCommandPtr lSharedPtr(new pmPersistentCommunicatorCommand(pPriority, pCommandType, pCommandTag, pDestination, pDataType, pCommandData, pDataUnits, pSecondaryData, pSecondaryDataUnits, pCallback));
	return lSharedPtr;
}


/* class pmTaskCommand */
bool pmTaskCommand::IsValid()
{
	if(mCommandType >= MAX_TASK_COMMAND_TYPES)
		return false;

	return true;
}

pmTaskCommandPtr pmTaskCommand::CreateSharedPtr(ushort pPriority, ushort pCommandType, void* pCommandData /* = NULL */, ulong pDataLength /* = 0 */)
{
	pmTaskCommandPtr lSharedPtr(new pmTaskCommand(pPriority, pCommandType, pCommandData, pDataLength));
	return lSharedPtr;
}


/* class pmSubtaskRangeCommand */
bool pmSubtaskRangeCommand::IsValid()
{
	if(mCommandType >= MAX_TASK_COMMAND_TYPES)
		return false;

	return true;
}

pmSubtaskRangeCommandPtr pmSubtaskRangeCommand::CreateSharedPtr(ushort pPriority, ushort pCommandType, void* pCommandData /* = NULL */, ulong pDataLength /* = 0 */)
{
	pmSubtaskRangeCommandPtr lSharedPtr(new pmSubtaskRangeCommand(pPriority, pCommandType, pCommandData, pDataLength));
	return lSharedPtr;
}


/* struct pmCommunicatorCommand::remoteTaskAssignStruct */
pmCommunicatorCommand::remoteTaskAssignStruct::remoteTaskAssignStruct(pmLocalTask* pLocalTask)
{
	if(!pLocalTask)
		return;

	pmMemSection* lInputSection = pLocalTask->GetMemSectionRO();
	pmMemSection* lOutputSection = pLocalTask->GetMemSectionRW();

	taskConfLength = pLocalTask->GetTaskConfigurationLength();
	taskId = pLocalTask->GetTaskId();
	inputMemLength = lInputSection?((ulong)(lInputSection->GetLength())):0;
	outputMemLength = lOutputSection?((ulong)(lOutputSection->GetLength())):0;
    
    inputMemInfo = lInputSection->GetMemInfo();
    outputMemInfo = lOutputSection->GetMemInfo();
    
	subtaskCount = pLocalTask->GetSubtaskCount();
	assignedDeviceCount = pLocalTask->GetAssignedDeviceCount();
	originatingHost = *(pLocalTask->GetOriginatingHost());
    sequenceNumber = pLocalTask->GetSequenceNumber();
	priority = pLocalTask->GetPriority();
	schedModel = (ushort)(pLocalTask->GetSchedulingModel());
	inputMemGenerationNumber = lInputSection?lInputSection->GetGenerationNumber():0;
	outputMemGenerationNumber = lOutputSection?lOutputSection->GetGenerationNumber():0;

    flags = 0;
    if(pLocalTask->IsMultiAssignEnabled())
        flags |= TASK_MULTI_ASSIGN_FLAG_VAL;

	strncpy(callbackKey, pLocalTask->GetCallbackUnit()->GetKey(), MAX_CB_KEY_LEN-1);
	callbackKey[MAX_CB_KEY_LEN-1] = '\0';
}

pmCommunicatorCommand::remoteTaskAssignStruct::remoteTaskAssignStruct()
{
	memset(this, 0, sizeof(*this));
}

pmCommunicatorCommand::remoteTaskAssignPacked::remoteTaskAssignPacked(pmLocalTask* pLocalTask /* = NULL */) : taskStruct(pLocalTask)
{
	if(!pLocalTask)
	{
        memset(this, 0, sizeof(*this));

		return;
	}

	taskConf.ptr = pLocalTask->GetTaskConfiguration();
	taskConf.length = pLocalTask->GetTaskConfigurationLength();

	// Transfer device list if the task scehduling model is pull or if reduction callback is defined
	if(taskStruct.assignedDeviceCount != 0 && (pLocalTask->GetSchedulingModel() == scheduler::PULL || pLocalTask->GetCallbackUnit()->GetDataReductionCB()))
	{
		uint* lDeviceArray = new uint[taskStruct.assignedDeviceCount];

		std::vector<pmProcessingElement*>& lDevices = pLocalTask->GetAssignedDevices();

		for(size_t i=0; i<taskStruct.assignedDeviceCount; ++i)
			lDeviceArray[i] = lDevices[i]->GetGlobalDeviceIndex();

		devices.ptr = lDeviceArray;
		devices.length = (uint)(sizeof(uint)) * taskStruct.assignedDeviceCount;
	}
	else
	{
		devices.ptr = NULL;
		devices.length = 0;
	}
}

pmCommunicatorCommand::remoteTaskAssignPacked::~remoteTaskAssignPacked()
{
	delete[] (uint*)(devices.ptr);
}

/* struct pmCommunicatorCommand::subtaskReducePacked */
pmCommunicatorCommand::subtaskReducePacked::subtaskReducePacked()
{
	memset(this, 0, sizeof(*this));
}

pmCommunicatorCommand::subtaskReducePacked::subtaskReducePacked(pmExecutionStub* pReducingStub, pmTask* pTask, ulong pSubtaskId)
{
	pmMachine* lOriginatingHost = pTask->GetOriginatingHost();
	this->reduceStruct.sequenceNumber = pTask->GetSequenceNumber();

	pmSubscriptionInfo lSubscriptionInfo;

	this->reduceStruct.originatingHost = *(lOriginatingHost);
	this->reduceStruct.subtaskId = pSubtaskId;

	if(pTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pReducingStub, pSubtaskId, lSubscriptionInfo))
    {
        void* lShadowMem = pTask->GetSubscriptionManager().GetSubtaskShadowMem(pReducingStub, pSubtaskId);
        this->subtaskMem.length = (uint)lSubscriptionInfo.length;
        this->subtaskMem.ptr = lShadowMem;
		this->reduceStruct.subscriptionOffset = lSubscriptionInfo.offset;
    }
	else
    {
        this->subtaskMem.length = 0;
        this->subtaskMem.ptr = NULL;
		this->reduceStruct.subscriptionOffset = 0;
    }
}

pmCommunicatorCommand::subtaskReducePacked::~subtaskReducePacked()
{
}


/* struct pmCommunicatorCommand::ownershipTransferPacked */
pmCommunicatorCommand::ownershipTransferPacked::ownershipTransferPacked()
{
	memset(this, 0, sizeof(*this));
}
    
pmCommunicatorCommand::ownershipTransferPacked::ownershipTransferPacked(pmMemSection* pMemSection, std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> >& pChangeData)
{
    this->memIdentifier.memOwnerHost = *(pMemSection->GetMemOwnerHost());
    this->memIdentifier.generationNumber = pMemSection->GetGenerationNumber();
    this->transferData = pChangeData;
}

pmCommunicatorCommand::ownershipTransferPacked::~ownershipTransferPacked()
{
}
 
    
/* struct pmCommunicatorCommand::memoryReceivePacked */
pmCommunicatorCommand::memoryReceivePacked::memoryReceivePacked()
{
	memset(this, 0, sizeof(*this));
}

pmCommunicatorCommand::memoryReceivePacked::memoryReceivePacked(uint pMemOwnerHost, ulong pGenerationNumber, ulong pOffset, ulong pLength, void* pMemPtr)
{
	this->receiveStruct.memOwnerHost = pMemOwnerHost;
	this->receiveStruct.generationNumber = pGenerationNumber;
	this->receiveStruct.offset = pOffset;
	this->receiveStruct.length = pLength;
	this->mem.ptr = pMemPtr;
	this->mem.length = (uint)pLength;
}

pmCommunicatorCommand::memoryReceivePacked::~memoryReceivePacked()
{
}

    
/* struct pmCommunicatorCommand::sendAcknowledgementPacked */    
pmCommunicatorCommand::sendAcknowledgementPacked::sendAcknowledgementPacked()
{
	memset(this, 0, sizeof(*this));
}
    
pmCommunicatorCommand::sendAcknowledgementPacked::sendAcknowledgementPacked(pmProcessingElement* pSourceDevice, pmSubtaskRange& pRange, pmCommunicatorCommand::ownershipDataStruct* pOwnershipData, uint pCount, pmStatus pExecStatus)
{
    this->ackStruct.sourceDeviceGlobalIndex = pSourceDevice->GetGlobalDeviceIndex();
    this->ackStruct.originatingHost = *(pRange.task->GetOriginatingHost());
    this->ackStruct.sequenceNumber = pRange.task->GetSequenceNumber();
    this->ackStruct.startSubtask = pRange.startSubtask;
    this->ackStruct.endSubtask = pRange.endSubtask;
    this->ackStruct.execStatus = (uint)pExecStatus;
    this->ackStruct.originalAllotteeGlobalIndex = (pRange.originalAllottee ? pRange.originalAllottee->GetGlobalDeviceIndex() : pSourceDevice->GetGlobalDeviceIndex());
    this->ackStruct.ownershipDataElements = pCount;
    this->ownershipData = pOwnershipData;
}

pmCommunicatorCommand::sendAcknowledgementPacked::~sendAcknowledgementPacked()
{
}

    
/* struct pmCommunicatorCommand::dataRedistributionPacked */
pmCommunicatorCommand::dataRedistributionPacked::dataRedistributionPacked()
{
	memset(this, 0, sizeof(*this));
}
    
pmCommunicatorCommand::dataRedistributionPacked::dataRedistributionPacked(pmTask* pTask, redistributionOrderStruct* pRedistributionData, uint pCount)
{
    this->redistributionStruct.originatingHost = *(pTask->GetOriginatingHost());
    this->redistributionStruct.sequenceNumber = pTask->GetSequenceNumber();
    this->redistributionStruct.remoteHost = *PM_LOCAL_MACHINE;
    this->redistributionStruct.subtasksAccounted = pTask->GetSubtasksExecuted();
    this->redistributionStruct.orderDataCount = pCount;
    this->redistributionData = pRedistributionData;
}
    
pmCommunicatorCommand::dataRedistributionPacked::~dataRedistributionPacked()
{
}

    
/* struct memoryIdebtifierStruct */
bool operator==(pmCommunicatorCommand::memoryIdentifierStruct& pIdentifier1, pmCommunicatorCommand::memoryIdentifierStruct& pIdentifier2)
{
    return (pIdentifier1.memOwnerHost == pIdentifier2.memOwnerHost && pIdentifier1.generationNumber == pIdentifier2.generationNumber);
}
    
} // end namespace pm



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

#include "pmNetwork.h"
#include "pmScheduler.h"
#include "pmCommand.h"
#include "pmResourceLock.h"
#include "pmDevicePool.h"
#include "pmCluster.h"
#include "pmHardware.h"
#include "pmCallbackUnit.h"
#include "pmStubManager.h"
#include "pmHeavyOperations.h"
#include "pmLogger.h"

namespace pm
{

using namespace network;
using namespace communicator;

pmCluster* PM_GLOBAL_CLUSTER = NULL;

//#define ENABLE_MPI_DEBUG_HOOK

#ifdef ENABLE_MPI_DEBUG_HOOK

#include <unistd.h>

void __mpi_debug_hook();
void __mpi_debug_hook()
{
	volatile bool hook = true;
	char hostname[256];

	gethostname(hostname, sizeof(hostname));
	std::cout << " MPI debug hook for process " << getpid() << " on host " << hostname << std::endl << std::flush;

	while(hook)
		sleep(5);
}

#define MPI_DEBUG_HOOK __mpi_debug_hook();
#else
#define MPI_DEBUG_HOOK
#endif

#ifdef DUMP_MPI_CALLS
bool __dump_mpi_call(const char* name, int line);
bool __dump_mpi_call(const char* name, int line)
{
    char lStr[512];
    sprintf(lStr, "MPI Call: %s (%s:%d)", name, __FILE__, line);
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, lStr);

	return true;
}

#define MPI_CALL(name, call) (__dump_mpi_call(name, __LINE__) && call)
#else
#define MPI_CALL(name, call) (call)
#endif

#define SAFE_GET_MACHINE_POOL(x) { x = pmMachinePool::GetMachinePool(); if(!x) PMTHROW(pmFatalErrorException()); }
#define SAFE_GET_MPI_COMMUNICATOR(x, y) \
	{ \
		const CLUSTER_IMPLEMENTATION_CLASS* dCluster = dynamic_cast<const CLUSTER_IMPLEMENTATION_CLASS*>(y); \
		if(dCluster) \
		{ \
			x = dCluster->GetCommunicator(); \
		} \
		else \
		{ \
			x = MPI_COMM_WORLD; \
		} \
	}
#define SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(x, y, z) \
	{ \
		const pmMachine* dMachine = dynamic_cast<const pmMachine*>(z); \
		if(dMachine) \
		{ \
			uint lId = *dMachine; \
			if(lId > __MAX(int)) \
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_MACHINES)); \
			x = MPI_COMM_WORLD; \
			y = lId; \
		} \
		else \
		{ \
			const CLUSTER_IMPLEMENTATION_CLASS* dCluster = dynamic_cast<const CLUSTER_IMPLEMENTATION_CLASS*>(z); \
			if(dCluster) \
			{ \
				x = dCluster->GetCommunicator(); \
				y = dCluster->GetRankInCommunicator(dMachine); \
			} \
			else \
			{ \
				x = MPI_COMM_WORLD; \
				y = MPI_ANY_SOURCE; \
			} \
		} \
	}
#define SAFE_GET_MPI_ADDRESS(x, y) \
	{ \
		if( MPI_CALL("MPI_Get_address", (MPI_Get_address(x, y) != MPI_SUCCESS)) ) \
			PMTHROW(pmFatalErrorException()); \
	}

/* class pmNetwork */
pmNetwork::pmNetwork()
{
}

pmNetwork::~pmNetwork()
{
}


/* class pmMPI */
pmNetwork* pmMPI::GetNetwork()
{
	static pmMPI lNetwork;
    return &lNetwork;
}

pmMPI::pmMPI()
    : pmNetwork()
    , mTotalHosts(0)
    , mHostId(0)
    , mDummyRequestInitiated(false)
    , mResourceLock __LOCK_NAME__("pmMPI::mResourceLock")
    , mDataTypesResourceLock __LOCK_NAME__("pmMPI::mDataTypesResourceLock")
	, mThreadTerminationFlag(false)
    , mReceiveThread(NULL)
    , mMPITypesAllocator(1)
{
	int lThreadSupport = 0, lMpiStatus = 0;

//	if(MPI_CALL("MPI_Init", ((lMpiStatus = MPI_Init(NULL, NULL)) != MPI_SUCCESS)))
//        PMTHROW(pmNetworkException(pmNetworkException::INIT_ERROR));
    
    if(MPI_CALL("MPI_Init_thread", ((lMpiStatus = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &lThreadSupport)) != MPI_SUCCESS))
       || lThreadSupport != MPI_THREAD_MULTIPLE)
	{
		MPI_CALL("MPI_Abort", MPI_Abort(MPI_COMM_WORLD, lMpiStatus));
		
        mTotalHosts = 0;
		mHostId = 0;

        PMTHROW(pmNetworkException(pmNetworkException::INIT_ERROR));
	}

	int lHosts = 0, lId = 0;

	MPI_CALL("MPI_Comm_size", MPI_Comm_size(MPI_COMM_WORLD, &lHosts));
	MPI_CALL("MPI_Comm_rank", MPI_Comm_rank(MPI_COMM_WORLD, &lId));

	pmLogger::GetLogger()->SetHostId(lId);

	mTotalHosts = lHosts;
	mHostId = lId;

	MPI_CALL("MPI_Comm_set_errhandler", MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

	PM_GLOBAL_CLUSTER = new pmClusterMPI();
    
    MPI_DEBUG_HOOK
    
	SwitchThread(std::shared_ptr<networkEvent>(new networkEvent()), MAX_PRIORITY_LEVEL);

	mReceiveThread = new pmUnknownLengthReceiveThread(this);
}

pmMPI::~pmMPI()
{
#ifdef SERIALIZE_DEFERRED_LOGS
    #ifdef ENABLE_ACCUMULATED_TIMINGS
        pmAccumulatedTimesSorter::GetAccumulatedTimesSorter()->FlushLogs();
    #endif

    if(mHostId == 0)
        pmLogger::GetLogger()->PrintDeferredLog();
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GlobalBarrier();

    uint lMachines = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
    for(uint i = 1; i < lMachines; ++i)
    {
        if(mHostId == 0)
        {
            MPI_Status lStatus;

            uint lLogLength = 0;
            if( MPI_CALL("MPI_Recv", (MPI_Recv(&lLogLength, 1, MPI_UNSIGNED, i, DEFERRED_LOG_LENGTH_TAG, MPI_COMM_WORLD, &lStatus) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
        
            if(lLogLength && ((lLogLength + 1) > lLogLength))   // Check for overflow and wrap around
            {
                std::unique_ptr<char> lLogAutoPtr(new char[lLogLength + 1]);
                
                if( MPI_CALL("MPI_Recv", (MPI_Recv(lLogAutoPtr.get(), lLogLength, MPI_CHAR, i, DEFERRED_LOG_TAG, MPI_COMM_WORLD, &lStatus) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
                
                (lLogAutoPtr.get())[lLogLength] = '\0';
                std::cerr << lLogAutoPtr.get() << std::endl;
            }
        }
        else if(mHostId == i)
        {
            const std::string& lDeferredLog = pmLogger::GetLogger()->GetDeferredLogStream().str();
            uint lLogLength = (uint)lDeferredLog.size();

            if( MPI_CALL("MPI_Send", (MPI_Send(&lLogLength, 1, MPI_UNSIGNED, 0, DEFERRED_LOG_LENGTH_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));

            if(lLogLength && ((lLogLength + 1) > lLogLength))   // Check for overflow and wrap around
            {
                char* lLog = const_cast<char*>(lDeferredLog.c_str());
                if( MPI_CALL("MPI_Send", (MPI_Send(lLog, lLogLength, MPI_CHAR, 0, DEFERRED_LOG_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
            }
            
            pmLogger::GetLogger()->ClearDeferredLog();
        }

        NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GlobalBarrier();
    }
#endif

	StopThreadExecution();

	delete mReceiveThread;

	delete static_cast<pmClusterMPI*>(PM_GLOBAL_CLUSTER);

	if( MPI_CALL("MPI_Finalize", (MPI_Finalize() != MPI_SUCCESS)) )
        PMTHROW(pmNetworkException(pmNetworkException::FINALIZE_ERROR));

    #ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down network thread");
	#endif
}

pmCommunicatorCommandPtr pmMPI::PackData(pmCommunicatorCommandPtr& pCommand)
{
	finalize_ptr<char, deleteArrayDeallocator<char>> lPackedDataAutoPtr;
	uint lTag = pCommand->GetTag();
	ulong lLength = sizeof(uint);

    int lPos = 0;
    MPI_Comm lCommunicator;
    SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());

	switch(pCommand->GetDataType())
	{
		case REMOTE_TASK_ASSIGN_PACKED:
		{
			remoteTaskAssignPacked* lData = (remoteTaskAssignPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);

			remoteTaskAssignStruct& lTaskStruct = lData->taskStruct;
			lLength += sizeof(lTaskStruct) + lData->taskStruct.taskConfLength + lData->taskStruct.taskMemCount * sizeof(taskMemoryStruct);
            
            if(lData->devices.get_ptr())
                lLength += lData->taskStruct.assignedDeviceCount * sizeof(uint);

			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTaskStruct, 1, GetDataTypeMPI(REMOTE_TASK_ASSIGN_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if(lData->taskConf.get_ptr())
			{
				if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->taskConf.get_ptr(), lData->taskStruct.taskConfLength, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			}

            if(!lData->taskMem.empty())
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(&(lData->taskMem)[0], lData->taskStruct.taskMemCount, GetDataTypeMPI(TASK_MEMORY_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

			if(lData->devices.get_ptr())
			{
				if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->devices.get_ptr(), lData->taskStruct.assignedDeviceCount, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			}
            
            lLength = lPos;

			break;
		}

		case SUBTASK_REDUCE_PACKED:
		{
			subtaskReducePacked* lData = (subtaskReducePacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
        
			subtaskReduceStruct& lStruct = lData->reduceStruct;
			lLength += sizeof(lStruct) + sizeof(shadowMemTransferStruct) * lStruct.shadowMemsCount;
            
            for(uint i = 0; i < lStruct.shadowMemsCount; ++i)
                lLength += lData->shadowMems[i].shadowMemData.subtaskMemLength;
            
            lLength += lData->reduceStruct.scratchBuffer1Length + lData->reduceStruct.scratchBuffer2Length + lData->reduceStruct.scratchBuffer3Length;
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(SUBTASK_REDUCE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

            if(lStruct.shadowMemsCount)
            {
                for(auto& lShadowMemTransfer: lData->shadowMems)
                {
                    if( MPI_CALL("MPI_Pack", (MPI_Pack(&lShadowMemTransfer.shadowMemData, 1, GetDataTypeMPI(SHADOW_MEM_TRANSFER_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                        PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

                    if(lShadowMemTransfer.shadowMemData.subtaskMemLength)
                    {
                        if( MPI_CALL("MPI_Pack", (MPI_Pack(lShadowMemTransfer.shadowMem.get_ptr(), lShadowMemTransfer.shadowMemData.subtaskMemLength, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                            PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
                    }
                }
            }

            if(lData->reduceStruct.scratchBuffer1Length)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->scratchBuffer1.get_ptr(), lData->reduceStruct.scratchBuffer1Length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            if(lData->reduceStruct.scratchBuffer2Length)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->scratchBuffer2.get_ptr(), lData->reduceStruct.scratchBuffer2Length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            if(lData->reduceStruct.scratchBuffer3Length)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->scratchBuffer3.get_ptr(), lData->reduceStruct.scratchBuffer3Length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            lLength = lPos;

			break;
		}

		case MEMORY_RECEIVE_PACKED:
		{
			memoryReceivePacked* lData = (memoryReceivePacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);

			memoryReceiveStruct& lStruct = lData->receiveStruct;
            ulong lDataLength = lData->receiveStruct.length * ((lData->receiveStruct.transferType == TRANSFER_GENERAL) ? 1 : lData->receiveStruct.count);
			lLength += sizeof(lStruct) + lDataLength;

			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(MEMORY_RECEIVE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if(lDataLength != 0)
			{
                if(lData->receiveStruct.transferType == TRANSFER_GENERAL)
                {
                    if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->mDataProducer(0), (int)lData->receiveStruct.length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                        PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
                }
                else
                {
                    DEBUG_EXCEPTION_ASSERT(lData->receiveStruct.transferType == TRANSFER_SCATTERED);
                    DEBUG_EXCEPTION_ASSERT(lData->mDataProducer);

                    for(ulong i = 0; i < lData->receiveStruct.count; ++i)
                    {
                        if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->mDataProducer(i), (int)lData->receiveStruct.length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                            PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
                    }
                }
			}

            lLength = lPos;

			break;
		}

		case DATA_REDISTRIBUTION_PACKED:
		{
			dataRedistributionPacked* lData = (dataRedistributionPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
            
            dataRedistributionStruct& lStruct = lData->redistributionStruct;
			lLength += sizeof(lStruct) + lData->redistributionStruct.orderDataCount * sizeof(redistributionOrderStruct);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(DATA_REDISTRIBUTION_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->redistributionData.get_ptr(), lData->redistributionStruct.orderDataCount, GetDataTypeMPI(REDISTRIBUTION_ORDER_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}

		case REDISTRIBUTION_OFFSETS_PACKED:
		{
			redistributionOffsetsPacked* lData = (redistributionOffsetsPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
            
			redistributionOffsetsStruct& lStruct = lData->redistributionStruct;
			lLength += sizeof(lStruct) + lData->redistributionStruct.offsetsDataCount * sizeof(ulong);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(REDISTRIBUTION_OFFSETS_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&(*lData->offsetsData.get_ptr())[0], lData->redistributionStruct.offsetsDataCount, MPI_UNSIGNED_LONG, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}

		case SEND_ACKNOWLEDGEMENT_PACKED:
		{
			sendAcknowledgementPacked* lData = (sendAcknowledgementPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
            
			sendAcknowledgementStruct& lStruct = lData->ackStruct;
			lLength += sizeof(lStruct) + lData->ackStruct.ownershipDataElements * sizeof(ownershipDataStruct) + lData->ackStruct.addressSpaceIndices * sizeof(uint);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(SEND_ACKNOWLEDGEMENT_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

            if(lData->ackStruct.ownershipDataElements)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(&(lData->ownershipVector[0]), (int)lData->ackStruct.ownershipDataElements, GetDataTypeMPI(OWNERSHIP_DATA_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }
            
            if(lData->ackStruct.addressSpaceIndices)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(&(lData->addressSpaceIndexVector[0]), (int)lData->ackStruct.addressSpaceIndices, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            lLength = lPos;
            
			break;
		}

		case OWNERSHIP_TRANSFER_PACKED:
		{
			ownershipTransferPacked* lData = (ownershipTransferPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
            
            DEBUG_EXCEPTION_ASSERT((uint)(lData->transferData->size()) == lData->transferDataElements);

			memoryIdentifierStruct& lStruct = lData->memIdentifier;
			lLength += sizeof(lStruct) + sizeof(uint) + lData->transferDataElements * sizeof(ownershipChangeStruct);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(MEMORY_IDENTIFIER_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lData->transferDataElements, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
        
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&(*lData->transferData.get())[0], lData->transferDataElements, GetDataTypeMPI(OWNERSHIP_CHANGE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}
        
        case MULTI_FILE_OPERATIONS_PACKED:
        {
			multiFileOperationsPacked* lData = (multiFileOperationsPacked*)(pCommand->GetData());
            EXCEPTION_ASSERT(lData);
            
			multiFileOperationsStruct& lStruct = lData->multiFileOpsStruct;
			lLength += sizeof(lStruct) + lStruct.fileCount * sizeof(uint) + lStruct.totalLength * sizeof(char);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
            lPackedDataAutoPtr.reset(new char[lLength]);
			char* lPackedData = lPackedDataAutoPtr.get_ptr();
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(MULTI_FILE_OPERATIONS_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            if(lStruct.fileCount)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->fileNameLengthsArray.get_ptr(), lStruct.fileCount, MPI_UNSIGNED_SHORT, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }
            
            if(lStruct.totalLength)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->fileNames.get_ptr(), lStruct.totalLength, MPI_CHAR, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }
            
            lLength = lPos;
            
            break;
        }

		default:
			PMTHROW(pmFatalErrorException());
	}

    return pmCommunicatorCommand<char, deleteArrayDeallocator<char>>::CreateSharedPtr(pCommand->GetPriority(), (communicatorCommandTypes)(pCommand->GetType()), pCommand->GetTag(), pCommand->GetDestination(), pCommand->GetDataType(), lPackedDataAutoPtr, lLength, pCommand->GetCommandCompletionCallback(), pCommand->GetUserIdentifier());
}

pmCommunicatorCommandPtr pmMPI::UnpackData(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pDataLength)
{
    void* lReceivedData = pPackedData.get_ptr();

    int lPos = 0;
    uint lInternalTag = (uint)MAX_COMMUNICATOR_COMMAND_TAGS;
    if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &lInternalTag, 1, MPI_UNSIGNED, MPI_COMM_WORLD) != MPI_SUCCESS)) )
        PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

    communicatorDataTypes lDataType = MAX_COMMUNICATOR_DATA_TYPES;
    communicatorCommandTags lTag = (communicatorCommandTags)lInternalTag;
    
    switch(lTag)
    {
        case REMOTE_TASK_ASSIGNMENT_TAG:
            lDataType = REMOTE_TASK_ASSIGN_PACKED;
            break;

        case SUBTASK_REDUCE_TAG:
            lDataType = SUBTASK_REDUCE_PACKED;
            break;
        
        case MEMORY_RECEIVE_TAG:
            lDataType = MEMORY_RECEIVE_PACKED;
            break;
        
        case DATA_REDISTRIBUTION_TAG:
            lDataType = DATA_REDISTRIBUTION_PACKED;
            break;
            
        case REDISTRIBUTION_OFFSETS_TAG:
            lDataType = REDISTRIBUTION_OFFSETS_PACKED;
            break;
            
        case SEND_ACKNOWLEDGEMENT_TAG:
            lDataType = SEND_ACKNOWLEDGEMENT_PACKED;
            break;
        
        case OWNERSHIP_TRANSFER_TAG:
            lDataType = OWNERSHIP_TRANSFER_PACKED;
            break;
        
        case MULTI_FILE_OPERATIONS_TAG:
            lDataType = MULTI_FILE_OPERATIONS_PACKED;
            break;

        default:
            PMTHROW(pmFatalErrorException());
    }

    pmCommunicatorCommandPtr lCommand;
    MPI_Comm lCommunicator = MPI_COMM_WORLD;    // When different communicators will be supported in future, then it should be explictly put in data preamble like tag

    pmCommandCompletionCallbackType lCompletionCallback = pmScheduler::GetScheduler()->GetSchedulerCommandCompletionCallback();

	switch(lDataType)
	{
		case REMOTE_TASK_ASSIGN_PACKED:
		{
			finalize_ptr<remoteTaskAssignPacked> lPackedData(new remoteTaskAssignPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->taskStruct), 1, GetDataTypeMPI(REMOTE_TASK_ASSIGN_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

			if(lPackedData->taskStruct.taskConfLength != 0)
			{
				lPackedData->taskConf.reset(new char[lPackedData->taskStruct.taskConfLength]);
				
				if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lPackedData->taskConf.get_ptr(), lPackedData->taskStruct.taskConfLength, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
			}
            
            if(lPackedData->taskStruct.taskMemCount != 0)
            {
                lPackedData->taskMem.resize(lPackedData->taskStruct.taskMemCount);

                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->taskMem[0]), lPackedData->taskStruct.taskMemCount, GetDataTypeMPI(TASK_MEMORY_STRUCT), lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }

			if(lPackedData->taskStruct.assignedDeviceCount)
			{
                lPackedData->devices.reset(new uint[lPackedData->taskStruct.assignedDeviceCount]);

                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lPackedData->devices.get_ptr(), lPackedData->taskStruct.assignedDeviceCount, MPI_UNSIGNED, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
			}
            
            lCommand = pmCommunicatorCommand<remoteTaskAssignPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);

			break;
		}

		case SUBTASK_REDUCE_PACKED:
		{
            finalize_ptr<subtaskReducePacked> lPackedData(new subtaskReducePacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->reduceStruct), 1, GetDataTypeMPI(SUBTASK_REDUCE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

            subtaskReduceStruct& lStruct = lPackedData->reduceStruct;

            if(lStruct.shadowMemsCount)
            {
                lPackedData->shadowMems.resize(lStruct.shadowMemsCount);

                for(auto& lShadowMemTransfer: lPackedData->shadowMems)
                {
                    if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &lShadowMemTransfer.shadowMemData, 1, GetDataTypeMPI(SHADOW_MEM_TRANSFER_STRUCT), lCommunicator) != MPI_SUCCESS)) )
                        PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

                    if(lShadowMemTransfer.shadowMemData.subtaskMemLength)
                    {
                        lShadowMemTransfer.shadowMem.reset(new char[lShadowMemTransfer.shadowMemData.subtaskMemLength]);

                        if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lShadowMemTransfer.shadowMem.get_ptr(), lShadowMemTransfer.shadowMemData.subtaskMemLength, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                            PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                    }
                }
            }
            
            if(lStruct.scratchBuffer1Length)
            {
                uint lScratchBuffer1Length = lStruct.scratchBuffer1Length;
                std::shared_ptr<finalize_ptr<char, deleteArrayDeallocator<char>>> lReceivedDataSharedPtr(new finalize_ptr<char, deleteArrayDeallocator<char>>(std::move(pPackedData)));
                std::function<void (char*)> lFunc([lReceivedDataSharedPtr, pDataLength, lPos, lCommunicator, lScratchBuffer1Length] (char* pMem) mutable
                                                         {
                                                             if( MPI_CALL("MPI_Unpack", (MPI_Unpack((void*)lReceivedDataSharedPtr->get_ptr(), pDataLength, &lPos, pMem, (int)lScratchBuffer1Length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                                                                 PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                                                         });
                
                lPackedData->scratchBuffer1Receiver = lFunc;
            }
            
            if(lStruct.scratchBuffer2Length)
            {
                uint lScratchBuffer2Length = lStruct.scratchBuffer2Length;
                std::shared_ptr<finalize_ptr<char, deleteArrayDeallocator<char>>> lReceivedDataSharedPtr(new finalize_ptr<char, deleteArrayDeallocator<char>>(std::move(pPackedData)));
                std::function<void (char*)> lFunc([lReceivedDataSharedPtr, pDataLength, lPos, lCommunicator, lScratchBuffer2Length] (char* pMem) mutable
                                                         {
                                                             if( MPI_CALL("MPI_Unpack", (MPI_Unpack((void*)lReceivedDataSharedPtr->get_ptr(), pDataLength, &lPos, pMem, (int)lScratchBuffer2Length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                                                                 PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                                                         });
                
                lPackedData->scratchBuffer2Receiver = lFunc;
            }

            if(lStruct.scratchBuffer3Length)
            {
                uint lScratchBuffer3Length = lStruct.scratchBuffer3Length;
                std::shared_ptr<finalize_ptr<char, deleteArrayDeallocator<char>>> lReceivedDataSharedPtr(new finalize_ptr<char, deleteArrayDeallocator<char>>(std::move(pPackedData)));
                std::function<void (char*)> lFunc([lReceivedDataSharedPtr, pDataLength, lPos, lCommunicator, lScratchBuffer3Length] (char* pMem) mutable
                                                         {
                                                             if( MPI_CALL("MPI_Unpack", (MPI_Unpack((void*)lReceivedDataSharedPtr->get_ptr(), pDataLength, &lPos, pMem, (int)lScratchBuffer3Length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                                                                 PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                                                         });
                
                lPackedData->scratchBuffer3Receiver = lFunc;
            }

            lCommand = pmCommunicatorCommand<subtaskReducePacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);

			break;
		}

		case MEMORY_RECEIVE_PACKED:
		{
            finalize_ptr<memoryReceivePacked> lPackedData(new memoryReceivePacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->receiveStruct), 1, GetDataTypeMPI(MEMORY_RECEIVE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

            ulong lDataLength = lPackedData->receiveStruct.length * ((lPackedData->receiveStruct.transferType == TRANSFER_GENERAL) ? 1 : lPackedData->receiveStruct.count);

			if(lDataLength != 0)
            {
                std::shared_ptr<finalize_ptr<char, deleteArrayDeallocator<char>>> lReceivedDataSharedPtr(new finalize_ptr<char, deleteArrayDeallocator<char>>(std::move(pPackedData)));
                std::function<void (char*, ulong)> lFunc([lReceivedDataSharedPtr, pDataLength, lPos, lCommunicator] (char* pMem, ulong pLength) mutable
                {
                    if( MPI_CALL("MPI_Unpack", (MPI_Unpack((void*)lReceivedDataSharedPtr->get_ptr(), pDataLength, &lPos, pMem, (int)pLength, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                        PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                });
                
                lPackedData->mDataReceiver = lFunc;
            }

            lCommand = pmCommunicatorCommand<memoryReceivePacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);

			break;
		}

		case DATA_REDISTRIBUTION_PACKED:
		{
            finalize_ptr<dataRedistributionPacked> lPackedData(new dataRedistributionPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->redistributionStruct), 1, GetDataTypeMPI(DATA_REDISTRIBUTION_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
			lPackedData->redistributionData.reset(new std::vector<redistributionOrderStruct>(lPackedData->redistributionStruct.orderDataCount));
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lPackedData->redistributionData.get_ptr(), lPackedData->redistributionStruct.orderDataCount, GetDataTypeMPI(REDISTRIBUTION_ORDER_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand = pmCommunicatorCommand<dataRedistributionPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);
            
			break;
		}

		case REDISTRIBUTION_OFFSETS_PACKED:
		{
            finalize_ptr<redistributionOffsetsPacked> lPackedData(new redistributionOffsetsPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->redistributionStruct), 1, GetDataTypeMPI(REDISTRIBUTION_OFFSETS_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
			lPackedData->offsetsData.reset(new std::vector<ulong>(lPackedData->redistributionStruct.offsetsDataCount));
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(*lPackedData->offsetsData.get_ptr())[0], lPackedData->redistributionStruct.offsetsDataCount, MPI_UNSIGNED_LONG, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand = pmCommunicatorCommand<redistributionOffsetsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);
            
			break;
		}

		case SEND_ACKNOWLEDGEMENT_PACKED:
		{
            finalize_ptr<sendAcknowledgementPacked> lPackedData(new sendAcknowledgementPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->ackStruct), 1, GetDataTypeMPI(SEND_ACKNOWLEDGEMENT_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

            if(lPackedData->ackStruct.ownershipDataElements)
            {
                lPackedData->ownershipVector.resize(lPackedData->ackStruct.ownershipDataElements);
                
                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->ownershipVector[0]), lPackedData->ackStruct.ownershipDataElements, GetDataTypeMPI(OWNERSHIP_DATA_STRUCT), lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }
            
            if(lPackedData->ackStruct.addressSpaceIndices)
            {
                lPackedData->addressSpaceIndexVector.resize(lPackedData->ackStruct.addressSpaceIndices);

                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->addressSpaceIndexVector[0]), lPackedData->ackStruct.addressSpaceIndices, MPI_UNSIGNED, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }

            lCommand = pmCommunicatorCommand<sendAcknowledgementPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);
            
			break;
		}

		case OWNERSHIP_TRANSFER_PACKED:
		{
            finalize_ptr<ownershipTransferPacked> lPackedData(new ownershipTransferPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->memIdentifier), 1, GetDataTypeMPI(MEMORY_IDENTIFIER_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
        
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->transferDataElements), 1, MPI_UNSIGNED, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

            lPackedData->transferData.reset(new std::vector<ownershipChangeStruct>(lPackedData->transferDataElements));
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(*lPackedData->transferData.get())[0], lPackedData->transferDataElements, GetDataTypeMPI(OWNERSHIP_CHANGE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand = pmCommunicatorCommand<ownershipTransferPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);
        
            break;
        }
            
        case MULTI_FILE_OPERATIONS_PACKED:
        {
            finalize_ptr<multiFileOperationsPacked> lPackedData(new multiFileOperationsPacked());

			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, &(lPackedData->multiFileOpsStruct), 1, GetDataTypeMPI(MULTI_FILE_OPERATIONS_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
        
            if(lPackedData->multiFileOpsStruct.fileCount)
            {
                lPackedData->fileNameLengthsArray.reset(new ushort[lPackedData->multiFileOpsStruct.fileCount]);
                
                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lPackedData->fileNameLengthsArray.get_ptr(), lPackedData->multiFileOpsStruct.fileCount, MPI_UNSIGNED_SHORT, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }

            if(lPackedData->multiFileOpsStruct.totalLength)
            {
                lPackedData->fileNames.reset(new char[lPackedData->multiFileOpsStruct.totalLength]);
                
                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(lReceivedData, pDataLength, &lPos, lPackedData->fileNames.get_ptr(), lPackedData->multiFileOpsStruct.totalLength, MPI_CHAR, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }
            
            lCommand = pmCommunicatorCommand<multiFileOperationsPacked>::CreateSharedPtr(MAX_CONTROL_PRIORITY, RECEIVE, lTag, NULL, lDataType, lPackedData, lPos, lCompletionCallback);
        
            break;
        }
        
		default:
			PMTHROW(pmFatalErrorException());
	}

	return lCommand;
}
    
bool pmMPI::IsUnknownLengthTag(communicatorCommandTags pTag)
{
    return (pTag == REMOTE_TASK_ASSIGNMENT_TAG ||
            pTag == SUBTASK_REDUCE_TAG ||
            pTag == MEMORY_RECEIVE_TAG ||
            pTag == DATA_REDISTRIBUTION_TAG ||
            pTag == REDISTRIBUTION_OFFSETS_TAG ||
            pTag == SEND_ACKNOWLEDGEMENT_TAG ||
            pTag == OWNERSHIP_TRANSFER_TAG ||
            pTag == MULTI_FILE_OPERATIONS_TAG);
}

void pmMPI::SendNonBlocking(pmCommunicatorCommandPtr& pCommand)
{
    void* lData = pCommand->GetData();
    ulong lLength = pCommand->GetDataUnits();

	if(!lData || lLength == 0)
		return;

#ifdef _DEBUG
	const pmHardware* lHardware = pCommand->GetDestination();

	if(!lHardware || !(dynamic_cast<const pmMachine*>(lHardware) || dynamic_cast<const pmProcessingElement*>(lHardware)))
		PMTHROW(pmFatalErrorException());
#endif

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	ulong lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	DEBUG_EXCEPTION_ASSERT(!pCommand->IsPersistent() || lBlocks == 0);
    
    pCommand->MarkExecutionStart();

	for(ulong i = 0; i < lBlocks; ++i)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + i *MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT);

	if(lLastBlockLength)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks *MPI_TRANSFER_MAX_LIMIT), (uint)lLastBlockLength);
}

/* MPI 2 currently does not support non-blocking collective messages */
void pmMPI::BroadcastNonBlocking(pmCommunicatorCommandPtr& pCommand)
{
	void* lData = pCommand->GetData();
	ulong lDataUnits = pCommand->GetDataUnits();

	if(!lData || lDataUnits == 0)
		PMTHROW(pmFatalErrorException());

	if(lDataUnits > MPI_TRANSFER_MAX_LIMIT)
		PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

	pCommand->MarkExecutionStart();

	MPI_Comm lCommunicator;
	int lRoot;
	SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lRoot, pCommand->GetDestination());

	MPI_Datatype lDataType = GetDataTypeMPI((communicatorDataTypes)(pCommand->GetDataType()));

	if( MPI_CALL("MPI_Bcast", (MPI_Bcast(lData, (int)lDataUnits, lDataType, lRoot, lCommunicator) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::BROADCAST_ERROR));

    pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(pCommand);
	pCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);
}

/* MPI 2 currently does not support non-blocking collective messages */
void pmMPI::All2AllNonBlocking(pmCommunicatorCommandPtr& pCommand)
{
    void* lSendData = NULL;
    void * lRecvData = NULL;
    ulong lSendUnits = 1;
    ulong lRecvUnits = 1;

    if(pCommand->GetTag() == MACHINE_POOL_TRANSFER_TAG)
    {
        all2AllWrapper<machinePool>* lWrapper = (all2AllWrapper<machinePool>*)pCommand->GetData();
        
        lSendData = &lWrapper->localData;
        lRecvData = &(lWrapper->all2AllData[0]);
    }

	if(!lSendData || !lRecvData)
		PMTHROW(pmFatalErrorException());

	if(lSendUnits > MPI_TRANSFER_MAX_LIMIT || lRecvUnits > MPI_TRANSFER_MAX_LIMIT)
		PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

	pCommand->MarkExecutionStart();

	MPI_Comm lCommunicator;
	SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());

	MPI_Datatype lDataType = GetDataTypeMPI((communicatorDataTypes)(pCommand->GetDataType()));

	if( MPI_CALL("MPI_Allgather", (MPI_Allgather(lSendData, (int)lSendUnits, lDataType, lRecvData, (int)lRecvUnits, lDataType, lCommunicator) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::ALL2ALL_ERROR));

    pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(pCommand);
	pCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);
}

void pmMPI::ReceiveNonBlocking(pmCommunicatorCommandPtr& pCommand)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataUnits();

	if(!lData || lLength == 0)
		return;

#ifdef _DEBUG
	// No hardware means receive from any machine
	const pmHardware* lHardware = pCommand->GetDestination();
	if(lHardware)
	{
		if(!(dynamic_cast<const pmMachine*>(lHardware) || dynamic_cast<const pmProcessingElement*>(lHardware)))
			PMTHROW(pmFatalErrorException());
	}
#endif

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	ulong lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	DEBUG_EXCEPTION_ASSERT(!pCommand->IsPersistent() || lBlocks == 0);

    pCommand->MarkExecutionStart();
    
	for(ulong i = 0; i < lBlocks; ++i)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT);

	if(lLastBlockLength)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), (uint)lLastBlockLength);
}

void pmMPI::SendNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength)
{
	MPI_Request* lRequest = NULL;
	MPI_Comm lCommunicator;
	int lDest;

	if(pCommand->IsPersistent())
	{
		lRequest = GetPersistentSendRequest(pCommand);
		if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
        
        DEBUG_EXCEPTION_ASSERT(!IsUnknownLengthTag(pCommand->GetTag()));
	}
	else
	{
		SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
		MPI_Datatype lDataType = GetDataTypeMPI((communicatorDataTypes)(pCommand->GetDataType()));
		
        communicatorCommandTags lTag = pCommand->GetTag();
        
        DEBUG_EXCEPTION_ASSERT(!IsUnknownLengthTag(lTag) || lDataType == MPI_PACKED);
        DEBUG_EXCEPTION_ASSERT(IsUnknownLengthTag(lTag) || lDataType != MPI_PACKED);
        
        if(IsUnknownLengthTag(lTag))
            lTag = UNKNOWN_LENGTH_TAG;
        
        if(pCommand->GetCommandCompletionCallback())
        {
            lRequest = (MPI_Request*)(mMPITypesAllocator.Allocate(sizeof(MPI_Request)));
            
            if( MPI_CALL("MPI_Isend", (MPI_Isend(pData, pLength, lDataType, lDest, (int)lTag, lCommunicator, lRequest) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
        }
        else
        {
            MPI_Request lActualRequest;
            
            if( MPI_CALL("MPI_Isend", (MPI_Isend(pData, pLength, lDataType, lDest, (int)lTag, lCommunicator, &lActualRequest) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
        }
	}

    // Not all send requests need to break the MPI_Waitany loop in network thread by completing a dummy command.
    // Only commands having the user callback registered need to be added to MPI_Waitany array.
    if(pCommand->GetCommandCompletionCallback())
    {
        EXCEPTION_ASSERT(lRequest);

        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        DEBUG_EXCEPTION_ASSERT(mNonBlockingRequestMap.find(lRequest) == mNonBlockingRequestMap.end());

        mNonBlockingRequestMap[lRequest] = pCommand;
        
        decltype(mRequestCountMap)::iterator lIter = mRequestCountMap.find(pCommand);
        if(lIter == mRequestCountMap.end())
            lIter = mRequestCountMap.emplace(std::piecewise_construct, std::forward_as_tuple(pCommand), std::forward_as_tuple(1)).first;
        else
            ++lIter->second;
        
        DEBUG_EXCEPTION_ASSERT(!pCommand->IsPersistent() || lIter->second == 1);

        CancelDummyRequest();	// Signal the other thread to handle the created request
    }
}

void pmMPI::ReceiveNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength)
{
	MPI_Request* lRequest = NULL;
	MPI_Comm lCommunicator;
	int lDest;

	if(pCommand->IsPersistent())
	{
		lRequest = GetPersistentRecvRequest(pCommand);
		if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
	}
	else
	{
		SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
		MPI_Datatype lDataType = GetDataTypeMPI((communicatorDataTypes)(pCommand->GetDataType()));

        lRequest = (MPI_Request*)(mMPITypesAllocator.Allocate(sizeof(MPI_Request)));
        
		if( MPI_CALL("MPI_Irecv", (MPI_Irecv(pData, pLength, lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
	}

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    DEBUG_EXCEPTION_ASSERT(mNonBlockingRequestMap.find(lRequest) == mNonBlockingRequestMap.end());
    
	mNonBlockingRequestMap[lRequest] = pCommand;
	
    decltype(mRequestCountMap)::iterator lIter = mRequestCountMap.find(pCommand);
	if(lIter == mRequestCountMap.end())
		lIter = mRequestCountMap.emplace(std::piecewise_construct, std::forward_as_tuple(pCommand), std::forward_as_tuple(1)).first;
	else
		++lIter->second;
    
    DEBUG_EXCEPTION_ASSERT(!pCommand->IsPersistent() || lIter->second == 1);

    if(!pCommand->IsPersistent())
        CancelDummyRequest();
}

void pmMPI::GlobalBarrier()
{
	if( MPI_CALL("MPI_Barrier", (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)) )
        PMTHROW(pmNetworkException(pmNetworkException::GLOBAL_BARRIER_ERROR));
}
    
void pmMPI::StartReceiving()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    CancelDummyRequest();
}

void pmMPI::InitializePersistentCommand(pmCommunicatorCommandPtr& pCommand)
{
    DEBUG_EXCEPTION_ASSERT(pCommand->IsPersistent());
    
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	MPI_Request* lRequest = (MPI_Request*)(mMPITypesAllocator.Allocate(sizeof(MPI_Request)));
    
	MPI_Comm lCommunicator;
	int lDest;

	communicatorCommandTypes lType = (communicatorCommandTypes)(pCommand->GetType());

	SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
	MPI_Datatype lDataType = GetDataTypeMPI((communicatorDataTypes)(pCommand->GetDataType()));

	if(lType == SEND)
	{
        DEBUG_EXCEPTION_ASSERT(mPersistentSendRequest.find(pCommand) == mPersistentSendRequest.end());

		if( MPI_CALL("MPI_Send_init", (MPI_Send_init(pCommand->GetData(), (uint)(pCommand->GetDataUnits()), lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));

		mPersistentSendRequest[pCommand] = lRequest;
	}
	else
	{
		if(lType == RECEIVE)
		{
            DEBUG_EXCEPTION_ASSERT(mPersistentRecvRequest.find(pCommand) == mPersistentRecvRequest.end());

			if( MPI_CALL("MPI_Recv_init", (MPI_Recv_init(pCommand->GetData(), (uint)(pCommand->GetDataUnits()), lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

			mPersistentRecvRequest[pCommand] = lRequest;
		}
	}
}

void pmMPI::TerminatePersistentCommand(pmCommunicatorCommandPtr& pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
	communicatorCommandTypes lType = (communicatorCommandTypes)(pCommand->GetType());
	MPI_Request* lRequest = NULL;

	if(lType == SEND)
    {
        decltype(mPersistentSendRequest)::iterator lIter = mPersistentSendRequest.find(pCommand);
        
        DEBUG_EXCEPTION_ASSERT(lIter != mPersistentSendRequest.end());

		lRequest = lIter->second;
        mPersistentSendRequest.erase(lIter);
    }
	else if(lType == RECEIVE)
    {
        decltype(mPersistentRecvRequest)::iterator lIter = mPersistentRecvRequest.find(pCommand);

        DEBUG_EXCEPTION_ASSERT(lIter != mPersistentRecvRequest.end());

		lRequest = lIter->second;
        mPersistentRecvRequest.erase(lIter);
    }

    mNonBlockingRequestMap.erase(lRequest);
    
    decltype(mRequestCountMap)::iterator lRequestCountIter = mRequestCountMap.find(pCommand);
    if(lRequestCountIter != mRequestCountMap.end())
    {
        --lRequestCountIter->second;
        
        DEBUG_EXCEPTION_ASSERT(lRequestCountIter->second == 0);

        mRequestCountMap.erase(pCommand);
    }

	if( MPI_CALL("MPI_Request_free", (MPI_Request_free(lRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::REQUEST_FREE_ERROR));

    mMPITypesAllocator.Deallocate(lRequest);
}

MPI_Request* pmMPI::GetPersistentSendRequest(pmCommunicatorCommandPtr& pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    return GetPersistentSendRequestInternal(pCommand);
}

MPI_Request* pmMPI::GetPersistentRecvRequest(pmCommunicatorCommandPtr& pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    return GetPersistentRecvRequestInternal(pCommand);
}

// Must be called with mResourceLock acquired
MPI_Request* pmMPI::GetPersistentSendRequestInternal(pmCommunicatorCommandPtr& pCommand)
{
	if(mPersistentSendRequest.find(pCommand) == mPersistentSendRequest.end())
		PMTHROW(pmFatalErrorException());

	return mPersistentSendRequest[pCommand];
}

// Must be called with mResourceLock acquired
MPI_Request* pmMPI::GetPersistentRecvRequestInternal(pmCommunicatorCommandPtr& pCommand)
{
	if(mPersistentRecvRequest.find(pCommand) == mPersistentRecvRequest.end())
		PMTHROW(pmFatalErrorException());

	return mPersistentRecvRequest[pCommand];
}

MPI_Datatype pmMPI::GetDataTypeMPI(communicatorDataTypes pDataType)
{
	switch(pDataType)
	{
		case BYTE:
			return MPI_BYTE;
			break;

		case INT:
			return MPI_INT;
			break;

		case UINT:
			return MPI_UNSIGNED;
			break;

		case REMOTE_TASK_ASSIGN_PACKED:
		case SUBTASK_REDUCE_PACKED:
		case MEMORY_RECEIVE_PACKED:
        case DATA_REDISTRIBUTION_PACKED:
        case REDISTRIBUTION_OFFSETS_PACKED:
        case SEND_ACKNOWLEDGEMENT_PACKED:
        case OWNERSHIP_TRANSFER_PACKED:
        case MULTI_FILE_OPERATIONS_PACKED:
			return MPI_PACKED;
			break;

		default:
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());

			std::map<communicatorDataTypes, MPI_Datatype*>::iterator lIter = mRegisteredDataTypes.find(pDataType);

			DEBUG_EXCEPTION_ASSERT(lIter != mRegisteredDataTypes.end());

            return *lIter->second;
		}
	}

	PMTHROW(pmFatalErrorException());
	return MPI_BYTE;
}


#define REGISTER_MPI_DATA_TYPE_HELPER_HEADER(cDataType, cName, headerMpiName) \
    cDataType cName; \
	MPI_Aint headerMpiName; \
	SAFE_GET_MPI_ADDRESS(&cName, &headerMpiName);

#define REGISTER_MPI_DATA_TYPE_HELPER(headerMpiName, cName, mpiName, mpiDataType, index, blockLength) \
    MPI_Aint mpiName = 0; \
	SAFE_GET_MPI_ADDRESS(&cName, &mpiName); \
	lBlockLength[index] = blockLength; \
	lDisplacement[index] = mpiName - headerMpiName; \
	lDataType[index] = mpiDataType;

void pmMPI::RegisterTransferDataType(communicatorDataTypes pDataType)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());
        
        EXCEPTION_ASSERT(mRegisteredDataTypes.find(pDataType) == mRegisteredDataTypes.end());
    }
#endif

	int lFieldCount = 0;

    switch(pDataType)
	{
		case MACHINE_POOL_STRUCT:
		{
			lFieldCount = machinePool::FIELD_COUNT_VALUE;
			break;
		}
            
		case DEVICE_POOL_STRUCT:
		{
			lFieldCount = devicePool::FIELD_COUNT_VALUE;
			break;
		}
            
		case MEMORY_IDENTIFIER_STRUCT:
		{
			lFieldCount = memoryIdentifierStruct::FIELD_COUNT_VALUE;
			break;
		}

		case MEMORY_DISTRIBUTION_STRUCT:
		{
			lFieldCount = memoryDistributionStruct::FIELD_COUNT_VALUE;
			break;
		}

		case TASK_MEMORY_STRUCT:
		{
			lFieldCount = taskMemoryStruct::FIELD_COUNT_VALUE;
			break;
		}

		case REMOTE_TASK_ASSIGN_STRUCT:
		{
			lFieldCount = remoteTaskAssignStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case REMOTE_SUBTASK_ASSIGN_STRUCT:
		{
			lFieldCount = remoteSubtaskAssignStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case OWNERSHIP_DATA_STRUCT:
		{
			lFieldCount = ownershipDataStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case OWNERSHIP_CHANGE_STRUCT:
		{
			lFieldCount = ownershipChangeStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case SEND_ACKNOWLEDGEMENT_STRUCT:
		{
			lFieldCount = sendAcknowledgementStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case TASK_EVENT_STRUCT:
		{
			lFieldCount = taskEventStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case STEAL_REQUEST_STRUCT:
		{
			lFieldCount = stealRequestStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case STEAL_RESPONSE_STRUCT:
		{
			lFieldCount = stealResponseStruct::FIELD_COUNT_VALUE;
			break;
		}

		case MEMORY_TRANSFER_REQUEST_STRUCT:
		{
			lFieldCount = memoryTransferRequest::FIELD_COUNT_VALUE;
			break;
		}
        
        case SHADOW_MEM_TRANSFER_STRUCT:
        {
            lFieldCount = shadowMemTransferStruct::FIELD_COUNT_VALUE;
            break;
        }

        case NO_REDUCTION_REQD_STRUCT:
        {
            lFieldCount = noReductionReqdStruct::FIELD_COUNT_VALUE;
            break;
        }

		case SUBTASK_REDUCE_STRUCT:
		{
			lFieldCount = subtaskReduceStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case MEMORY_RECEIVE_STRUCT:
		{
			lFieldCount = memoryReceiveStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case HOST_FINALIZATION_STRUCT:
		{
			lFieldCount = hostFinalizationStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case REDISTRIBUTION_ORDER_STRUCT:
		{
			lFieldCount = redistributionOrderStruct::FIELD_COUNT_VALUE;
			break;
		}

		case DATA_REDISTRIBUTION_STRUCT:
		{
			lFieldCount = dataRedistributionStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case REDISTRIBUTION_OFFSETS_STRUCT:
		{
			lFieldCount = redistributionOffsetsStruct::FIELD_COUNT_VALUE;
			break;
		}

        case SUBTASK_RANGE_CANCEL_STRUCT:
		{
			lFieldCount = subtaskRangeCancelStruct::FIELD_COUNT_VALUE;
			break;
		}

        case FILE_OPERATIONS_STRUCT:
		{
			lFieldCount = fileOperationsStruct::FIELD_COUNT_VALUE;
			break;
		}

        case MULTI_FILE_OPERATIONS_STRUCT:
		{
			lFieldCount = multiFileOperationsStruct::FIELD_COUNT_VALUE;
			break;
		}

		default:
			PMTHROW(pmFatalErrorException());
	}
    
    std::vector<int> lBlockLengthVector(lFieldCount);
	int* lBlockLength = &lBlockLengthVector[0];
    std::vector<MPI_Aint> lDisplacementVector(lFieldCount);
	MPI_Aint* lDisplacement = &lDisplacementVector[0];
    std::vector<MPI_Datatype> lDataTypeVector(lFieldCount);
	MPI_Datatype* lDataType = &lDataTypeVector[0];

	switch(pDataType)
	{
		case MACHINE_POOL_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(machinePool, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.cpuCores, lDataCoresMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.gpuCards, lDataCardsMPI, MPI_UNSIGNED, 1, 1);

			break;
		}

		case DEVICE_POOL_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(devicePool, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.name, lDataNameMPI, MPI_CHAR, 0, MAX_NAME_STR_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.description, lDataDescMPI, MPI_CHAR, 1, MAX_DESC_STR_LEN);

			break;
		}

        case MEMORY_IDENTIFIER_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(memoryIdentifierStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memOwnerHost, lMemOwnerHostMPI, MPI_UNSIGNED, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.generationNumber, lGenerationNumberMPI, MPI_UNSIGNED_LONG, 1, 1);

            break;
        }

        case MEMORY_DISTRIBUTION_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(memoryDistributionStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.distType, lDistTypeMPI, MPI_UNSIGNED_SHORT, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.blockDim, lBlockDimMPI, MPI_UNSIGNED, 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.matrixWidth, lMatrixWidthMPI, MPI_UNSIGNED, 2, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.matrixHeight, lMatrixHeightMPI, MPI_UNSIGNED, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.randomize, lRandomizeMPI, MPI_UNSIGNED_SHORT, 4, 1);

            break;
        }

        case TASK_MEMORY_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(taskMemoryStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memIdentifier, lMemIdentifierMPI, GetDataTypeMPI(MEMORY_IDENTIFIER_STRUCT), 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.distStruct, lDistStructMPI, GetDataTypeMPI(MEMORY_DISTRIBUTION_STRUCT), 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memLength, lMemLengthMPI, MPI_UNSIGNED_LONG, 2, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memType, lMemTypeMPI, MPI_UNSIGNED_SHORT, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subscriptionVisibility, lSubscriptionVisibilityMPI, MPI_UNSIGNED_SHORT, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.flags, lFlagsMPI, MPI_UNSIGNED_SHORT, 5, 1);

            break;
        }

		case REMOTE_TASK_ASSIGN_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(remoteTaskAssignStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskConfLength, lTaskConfLengthMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskMemCount, lTaskMemCountMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskId, lTaskIdMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskCount, lSubtaskCountMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.callbackKey, lCallbackKeyMPI, MPI_CHAR, 4, MAX_CB_KEY_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.assignedDeviceCount, lAssignedDeviceCountMPI, MPI_UNSIGNED, 5, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 6, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 7, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.priority, lPriorityMPI, MPI_UNSIGNED_SHORT, 8, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.schedModel, lSchedModelMPI, MPI_UNSIGNED_SHORT, 9, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.flags, lFlagsMPI, MPI_UNSIGNED_SHORT, 10, 1);

			break;
		}

		case REMOTE_SUBTASK_ASSIGN_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(remoteSubtaskAssignStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.assignmentType, lAssignmentTypeMPI, MPI_UNSIGNED_SHORT, 6, 1);

			break;
		}

		case OWNERSHIP_DATA_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(ownershipDataStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);

			break;
		}

		case OWNERSHIP_CHANGE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(ownershipChangeStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.newOwnerHost, lNewOwnerHostMPI, MPI_UNSIGNED, 2, 1);

			break;
		}
        
		case SEND_ACKNOWLEDGEMENT_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(sendAcknowledgementStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceDeviceGlobalIndex, lSourceDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.execStatus, lExecStatusMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 6, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.ownershipDataElements, lOwnershipDataElementsMPI, MPI_UNSIGNED, 7, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.addressSpaceIndices, lAddressSpaceIndicesMPI, MPI_UNSIGNED, 8, 1);

			break;
		}

		case TASK_EVENT_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(taskEventStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskEvent, lTaskEventMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);

			break;
		}

		case STEAL_REQUEST_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(stealRequestStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.stealingDeviceGlobalIndex, lStealingDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.stealingDeviceExecutionRate, lStealingDeviceExecutionRateMPI, MPI_DOUBLE, 4, 1);

			break;
		}

		case STEAL_RESPONSE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(stealResponseStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.stealingDeviceGlobalIndex, lStealingDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.success, lSuccessMPI, MPI_UNSIGNED_SHORT, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 6, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 7, 1);

			break;
		}

		case MEMORY_TRANSFER_REQUEST_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(memoryTransferRequest, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceMemIdentifier, lSourceMemIdentifierMPI, GetDataTypeMPI(MEMORY_IDENTIFIER_STRUCT), 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.destMemIdentifier, lDestMemIdentifierMPI, GetDataTypeMPI(MEMORY_IDENTIFIER_STRUCT), 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.transferType, lTransferTypeMPI, MPI_UNSIGNED_SHORT, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.receiverOffset, lReceiverOffsetMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 5, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.step, lStepMPI, MPI_UNSIGNED_LONG, 6, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.count, lCountMPI, MPI_UNSIGNED_LONG, 7, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.destHost, lDestHostMPI, MPI_UNSIGNED_LONG, 8, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.isForwarded, lIsForwardedMPI, MPI_UNSIGNED_SHORT, 9, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.isTaskOriginated, lIsTaskOriginatedMPI, MPI_UNSIGNED_SHORT, 10, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 11, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 12, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.priority, lPriorityMPI, MPI_UNSIGNED_SHORT, 13, 1);

			break;
		}

		case SHADOW_MEM_TRANSFER_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(shadowMemTransferStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.writeOnlyUnprotectedPageRangesCount, lWriteOnlyUnprotectedPageRangesCountMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskMemLength, lSubtaskMemLengthMPI, MPI_UNSIGNED, 1, 1);

			break;
		}

        case NO_REDUCTION_REQD_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(noReductionReqdStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);

            break;
        }
        
		case SUBTASK_REDUCE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(subtaskReduceStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskId, lSubtaskIdMPI, MPI_UNSIGNED_LONG, 2, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.shadowMemsCount, lShadowMemsCountMPI, MPI_UNSIGNED, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.scratchBuffer1Length, lScratchBuffer1LengthMPI, MPI_UNSIGNED, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.scratchBuffer2Length, lScratchBuffer2LengthMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.scratchBuffer3Length, lScratchBuffer3LengthMPI, MPI_UNSIGNED, 6, 1);

			break;
		}

		case MEMORY_RECEIVE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(memoryReceiveStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memOwnerHost, lMemOwnerHostMPI, MPI_UNSIGNED, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.generationNumber, lGenerationNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.transferType, lTransferTypeMPI, MPI_UNSIGNED_SHORT, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.step, lStepMPI, MPI_UNSIGNED_LONG, 5, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.count, lCountMPI, MPI_UNSIGNED_LONG, 6, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.isTaskOriginated, lIsTaskOriginatedMPI, MPI_UNSIGNED_SHORT, 7, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 8, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 9, 1);

			break;
		}
            
		case HOST_FINALIZATION_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(hostFinalizationStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.terminate, lTerminateMPI, MPI_UNSIGNED_SHORT, 0, 1);

			break;
		}

		case REDISTRIBUTION_ORDER_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(redistributionOrderStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.order, lOrderMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);
            
			break;
		}

		case DATA_REDISTRIBUTION_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(dataRedistributionStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.remoteHost, lRemoteHostMPI, MPI_UNSIGNED, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtasksAccounted, lSubtasksAccountedMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.orderDataCount, lOrderDataCountMPI, MPI_UNSIGNED, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.addressSpaceIndex, lAddressSpaceIndexMPI, MPI_UNSIGNED, 5, 1);
            
			break;
		}
        
		case REDISTRIBUTION_OFFSETS_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(redistributionOffsetsStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.redistributedMemGenerationNumber, lRedistributedMemGenerationNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offsetsDataCount, lOffsetsDataCountMPI, MPI_UNSIGNED, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.addressSpaceIndex, lAddressSpaceIndexMPI, MPI_UNSIGNED, 4, 1);
            
			break;
		}

        case SUBTASK_RANGE_CANCEL_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(subtaskRangeCancelStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 5, 1);

			break;        
        }

        case FILE_OPERATIONS_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(fileOperationsStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileName, lFileNameMPI, MPI_CHAR, 0, MAX_FILE_SIZE_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileOp, lFileOpMPI, MPI_UNSIGNED_SHORT, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceHost, lSourceHostMPI, MPI_UNSIGNED, 2, 1);

			break;        
        }

        case MULTI_FILE_OPERATIONS_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(multiFileOperationsStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileOp, lFileOpMPI, MPI_UNSIGNED_SHORT, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceHost, lSourceHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.userId, lUserIdMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileCount, lFileCountMPI, MPI_UNSIGNED, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.totalLength, lTotalLengthMPI, MPI_UNSIGNED, 4, 1);

			break;        
        }

		default:
			PMTHROW(pmFatalErrorException());
	}

	bool lError = false;
	MPI_Datatype* lNewType = (MPI_Datatype*)(mMPITypesAllocator.Allocate(sizeof(MPI_Datatype)));

	if( (MPI_CALL("MPI_Type_create_struct", (MPI_Type_create_struct(lFieldCount, lBlockLength, lDisplacement, lDataType, lNewType) != MPI_SUCCESS))) || (MPI_CALL("MPI_Type_commit", (MPI_Type_commit(lNewType) != MPI_SUCCESS))) )
	{
		lError = true;
        
		mMPITypesAllocator.Deallocate(lNewType);
	}

	if(lError || !lNewType)
		PMTHROW(pmNetworkException(pmNetworkException::DATA_TYPE_REGISTRATION));

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());
	mRegisteredDataTypes[pDataType] = lNewType;
}

void pmMPI::UnregisterTransferDataType(communicatorDataTypes pDataType)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());

	std::map<communicatorDataTypes, MPI_Datatype*>::iterator lIter = mRegisteredDataTypes.find(pDataType);
	DEBUG_EXCEPTION_ASSERT(lIter != mRegisteredDataTypes.end());
    
    DEBUG_EXCEPTION_ASSERT(lIter->second);

    MPI_CALL("MPI_Type_free", MPI_Type_free(lIter->second));
    
    mMPITypesAllocator.Deallocate(lIter->second);

    mRegisteredDataTypes.erase(lIter);
}
    
// Must be called with mResourceLock acquired
void pmMPI::CommandComplete(pmCommunicatorCommandPtr& pCommand, pmStatus pStatus)
{
    ushort lCommandType = pCommand->GetType();

    const pmCommandCompletionCallbackType lCallback = pCommand->GetCommandCompletionCallback();
    bool lIsPersistent = pCommand->IsPersistent();

    if(lIsPersistent && lCallback)
        pCommand->SetCommandCompletionCallback(NULL);

    pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(pCommand);
    pCommand->MarkExecutionEnd(pStatus, lCommandPtr);

#if 0
	const pmHardware* lHardware = pCommand->GetDestination();
	const pmMachine* lMachine = dynamic_cast<const pmMachine*>(lHardware);

	if(lMachine)
	{
		pmMachinePool* lMachinePool;
		SAFE_GET_MACHINE_POOL(lMachinePool);

        if(lCommandType == SEND || lCommandType == BROADCAST)
            lMachinePool->RegisterSendCompletion(lMachine, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
        else if(lCommandType == RECEIVE)
            lMachinePool->RegisterReceiveCompletion(lMachine, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	}
#endif
    
    if(lIsPersistent)
    {
        EXCEPTION_ASSERT(lCommandType == SEND || lCommandType == RECEIVE);

        if(lCallback)
        {
            pmCommunicatorCommandPtr lClonePtr = pCommand->Clone(); // This creates a copy of the data
            lCallback(lClonePtr);
        }
        
        pCommand->SetCommandCompletionCallback(lCallback);
        pCommand->MarkExecutionStart();

        MPI_Request* lRequest = NULL;

        if(lCommandType == SEND)
        {
            lRequest = GetPersistentSendRequestInternal(pCommand);
            if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
        }
        else
        {
            lRequest = GetPersistentRecvRequestInternal(pCommand);
            if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
        }

        DEBUG_EXCEPTION_ASSERT(mNonBlockingRequestMap.find(lRequest) == mNonBlockingRequestMap.end());
        
        mNonBlockingRequestMap[lRequest] = pCommand;
        
        DEBUG_EXCEPTION_ASSERT(mRequestCountMap.find(pCommand) == mRequestCountMap.end());

        mRequestCountMap[pCommand] = 1;
    }
}

/* Must be called with mResourceLock acquired */
void pmMPI::SetupDummyRequest()
{
	if(!mPersistentDummyRecvRequest.get())
	{
        EXCEPTION_ASSERT(!mDummyRequestInitiated);

		mPersistentDummyRecvRequest.reset(new MPI_Request());

        if( MPI_CALL("MPI_Recv_init", (MPI_Recv_init(NULL, 0, MPI_BYTE, mHostId, PM_MPI_DUMMY_TAG, MPI_COMM_WORLD, mPersistentDummyRecvRequest.get()) != MPI_SUCCESS)) )
            PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CREATION_ERROR));
    }

    if(!mDummyRequestInitiated)
    {
		if( MPI_CALL("MPI_Start", (MPI_Start(mPersistentDummyRecvRequest.get()) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CREATION_ERROR));
        
        mDummyRequestInitiated = true;
	}

    DEBUG_EXCEPTION_ASSERT(mPersistentDummyRecvRequest);
}

/* Must be called with mResourceLock acquired */
void pmMPI::CancelDummyRequest()
{
	if(mPersistentDummyRecvRequest.get() && mDummyRequestInitiated)
	{
        // For some reason, persistent send request has proved to be slower than non-persistent one here. So, using the latter variant.
        MPI_Request lRequest;

        if( MPI_CALL("MPI_Isend", (MPI_Isend(NULL, 0, MPI_BYTE, mHostId, PM_MPI_DUMMY_TAG, MPI_COMM_WORLD, &lRequest) != MPI_SUCCESS)) )
            PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CANCEL_ERROR));
        
        mDummyRequestInitiated = false;
	}
}

void pmMPI::StopThreadExecution()
{
	// Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        if(mThreadTerminationFlag)
            return;   // Already stopped
    }
    
    mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));

	// Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        mThreadTerminationFlag = true;
        CancelDummyRequest();
    }

	mSignalWait->Wait();
}

void pmMPI::ThreadSwitchCallback(std::shared_ptr<networkEvent>& pCommand)
{
	/* Do not derefernce pCommand.get() in this function as it is NULL (passed in the constructor above) */
	
	// This loop terminates with the pmThread's destruction
	while(1)
	{
		try
		{
            size_t lRequestCount = 0;
            
            std::vector<MPI_Request> lRequestVector;
            std::vector<MPI_Request*> lKeyVector;

			// Auto lock/unlock scope
			{
				FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

				if(mThreadTerminationFlag)
                {
                    mSignalWait->Signal();
					return;
                }

				SetupDummyRequest();

                lRequestCount = mNonBlockingRequestMap.size();
				++lRequestCount; // Adding one for dummy request
                
                lRequestVector.reserve(lRequestCount);
                lKeyVector.reserve(lRequestCount - 1);
                
                DEBUG_EXCEPTION_ASSERT(mPersistentDummyRecvRequest.get() && *mPersistentDummyRecvRequest.get());
				lRequestVector.emplace_back(*mPersistentDummyRecvRequest.get());
                
                typename decltype(mNonBlockingRequestMap)::iterator lIter = mNonBlockingRequestMap.begin(), lEndIter = mNonBlockingRequestMap.end();
                for(; lIter != lEndIter; ++lIter)
                {
                    DEBUG_EXCEPTION_ASSERT(lIter->first && *lIter->first);
                    
                    lRequestVector.push_back(*lIter->first);
                    lKeyVector.push_back(lIter->first);
                }
            }

			int lFinishingRequestIndex = 0;
			MPI_Status lFinishingRequestStatus;
            
			if( MPI_CALL("MPI_Waitany", (MPI_Waitany((int)lRequestCount, &lRequestVector[0], &lFinishingRequestIndex, &lFinishingRequestStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::WAIT_ERROR));

            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            
            if(mThreadTerminationFlag)
            {
                mSignalWait->Signal();
                return;
            }

			if(lFinishingRequestIndex != 0)		// Not Dummy Request
			{
                typename decltype(mNonBlockingRequestMap)::iterator lFinishingIter = mNonBlockingRequestMap.find(lKeyVector[lFinishingRequestIndex - 1]);

                DEBUG_EXCEPTION_ASSERT(lFinishingIter != mNonBlockingRequestMap.end());

                MPI_Request* lMpiRequest = lFinishingIter->first;
				pmCommunicatorCommandPtr lCommand = lFinishingIter->second;

                typename decltype(mRequestCountMap)::iterator lRequestCountIter = mRequestCountMap.find(lCommand);
                DEBUG_EXCEPTION_ASSERT(lRequestCountIter != mRequestCountMap.end());

                --lRequestCountIter->second;
                
                mNonBlockingRequestMap.erase(lFinishingIter);

                if(!lCommand->IsPersistent())
                    mMPITypesAllocator.Deallocate(lMpiRequest);

				if(lRequestCountIter->second == 0)
				{
					mRequestCountMap.erase(lRequestCountIter);

					ushort lCommandType = lCommand->GetType();
                    EXCEPTION_ASSERT(lCommandType == SEND || lCommandType == BROADCAST || lCommandType == RECEIVE);

                    CommandComplete(lCommand, pmSuccess);
				}
			}
		}
		catch(pmException& e)
		{
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from primary network thread");
		}
	}
}
    
void pmMPI::FreezeReceptionAndFinishCommands()
{
    StopThreadExecution();
    
    if(mReceiveThread)
        mReceiveThread->StopThreadExecution();
}


/* class pmMPI::pmUnknownLengthReceiveThread */
pmMPI::pmUnknownLengthReceiveThread::pmUnknownLengthReceiveThread(pmMPI* pMPI)
	: mMPI(pMPI)
	, mThreadTerminationFlag(false)
    , mResourceLock __LOCK_NAME__("pmMPI::pmUnknownLengthReceiveThread::mResourceLock")
{
	SwitchThread(std::shared_ptr<networkEvent>(new networkEvent()), MAX_PRIORITY_LEVEL);
}

pmMPI::pmUnknownLengthReceiveThread::~pmUnknownLengthReceiveThread()
{
	StopThreadExecution();

	#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down unknown length network thread");
	#endif
}

void pmMPI::pmUnknownLengthReceiveThread::StopThreadExecution()
{
	MPI_Request lRequest;
    
	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        if(mThreadTerminationFlag)
            return;   // Already stopped
    }
        
    mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));

	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
		mThreadTerminationFlag = true;
		SendDummyProbeCancellationMessage(lRequest);
	}

	mSignalWait->Wait();
}

pmStatus pmMPI::pmUnknownLengthReceiveThread::SendDummyProbeCancellationMessage(MPI_Request& pRequest)
{
	if( MPI_CALL("MPI_Isend", (MPI_Isend(NULL, 0, MPI_CHAR, mMPI->GetHostId(), UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &pRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
    
	return pmSuccess;
}

pmStatus pmMPI::pmUnknownLengthReceiveThread::ReceiveDummyProbeCancellationMessage()
{
	MPI_Request lRequest;
	MPI_Status lStatus;

	if( MPI_CALL("MPI_Irecv", (MPI_Irecv(NULL, 0, MPI_CHAR, mMPI->GetHostId(), UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &lRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

	if( MPI_CALL("MPI_Wait", (MPI_Wait(&lRequest, &lStatus) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

	return pmSuccess;
}
    
void pmMPI::pmUnknownLengthReceiveThread::ThreadSwitchCallback(std::shared_ptr<networkEvent>& pCommand)
{
	/* Do not use pCommand in this function as it is NULL (passed in the constructor above) */
	
	// This loop terminates with the pmThread's destruction
	while(1)
	{
		try
		{
			MPI_Status lProbeStatus, lRecvStatus;

			// Auto lock/unlock scope
			{
				FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
				if(mThreadTerminationFlag)
				{
					ReceiveDummyProbeCancellationMessage();
                    mSignalWait->Signal();
					//SendReverseDummyProbeCancellationMessage();
					return;
				}
			}
			
			if( MPI_CALL("MPI_Probe", (MPI_Probe(MPI_ANY_SOURCE, UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &lProbeStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::PROBE_ERROR));
			
			// Auto lock/unlock scope
			{
				FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
				if(mThreadTerminationFlag)
				{
					ReceiveDummyProbeCancellationMessage();
                    mSignalWait->Signal();
					//SendReverseDummyProbeCancellationMessage();
					return;
				}
			}

			int lLength = 0;
			if( MPI_CALL("MPI_Get_count", (MPI_Get_count(&lProbeStatus, MPI_PACKED, &lLength) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::GET_COUNT_ERROR));

            finalize_ptr<char, deleteArrayDeallocator<char>> lPackedData(new char[lLength]);

			if( MPI_CALL("MPI_Recv", (MPI_Recv(lPackedData.get_ptr(), lLength, MPI_PACKED, lProbeStatus.MPI_SOURCE, (int)(UNKNOWN_LENGTH_TAG), MPI_COMM_WORLD, &lRecvStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->UnpackDataEvent(std::move(lPackedData), lLength, MAX_CONTROL_PRIORITY);
		}
		catch(pmException& e)
		{
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from secondary network thread");
		}
	}
}

} // end namespace pm




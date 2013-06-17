
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

#include "pmNetwork.h"
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

pmCluster* PM_GLOBAL_CLUSTER = NULL;

//#define ENABLE_MPI_DEBUG_HOOK

#ifdef ENABLE_MPI_DEBUG_HOOK
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
		CLUSTER_IMPLEMENTATION_CLASS* dCluster = dynamic_cast<CLUSTER_IMPLEMENTATION_CLASS*>(y); \
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
		pmMachine* dMachine = dynamic_cast<pmMachine*>(z); \
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
			CLUSTER_IMPLEMENTATION_CLASS* dCluster = dynamic_cast<CLUSTER_IMPLEMENTATION_CLASS*>(z); \
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

pmStatus pmNetwork::SendComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus, std::tr1::static_pointer_cast<pmCommand>(pCommand));

	pmHardware* lHardware = pCommand->GetDestination();
	pmMachine* lMachine = dynamic_cast<pmMachine*>(lHardware);

	if(lMachine)
	{
		pmMachinePool* lMachinePool;
		SAFE_GET_MACHINE_POOL(lMachinePool);

		lMachinePool->RegisterSendCompletion(lMachine, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	}

	return pmSuccess;
}

pmStatus pmNetwork::ReceiveComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus, std::tr1::static_pointer_cast<pmCommand>(pCommand));

	pmHardware* lHardware = pCommand->GetDestination();
	pmMachine* lMachine = dynamic_cast<pmMachine*>(lHardware);

	if(lMachine)
	{
		pmMachinePool* lMachinePool;
		SAFE_GET_MACHINE_POOL(lMachinePool);

		lMachinePool->RegisterReceiveCompletion(lMachine, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	}

	return pmSuccess;
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
    , mDummyReceiveRequest(NULL)
    , mDataTypesResourceLock __LOCK_NAME__("pmMPI::mDataTypesResourceLock")
    , mResourceLock __LOCK_NAME__("pmMPI::mResourceLock")
    , mSignalWait(NULL)
	, mThreadTerminationFlag(false)
    , mPersistentReceptionFreezed(false)
    , mReceiveThread(NULL)
#ifdef PROGRESSIVE_SLEEP_NETWORK_THREAD
	, mProgressiveSleepTime(MIN_PROGRESSIVE_SLEEP_TIME_MILLI_SECS)
#endif
{
	int lThreadSupport, lMpiStatus;

	//if(MPI_CALL("MPI_Init", ((lMpiStatus = MPI_Init(NULL, NULL)) != MPI_SUCCESS)))
    if(MPI_CALL("MPI_Init_thread", ((lMpiStatus = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &lThreadSupport)) != MPI_SUCCESS))
       || lThreadSupport != MPI_THREAD_MULTIPLE)
	{
		MPI_CALL("MPI_Abort", MPI_Abort(MPI_COMM_WORLD, lMpiStatus));
		mTotalHosts = 0;
		mHostId = 0;
		PMTHROW(pmMpiInitException());
	}

	int lHosts, lId;

	MPI_CALL("MPI_Comm_size", MPI_Comm_size(MPI_COMM_WORLD, &lHosts));
	MPI_CALL("MPI_Comm_rank", MPI_Comm_rank(MPI_COMM_WORLD, &lId));

	pmLogger::GetLogger()->SetHostId(lId);

	mTotalHosts = lHosts;
	mHostId = lId;

	MPI_CALL("MPI_Comm_set_errhandler", MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

	PM_GLOBAL_CLUSTER = new pmClusterMPI();
    
    MPI_DEBUG_HOOK
    
	networkEvent lNetworkEvent; 
	SwitchThread(lNetworkEvent, MAX_PRIORITY_LEVEL);

	mReceiveThread = new pmUnknownLengthReceiveThread(this);
}

pmMPI::~pmMPI()
{
#ifdef SERIALIZE_DEFERRED_LOGS
    #ifdef ENABLE_ACCUMULATED_TIMINGS
        pmAccumulatedTimesSorter::GetAccumulatedTimesSorter()->FlushLogs();
    #endif
    
    if(GetHostId() == 0)
    {
        MPI_Status lStatus;

        pmLogger::GetLogger()->PrintDeferredLog();

        uint lMachines = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();
        for(uint i = 1; i < lMachines; ++i)
        {
            uint lLogLength = 0;
            if( MPI_CALL("MPI_Recv", (MPI_Recv(&lLogLength, 1, MPI_UNSIGNED, i, pmCommunicatorCommand::DEFERRED_LOG_LENGTH_TAG, MPI_COMM_WORLD, &lStatus) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
        
            if(lLogLength && ((lLogLength + 1) > lLogLength))   // Check for overflow and wrap around
            {
                std::auto_ptr<char> lLogAutoPtr(new char[lLogLength + 1]);
                
                if( MPI_CALL("MPI_Recv", (MPI_Recv(lLogAutoPtr.get(), lLogLength, MPI_CHAR, i, pmCommunicatorCommand::DEFERRED_LOG_TAG, MPI_COMM_WORLD, &lStatus) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
                
                (lLogAutoPtr.get())[lLogLength] = '\0';
                std::cerr << lLogAutoPtr.get() << std::endl;
            }
        }
    }
    else
    {
        const std::string& lDeferredLog = pmLogger::GetLogger()->GetDeferredLogStream().str();
        uint lLogLength = (uint)lDeferredLog.size();

        if( MPI_CALL("MPI_Send", (MPI_Send(&lLogLength, 1, MPI_UNSIGNED, 0, pmCommunicatorCommand::DEFERRED_LOG_LENGTH_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)) )
            PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));

        if(lLogLength && ((lLogLength + 1) > lLogLength))   // Check for overflow and wrap around
        {
            char* lLog = const_cast<char*>(lDeferredLog.c_str());
            if( MPI_CALL("MPI_Send", (MPI_Send(lLog, lLogLength, MPI_CHAR, 0, pmCommunicatorCommand::DEFERRED_LOG_TAG, MPI_COMM_WORLD) != MPI_SUCCESS)) )
                PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
        }
        
        pmLogger::GetLogger()->ClearDeferredLog();
    }
    
    NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GlobalBarrier();
#endif

	StopThreadExecution();

	delete mReceiveThread;

	delete dynamic_cast<pmClusterMPI*>(PM_GLOBAL_CLUSTER);

	MPI_CALL("MPI_Finalize", (MPI_Finalize() == MPI_SUCCESS));

    #ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down network thread");
	#endif
}

pmStatus pmMPI::PackData(pmCommunicatorCommandPtr pCommand)
{
	void* lPackedData = NULL;
	uint lTag = pCommand->GetTag();
	ulong lLength = sizeof(uint);

	switch(pCommand->GetDataType())
	{
		case pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED:
		{
			pmCommunicatorCommand::remoteTaskAssignPacked* lData = (pmCommunicatorCommand::remoteTaskAssignPacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());

			pmCommunicatorCommand::remoteTaskAssignStruct& lTaskStruct = lData->taskStruct;
			lLength += sizeof(lTaskStruct) + lData->taskConf.length + lData->devices.length;

			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
	
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTaskStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			
			if(lData->taskConf.ptr)
			{
				if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->taskConf.ptr, lData->taskConf.length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			}
			
			if(lData->devices.ptr)
			{
				if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->devices.ptr, lData->devices.length/((uint)(sizeof(uint))), MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			}
            
            lLength = lPos;

			break;
		}

		case pmCommunicatorCommand::SUBTASK_REDUCE_PACKED:
		{
			pmCommunicatorCommand::subtaskReducePacked::subtaskReduceInfo* lInfo = (pmCommunicatorCommand::subtaskReducePacked::subtaskReduceInfo*)(pCommand->GetData());
			if(!lInfo)
				PMTHROW(pmFatalErrorException());
        
            pmCommunicatorCommand::subtaskReducePacked* lData = lInfo->packedData;

			pmCommunicatorCommand::subtaskReduceStruct& lStruct = lData->reduceStruct;
            uint lTotalSubscriptions = lStruct.inputMemSubscriptionCount + lStruct.outputMemReadSubscriptionCount + lStruct.outputMemWriteSubscriptionCount;
			lLength += sizeof(lStruct) + sizeof(pmCommunicatorCommand::ownershipDataStruct) * lTotalSubscriptions + lData->subtaskMem.length;

			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
	
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

            if(lTotalSubscriptions)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->subscriptions, lTotalSubscriptions, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            if(lStruct.subtaskMemLength)
            {
                if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->subtaskMem.ptr, lData->subtaskMem.length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            }

            lLength = lPos;

			break;
		}

		case pmCommunicatorCommand::MEMORY_RECEIVE_PACKED:
		{
			pmCommunicatorCommand::memoryReceivePacked* lData = (pmCommunicatorCommand::memoryReceivePacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());

			pmCommunicatorCommand::memoryReceiveStruct& lStruct = lData->receiveStruct;
			lLength += sizeof(lStruct) + lData->mem.length;

			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
	
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));

			if(lData->mem.length != 0)
			{
				if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->mem.ptr, lData->mem.length, MPI_BYTE, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
			}

            lLength = lPos;

			break;
		}

		case pmCommunicatorCommand::DATA_REDISTRIBUTION_PACKED:
		{
			pmCommunicatorCommand::dataRedistributionPacked* lData = (pmCommunicatorCommand::dataRedistributionPacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());
            
			pmCommunicatorCommand::dataRedistributionStruct& lStruct = lData->redistributionStruct;
			lLength += sizeof(lStruct) + lData->redistributionStruct.orderDataCount * sizeof(pmCommunicatorCommand::redistributionOrderStruct);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::DATA_REDISTRIBUTION_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->redistributionData, lData->redistributionStruct.orderDataCount, GetDataTypeMPI(pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}

		case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_PACKED:
		{
			pmCommunicatorCommand::redistributionOffsetsPacked* lData = (pmCommunicatorCommand::redistributionOffsetsPacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());
            
			pmCommunicatorCommand::redistributionOffsetsStruct& lStruct = lData->redistributionStruct;
			lLength += sizeof(lStruct) + lData->redistributionStruct.offsetsDataCount * sizeof(ulong);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->offsetsData, lData->redistributionStruct.offsetsDataCount, MPI_UNSIGNED_LONG, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}

		case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_PACKED:
		{
			pmCommunicatorCommand::sendAcknowledgementPacked* lData = (pmCommunicatorCommand::sendAcknowledgementPacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());
            
			pmCommunicatorCommand::sendAcknowledgementStruct& lStruct = lData->ackStruct;
			lLength += sizeof(lStruct) + lData->ackStruct.ownershipDataElements * sizeof(pmCommunicatorCommand::ownershipDataStruct);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(lData->ownershipData, lData->ackStruct.ownershipDataElements, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}

		case pmCommunicatorCommand::OWNERSHIP_TRANSFER_PACKED:
		{
			pmCommunicatorCommand::ownershipTransferPacked* lData = (pmCommunicatorCommand::ownershipTransferPacked*)(pCommand->GetData());
			if(!lData)
				PMTHROW(pmFatalErrorException());
            
			pmCommunicatorCommand::memoryIdentifierStruct& lStruct = lData->memIdentifier;
            uint lDataElementCount = (uint)(lData->transferData->size());
			lLength += sizeof(lStruct) + sizeof(uint) + lDataElementCount * sizeof(pmCommunicatorCommand::ownershipChangeStruct);
            
			if(lLength > __MAX_SIGNED(int))
				PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));
            
			lPackedData = new char[lLength];
			int lPos = 0;
			MPI_Comm lCommunicator;
			SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lTag, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lStruct, 1, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&lDataElementCount, 1, MPI_UNSIGNED, lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
        
			if( MPI_CALL("MPI_Pack", (MPI_Pack(&(lData->transferData.get())[0], lDataElementCount, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT), lPackedData, (int)lLength, &lPos, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_PACK_ERROR));
            
            lLength = lPos;
            
			break;
		}
        
		default:
			PMTHROW(pmFatalErrorException());
	}

	pCommand->SetSecondaryData(lPackedData, lLength);

	return pmSuccess;
}

pmCommunicatorCommandPtr pmMPI::UnpackData(void* pPackedData, int pDataLength)
{
    int lPos = 0;
    uint lInternalTag = (uint)pmCommunicatorCommand::MAX_COMMUNICATOR_COMMAND_TAGS;
    if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &lInternalTag, 1, MPI_UNSIGNED, MPI_COMM_WORLD) != MPI_SUCCESS)) )
        PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

    pmCommunicatorCommand::communicatorDataTypes lDataType = pmCommunicatorCommand::MAX_COMMUNICATOR_DATA_TYPES;
    pmCommunicatorCommand::communicatorCommandTags lTag = (pmCommunicatorCommand::communicatorCommandTags)lInternalTag;
    switch(lTag)
    {
        case pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT_TAG:
            lDataType = pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED;
            break;

        case pmCommunicatorCommand::SUBTASK_REDUCE_TAG:
            lDataType = pmCommunicatorCommand::SUBTASK_REDUCE_PACKED;
            break;
        
        case pmCommunicatorCommand::MEMORY_RECEIVE_TAG:
            lDataType = pmCommunicatorCommand::MEMORY_RECEIVE_PACKED;
            break;
        
        case pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG:
            lDataType = pmCommunicatorCommand::DATA_REDISTRIBUTION_PACKED;
            break;
            
        case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_TAG:
            lDataType = pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_PACKED;
            break;
            
        case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG:
            lDataType = pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_PACKED;
            break;
        
        case pmCommunicatorCommand::OWNERSHIP_TRANSFER_TAG:
            lDataType = pmCommunicatorCommand::OWNERSHIP_TRANSFER_PACKED;
            break;
        
        default:
            PMTHROW(pmFatalErrorException());
    }

    pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_CONTROL_PRIORITY, pmCommunicatorCommand::RECEIVE, lTag,	NULL, lDataType, NULL, 0, NULL, 0, pmScheduler::GetScheduler()->GetUnknownLengthCommandCompletionCallback());

	MPI_Comm lCommunicator;
	SAFE_GET_MPI_COMMUNICATOR(lCommunicator, lCommand->GetDestination());

	switch(lDataType)
	{
		case pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED:
		{
			pmCommunicatorCommand::remoteTaskAssignPacked* lPackedData = new pmCommunicatorCommand::remoteTaskAssignPacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->taskStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

			if(lPackedData->taskStruct.taskConfLength != 0)
			{
				lPackedData->taskConf.ptr = new char[lPackedData->taskStruct.taskConfLength];
				lPackedData->taskConf.length = lPackedData->taskStruct.taskConfLength;
				
				if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->taskConf.ptr, lPackedData->taskConf.length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
			}

			if(lPackedData->taskStruct.assignedDeviceCount != 0)
			{
                bool lReductionCB = false;
                
                try {
                    pmCallbackUnit* lCallbackUnit = pmCallbackUnit::FindCallbackUnit(lPackedData->taskStruct.callbackKey);	// throws exception if key unregistered
                    
                    lReductionCB = (lCallbackUnit && lCallbackUnit->GetDataReductionCB());
                } catch(pmException& e) {}
                
                
                if(lPackedData->taskStruct.schedModel == scheduler::PULL || lReductionCB)
                {
                    lPackedData->devices.ptr = new uint[lPackedData->taskStruct.assignedDeviceCount];
                    lPackedData->devices.length = lPackedData->taskStruct.assignedDeviceCount * (uint)(sizeof(uint));

                    if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->devices.ptr, lPackedData->taskStruct.assignedDeviceCount, MPI_UNSIGNED, lCommunicator) != MPI_SUCCESS)) )
                        PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
                }
			}
            
            lCommand->SetData(lPackedData, lPos);

			break;
		}

		case pmCommunicatorCommand::SUBTASK_REDUCE_PACKED:
		{
			pmCommunicatorCommand::subtaskReducePacked* lPackedData = new pmCommunicatorCommand::subtaskReducePacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->reduceStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

            pmCommunicatorCommand::subtaskReduceStruct& lStruct = lPackedData->reduceStruct;
            uint lTotalSubscriptions = lStruct.inputMemSubscriptionCount + lStruct.outputMemReadSubscriptionCount + lStruct.outputMemWriteSubscriptionCount;

            if(lTotalSubscriptions)
            {
                lPackedData->subscriptions = new pmCommunicatorCommand::ownershipDataStruct[lTotalSubscriptions];
                
                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->subscriptions, lTotalSubscriptions, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT), lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }
        
            if(lStruct.subtaskMemLength)
            {
                lPackedData->subtaskMem.ptr = new char[lStruct.subtaskMemLength];
                lPackedData->subtaskMem.length = (uint)lStruct.subtaskMemLength;
                    
                if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->subtaskMem.ptr, lPackedData->subtaskMem.length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
                    PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            }

            lCommand->SetData(lPackedData, lPos);

			break;
		}

		case pmCommunicatorCommand::MEMORY_RECEIVE_PACKED:
		{
			pmCommunicatorCommand::memoryReceivePacked* lPackedData = new pmCommunicatorCommand::memoryReceivePacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->receiveStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));

			if(lPackedData->receiveStruct.length == 0)
			{
				lPackedData->mem.ptr = NULL;
				lPackedData->mem.length = 0;
			}
			else
			{
				lPackedData->mem.ptr = new char[lPackedData->receiveStruct.length];
				lPackedData->mem.length = (uint)(lPackedData->receiveStruct.length);
				
				if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->mem.ptr, lPackedData->mem.length, MPI_BYTE, lCommunicator) != MPI_SUCCESS)) )
					PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
			}

            lCommand->SetData(lPackedData, lPos);

			break;
		}

		case pmCommunicatorCommand::DATA_REDISTRIBUTION_PACKED:
		{
			pmCommunicatorCommand::dataRedistributionPacked* lPackedData = new pmCommunicatorCommand::dataRedistributionPacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->redistributionStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::DATA_REDISTRIBUTION_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
			lPackedData->redistributionData = new pmCommunicatorCommand::redistributionOrderStruct[lPackedData->redistributionStruct.orderDataCount];
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->redistributionData, lPackedData->redistributionStruct.orderDataCount, GetDataTypeMPI(pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand->SetData(lPackedData, lPos);
            
			break;
		}

		case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_PACKED:
		{
			pmCommunicatorCommand::redistributionOffsetsPacked* lPackedData = new pmCommunicatorCommand::redistributionOffsetsPacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->redistributionStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
			lPackedData->offsetsData = new ulong[lPackedData->redistributionStruct.offsetsDataCount];
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->offsetsData, lPackedData->redistributionStruct.offsetsDataCount, MPI_UNSIGNED_LONG, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand->SetData(lPackedData, lPos);
            
			break;
		}

		case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_PACKED:
		{
			pmCommunicatorCommand::sendAcknowledgementPacked* lPackedData = new pmCommunicatorCommand::sendAcknowledgementPacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->ackStruct), 1, GetDataTypeMPI(pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
			lPackedData->ownershipData = new pmCommunicatorCommand::ownershipDataStruct[lPackedData->ackStruct.ownershipDataElements];
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, lPackedData->ownershipData, lPackedData->ackStruct.ownershipDataElements, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand->SetData(lPackedData, lPos);
            
			break;
		}

		case pmCommunicatorCommand::OWNERSHIP_TRANSFER_PACKED:
		{
			pmCommunicatorCommand::ownershipTransferPacked* lPackedData = new pmCommunicatorCommand::ownershipTransferPacked();
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->memIdentifier), 1, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
        
            uint lDataElementCount = 0;
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &lDataElementCount, 1, MPI_UNSIGNED, lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
        
			lPackedData->transferData.reset(new std::vector<pmCommunicatorCommand::ownershipChangeStruct>());
            
			if( MPI_CALL("MPI_Unpack", (MPI_Unpack(pPackedData, pDataLength, &lPos, &(lPackedData->transferData.get())[0], lDataElementCount, GetDataTypeMPI(pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT), lCommunicator) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::DATA_UNPACK_ERROR));
            
            lCommand->SetData(lPackedData, lPos);
        
            break;
        }
        
		default:
			PMTHROW(pmFatalErrorException());
	}

	return lCommand;
}
    
bool pmMPI::IsUnknownLengthTag(pmCommunicatorCommand::communicatorCommandTags pTag)
{
    return (pTag == pmCommunicatorCommand::REMOTE_TASK_ASSIGNMENT_TAG || 
            pTag == pmCommunicatorCommand::SUBTASK_REDUCE_TAG ||
            pTag == pmCommunicatorCommand::MEMORY_RECEIVE_TAG ||
            pTag == pmCommunicatorCommand::DATA_REDISTRIBUTION_TAG ||
            pTag == pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_TAG ||
            pTag == pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_TAG ||
            pTag == pmCommunicatorCommand::OWNERSHIP_TRANSFER_TAG);
}

pmStatus pmMPI::SendNonBlocking(pmCommunicatorCommandPtr pCommand)
{
	void* lData = NULL;
	ulong lLength = 0;

	if(IsUnknownLengthTag(pCommand->GetTag()))
	{
		lData = pCommand->GetSecondaryData();
		lLength = pCommand->GetSecondaryDataLength();
	}
	else
	{
		lData = pCommand->GetData();
		lLength = pCommand->GetDataLength();
	}

	if(!lData || lLength == 0)
		return pmSuccess;

	pmHardware* lHardware = pCommand->GetDestination();
	if(!lHardware)
		PMTHROW(pmFatalErrorException());

	if(!(dynamic_cast<pmMachine*>(lHardware) || dynamic_cast<pmProcessingElement*>(lHardware)))
		PMTHROW(pmFatalErrorException());

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	ulong lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	if(std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(pCommand) && lBlocks != 0)
		PMTHROW(pmFatalErrorException());

	for(ulong i=0; i<lBlocks; ++i)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT);

	if(lLastBlockLength)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), (uint)lLastBlockLength);

	return pmSuccess;
}

pmStatus pmMPI::BroadcastNonBlocking(pmCommunicatorCommandPtr pCommand)
{
	void* lData = pCommand->GetData();
	ulong lDataLength = pCommand->GetDataLength();

	if(!lData || lDataLength == 0)
		PMTHROW(pmFatalErrorException());

	if(lDataLength > MPI_TRANSFER_MAX_LIMIT)
		PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

	pCommand->MarkExecutionStart();

	MPI_Comm lCommunicator;
	int lRoot;
	SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lRoot, pCommand->GetDestination());

	MPI_Datatype lDataType = GetDataTypeMPI((pmCommunicatorCommand::communicatorDataTypes)(pCommand->GetDataType()));

	if( MPI_CALL("MPI_Bcast", (MPI_Bcast(lData, (uint)lDataLength, lDataType, lRoot, lCommunicator) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::BROADCAST_ERROR));

	pCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(pCommand));

	return pmSuccess;
}

pmStatus pmMPI::All2AllNonBlocking(pmCommunicatorCommandPtr pCommand)
{
	void* lSendData = pCommand->GetData();
	void* lRecvData = pCommand->GetSecondaryData();
	ulong lSendLength = pCommand->GetDataLength();
	ulong lRecvLength = pCommand->GetSecondaryDataLength();

	if(!lSendData || !lRecvData || lSendLength == 0 || lRecvLength == 0)
		PMTHROW(pmFatalErrorException());

	if(lSendLength > MPI_TRANSFER_MAX_LIMIT || lRecvLength > MPI_TRANSFER_MAX_LIMIT)
		PMTHROW(pmBeyondComputationalLimitsException(pmBeyondComputationalLimitsException::MPI_MAX_TRANSFER_LENGTH));

	pCommand->MarkExecutionStart();

	MPI_Comm lCommunicator;
	SAFE_GET_MPI_COMMUNICATOR(lCommunicator, pCommand->GetDestination());

	MPI_Datatype lDataType = GetDataTypeMPI((pmCommunicatorCommand::communicatorDataTypes)(pCommand->GetDataType()));

	if( MPI_CALL("MPI_Allgather", (MPI_Allgather(lSendData, (uint)lSendLength, lDataType, lRecvData, (uint)lRecvLength, lDataType, lCommunicator) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::ALL2ALL_ERROR));

	pCommand->MarkExecutionEnd(pmSuccess, std::tr1::static_pointer_cast<pmCommand>(pCommand));

	return pmSuccess;
}

pmStatus pmMPI::ReceiveNonBlocking(pmCommunicatorCommandPtr pCommand)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataLength();
	pmHardware* lHardware = pCommand->GetDestination();

	if(!lData || lLength == 0)
		return pmSuccess;

	// No hardware means receive from any machine
	if(lHardware)
	{
		if(!(dynamic_cast<pmMachine*>(lHardware) || dynamic_cast<pmProcessingElement*>(lHardware)))
			PMTHROW(pmFatalErrorException());
	}

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	ulong lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	if(std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(pCommand) && lBlocks != 0)
		PMTHROW(pmFatalErrorException());

	for(ulong i=0; i<lBlocks; ++i)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT);

	if(lLastBlockLength)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), (uint)lLastBlockLength);

	return pmSuccess;
}

pmStatus pmMPI::SendNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength)
{
	pCommand->MarkExecutionStart();

	MPI_Request* lRequest = NULL;
	MPI_Comm lCommunicator;
	int lDest;

	pmPersistentCommunicatorCommandPtr lPersistentCommand;
	if((lPersistentCommand = std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(pCommand)))
	{
		lRequest = GetPersistentSendRequest(lPersistentCommand.get());
		if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
	}
	else
	{
		SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
		MPI_Datatype lDataType = GetDataTypeMPI((pmCommunicatorCommand::communicatorDataTypes)(pCommand->GetDataType()));
		
        lRequest = new MPI_Request();
        pmCommunicatorCommand::communicatorCommandTags lTag = pCommand->GetTag();
        if(IsUnknownLengthTag(lTag))
            lTag = pmCommunicatorCommand::UNKNOWN_LENGTH_TAG;
        
		if( MPI_CALL("MPI_Isend", (MPI_Isend(pData, pLength, lDataType, lDest, (int)lTag, lCommunicator, lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
	}

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
	mNonBlockingRequestMap[lRequest] = pCommand;
	
	if(mRequestCountMap.find(pCommand) == mRequestCountMap.end())
		mRequestCountMap[pCommand] = 1;
	else
		mRequestCountMap[pCommand] = mRequestCountMap[pCommand] + 1;

	CancelDummyRequest();	// Signal the other thread to handle the created request

	return pmSuccess;
}

pmStatus pmMPI::ReceiveNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength)
{
	pCommand->MarkExecutionStart();

	MPI_Request* lRequest = NULL;
	MPI_Comm lCommunicator;
	int lDest;

	pmPersistentCommunicatorCommandPtr lPersistentCommand;
	if((lPersistentCommand = std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(pCommand)))
	{
		lRequest = GetPersistentRecvRequest(lPersistentCommand.get());
		if( MPI_CALL("MPI_Start", (MPI_Start(lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
	}
	else
	{
		SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
		MPI_Datatype lDataType = GetDataTypeMPI((pmCommunicatorCommand::communicatorDataTypes)(pCommand->GetDataType()));

        lRequest = new MPI_Request();
		if( MPI_CALL("MPI_Irecv", (MPI_Irecv(pData, pLength, lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));
	}

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
	mNonBlockingRequestMap[lRequest] = pCommand;
	
	if(mRequestCountMap.find(pCommand) == mRequestCountMap.end())
		mRequestCountMap[pCommand] = 1;
	else
		mRequestCountMap[pCommand] = mRequestCountMap[pCommand] + 1;

	CancelDummyRequest();	// Signal the other thread to handle the created request

	return pmSuccess;
}

pmStatus pmMPI::GlobalBarrier()
{
	if( MPI_CALL("MPI_Barrier", (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)) )
        PMTHROW(pmNetworkException(pmNetworkException::GLOBAL_BARRIER_ERROR));

	return pmSuccess;
}

pmStatus pmMPI::InitializePersistentCommand(pmPersistentCommunicatorCommand* pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	MPI_Request* lRequest = new MPI_Request();
	MPI_Comm lCommunicator;
	int lDest;

	pmCommunicatorCommand::communicatorCommandTypes lType = (pmCommunicatorCommand::communicatorCommandTypes)(pCommand->GetType());

	SAFE_GET_MPI_COMMUNICATOR_AND_DESTINATION(lCommunicator, lDest, pCommand->GetDestination());
	MPI_Datatype lDataType = GetDataTypeMPI((pmCommunicatorCommand::communicatorDataTypes)(pCommand->GetDataType()));

	if(lType == pmCommunicatorCommand::SEND)
	{
		if(mPersistentSendRequest.find(pCommand) != mPersistentSendRequest.end())
			PMTHROW(pmFatalErrorException());

		if( MPI_CALL("MPI_Send_init", (MPI_Send_init(pCommand->GetData(), (uint)(pCommand->GetDataLength()), lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));

		mPersistentSendRequest[pCommand] = lRequest;
	}
	else
	{
		if(lType == pmCommunicatorCommand::RECEIVE)
		{
			if(mPersistentRecvRequest.find(pCommand) != mPersistentRecvRequest.end())
				PMTHROW(pmFatalErrorException());

			if( MPI_CALL("MPI_Recv_init", (MPI_Recv_init(pCommand->GetData(), (uint)(pCommand->GetDataLength()), lDataType, lDest, (int)(pCommand->GetTag()), lCommunicator, lRequest) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

			mPersistentRecvRequest[pCommand] = lRequest;
		}
	}
    
	return pmSuccess;
}

pmStatus pmMPI::TerminatePersistentCommand(pmPersistentCommunicatorCommand* pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    CancelDummyRequest();

	pmCommunicatorCommand::communicatorCommandTypes lType = (pmCommunicatorCommand::communicatorCommandTypes)(pCommand->GetType());
	MPI_Request* lRequest = NULL;

	if(lType == pmCommunicatorCommand::SEND && mPersistentSendRequest.find(pCommand) != mPersistentSendRequest.end())
    {
		lRequest = mPersistentSendRequest[pCommand];
        mPersistentSendRequest.erase(pCommand);
    }
	else if(lType == pmCommunicatorCommand::RECEIVE && mPersistentRecvRequest.find(pCommand) != mPersistentRecvRequest.end())
    {
		lRequest = mPersistentRecvRequest[pCommand];
        mPersistentRecvRequest.erase(pCommand);
    }

    pmCommunicatorCommandPtr lCommand = mNonBlockingRequestMap[lRequest];
    
    if(mRequestCountMap.find(lCommand) != mRequestCountMap.end())
    {
        mRequestCountMap[lCommand] -= 1;
        if(mRequestCountMap[lCommand] == 0)
            mRequestCountMap.erase(lCommand);
    }
        
    mNonBlockingRequestMap.erase(lRequest);
    
	if( MPI_CALL("MPI_Request_free", (MPI_Request_free(lRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::REQUEST_FREE_ERROR));

    delete lRequest;
    
	return pmSuccess;
}

MPI_Request* pmMPI::GetPersistentSendRequest(pmPersistentCommunicatorCommand* pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mPersistentSendRequest.find(pCommand) == mPersistentSendRequest.end())
		PMTHROW(pmFatalErrorException());

	return mPersistentSendRequest[pCommand];
}

MPI_Request* pmMPI::GetPersistentRecvRequest(pmPersistentCommunicatorCommand* pCommand)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mPersistentRecvRequest.find(pCommand) == mPersistentRecvRequest.end())
		PMTHROW(pmFatalErrorException());

	return mPersistentRecvRequest[pCommand];
}

MPI_Datatype pmMPI::GetDataTypeMPI(pmCommunicatorCommand::communicatorDataTypes pDataType)
{
	switch(pDataType)
	{
		case pmCommunicatorCommand::BYTE:
			return MPI_BYTE;
			break;

		case pmCommunicatorCommand::INT:
			return MPI_INT;
			break;

		case pmCommunicatorCommand::UINT:
			return MPI_UNSIGNED;
			break;

		case pmCommunicatorCommand::REMOTE_TASK_ASSIGN_PACKED:
		case pmCommunicatorCommand::SUBTASK_REDUCE_PACKED:
		case pmCommunicatorCommand::MEMORY_RECEIVE_PACKED:
        case pmCommunicatorCommand::DATA_REDISTRIBUTION_PACKED:
        case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_PACKED:
        case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_PACKED:
        case pmCommunicatorCommand::OWNERSHIP_TRANSFER_PACKED:
			return MPI_PACKED;
			break;

		default:
		{
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());

			std::map<pmCommunicatorCommand::communicatorDataTypes, MPI_Datatype*>::iterator lIter = mRegisteredDataTypes.find(pDataType);
			if(lIter != mRegisteredDataTypes.end())
				return *(mRegisteredDataTypes[pDataType]);

			break;
		}
	}

	PMTHROW(pmFatalErrorException());
	return MPI_BYTE;
}


#define REGISTER_MPI_DATA_TYPE_HELPER_HEADER(cDataType, cName, headerMpiName) cDataType cName; \
	MPI_Aint headerMpiName; \
	SAFE_GET_MPI_ADDRESS(&cName, &headerMpiName);

#define REGISTER_MPI_DATA_TYPE_HELPER(headerMpiName, cName, mpiName, mpiDataType, index, blockLength) MPI_Aint mpiName; \
	SAFE_GET_MPI_ADDRESS(&cName, &mpiName); \
	lBlockLength[index] = blockLength; \
	lDisplacement[index] = mpiName - headerMpiName; \
	lDataType[index] = mpiDataType;

pmStatus pmMPI::RegisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType)
{
#ifdef _DEBUG
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());
        
        std::map<pmCommunicatorCommand::communicatorDataTypes, MPI_Datatype*>::iterator lIter = mRegisteredDataTypes.find(pDataType);
        if(lIter != mRegisteredDataTypes.end())
            PMTHROW(pmFatalErrorException());
    }
#endif

	int lFieldCount = 0;

    switch(pDataType)
	{
		case pmCommunicatorCommand::MACHINE_POOL_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::machinePool::FIELD_COUNT_VALUE;            
			break;
		}
            
		case pmCommunicatorCommand::DEVICE_POOL_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::devicePool::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::remoteTaskAssignStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::remoteSubtaskAssignStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::ownershipDataStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::ownershipChangeStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::sendAcknowledgementStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::TASK_EVENT_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::taskEventStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::STEAL_REQUEST_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::stealRequestStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::STEAL_RESPONSE_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::stealResponseStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::memoryIdentifierStruct::FIELD_COUNT_VALUE;
			break;
		}

		case pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::memoryTransferRequest::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::subtaskReduceStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::memoryReceiveStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::HOST_FINALIZATION_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::hostFinalizationStruct::FIELD_COUNT_VALUE;
			break;
		}
            
		case pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::redistributionOrderStruct::FIELD_COUNT_VALUE;
			break;
		}

		case pmCommunicatorCommand::DATA_REDISTRIBUTION_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::dataRedistributionStruct::FIELD_COUNT_VALUE;
			break;
		}
        
		case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::redistributionOffsetsStruct::FIELD_COUNT_VALUE;
			break;
		}

        case pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::subtaskRangeCancelStruct::FIELD_COUNT_VALUE;
			break;
		}

        case pmCommunicatorCommand::FILE_OPERATIONS_STRUCT:
		{
			lFieldCount = pmCommunicatorCommand::fileOperationsStruct::FIELD_COUNT_VALUE;
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
		case pmCommunicatorCommand::MACHINE_POOL_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::machinePool, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.cpuCores, lDataCoresMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.gpuCards, lDataCardsMPI, MPI_UNSIGNED, 1, 1);

			break;
		}

		case pmCommunicatorCommand::DEVICE_POOL_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::devicePool, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.name, lDataNameMPI, MPI_CHAR, 0, MAX_NAME_STR_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.description, lDataDescMPI, MPI_CHAR, 1, MAX_DESC_STR_LEN);

			break;
		}

		case pmCommunicatorCommand::REMOTE_TASK_ASSIGN_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::remoteTaskAssignStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskConfLength, lTaskConfLengthMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskId, lTaskIdMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.inputMemLength, lInputMemLengthMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.outputMemLength, lOutputMemLengthMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.inputMemInfo, lInputMemInfo, MPI_UNSIGNED_SHORT, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.outputMemInfo, lOutputMemInfo, MPI_UNSIGNED_SHORT, 5, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskCount, lSubtaskCountMPI, MPI_UNSIGNED_LONG, 6, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.callbackKey, lCallbackKeyMPI, MPI_CHAR, 7, MAX_CB_KEY_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.assignedDeviceCount, lAssignedDeviceCountMPI, MPI_UNSIGNED, 8, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 9, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 10, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.priority, lPriorityMPI, MPI_UNSIGNED_SHORT, 11, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.schedModel, lSchedModelMPI, MPI_UNSIGNED_SHORT, 12, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.inputMemGenerationNumber, lInputMemGenerationNumberMPI, MPI_UNSIGNED_LONG, 13, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.outputMemGenerationNumber, lOutputMemGenerationNumberMPI, MPI_UNSIGNED_LONG, 14, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.flags, lFlagsMPI, MPI_UNSIGNED_SHORT, 15, 1);

			break;
		}

		case pmCommunicatorCommand::REMOTE_SUBTASK_ASSIGN_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::remoteSubtaskAssignStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.assignmentType, lAssignmentTypeMPI, MPI_UNSIGNED_SHORT, 6, 1);

			break;
		}

		case pmCommunicatorCommand::OWNERSHIP_DATA_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::ownershipDataStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);

			break;
		}

		case pmCommunicatorCommand::OWNERSHIP_CHANGE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::ownershipChangeStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.newOwnerHost, lNewOwnerHostMPI, MPI_UNSIGNED, 2, 1);

			break;
		}
        
		case pmCommunicatorCommand::SEND_ACKNOWLEDGEMENT_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::sendAcknowledgementStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceDeviceGlobalIndex, lSourceDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.execStatus, lExecStatusMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 6, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.ownershipDataElements, lOwnershipDataElementsMPI, MPI_UNSIGNED, 7, 1);

			break;
		}

		case pmCommunicatorCommand::TASK_EVENT_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::taskEventStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.taskEvent, lTaskEventMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);

			break;
		}

		case pmCommunicatorCommand::STEAL_REQUEST_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::stealRequestStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.stealingDeviceGlobalIndex, lStealingDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.stealingDeviceExecutionRate, lStealingDeviceExecutionRateMPI, MPI_DOUBLE, 4, 1);

			break;
		}

		case pmCommunicatorCommand::STEAL_RESPONSE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::stealResponseStruct, lData, lDataMPI);
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

        case pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::memoryIdentifierStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memOwnerHost, lMemOwnerHostMPI, MPI_UNSIGNED, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.generationNumber, lGenerationNumberMPI, MPI_UNSIGNED_LONG, 1, 1);

            break;
        }

		case pmCommunicatorCommand::MEMORY_TRANSFER_REQUEST_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::memoryTransferRequest, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceMemIdentifier, lSourceMemIdentifierMPI, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT), 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.destMemIdentifier, lDestMemIdentifierMPI, GetDataTypeMPI(pmCommunicatorCommand::MEMORY_IDENTIFIER_STRUCT), 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.receiverOffset, lReceiverOffsetMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 4, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.destHost, lDestHostMPI, MPI_UNSIGNED_LONG, 5, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.isForwarded, lIsForwardedMPI, MPI_UNSIGNED_SHORT, 6, 1);

			break;
		}

		case pmCommunicatorCommand::SUBTASK_REDUCE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::subtaskReduceStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskId, lSubtaskIdMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.inputMemSubscriptionCount, lInputMemSubscriptionCountMPI, MPI_UNSIGNED, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.outputMemReadSubscriptionCount, lOutputMemReadSubscriptionCountMPI, MPI_UNSIGNED, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.outputMemWriteSubscriptionCount, lOutputMemWriteSubscriptionCountMPI, MPI_UNSIGNED, 5, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.writeOnlyUnprotectedPageRangesCount, lWriteOnlyUnprotectedPageRangesCountMPI, MPI_UNSIGNED, 6, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtaskMemLength, lSubtaskMemLengthMPI, MPI_UNSIGNED, 7, 1);

			break;
		}

		case pmCommunicatorCommand::MEMORY_RECEIVE_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::memoryReceiveStruct, lData, lDataMPI);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.memOwnerHost, lMemOwnerHostMPI, MPI_UNSIGNED, 0, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.generationNumber, lGenerationNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offset, lOffsetMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 3, 1);

			break;
		}
            
		case pmCommunicatorCommand::HOST_FINALIZATION_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::hostFinalizationStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.terminate, lTerminateMPI, MPI_UNSIGNED_SHORT, 0, 1);

			break;
		}

		case pmCommunicatorCommand::REDISTRIBUTION_ORDER_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::redistributionOrderStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.order, lOrderMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.length, lLengthMPI, MPI_UNSIGNED_LONG, 1, 1);
            
			break;
		}

		case pmCommunicatorCommand::DATA_REDISTRIBUTION_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::dataRedistributionStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.remoteHost, lRemoteHostMPI, MPI_UNSIGNED, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.subtasksAccounted, lSubtasksAccountedMPI, MPI_UNSIGNED_LONG, 3, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.orderDataCount, lOrderDataCountMPI, MPI_UNSIGNED, 4, 1);
            
			break;
		}
        
		case pmCommunicatorCommand::REDISTRIBUTION_OFFSETS_STRUCT:
		{
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::redistributionOffsetsStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.redistributedMemGenerationNumber, lRedistributedMemGenerationNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.offsetsDataCount, lOffsetsDataCountMPI, MPI_UNSIGNED, 3, 1);
            
			break;
		}

        case pmCommunicatorCommand::SUBTASK_RANGE_CANCEL_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::subtaskRangeCancelStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.targetDeviceGlobalIndex, lTargetDeviceGlobalIndexMPI, MPI_UNSIGNED, 0, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originatingHost, lOriginatingHostMPI, MPI_UNSIGNED, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sequenceNumber, lSequenceNumberMPI, MPI_UNSIGNED_LONG, 2, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.startSubtask, lStartSubtaskMPI, MPI_UNSIGNED_LONG, 3, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.endSubtask, lEndSubtaskMPI, MPI_UNSIGNED_LONG, 4, 1);
            REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.originalAllotteeGlobalIndex, lOriginalAllotteeGlobalIndexMPI, MPI_UNSIGNED, 5, 1);

			break;        
        }

        case pmCommunicatorCommand::FILE_OPERATIONS_STRUCT:
        {
			REGISTER_MPI_DATA_TYPE_HELPER_HEADER(pmCommunicatorCommand::fileOperationsStruct, lData, lDataMPI);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileName, lFileNameMPI, MPI_CHAR, 0, MAX_FILE_SIZE_LEN);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.fileOp, lFileOpMPI, MPI_UNSIGNED_SHORT, 1, 1);
			REGISTER_MPI_DATA_TYPE_HELPER(lDataMPI, lData.sourceHost, lSourceHostMPI, MPI_UNSIGNED, 2, 1);

			break;        
        }

		default:
			PMTHROW(pmFatalErrorException());
	}

	bool lError = false;
	MPI_Datatype* lNewType = new MPI_Datatype();

	if( (MPI_CALL("MPI_Type_create_struct", (MPI_Type_create_struct(lFieldCount, lBlockLength, lDisplacement, lDataType, lNewType) != MPI_SUCCESS))) || (MPI_CALL("MPI_Type_commit", (MPI_Type_commit(lNewType) != MPI_SUCCESS))) )
	{
		lError = true;
		delete lNewType;
	}

	if(lError)
		PMTHROW(pmNetworkException(pmNetworkException::DATA_TYPE_REGISTRATION));

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());
	mRegisteredDataTypes[pDataType] = lNewType;

	return pmSuccess;
}

pmStatus pmMPI::UnregisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mDataTypesResourceLock, Lock(), Unlock());
	
	std::map<pmCommunicatorCommand::communicatorDataTypes, MPI_Datatype*>::iterator lIter = mRegisteredDataTypes.find(pDataType);
	if(lIter != mRegisteredDataTypes.end())
	{
		MPI_CALL("MPI_Type_free", MPI_Type_free(mRegisteredDataTypes[pDataType]));
		delete mRegisteredDataTypes[pDataType];

		mRegisteredDataTypes.erase(pDataType);
	}

	return pmSuccess;
}

/* Must be called with mResourceLock acquired */
pmStatus pmMPI::SetupDummyRequest()
{
	if(!mDummyReceiveRequest)
	{
		mDummyReceiveRequest = new MPI_Request();

		if( MPI_CALL("MPI_Irecv", (MPI_Irecv(NULL, 0, MPI_BYTE, mHostId, PM_MPI_DUMMY_TAG, MPI_COMM_WORLD, mDummyReceiveRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CREATION_ERROR));
	}

	return pmSuccess;
}

/* Must be called with mResourceLock acquired */
pmStatus pmMPI::CancelDummyRequest()
{
	if(mDummyReceiveRequest)
	{
	        MPI_Request lRequest;
        
		if( MPI_CALL("MPI_Isend", (MPI_Isend(NULL, 0, MPI_BYTE, mHostId, PM_MPI_DUMMY_TAG, MPI_COMM_WORLD, &lRequest) != MPI_SUCCESS)) )
			PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CANCEL_ERROR));

//		if( MPI_CALL("MPI_Cancel", (MPI_Cancel(mDummyReceiveRequest) != MPI_SUCCESS)) )
//			PMTHROW(pmNetworkException(pmNetworkException::DUMMY_REQUEST_CANCEL_ERROR));
	}

	delete mDummyReceiveRequest;

	mDummyReceiveRequest = NULL;

	return pmSuccess;
}

pmStatus pmMPI::StopThreadExecution()
{
    if(mSignalWait)
        PMTHROW(pmFatalErrorException());

    mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();

	// Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        mThreadTerminationFlag = true;
        CancelDummyRequest();
    }

	mSignalWait->Wait();
    
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    delete mSignalWait;

	return pmSuccess;
}

#ifdef PROGRESSIVE_SLEEP_NETWORK_THREAD
pmStatus pmMPI::ThreadSwitchCallback(networkEvent& pCommand)
{
    /* Do not use pCommand in this function as it is NULL (passed in the constructor above) */

    // This loop terminates with the pmThread's destruction
    while(1)
    {
		if(mThreadTerminationFlag)
        {
            mSignalWait->Signal();
            return pmSuccess;
        }

		struct timespec lReqTime, lRemTime;
		lReqTime.tv_sec = mProgressiveSleepTime/1000;
		lReqTime.tv_nsec = (mProgressiveSleepTime%1000)*1000;
		nanosleep(&lReqTime, &lRemTime);

		if(mThreadTerminationFlag)
        {
            mSignalWait->Signal();
            return pmSuccess;
        }

        try
        {               
			FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

			size_t lRequestCount = mNonBlockingRequestMap.size();
			if(!lRequestCount)
				continue;

			std::vector<MPI_Request> lRequestVector(lRequestCount);
			std::vector<MPI_Request*> lRequestPtrVector(lRequestCount);
			std::vector<pmCommunicatorCommandPtr> lCommandVector(lRequestCount);

			MPI_Request* lRequestArray = &lRequestVector[0];
			pmCommunicatorCommandPtr* lCommandArray = &lCommandVector[0];
	
			size_t lIndex = 0;	
			std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lIter = mNonBlockingRequestMap.begin();
			std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lEndIter = mNonBlockingRequestMap.end();
			for(; lIter != lEndIter; ++lIter, ++lIndex)
			{
                lRequestPtrVector[lIndex] = lIter->first;
				lRequestArray[lIndex] = *(lIter->first);
				lCommandArray[lIndex] = lIter->second;
			}
			
			int lFlag, lFinishingRequestIndex;
			MPI_Status lFinishingRequestStatus;

			if( MPI_CALL("MPI_Testany", (MPI_Testany(lRequestCount, lRequestArray, &lFinishingRequestIndex, &lFlag, &lFinishingRequestStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::TEST_ERROR));

			if(lFlag && lFinishingRequestIndex != MPI_UNDEFINED)
			{
				pmCommunicatorCommandPtr lCommand = lCommandArray[lFinishingRequestIndex];

				mNonBlockingRequestMap.erase(lRequestPtrVector[lFinishingRequestIndex]);
                if(!std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand))
                    delete lRequestPtrVector[lFinishingRequestIndex];
                
                if(mRequestCountMap.find(lCommand) == mRequestCountMap.end())
                    PMTHROW(pmFatalErrorException());
                
                if(mRequestCountMap[lCommand] == 0)
                    PMTHROW(pmFatalErrorException());

				mRequestCountMap[lCommand] = (mRequestCountMap[lCommand] - 1);
				size_t lRemainingCount = mRequestCountMap[lCommand];

                if(lRemainingCount == 0)
                {
					mRequestCountMap.erase(lCommand);
                    if(std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand) && mPersistentReceptionFreezed)
                        continue;
                    
                    ushort lCommandType = lCommand->GetType();

                    switch(lCommandType)
                    {
                        case pmCommunicatorCommand::SEND:
                        case pmCommunicatorCommand::BROADCAST:
                        {
                            SendComplete(lCommand, pmSuccess);
                            break;
                        }

                        case pmCommunicatorCommand::RECEIVE:
                        {
                            ReceiveComplete(lCommand, pmSuccess);
                            break;
                        }

                        default:
                            PMTHROW(pmFatalErrorException());
                    }
                    
                    if(!std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand))
                        mCommandCompletionSignalWait.Signal();
                }

				mProgressiveSleepTime = MIN_PROGRESSIVE_SLEEP_TIME_MILLI_SECS;
			}
			else
			{
				mProgressiveSleepTime += PROGRESSIVE_SLEEP_TIME_INCREMENT_MILLI_SECS;
				if(mProgressiveSleepTime > MAX_PROGRESSIVE_SLEEP_TIME_MILLI_SECS)
					mProgressiveSleepTime = MAX_PROGRESSIVE_SLEEP_TIME_MILLI_SECS;
			}
        }
        catch(pmException& e)
        {
            pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from primary network thread");
        }
    }

    return pmSuccess;
}
#else
pmStatus pmMPI::ThreadSwitchCallback(networkEvent& pCommand)
{
	/* Do not use pCommand in this function as it is NULL (passed in the constructor above) */
	
	// This loop terminates with the pmThread's destruction
	while(1)
	{
		try
		{
            size_t lRequestCount = 0;
            
            std::vector<MPI_Request> lRequestVector;
            std::vector<MPI_Request*> lRequestPtrVector;
            std::vector<pmCommunicatorCommandPtr> lCommandVector;

            MPI_Request* lRequestArray = NULL;
            pmCommunicatorCommandPtr* lCommandArray = NULL;
            
			// Auto lock/unlock scope
			{
				FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

				if(mThreadTerminationFlag)
                {
                    mSignalWait->Signal();
					return pmSuccess;
                }

				SetupDummyRequest();

                lRequestCount = mNonBlockingRequestMap.size();
				++lRequestCount; // Adding one for dummy request
                
                lRequestVector.resize(lRequestCount);
                lRequestPtrVector.resize(lRequestCount);
                lCommandVector.resize(lRequestCount);
                
                lRequestArray = &lRequestVector[0];
                lCommandArray = &lCommandVector[0];
                
				lRequestArray[0] = *mDummyReceiveRequest;
                
                size_t lIndex = 1;
                std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lIter = mNonBlockingRequestMap.begin();
                std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lEndIter = mNonBlockingRequestMap.end();
                for(; lIter != lEndIter; ++lIter, ++lIndex)
                {
                    lRequestPtrVector[lIndex] = lIter->first;
                    lRequestArray[lIndex] = *(lIter->first);
                    lCommandArray[lIndex] = lIter->second;
                }
			}

			int lFinishingRequestIndex;
			MPI_Status lFinishingRequestStatus;

			if( MPI_CALL("MPI_Waitany", (MPI_Waitany((int)lRequestCount, lRequestArray, &lFinishingRequestIndex, &lFinishingRequestStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::WAIT_ERROR));

            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            
			if(lFinishingRequestIndex == 0)		// Dummy Request
			{
                if(mThreadTerminationFlag)
                {
                    mSignalWait->Signal();
                    return pmSuccess;
                }
			}
			else
			{
				pmCommunicatorCommandPtr lCommand = lCommandArray[lFinishingRequestIndex];

                mNonBlockingRequestMap.erase(lRequestPtrVector[lFinishingRequestIndex]);
                if(!std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand))
                    delete lRequestPtrVector[lFinishingRequestIndex];

                if(mRequestCountMap.find(lCommand) == mRequestCountMap.end())
                    PMTHROW(pmFatalErrorException());

                size_t lRemainingCount = mRequestCountMap[lCommand];
                --lRemainingCount;
                
                mRequestCountMap[lCommand] = lRemainingCount;

				if(lRemainingCount == 0)
				{
					mRequestCountMap.erase(lCommand);
                    if(std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand) && mPersistentReceptionFreezed)
                        continue;

					ushort lCommandType = lCommand->GetType();

					switch(lCommandType)
					{
						case pmCommunicatorCommand::SEND:
						case pmCommunicatorCommand::BROADCAST:
						{
							SendComplete(lCommand, pmSuccess);
							break;
						}

						case pmCommunicatorCommand::RECEIVE:
						{
							ReceiveComplete(lCommand, pmSuccess);
							break;
						}

						default:
							PMTHROW(pmFatalErrorException());
					}
                    
                    if(!std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lCommand))
                        mCommandCompletionSignalWait.Signal();
				}
			}
		}
		catch(pmException& e)
		{
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from primary network thread");
		}
	}

	return pmSuccess;
}
#endif
    
pmStatus pmMPI::FreezeReceptionAndFinishCommands()
{
    /* Set no more persistent command reception; steal commands may be active even after tasks are destroyed */

	// Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        mPersistentReceptionFreezed = true;
    }

    // Wait for all non-persistent commands to finish
    while(1)
    {
        bool lFound = false;

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

            std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lIter = mNonBlockingRequestMap.begin();
            std::map<MPI_Request*, pmCommunicatorCommandPtr>::iterator lEndIter = mNonBlockingRequestMap.end();
            for(; lIter != lEndIter; ++lIter)
            {
                if(!std::tr1::dynamic_pointer_cast<pmPersistentCommunicatorCommand>(lIter->second))
                {
                    lFound = true;
                    break;
                }
            }
            
            if(!lFound)
                break;
        }
        
        mCommandCompletionSignalWait.Wait();
    }

    return pmSuccess;
}


/* class pmMPI::pmUnknownLengthReceiveThread */
pmMPI::pmUnknownLengthReceiveThread::pmUnknownLengthReceiveThread(pmMPI* pMPI)
	: mMPI(pMPI)
	, mThreadTerminationFlag(false)
    , mSignalWait(NULL)
    , mResourceLock __LOCK_NAME__("pmMPI::pmUnknownLengthReceiveThread::mResourceLock")
{
	networkEvent lNetworkEvent; 
	
	SwitchThread(lNetworkEvent, MAX_PRIORITY_LEVEL);
}

pmMPI::pmUnknownLengthReceiveThread::~pmUnknownLengthReceiveThread()
{
	StopThreadExecution();

	#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down unknown length network thread");
	#endif
}

pmStatus pmMPI::pmUnknownLengthReceiveThread::StopThreadExecution()
{
	MPI_Request lRequest;
    
    if(mSignalWait)
        PMTHROW(pmFatalErrorException());
    
    mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();

	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
		mThreadTerminationFlag = true;
		SendDummyProbeCancellationMessage(lRequest);
	}

	mSignalWait->Wait();
    
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    //delete[] lData;
    delete mSignalWait;
    
	return pmSuccess;
}

pmStatus pmMPI::pmUnknownLengthReceiveThread::SendDummyProbeCancellationMessage(MPI_Request& pRequest)
{
	if( MPI_CALL("MPI_Isend", (MPI_Isend(NULL, 0, MPI_CHAR, mMPI->GetHostId(), pmCommunicatorCommand::UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &pRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::SEND_ERROR));
    
	return pmSuccess;
}

pmStatus pmMPI::pmUnknownLengthReceiveThread::ReceiveDummyProbeCancellationMessage()
{
	MPI_Request lRequest;
	MPI_Status lStatus;

	if( MPI_CALL("MPI_Irecv", (MPI_Irecv(NULL, 0, MPI_CHAR, mMPI->GetHostId(), pmCommunicatorCommand::UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &lRequest) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

	if( MPI_CALL("MPI_Wait", (MPI_Wait(&lRequest, &lStatus) != MPI_SUCCESS)) )
		PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

	return pmSuccess;
}
    
pmStatus pmMPI::pmUnknownLengthReceiveThread::ThreadSwitchCallback(networkEvent& pCommand)
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
					return pmSuccess;
				}
			}
			
			if( MPI_CALL("MPI_Probe", (MPI_Probe(MPI_ANY_SOURCE, pmCommunicatorCommand::UNKNOWN_LENGTH_TAG, MPI_COMM_WORLD, &lProbeStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::PROBE_ERROR));
			
			// Auto lock/unlock scope
			{
				FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
				if(mThreadTerminationFlag)
				{
					ReceiveDummyProbeCancellationMessage();
                    mSignalWait->Signal();
					//SendReverseDummyProbeCancellationMessage();
					return pmSuccess;
				}
			}

			int lLength;
			if( MPI_CALL("MPI_Get_count", (MPI_Get_count(&lProbeStatus, MPI_PACKED, &lLength) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::GET_COUNT_ERROR));

            char* lPackedData = new char[lLength];

			if( MPI_CALL("MPI_Recv", (MPI_Recv(lPackedData, lLength, MPI_PACKED, lProbeStatus.MPI_SOURCE, (int)(pmCommunicatorCommand::UNKNOWN_LENGTH_TAG), MPI_COMM_WORLD, &lRecvStatus) != MPI_SUCCESS)) )
				PMTHROW(pmNetworkException(pmNetworkException::RECEIVE_ERROR));

            pmHeavyOperationsThreadPool::GetHeavyOperationsThreadPool()->UnpackDataEvent(lPackedData, lLength, MAX_CONTROL_PRIORITY);
		}
		catch(pmException& e)
		{
			pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, "Exception generated from secondary network thread");
		}
	}

	return pmSuccess;
}

} // end namespace pm




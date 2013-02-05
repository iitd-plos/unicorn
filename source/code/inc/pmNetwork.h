
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

#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"
#include "pmThread.h"
#include "pmCommand.h"

#include "mpi.h"
#include <map>
#include <vector>

namespace pm
{

class pmCluster;

namespace network
{

typedef struct networkEvent : public pmBasicThreadEvent
{
} networkEvent;

}

/**
 * \brief The base network class of PMLIB.
 * This class implements non-blocking send, receive and broadcast operations
 * and also calls MarkExecutionStart and MarkExecutionEnd for timing and signal
 * operations on command objects. Callers call WaitForFinish on command objects
 * to simulate blocking send or receive operations.
 * This class serves as a factory class to various network implementations.
 * This class has interface to pmCommunicator. The communicator talks in pmCommand
 * objects only while the pmNetwork deals with the actual bytes transferred.
 * Only one instance of pmNetwork class is created on each machine.
*/

class pmNetwork : public pmBase
{
    friend class pmHeavyOperationsThread;

	public:
		virtual pmStatus SendNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus All2AllNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
        virtual pmStatus FreezeReceptionAndFinishCommands() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;

		virtual pmStatus RegisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType) = 0;
		virtual pmStatus UnregisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType) = 0;
	
		virtual pmStatus PackData(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmCommunicatorCommandPtr UnpackData(void* pPackedData, int pDataLength) = 0;

		virtual pmStatus InitializePersistentCommand(pmPersistentCommunicatorCommand* pCommand) = 0;
		virtual pmStatus TerminatePersistentCommand(pmPersistentCommunicatorCommand* pCommand) = 0;

		virtual pmStatus GlobalBarrier() = 0;

		pmNetwork();
		virtual ~pmNetwork();

    protected:
		pmStatus SendComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus);
		pmStatus ReceiveComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus);
    
        static pmNetwork* mNetwork;
};

class pmMPI : public pmNetwork, public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
{
    friend class pmController;

    public:
		enum pmRequestTag
		{
			PM_MPI_DUMMY_TAG = pmCommunicatorCommand::MAX_COMMUNICATOR_COMMAND_TAGS,
			PM_MPI_REVERSE_DUMMY_TAG = pmCommunicatorCommand::MAX_COMMUNICATOR_COMMAND_TAGS
		};

		typedef class pmUnknownLengthReceiveThread : public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
		{
			public:
				pmUnknownLengthReceiveThread(pmMPI* mMPI);
				virtual ~pmUnknownLengthReceiveThread();

				virtual pmStatus ThreadSwitchCallback(network::networkEvent& pCommand);

			private:
				pmStatus StopThreadExecution();
				pmStatus SendDummyProbeCancellationMessage(MPI_Request& pRequest);
				pmStatus ReceiveDummyProbeCancellationMessage();

				pmMPI* mMPI;
				bool mThreadTerminationFlag;
                pmSignalWait* mSignalWait;
				std::vector<pmCommunicatorCommandPtr> mReceiveCommands;
				RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		} pmUnknownLengthReceiveThread;

		static pmNetwork* GetNetwork();

		virtual pmStatus SendNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus All2AllNonBlocking(pmCommunicatorCommandPtr pCommand);
        virtual pmStatus FreezeReceptionAndFinishCommands();

		virtual pmStatus PackData(pmCommunicatorCommandPtr pCommand);
		virtual pmCommunicatorCommandPtr UnpackData(void* pPackedData, int pDataLength);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

		MPI_Datatype GetDataTypeMPI(pmCommunicatorCommand::communicatorDataTypes pDataType);
		virtual pmStatus RegisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType);
		virtual pmStatus UnregisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType);

		virtual pmStatus ThreadSwitchCallback(network::networkEvent& pCommand);

		virtual pmStatus InitializePersistentCommand(pmPersistentCommunicatorCommand* pCommand);
		virtual pmStatus TerminatePersistentCommand(pmPersistentCommunicatorCommand* pCommand);

		virtual pmStatus GlobalBarrier();

	private:
		pmMPI();
		virtual ~pmMPI();

		pmStatus StopThreadExecution();

        bool IsUnknownLengthTag(pmCommunicatorCommand::communicatorCommandTags pTag);
        pmStatus SendNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength);
		pmStatus ReceiveNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength);

		virtual MPI_Request* GetPersistentSendRequest(pmPersistentCommunicatorCommand* pCommand);
		virtual MPI_Request* GetPersistentRecvRequest(pmPersistentCommunicatorCommand* pCommand);

		pmStatus SetupDummyRequest();
		pmStatus CancelDummyRequest();

		uint mTotalHosts;
		uint mHostId;

		std::map<MPI_Request*, pmCommunicatorCommandPtr> mNonBlockingRequestMap;	// Map of MPI_Request objects and corresponding pmCommunicatorCommand objects
		std::map<pmCommunicatorCommandPtr, size_t> mRequestCountMap;	// Maps MpiCommunicatorCommand object to the number of MPI_Requests issued
		MPI_Request* mDummyReceiveRequest;
		std::map<pmCommunicatorCommand::communicatorDataTypes, MPI_Datatype*> mRegisteredDataTypes;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mDataTypesResourceLock;	   // Resource lock on mRegisteredDataTypes

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	   // Resource lock on mDummyReceiveRequest, mNonBlockingRequestMap, mResourceCountMap, mPersistentSendRequest, mPersistentRecvRequest

		std::map<pmPersistentCommunicatorCommand*, MPI_Request*> mPersistentSendRequest;
		std::map<pmPersistentCommunicatorCommand*, MPI_Request*> mPersistentRecvRequest;

        pmSignalWait* mSignalWait;
    
		bool mThreadTerminationFlag;
        bool mPersistentReceptionFreezed;
    
#ifdef PROGRESSIVE_SLEEP_NETWORK_THREAD
		long mProgressiveSleepTime;	// in ms
#endif

		pmUnknownLengthReceiveThread* mReceiveThread;
    
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mCommandCompletionSignalWait;
};

} // end namespace pm

#endif

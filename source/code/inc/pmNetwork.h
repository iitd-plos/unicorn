
#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmInternalDefinitions.h"
#include "pmResourceLock.h"
#include "pmThread.h"
#include "pmCommand.h"

#include "mpi.h"
#include <map>
#include <vector>

namespace pm
{

class pmCluster;

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
	public:
		virtual pmStatus SendNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus All2AllNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus ReceiveAllocateUnpackNonBlocking(pmCommunicatorCommandPtr pCommand) = 0;		// Receive Packed Data of unknown size

		virtual pmStatus DestroyNetwork() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;

		virtual pmStatus RegisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType) = 0;
		virtual pmStatus UnregisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType) = 0;
	
		virtual pmStatus PackData(pmCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus UnpackData(pmCommunicatorCommandPtr pCommand, void* pPackedData, int pDataLength) = 0;

		virtual pmStatus InitializePersistentCommand(pmPersistentCommunicatorCommandPtr pCommand) = 0;
		virtual pmStatus TerminatePersistentCommand(pmPersistentCommunicatorCommandPtr pCommand) = 0;

		pmNetwork();
		virtual ~pmNetwork();

	private:
		virtual MPI_Request GetPersistentSendRequest(pmPersistentCommunicatorCommandPtr pCommand) = 0;
		virtual MPI_Request GetPersistentRecvRequest(pmPersistentCommunicatorCommandPtr pCommand) = 0;
};

class pmMPI : public pmNetwork, public THREADING_IMPLEMENTATION_CLASS
{
	public:
		enum pmRequestTag
		{
			PM_MPI_DUMMY_TAG = pmCommunicatorCommand::MAX_COMMUNICATOR_COMMAND_TAGS
		};

		typedef class pmUnknownLengthReceiveThread : public THREADING_IMPLEMENTATION_CLASS
		{
			public:
				pmUnknownLengthReceiveThread(pmMPI* pMPI);
				virtual ~pmUnknownLengthReceiveThread();

				virtual pmStatus ThreadSwitchCallback(pmThreadCommandPtr pCommand);
				//pmStatus EnqueueReceiveCommand(pmCommunicatorCommandPtr pCommand);

			private:
				pmMPI* mMPI;
				std::vector<pmCommunicatorCommandPtr> mReceiveCommands;
				RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		} pmUnknownLengthReceiveThread;

		static pmNetwork* GetNetwork();
		virtual pmStatus DestroyNetwork();

		virtual pmStatus SendNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus All2AllNonBlocking(pmCommunicatorCommandPtr pCommand);

		virtual pmStatus PackData(pmCommunicatorCommandPtr pCommand);
		virtual pmStatus UnpackData(pmCommunicatorCommandPtr pCommand, void* pPackedData, int pDataLength, int pPos);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

		MPI_Datatype GetDataTypeMPI(pmCommunicatorCommand::communicatorDataTypes pDataType);
		virtual pmStatus RegisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType);
		virtual pmStatus UnregisterTransferDataType(pmCommunicatorCommand::communicatorDataTypes pDataType);

		virtual pmStatus ThreadSwitchCallback(pmThreadCommandPtr pCommand);

		virtual pmStatus ReceiveAllocateUnpackNonBlocking(pmCommunicatorCommandPtr pCommand);		// Receive Packed Data of unknown size

		virtual pmStatus InitializePersistentCommand(pmPersistentCommunicatorCommandPtr pCommand);
		virtual pmStatus TerminatePersistentCommand(pmPersistentCommunicatorCommandPtr pCommand);

	private:
		pmMPI();
		virtual ~pmMPI();

		pmStatus SendNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength);
		pmStatus ReceiveNonBlockingInternal(pmCommunicatorCommandPtr pCommand, void* pData, int pLength);

		pmStatus SendComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus);
		pmStatus ReceiveComplete(pmCommunicatorCommandPtr pCommand, pmStatus pStatus);

		virtual MPI_Request GetPersistentSendRequest(pmPersistentCommunicatorCommandPtr pCommand);
		virtual MPI_Request GetPersistentRecvRequest(pmPersistentCommunicatorCommandPtr pCommand);

		pmStatus SetupDummyRequest();
		pmStatus CancelDummyRequest();

		static pmNetwork* mNetwork;
		uint mTotalHosts;
		uint mHostId;

		std::vector< std::pair<MPI_Request, pmCommunicatorCommandPtr> > mNonBlockingRequestVector;	// Vector of MPI_Request objects and corresponding pmCommunicatorCommand objects
		std::map<pmCommunicatorCommandPtr, size_t> mRequestCountMap;	// Maps MpiCommunicatorCommand object to the number of MPI_Requests issued
		MPI_Request* mDummyReceiveRequest;
		std::map<pmCommunicatorCommand::communicatorDataTypes, MPI_Datatype*> mRegisteredDataTypes;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mDataTypesResourceLock;	   // Resource lock on mRegisteredDataTypes

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	   // Resource lock on mDummyReceiveRequest, mNonBlockingRequestVector and mResourceCountMap

		std::map<pmPersistentCommunicatorCommandPtr, MPI_Request> mPersistentSendRequest;
		std::map<pmPersistentCommunicatorCommandPtr, MPI_Request> mPersistentRecvRequest;

		pmUnknownLengthReceiveThread mReceiveThread;
};

} // end namespace pm

#endif

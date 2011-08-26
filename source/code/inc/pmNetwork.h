
#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmInternalDefinitions.h"
#include "pmResourceLock.h"
#include "pmThread.h"

#include "mpi.h"
#include <map>
#include <vector>

namespace pm
{

class pmCluster;
class pmCommunicatorCommand;
class pmThreadCommand;
class pmSignalWait;

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
		virtual pmStatus SendNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware) = 0;
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware) = 0;
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware) = 0;

		virtual pmStatus DestroyNetwork() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;
	private:
};

class pmMPI : public pmNetwork, public THREADING_IMPLEMENTATION_CLASS
{
	public:
		enum pmRequestTag
		{
			PM_MPI_DUMMY_TAG
		};

		static pmNetwork* GetNetwork();
		virtual pmStatus DestroyNetwork();

		virtual pmStatus SendNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware);
		virtual pmStatus BroadcastNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware);
		virtual pmStatus ReceiveNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand);

	private:
		pmMPI();

		pmStatus SendNonBlockingInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware);
		pmStatus ReceiveNonBlockingInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware);

		pmStatus SendComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus);
		pmStatus ReceiveComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus);

		pmStatus SetupDummyRequest();
		pmStatus CancelDummyRequest();

		static pmNetwork* mNetwork;
		uint mTotalHosts;
		uint mHostId;

		std::vector< std::pair<MPI_Request, pmCommunicatorCommand*> > mNonBlockingRequestVector;	// Vector of MPI_Request objects and corresponding pmCommunicatorCommand objects
		std::map<pmCommunicatorCommand*, size_t> mRequestCountMap;	// Maps MpmCommunicatorCommand object to the number of MPI_Requests issued
		MPI_Request* mDummyReceiveRequest;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	   // Resource lock on mDummyReceiveRequest, mNonBlockingRequestMap and mResourceCountMap
};

} // end namespace pm

#endif

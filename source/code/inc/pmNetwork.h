
#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmInternalDefinitions.h"
#include "pmResourceLock.h"

#include "mpi.h"

#include <vector>
#include <map>

namespace pm
{

class pmCluster;
class pmCommunicatorCommand;

/**
 * \brief The base network class of PMLIB.
 * This class serves as a factory class to various network implementations.
 * This class has interface to pmCommunicator. The communicator talks in pmCommand
 * objects only while the pmNetwork deals with the actual bytes transferred.
 * Only one instance of pmNetwork class is created on each machine.
*/

class pmNetwork
{
	public:
		virtual pmStatus Send(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false) = 0;
		virtual pmStatus Broadcast(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false) = 0;
		virtual pmStatus Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false) = 0;

		virtual pmStatus DestroyNetwork() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;
	private:
};

class pmMPI : public pmNetwork
{
	public:
		static pmNetwork* GetNetwork();
		virtual pmStatus DestroyNetwork();

		virtual pmStatus Send(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		virtual pmStatus Broadcast(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		virtual pmStatus Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

	private:
		pmMPI();

		pmStatus SendInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware, bool pBlocking = false);
		pmStatus ReceiveInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware, bool pBlocking = false);

		pmStatus SendComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus);
		pmStatus ReceiveComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus);

		static pmNetwork* mNetwork;
		uint mTotalHosts;
		uint mHostId;

		std::map<MPI_Request*, pmCommunicatorCommand*> mNonBlockingRequestMap;	// Maps MPI_Request objects to pmCommunicatorCommand object
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

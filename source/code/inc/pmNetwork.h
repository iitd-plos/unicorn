
#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmInternalDefinitions.h"
#include "mpi.h"

namespace pm
{

class pmCluster;

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
		virtual pmStatus SendByteArrayToHost(char* pArray, ulong pLength, bool pBlocking, uint pHost) = 0;
		virtual pmStatus SendByteArrayToCluster(char* pArray, ulong pLength, bool pBlocking, pmCluster& pCluster) = 0;
		virtual pmStatus BroadcastByteArray(char* pArray, ulong pLength, bool pBlocking) = 0;

		virtual pmStatus ReceiveByteArrayFromHost(char* pArray, ulong pLength, bool pBlocking, uint pHost) = 0;
		virtual pmStatus ReceiveBroadcastedByteArray(char* pArray, ulong pLength, bool pBlocking) = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;

		virtual pmStatus DestroyNetwork() = 0;
	private:
};

class pmMPI : public pmNetwork
{
	public:
		static pmNetwork* GetNetwork();
		virtual pmStatus DestroyNetwork();

		virtual pmStatus SendByteArrayToHost(char* pArray, ulong pLength, bool pBlocking, uint pHost);
		virtual pmStatus SendByteArrayToCluster(char* pArray, ulong pLength, bool pBlocking, pmCluster& pCluster);
		virtual pmStatus BroadcastByteArray(char* pArray, ulong pLength, bool pBlocking);

		virtual pmStatus ReceiveByteArrayFromHost(char* pArray, ulong pLength, bool pBlocking, uint pHost);
		virtual pmStatus ReceiveBroadcastedByteArray(char* pArray, ulong pLength, bool pBlocking);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

	private:
		pmMPI();

		static pmNetwork* mNetwork;
		uint mTotalHosts;
		uint mHostId;
};

} // end namespace pm

#endif

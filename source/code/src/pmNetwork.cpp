
#include "pmNetwork.h"

namespace pm
{

/* pmMPI Class */
pmNetwork* pmMPI::mNetwork = NULL;

pmNetwork* pmMPI::GetNetwork()
{
	if(!mNetwork)
		mNetwork = new pmMPI();

	return mNetwork;
}

pmStatus pmMPI::DestroyNetwork()
{
	delete mNetwork;
	mNetwork = NULL;

	if(MPI_Finalize() == MPI_SUCCESS)
		return pmSuccess;

	return pmNetworkTerminationError;
}

pmMPI::pmMPI()
{
	int lMpiStatus;
	int lArgc = 0;

	if((lMpiStatus = MPI_Init(&lArgc, NULL)) != MPI_SUCCESS)
	{
		MPI_Abort(MPI_COMM_WORLD, lMpiStatus);
		mTotalHosts = 0;
		mHostId = 0;
		throw pmInitExceptionMPI();
	}

	int lHosts, lId;

	MPI_Comm_size(MPI_COMM_WORLD, &lHosts);
	MPI_Comm_rank(MPI_COMM_WORLD, &lId);

	mTotalHosts = lHosts;
	mHostId = lId;
}

pmStatus pmMPI::SendByteArrayToHost(char* pArray, ulong pLength, bool pBlocking, uint pHost)
{
	return pmSuccess;
}

pmStatus pmMPI::SendByteArrayToCluster(char* pArray, ulong pLength, bool pBlocking, pmCluster& pCluster)
{
	return pmSuccess;
}

pmStatus pmMPI::BroadcastByteArray(char* pArray, ulong pLength, bool pBlocking)
{
	return pmSuccess;
}

pmStatus pmMPI::ReceiveByteArrayFromHost(char* pArray, ulong pLength, bool pBlocking, uint pHost)
{
	return pmSuccess;
}

pmStatus pmMPI::ReceiveBroadcastedByteArray(char* pArray, ulong pLength, bool pBlocking)
{
	return pmSuccess;
}

} // end namespace pm




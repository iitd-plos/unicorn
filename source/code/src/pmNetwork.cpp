
#include "pmNetwork.h"
#include "pmCommand.h"
#include "pmResourceLock.h"
#include "pmDevicePool.h"

namespace pm
{

#define SAFE_GET_DEVICE_POOL(x) { x = pmDevicePool::GetDevicePool(); if(!x) throw pmFatalErrorException(); }

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
		throw pmMpiInitException();
	}

	int lHosts, lId;

	MPI_Comm_size(MPI_COMM_WORLD, &lHosts);
	MPI_Comm_rank(MPI_COMM_WORLD, &lId);

	mTotalHosts = lHosts;
	mHostId = lId;
}

pmStatus pmMPI::Send(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataLength();

	if(!lData || lLength == 0)
		return pmSuccess;

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	int lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	for(ulong i=0; i<lBlocks; ++i)
		SendInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT, pHardware, pBlocking);

	if(lLastBlockLength)
		SendInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), lLastBlockLength, pHardware, pBlocking);

	if(pBlocking)
		SendComplete(pCommand, pmSuccess);

	return pmSuccess;
}

pmStatus pmMPI::Broadcast(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	// Call pmMPI::Send from here on MPI_COMM_WORLD or use an MPI API directly and add other code as in pmMPI::Send

	return pmSuccess;
}

pmStatus pmMPI::Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataLength();

	if(!lData || lLength == 0)
		return pmSuccess;

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	int lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	for(ulong i=0; i<lBlocks; ++i)
		ReceiveInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT, pHardware, pBlocking);

	if(lLastBlockLength)
		ReceiveInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), lLastBlockLength, pHardware, pBlocking);

	if(pBlocking)
		ReceiveComplete(pCommand, pmSuccess);

	return pmSuccess;
}

pmStatus pmMPI::SendInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware, bool pBlocking /* = false */)
{
	pCommand->MarkExecutionStart();

	if(pBlocking)
	{
		//int lStatus = MPI_Send(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm);
		// Throw from here if any error
	}
	else
	{
		MPI_Request* lRequest = NULL;
	
		//int lStatus = MPI_Isend(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm, lRequest);
		// Throw from here if any error

		mResourceLock.Lock();
		mNonBlockingRequestMap[lRequest] = pCommand;
		mResourceLock.Unlock();
	}

	// Call SendComplete and throw from here if any error

	return pmSuccess;
}

pmStatus pmMPI::ReceiveInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware, bool pBlocking /* = false */)
{
	pCommand->MarkExecutionStart();

	if(pBlocking)
	{
		//MPI_Status lMpiStatus;

		//int lStatus = MPI_Recv(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm, lMpiStatus);
	}
	else
	{
		MPI_Request* lRequest = NULL;

		//int lStatus = MPI_Irecv(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm, lRequest);
		// Throw from here if any error

		mResourceLock.Lock();
		mNonBlockingRequestMap[lRequest] = pCommand;
		mResourceLock.Unlock();
	}

	// Call ReceiveComplete and throw from here if any error

	return pmSuccess;
}

pmStatus pmMPI::SendComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus);

	/*
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

	lDevicePool->RegisterSendCompletion(lMachineId, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	*/

	return pmSuccess;
}

pmStatus pmMPI::ReceiveComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus);

	/*
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

	lDevicePool->RegisterReceiveCompletion(lMachineId, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	*/

	return pmSuccess;
}

} // end namespace pm




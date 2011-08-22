
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

	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	mDummyReceiveRequest = NULL;

	SwitchThread(NULL);	// Create an infinite loop in a new thread
}

pmStatus pmMPI::SendNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataLength();

	if(!lData || lLength == 0)
		return pmSuccess;

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	int lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	for(ulong i=0; i<lBlocks; ++i)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT, pHardware);

	if(lLastBlockLength)
		SendNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), lLastBlockLength, pHardware);

	return pmSuccess;
}

pmStatus pmMPI::BroadcastNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware)
{
	// Call pmMPI::Send from here on MPI_COMM_WORLD or use an MPI API directly and add other code as in pmMPI::Send

	return pmSuccess;
}

pmStatus pmMPI::ReceiveNonBlocking(pmCommunicatorCommand* pCommand, pmHardware pHardware)
{
	void* lData = pCommand->GetData();
	ulong lLength = pCommand->GetDataLength();

	if(!lData || lLength == 0)
		return pmSuccess;

	ulong lBlocks = lLength/MPI_TRANSFER_MAX_LIMIT;
	
	int lLastBlockLength = lLength - lBlocks * MPI_TRANSFER_MAX_LIMIT;

	for(ulong i=0; i<lBlocks; ++i)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + i*MPI_TRANSFER_MAX_LIMIT), MPI_TRANSFER_MAX_LIMIT, pHardware);

	if(lLastBlockLength)
		ReceiveNonBlockingInternal(pCommand, (void*)((char*)lData + lBlocks*MPI_TRANSFER_MAX_LIMIT), lLastBlockLength, pHardware);

	return pmSuccess;
}

pmStatus pmMPI::SendNonBlockingInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware)
{
	std::map<pmCommunicatorCommand*, int>::iterator lIter;

	pCommand->MarkExecutionStart();

	MPI_Request lRequest;
	
	//int lStatus = MPI_Isend(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm, &lRequest);
	// Throw from here if any error

	if(!lRequest)
		throw pmNetworkException(pmNetworkException::SEND_ERROR);

	mResourceLock.Lock();
	mNonBlockingRequestVector.push_back(std::pair<MPI_Request, pmCommunicatorCommand*>(lRequest, pCommand));
	
	std::map<pmCommunicatorCommand*, size_t>::iterator lIter = mRequestCountMap.find(pCommand);
	if(lIter == mRequestCountMap.end())
		mRequestCountMap[pCommand] = 0;
	else
		mRequestCountMap[pCommand] = mRequestCountMap[pCommand] + 1;

	CancelDummyRequest();	// Signal the other thread to handle the created request

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmMPI::ReceiveNonBlockingInternal(pmCommunicatorCommand* pCommand, void* pData, int pLength, pmHardware pHardware)
{
	pCommand->MarkExecutionStart();

	MPI_Request lRequest = NULL;

	//int lStatus = MPI_Irecv(pData, pLength, MPI_BYTE, lDest, lTag, lMpiComm, &lRequest);
	// Throw from here if any error

	if(!lRequest)
		throw pmNetworkException(pmNetworkException::RECEIVE_ERROR);

	mResourceLock.Lock();
	mNonBlockingRequestVector.push_back(std::pair<MPI_Request, pmCommunicatorCommand*>(lRequest, pCommand));
	
	std::map<pmCommunicatorCommand*, size_t>::iterator lIter = mRequestCountMap.find(pCommand);
	if(lIter == mRequestCountMap.end())
		mRequestCountMap[pCommand] = 0;
	else
		mRequestCountMap[pCommand] = mRequestCountMap[pCommand] + 1;

	CancelDummyRequest();	// Signal the other thread to handle the created request

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmMPI::SendComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus);

	/*
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

	lDevicePool->RegisterSendCompletion(lDestMachineId, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	*/

	return pmSuccess;
}

pmStatus pmMPI::ReceiveComplete(pmCommunicatorCommand* pCommand, pmStatus pStatus)
{
	pCommand->MarkExecutionEnd(pStatus);

	/*
	pmDevicePool* lDevicePool;
	SAFE_GET_DEVICE_POOL(lDevicePool);

	lDevicePool->RegisterReceiveCompletion(lSourceMachineId, pCommand->GetDataLength(), pCommand->GetExecutionTimeInSecs());
	*/

	return pmSuccess;
}

/* Must be called with mResourceLock acquired */
pmStatus pmMPI::SetupDummyRequest()
{
	if(!mDummyReceiveRequest)
	{
		void* lDummyBuffer;
		if( MPI_Irecv(lDummyBuffer, 1, MPI_BYTE, mHostId, PM_MPI_DUMMY_TAG, MPI_COMM_WORLD, mDummyReceiveRequest) != MPI_SUCCESS )
			throw pmNetworkException(pmNetworkException::DUMMY_REQUEST_CREATION_ERROR);
	}

	return pmSuccess;
}

/* Must be called with mResourceLock acquired */
pmStatus pmMPI::CancelDummyRequest()
{
	if(mDummyReceiveRequest)
	{
		if( MPI_Cancel(mDummyReceiveRequest) != MPI_SUCCESS )
			throw pmNetworkException(pmNetworkException::DUMMY_REQUEST_CANCEL_ERROR);
	}

	mDummyReceiveRequest = NULL;
}

pmStatus pmMPI::ThreadSwitchCallback(pmThreadCommand* pCommand)
{
	/* Do not use pCommand in this function as it is NULL (passed in the constructor above) */
	
	// This loop terminates with the pmThread's destruction
	while(1)
	{
		mResourceLock.Lock();

		SetupDummyRequest();

		size_t lRequestCount = mNonBlockingRequestVector.size();
		++lRequestCount; // Adding one for dummy request

		MPI_Request* lRequestArray = new MPI_Request[lRequestCount];
		
		typedef pmCommunicatorCommand* pmCommunicatorCommandPtr;
		pmCommunicatorCommandPtr* lCommandArray = new pmCommunicatorCommandPtr[lRequestCount];

		lRequestArray[0] = *mDummyReceiveRequest;
		lCommandArray[0] = NULL;

		for(size_t i=0; i<lRequestCount-1; ++i)
		{
			lRequestArray[i+1] = mNonBlockingRequestVector[i].first;
			lCommandArray[i+1] = mNonBlockingRequestVector[i].second;
		}

		mResourceLock.Unlock();

		size_t lFinishingRequestIndex;
		MPI_Status lFinishingRequestStatus;

		if( MPI_Waitany(lRequestCount, lRequestArray, &lFinishingRequestIndex, &lFinishingRequestStatus) != MPI_SUCCESS )
		{
			delete[] lCommandArray;
			delete[] lRequestArray;
			throw pmNetworkException(pmNetworkException::WAIT_ERROR);
		}

		delete[] lRequestArray;

		if(lFinishingRequestIndex == 0)		// Dummy Request
		{
			int lFlag = 0;
			if( MPI_Test_cancelled(&lFinishingRequestStatus, &lFlag) != MPI_SUCCESS )
			{
				delete[] lCommandArray;
				throw pmNetworkException(pmNetworkException::CANCELLATION_TEST_ERROR);
			}

			if(!lFlag)
			{
				delete[] lCommandArray;
				throw pmNetworkException(pmNetworkException::INVALID_DUMMY_REQUEST_STATUS_ERROR);
			}
		}
		else
		{
			pmCommunicatorCommand* lCommand = lCommandArray[lFinishingRequestIndex];
			if(mRequestCountMap.find(lCommand) == mRequestCountMap.end())
			{
				delete[] lCommandArray;
				throw pmFatalErrorException();
			}

			mRequestCountMap[lCommand] = mRequestCountMap[lCommand] - 1;
			if(mRequestCountMap[lCommand] == 0)
			{
				ushort lCommandId = lCommand->GetId();

				switch(lCommandId)
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
					{
						delete[] lCommandArray;
						throw pmFatalErrorException();
					}
				}
			}
		}

		delete[] lCommandArray;
	}

	return pmSuccess;
}

} // end namespace pm




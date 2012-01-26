
#include "pmCommunicator.h"
#include "pmCommand.h"
#include "pmNetwork.h"

namespace pm
{

#define SAFE_GET_NETWORK(x) { x = NETWORK_IMPLEMENTATION_CLASS::GetNetwork(); if(!x) throw pmFatalErrorException(); }

pmCommunicator* pmCommunicator::mCommunicator = NULL;

pmCommunicator::pmCommunicator()
{
}

pmCommunicator* pmCommunicator::GetCommunicator()
{
	if(!mCommunicator)
		mCommunicator = new pmCommunicator();

	return mCommunicator;
}

pmStatus pmCommunicator::DestroyCommunicator()
{
	delete mCommunicator;
	mCommunicator = NULL;

	return pmSuccess;
}

pmStatus pmCommunicator::Send(pmCommunicatorCommandPtr pCommand, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);

	pmStatus lStatus = lNetwork->SendNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

pmStatus pmCommunicator::SendPacked(pmCommunicatorCommandPtr pCommand, bool pBlocking /* = false */)
{
	pmStatus lStatus;
	pmNetwork* lNetwork;

	SAFE_GET_NETWORK(lNetwork);

	pCommand->MarkExecutionStart();

	if(lNetwork->PackData(pCommand) != pmSuccess)
		throw pmDataProcessingException(pmDataProcessingException::DATA_PACKING_FAILED);

	lStatus = lNetwork->SendNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

pmStatus pmCommunicator::Broadcast(pmCommunicatorCommandPtr pCommand, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->BroadcastNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

pmStatus pmCommunicator::Receive(pmCommunicatorCommandPtr pCommand, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->ReceiveNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

/*
pmStatus pmCommunicator::ReceivePacked(pmCommunicatorCommandPtr pCommand, bool pBlocking)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->ReceiveAllocateUnpackNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}
*/

pmStatus pmCommunicator::All2All(pmCommunicatorCommandPtr pCommand, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->All2AllNonBlocking(pCommand);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

} // end namespace pm




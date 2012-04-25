
#include "pmCommunicator.h"
#include "pmCommand.h"
#include "pmNetwork.h"

namespace pm
{

#define SAFE_GET_NETWORK(x) { x = NETWORK_IMPLEMENTATION_CLASS::GetNetwork(); if(!x) PMTHROW(pmFatalErrorException()); }

pmCommunicator* pmCommunicator::mCommunicator = NULL;

pmCommunicator::pmCommunicator()
{
    if(mCommunicator)
        PMTHROW(pmFatalErrorException());
    
    mCommunicator = this;
}

pmCommunicator* pmCommunicator::GetCommunicator()
{
	return mCommunicator;
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

	if(lNetwork->PackData(pCommand) != pmSuccess)
		PMTHROW(pmDataProcessingException(pmDataProcessingException::DATA_PACKING_FAILED));

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




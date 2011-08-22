
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

pmStatus pmCommunicator::Send(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);

	pmStatus lStatus = lNetwork->SendNonBlocking(pCommand, pHardware);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

pmStatus pmCommunicator::Broadcast(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->BroadcastNonBlocking(pCommand, pHardware);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

pmStatus pmCommunicator::Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);
	
	pmStatus lStatus = lNetwork->ReceiveNonBlocking(pCommand, pHardware);

	if(pBlocking)
		lStatus = pCommand->WaitForFinish();

	return lStatus;
}

} // end namespace pm




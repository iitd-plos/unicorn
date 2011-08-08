
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

pmStatus pmCommunicator::BuildThreadCommand(const pmCommunicatorCommand* pCommunicatorCommand, const pmThreadCommand* pThreadCommand, internalIdentifier pId, pmHardware pHardware, bool pBlocking)
{
	/* All commands must be allocated on heap as they are usually passed across threads */
	commandWrapper* lWrapper = new commandWrapper();
	lWrapper->id = pId;
	lWrapper->hardware = pHardware;
	lWrapper->blocking = pBlocking;
	lWrapper->communicatorCommand = pCommunicatorCommand;

	pThreadCommand = new pmThreadCommand(CONTROLLER_COMMAND_WRAPPER, lWrapper);

	return pmSuccess;
}

pmStatus pmCommunicator::ThreadSwitchCallback(const pmThreadCommand* pThreadCommand)
{
	ushort lCommandId = pThreadCommand->GetId();
	commandWrapper* lWrapper = (commandWrapper*)(pThreadCommand->GetData());
	internalIdentifier lId = lWrapper->id;
	pmHardware lHardware = lWrapper->hardware;
	bool lBlocking = lWrapper->blocking;
	pmCommunicatorCommand* lCommunicatorCommand = lWrapper->communicatorCommand;

	delete lWrapper;
	delete pThreadCommand;

	if(lCommandId != CONTROLLER_COMMAND_WRAPPER)
		throw pmInvalidCommandIdException(lCommandId);

	switch(lId)
	{
		case SEND:
			return SendInternal(lCommunicatorCommand, lHardware, lBlocking);
			break;

		case BROADCAST:
			return BroadcastInternal(lCommunicatorCommand, lHardware, lBlocking);
			break;

		case RECEIVE:
			return ReceiveInternal(lCommunicatorCommand, lHardware, lBlocking);
			break;

		default:
			throw pmFatalErrorException();
	}

	return pmSuccess;
}

pmStatus pmCommunicator::Send(const pmCommunicatorCommand* pCommunicatorCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmThreadCommand* lThreadCommand;
	
	if(pBlocking)
	{
		return SendInternal(pCommunicatorCommand, pHardware, pBlocking);
	}
	else
	{
		if( BuildThreadCommand(pCommunicatorCommand, lThreadCommand, SEND, pHardware, pBlocking) == pmSuccess )
			return SwitchThread(lThreadCommand);
	}
	
	return pmSuccess;
}

pmStatus pmCommunicator::Broadcast(const pmCommunicatorCommand& pCommunicatorCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmThreadCommand* lThreadCommand;
		
	if(pBlocking)
	{
		return BroadcastInternal(pCommunicatorCommand, pHardware, pBlocking);
	}
	else
	{
		if( BuildThreadCommand(pCommunicatorCommand, lThreadCommand, BROADCAST, pHardware, pBlocking) == pmSuccess )
			return SwitchThread(lThreadCommand);
	}
	
	return pmSuccess;
}

pmStatus pmCommunicator::Receive(pmCommunicatorCommand& pCommunicatorCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmThreadCommand* lThreadCommand;
		
	if(pBlocking)
	{
		return ReceiveInternal(pCommunicatorCommand, pHardware, pBlocking);
	}
	else
	{
		if( BuildThreadCommand(pCommunicatorCommand, lThreadCommand, RECEIVE, pHardware, pBlocking) == pmSuccess )
			return SwitchThread(lThreadCommand);
	}
	
	return pmSuccess;
}

pmStatus pmCommunicator::SendInternal(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	pmStatus lStatus = pmSuccess;
	pmNetwork* lNetwork;
	SAFE_GET_NETWORK(lNetwork);

	if(pHardware.IsHost())
	{
		lStatus = lNetwork->SendByteArrayToHost(pCommand->GetData(), pCommand->GetDataLength(), pBlocking, pHardware.GetHost());
	}
	else
	if(pHardware.isCluster())
	{
		lStatus = lNetwork->SendByteArrayToCluster(pCommand->GetData(), pCommand->GetDataLength(), pBlocking, pHardware.GetCluster());
	}
	else
	{
		throw pmFatalErrorException();
	}

	pCommand->SetStatus(lStatus);

	return lStatus;
}

pmStatus pmCommunicator::BroadcastInternal(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	return pmSuccess;
}

pmStatus pmCommunicator::ReceiveInternal(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking /* = false */)
{
	return pmSuccess;
}

} // end namespace pm




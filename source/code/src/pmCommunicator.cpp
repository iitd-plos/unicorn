
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

#include "pmCommunicator.h"
#include "pmCommand.h"
#include "pmNetwork.h"
#include "pmHeavyOperations.h"

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




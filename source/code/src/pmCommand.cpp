
#include "pmCommand.h"
#include "pmSignalWait.h"

namespace pm
{

/* class pmCommand */
pmCommand::pmCommand(ushort pCommandId, void* pCommandData /* = NULL */, ulong pDataLength /* = 0 */)
{
	mCommandId = pCommandId;
	mCommandData = pCommandData;
	mDataLength = pDataLength;
	mStatus = pmStatusUnavailable;
	mSignalWait = NULL;
}

pmStatus pmCommand::SetData(void* pCommandData, ulong pDataLength)
{
	mCommandData = pCommandData;
	mDataLength = pDataLength;
	
	return pmSuccess;
}

pmStatus pmCommand::SetStatus(pmStatus pStatus)
{
	mStatus = pStatus;

	if(mStatus != pmStatusUnavailable)
	{
		if(mSignalWait)
		{
			mSignalWait->Signal();

			delete mSignalWait;
			mSignalWait = NULL;
		}
	}

	return pmSuccess;
}

pmStatus pmCommand::WaitForStatus()
{
	if(mStatus == pmStatusUnavailable)
	{
		mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();
		mSignalWait->Wait();
	}

	return pmSuccess;
}

/* class pmCommunicatorCommand */
bool pmCommunicatorCommand::IsValid()
{
	if(mCommandId >= MAX_COMMUNICATOR_COMMANDS)
		return false;

	return true;
}

/* class pmThreadCommand */
bool pmThreadCommand::IsValid()
{
	if(mCommandId >= MAX_THREAD_COMMANDS)
		return false;

	return true;
}

} // end namespace pm




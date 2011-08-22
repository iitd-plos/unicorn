
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

ushort pmCommand::GetId()
{
	mResourceLock.Lock();

	ushort lCommandId = mCommandId;

	mResourceLock.Unlock();

	return lCommandId;
}

void* pmCommand::GetData()
{
	mResourceLock.Lock();

	void* lCommandData = mCommandData;

	mResourceLock.Unlock();

	return lCommandData;
}

ulong pmCommand::GetDataLength()
{
	mResourceLock.Lock();

	ulong lDataLength = mDataLength;

	mResourceLock.Unlock();

	return lDataLength;
}

pmStatus pmCommand::GetStatus()
{
	mResourceLock.Lock();

	pmStatus lStatus = mStatus;

	mResourceLock.Unlock();

	return lStatus;
}

pmStatus pmCommand::SetData(void* pCommandData, ulong pDataLength)
{
	mResourceLock.Lock();

	mCommandData = pCommandData;
	mDataLength = pDataLength;

	mResourceLock.Unlock();
	
	return pmSuccess;
}

pmStatus pmCommand::SetStatus(pmStatus pStatus)
{
	mResourceLock.Lock();
	mStatus = pStatus;
	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmCommand::WaitForFinish()
{
	mResourceLock.Lock();

	if(mStatus == pmStatusUnavailable)
	{
		mSignalWait = new SIGNAL_WAIT_IMPLEMENTATION_CLASS();
		
		mResourceLock.Unlock();

		mSignalWait->Wait();

		return mStatus;
	}

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmCommand::MarkExecutionStart()
{
	mResourceLock.Lock();
	mTimer.Start();
	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmCommand::MarkExecutionEnd(pmStatus pStatus)
{
	mResourceLock.Lock();
	mTimer.Stop();

	mStatus = pStatus;

	if(mSignalWait)
	{
		mSignalWait->Signal();

		delete mSignalWait;
		mSignalWait = NULL;
	}

	mResourceLock.Unlock();

	return pmSuccess;
}

double pmCommand::GetExecutionTimeInSecs()
{
	mResourceLock.Lock();
	double lTime = mTimer.GetElapsedTimeInSecs();
	mResourceLock.Unlock();

	return lTime;
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




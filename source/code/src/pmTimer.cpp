
#include "pmTimer.h"

namespace pm
{

/* class pmTimer */
pmTimer::pmTimer()
{
	mState = pmTimer::NOT_STARTED;
}

pmStatus pmTimer::SetState(pmTimer::timerState pState)
{
	if(pState == pmTimer::NOT_STARTED || pState >= pmTimer::MAX_STATES)
		throw pmTimerException(pmTimerException::INVALID_STATE);

	mState = pState;

	return pmSuccess;
}

pmTimer::timerState pmTimer::GetState()
{
	return mState;
}

/* class pmLinuxTimer */
pmLinuxTimer::pmLinuxTimer()
{
	mStartTime = (double)0;
	mUnpausedTime = (double)0;
}

pmStatus pmLinuxTimer::Start()
{
	if(GetState() != pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::ALREADY_RUNNING);

	mStartTime = GetCurrentTimeInSecs();

	SetState(pmTimer::STARTED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Stop()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::NOT_STARTED);

	SetState(pmTimer::STOPPED);

	if(lState == pmTimer::PAUSED)
		return pmSuccess;

	mUnpausedTime += GetElapsedTimeInSecs();

	return pmSuccess;
}

pmStatus pmLinuxTimer::Reset()
{
	if(GetState() == pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::NOT_STARTED);

	mStartTime = GetCurrentTimeInSecs();
	mUnpausedTime = 0;

	SetState(pmTimer::STARTED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Pause()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::NOT_STARTED);

	if(lState == pmTimer::PAUSED)
		throw pmTimerException(pmTimerException::ALREADY_PAUSED);

	if(lState == pmTimer::STOPPED)
		throw pmTimerException(pmTimerException::ALREADY_STOPPED);

	mUnpausedTime += GetElapsedTimeInSecs();

	SetState(pmTimer::PAUSED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Resume()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::NOT_STARTED);

	if(lState == pmTimer::STOPPED)
		throw pmTimerException(pmTimerException::ALREADY_STOPPED);

	if(lState != pmTimer::PAUSED)
		throw pmTimerException(pmTimerException::NOT_PAUSED);

	mStartTime = GetCurrentTimeInSecs();
	
	SetState(pmTimer::STARTED);

	return pmSuccess;
}

double pmLinuxTimer::GetElapsedTimeInSecs()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		throw pmTimerException(pmTimerException::NOT_STARTED);

	if(lState == pmTimer::PAUSED || lState == pmTimer::STOPPED)
		return mUnpausedTime;

	return GetCurrentTimeInSecs() - mStartTime;
}

double pmLinuxTimer::GetCurrentTimeInSecs()
{
	struct timeval lTimeVal;
	struct timezone lTimeZone;

	::gettimeofday(&lTimeVal, &lTimeZone);

	double lCurrentTime = ((double)(lTimeVal.tv_sec * 1000000 + lTimeVal.tv_usec))/1000000;

	return lCurrentTime;
}

}

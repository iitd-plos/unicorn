
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

#include "pmTimer.h"

namespace pm
{

/* class pmTimer */
pmTimer::pmTimer()
{
	mState = pmTimer::NOT_STARTED;
}
    
pmTimer::~pmTimer()
{
}

pmStatus pmTimer::SetState(pmTimer::timerState pState)
{
	if(pState == pmTimer::NOT_STARTED || pState >= pmTimer::MAX_STATES)
		PMTHROW(pmTimerException(pmTimerException::INVALID_STATE));

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
    
pmLinuxTimer::~pmLinuxTimer()
{
}

pmStatus pmLinuxTimer::Start()
{
	if(GetState() == pmTimer::STOPPED)
        return Reset();
        
    if(GetState() != pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::ALREADY_RUNNING));

	mStartTime = GetCurrentTimeInSecs();

	SetState(pmTimer::STARTED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Stop()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	if(lState == pmTimer::PAUSED)
    {
        SetState(pmTimer::STOPPED);
		return pmSuccess;
    }

	mUnpausedTime += GetElapsedTimeInSecs();

	SetState(pmTimer::STOPPED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Reset()
{
	if(GetState() == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	mStartTime = GetCurrentTimeInSecs();
	mUnpausedTime = 0;

	SetState(pmTimer::STARTED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Pause()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	if(lState == pmTimer::PAUSED)
		PMTHROW(pmTimerException(pmTimerException::ALREADY_PAUSED));

	if(lState == pmTimer::STOPPED)
		PMTHROW(pmTimerException(pmTimerException::ALREADY_STOPPED));

	mUnpausedTime += GetElapsedTimeInSecs();

	SetState(pmTimer::PAUSED);

	return pmSuccess;
}

pmStatus pmLinuxTimer::Resume()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	if(lState == pmTimer::STOPPED)
		PMTHROW(pmTimerException(pmTimerException::ALREADY_STOPPED));

	if(lState != pmTimer::PAUSED)
		PMTHROW(pmTimerException(pmTimerException::NOT_PAUSED));

	mStartTime = GetCurrentTimeInSecs();
	
	SetState(pmTimer::STARTED);

	return pmSuccess;
}

double pmLinuxTimer::GetElapsedTimeInSecs()
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	if(lState == pmTimer::PAUSED || lState == pmTimer::STOPPED)
		return mUnpausedTime;

	return GetCurrentTimeInSecs() - mStartTime;
}

}

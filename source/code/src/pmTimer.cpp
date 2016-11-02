
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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

pmTimer::timerState pmTimer::GetState() const
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
    
    if(lState == pmTimer::STOPPED)
		PMTHROW(pmTimerException(pmTimerException::ALREADY_STOPPED));

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

double pmLinuxTimer::GetElapsedTimeInSecs() const
{
	pmTimer::timerState lState = GetState();

	if(lState == pmTimer::NOT_STARTED)
		PMTHROW(pmTimerException(pmTimerException::NOT_STARTED));

	if(lState == pmTimer::PAUSED || lState == pmTimer::STOPPED)
		return mUnpausedTime;

	return GetCurrentTimeInSecs() - mStartTime;
}

}


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

#include "pmSignalWait.h"
#include TIMER_IMPLEMENTATION_HEADER

#ifdef TRACK_THREADS
#include <pthread.h>
#endif

#ifdef DUMP_THREADS
#include "pmLogger.h"
#endif

namespace pm
{

#ifdef DUMP_THREADS
void __dump_thread_state(bool pWait);
void __dump_thread_state(bool pWait)
{
    char lStr[512];
    if(pWait)
        sprintf(lStr, "Thread %p entering wait", pthread_self());
    else
        sprintf(lStr, "Thread %p resuming", pthread_self());
        
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::WARNING, lStr);
}
#define THREAD_WAIT_ENTER __dump_thread_state(true);
#define THREAD_WAIT_EXIT __dump_thread_state(false);
#else
#define THREAD_WAIT_ENTER
#define THREAD_WAIT_EXIT
#endif

    
/* class pmPThreadSignalWait */
pmPThreadSignalWait::pmPThreadSignalWait(bool pOnceUse)
	: pmSignalWait(pOnceUse)
    , mExiting(false)
    , mCondEnforcer(false)
	, mWaitingThreadCount(0)
    , mResourceLock __LOCK_NAME__("pmPThreadSignalWait::mResourceLock")
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailureException, pmThreadFailureException::COND_VAR_INIT_FAILURE );
}

pmPThreadSignalWait::~pmPThreadSignalWait()
{
	WaitTillAllBlockedThreadsWakeup();

	// Auto lock/unlock scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
		THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_DESTROY_FAILURE );
	}
}

pmStatus pmPThreadSignalWait::Wait()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mExiting)
		return pmSuccess;

	++mWaitingThreadCount;
    
    THREAD_WAIT_ENTER

	while(!mCondEnforcer)
		THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, mResourceLock.GetMutex()), pmThreadFailureException, pmThreadFailureException::COND_VAR_WAIT_FAILURE );

    THREAD_WAIT_EXIT

	--mWaitingThreadCount;

	if(!mOnceUse && mWaitingThreadCount == 0)
		mCondEnforcer = false;

	return pmSuccess;
}

bool pmPThreadSignalWait::WaitWithTimeOut(ulong pTriggerTime)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    bool lRetVal = false;
    
	if(mExiting)
		return pmSuccess;
    
	++mWaitingThreadCount;
    
    THREAD_WAIT_ENTER
    
    struct timespec lTimespec;
    lTimespec.tv_sec = pTriggerTime;
    lTimespec.tv_nsec = 0;

    while(!mCondEnforcer)
    {
		int lError = pthread_cond_timedwait(&mCondVariable, mResourceLock.GetMutex(), &lTimespec);
        if(lError)
        {
            if(lError != ETIMEDOUT)
                PMTHROW(pmThreadFailureException(pmThreadFailureException::COND_VAR_WAIT_FAILURE, lError));

            lRetVal = true;
            break;
        }
    }
    
    THREAD_WAIT_EXIT
    
	--mWaitingThreadCount;
    
	if(!mOnceUse && mWaitingThreadCount == 0)
		mCondEnforcer = false;
    
	return lRetVal;
}

pmStatus pmPThreadSignalWait::Signal()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    EXCEPTION_ASSERT(!mOnceUse || !mCondEnforcer);

	mCondEnforcer = true;

	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_broadcast(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_SIGNAL_FAILURE );

	return pmSuccess;
}

pmStatus pmPThreadSignalWait::WaitTillAllBlockedThreadsWakeup()
{
    THREAD_WAIT_ENTER

	while(1)
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

		if(mWaitingThreadCount == 0)
		{
			mExiting = true;

            THREAD_WAIT_EXIT
            
			return pmSuccess;
		}
	}

	return pmSuccess;
}
    
}

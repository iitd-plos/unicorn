
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

#include "pmSignalWait.h"
#include TIMER_IMPLEMENTATION_HEADER
#include STANDARD_ERROR_HEADER

#ifdef TRACK_THREADS
#include <pthread.h>
#endif

namespace pm
{

#ifdef DUMP_THREADS
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
pmPThreadSignalWait::pmPThreadSignalWait()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailureException, pmThreadFailureException::COND_VAR_INIT_FAILURE );

	mExiting = false;	
	mCondEnforcer = false;
	mWaitingThreadCount = 0;
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

	if(mWaitingThreadCount == 0)
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
    
	if(mWaitingThreadCount == 0)
		mCondEnforcer = false;
    
	return lRetVal;
}

pmStatus pmPThreadSignalWait::Signal()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
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

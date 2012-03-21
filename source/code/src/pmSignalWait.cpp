
#include "pmSignalWait.h"

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

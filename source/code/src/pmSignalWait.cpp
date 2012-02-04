
#include "pmSignalWait.h"

namespace pm
{

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

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_DESTROY_FAILURE );
}

pmStatus pmPThreadSignalWait::Wait()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mExiting)
		return pmSuccess;

	++mWaitingThreadCount;

	while(!mCondEnforcer)
		THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, mResourceLock.GetMutex()), pmThreadFailureException, pmThreadFailureException::COND_VAR_WAIT_FAILURE );

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
	while(1)
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

		if(mWaitingThreadCount == 0)
		{
			mExiting = true;
			return pmSuccess;
		}
	}

	return pmSuccess;
}

}

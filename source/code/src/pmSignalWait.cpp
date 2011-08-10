
#include "pmSignalWait.h"

namespace pm
{

/* class pmPThreadSignalWait */
pmPThreadSignalWait::pmPThreadSignalWait()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailureException, pmThreadFailureException::COND_VAR_INIT_FAILURE );
	mCondEnforcer = false;
}

pmPThreadSignalWait::~pmPThreadSignalWait()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_DESTROY_FAILURE );
}

pmStatus pmPThreadSignalWait::Wait()
{
	while(!mCondEnforcer)
		THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, &mMutex), pmThreadFailureException, pmThreadFailureException::COND_VAR_WAIT_FAILURE );

	mCondEnforcer = false;

	return pmSuccess;
}

pmStatus pmPThreadSignalWait::Signal()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
	mCondEnforcer = true;

	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_signal(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_SIGNAL_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

}
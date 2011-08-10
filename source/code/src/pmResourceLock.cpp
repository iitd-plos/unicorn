
#include "pmResourceLock.h"

namespace pm
{

/* class pmPThreadResourceLock */
pmPThreadResourceLock::pmPThreadResourceLock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
}

pmPThreadResourceLock::~pmPThreadResourceLock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
}

pmStatus pmPThreadResourceLock::Lock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );

	return pmSuccess;
}

pmStatus pmPThreadResourceLock::Unlock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

}
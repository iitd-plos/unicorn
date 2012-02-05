
#include "pmResourceLock.h"
#include "pmLogger.h"

namespace pm
{

//#define TRACK_MUTEXES

#ifdef TRACK_MUTEXES

void __dump_mutex(pthread_mutex_t* mutex, const char* name)
{
        char lStr[512];
        sprintf(lStr, "Mutex State Change: %p (%s)", mutex, name);
        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define DUMP_MUTEX(a, b) __dump_mutex(a, b)

#else
#define DUMP_MUTEX(a, b)
#endif

/* class pmPThreadResourceLock */
pmPThreadResourceLock::pmPThreadResourceLock()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
	DUMP_MUTEX(&mMutex, "Created");
}

pmPThreadResourceLock::~pmPThreadResourceLock()
{
	DUMP_MUTEX(&mMutex, "Destroying");
	Lock();
	Unlock();
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
}

pmStatus pmPThreadResourceLock::Lock()
{
	DUMP_MUTEX(&mMutex, "Locking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );

	return pmSuccess;
}

pmStatus pmPThreadResourceLock::Unlock()
{
	DUMP_MUTEX(&mMutex, "Unlocking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

}

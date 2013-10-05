
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

#include "pmResourceLock.h"
#include "pmLogger.h"
#include "pmTimer.h"

#ifdef TRACK_MUTEX_TIMINGS
#include <string.h>
#endif

namespace pm
{

//#define TRACK_MUTEXES

#ifdef TRACK_MUTEXES

void __dump_mutex(pthread_mutex_t* mutex, const char* state)
{
    char lStr[512];
    sprintf(lStr, "Mutex State Change: %p (%s)", mutex, state);
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define DUMP_MUTEX(a, b) __dump_mutex(a, b)

#else
#define DUMP_MUTEX(a, b)
#endif

    
/* class pmPThreadResourceLock */
pmPThreadResourceLock::pmPThreadResourceLock(
                                        #ifdef TRACK_MUTEX_TIMINGS
                                              const char* pName /* = "" */
                                        #endif
                                             )
#ifdef RECORD_LOCK_ACQUISITIONS
    : mLine(-1)
    , mIsCurrentlyAcquired(false)
#endif
#ifdef TRACK_MUTEX_TIMINGS
#ifdef RECORD_LOCK_ACQUISITIONS
    ,
#else
    :
#endif
     mLockTimer(strlen(pName) ? (std::string(pName).append(" [Lock]")) : pName)
    , mUnlockTimer(strlen(pName) ? std::string(pName).append(" [Unlock]") : pName)
#endif
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

void pmPThreadResourceLock::Lock()
{
#ifdef TRACK_MUTEX_TIMINGS
    pmAccumulationTimerHelper lHelperTimer(&mLockTimer);
#endif

	DUMP_MUTEX(&mMutex, "Locking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
}

void pmPThreadResourceLock::Unlock()
{
#ifdef TRACK_MUTEX_TIMINGS
    pmAccumulationTimerHelper lHelperTimer(&mUnlockTimer);
#endif

	DUMP_MUTEX(&mMutex, "Unlocking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );
}

#ifdef RECORD_LOCK_ACQUISITIONS
void pmPThreadResourceLock::RecordAcquisition(const char* pFile, int pLine)
{
    mFile = pFile;
    mLine = pLine;
    mIsCurrentlyAcquired = true;
    mThread = pthread_self();
}

void pmPThreadResourceLock::ResetAcquisition()
{
    mIsCurrentlyAcquired = false;
}

bool pmPThreadResourceLock::IsLockSelfAcquired()
{
    return (mIsCurrentlyAcquired && mThread == pthread_self());
}
#endif

}

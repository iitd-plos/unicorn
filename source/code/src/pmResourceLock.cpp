
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

void __dump_rwlock(pthread_rwlock_t* rwlock, const char* state)
{
    char lStr[512];
    sprintf(lStr, "RWLock State Change: %p (%s)", rwlock, state);
    pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, lStr);
}

#define DUMP_MUTEX(a, b) __dump_mutex(a, b)
#define DUMP_RWLOCK(a, b) __dump_rwlock(a, b)

#else
#define DUMP_MUTEX(a, b)
#define DUMP_RWLOCK(a, b)
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


/* class pmPThreadRWResourceLock */
pmPThreadRWResourceLock::pmPThreadRWResourceLock(
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
	THROW_ON_NON_ZERO_RET_VAL( pthread_rwlock_init(&mRWLock, NULL), pmThreadFailureException, pmThreadFailureException::RWLOCK_INIT_FAILURE );
	DUMP_RWLOCK(&mRWLock, "Created");
}

pmPThreadRWResourceLock::~pmPThreadRWResourceLock()
{
	DUMP_RWLOCK(&mRWLock, "Destroying");

	ReadLock();
	Unlock();
	
    THROW_ON_NON_ZERO_RET_VAL( pthread_rwlock_destroy(&mRWLock), pmThreadFailureException, pmThreadFailureException::RWLOCK_DESTROY_FAILURE );
}

void pmPThreadRWResourceLock::ReadLock()
{
#ifdef TRACK_MUTEX_TIMINGS
    pmAccumulationTimerHelper lHelperTimer(&mLockTimer);
#endif

	DUMP_RWLOCK(&mRWLock, "Read_Locking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_rwlock_rdlock(&mRWLock), pmThreadFailureException, pmThreadFailureException::RWLOCK_READ_LOCK_FAILURE );
}

void pmPThreadRWResourceLock::WriteLock()
{
#ifdef TRACK_MUTEX_TIMINGS
    pmAccumulationTimerHelper lHelperTimer(&mLockTimer);
#endif

	DUMP_RWLOCK(&mRWLock, "Write_Locking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_rwlock_wrlock(&mRWLock), pmThreadFailureException, pmThreadFailureException::RWLOCK_WRITE_LOCK_FAILURE );
}

void pmPThreadRWResourceLock::Unlock()
{
#ifdef TRACK_MUTEX_TIMINGS
    pmAccumulationTimerHelper lHelperTimer(&mUnlockTimer);
#endif

	DUMP_RWLOCK(&mRWLock, "Unlocking");
	THROW_ON_NON_ZERO_RET_VAL( pthread_rwlock_unlock(&mRWLock), pmThreadFailureException, pmThreadFailureException::RWLOCK_UNLOCK_FAILURE );
}

#ifdef RECORD_LOCK_ACQUISITIONS
void pmPThreadRWResourceLock::RecordAcquisition(const char* pFile, int pLine)
{
    mFile = pFile;
    mLine = pLine;
    mIsCurrentlyAcquired = true;
    mThread = pthread_self();
}

void pmPThreadRWResourceLock::ResetAcquisition()
{
    mIsCurrentlyAcquired = false;
}

bool pmPThreadRWResourceLock::IsLockSelfAcquired()
{
    return (mIsCurrentlyAcquired && mThread == pthread_self());
}
#endif

}

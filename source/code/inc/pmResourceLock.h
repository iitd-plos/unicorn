
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

#ifndef __PM_RESOURCE_LOCK__
#define __PM_RESOURCE_LOCK__

#include "pmBase.h"

#ifdef RECORD_LOCK_ACQUISITIONS
#include <string>
#endif

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief An implementation of resource locking and unlocking mechanism.
 * The locking and unlocking mechanism prevents corruption of a resource
 * being simultaneously modified by more than one threads. Clients must
 * guard all accesses to shared resources inside Lock/Unlock calls.
*/

class pmResourceLockBase : public pmBase
{
	public:
    #ifdef RECORD_LOCK_ACQUISITIONS
        virtual void RecordAcquisition(const char* pFile, int pLine) = 0;
        virtual void ResetAcquisition() = 0;
        virtual bool IsLockSelfAcquired() = 0;
    #endif

	private:
};

class pmResourceLock : public pmResourceLockBase
{
	public:
        virtual void Lock() = 0;
		virtual void Unlock() = 0;

	private:
};

class pmPThreadResourceLock : public pmResourceLockBase
{
	public:
		pmPThreadResourceLock(
                        #ifdef TRACK_MUTEX_TIMINGS
                              const char* pName = ""
                        #endif
                              );
    
		virtual ~pmPThreadResourceLock();
    
		virtual void Lock();
		virtual void Unlock();
    
    #ifdef RECORD_LOCK_ACQUISITIONS
        virtual void RecordAcquisition(const char* pFile, int pLine);
        virtual void ResetAcquisition();
        virtual bool IsLockSelfAcquired();
    #endif
    
		pthread_mutex_t* GetMutex() {return &mMutex;}

	private:
		pthread_mutex_t mMutex;
    
    #ifdef RECORD_LOCK_ACQUISITIONS
        std::string mFile;
        int mLine;
        bool mIsCurrentlyAcquired;
        pthread_t mThread;
    #endif

    #ifdef TRACK_MUTEX_TIMINGS
        pmAccumulationTimer mLockTimer;
        pmAccumulationTimer mUnlockTimer;
    #endif
};

class pmRWResourceLock : public pmResourceLockBase
{
	public:
        virtual void ReadLock() = 0;
        virtual void WriteLock() = 0;
		virtual void Unlock() = 0;

	private:
};

class pmPThreadRWResourceLock : public pmRWResourceLock
{
	public:
		pmPThreadRWResourceLock(
                        #ifdef TRACK_MUTEX_TIMINGS
                              const char* pName = ""
                        #endif
                              );
    
		virtual ~pmPThreadRWResourceLock();
    
        virtual void ReadLock();
        virtual void WriteLock();
		virtual void Unlock();
    
    #ifdef RECORD_LOCK_ACQUISITIONS
        virtual void RecordAcquisition(const char* pFile, int pLine);
        virtual void ResetAcquisition();
        virtual bool IsLockSelfAcquired();
    #endif
    
	private:
		pthread_rwlock_t mRWLock;
    
    #ifdef RECORD_LOCK_ACQUISITIONS
        std::string mFile;
        int mLine;
        bool mIsCurrentlyAcquired;
        pthread_t mThread;
    #endif

    #ifdef TRACK_MUTEX_TIMINGS
        pmAccumulationTimer mLockTimer;
        pmAccumulationTimer mUnlockTimer;
    #endif
};
    
} // end namespace pm

#endif


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

class pmResourceLock : public pmBase
{
	public:
        virtual pmStatus Lock() = 0;
		virtual pmStatus Unlock() = 0;

    #ifdef RECORD_LOCK_ACQUISITIONS
        virtual void RecordAcquisition(const char* pFile, int pLine) = 0;
        virtual void ResetAcquisition() = 0;
        virtual bool IsLockSelfAcquired() = 0;
    #endif

	private:
};

class pmPThreadResourceLock : public pmResourceLock
{
	public:
		pmPThreadResourceLock(
                        #ifdef TRACK_MUTEX_TIMINGS
                              const char* pName = ""
                        #endif
                              );
    
		virtual ~pmPThreadResourceLock();
    
		virtual pmStatus Lock();
		virtual pmStatus Unlock();
    
    #ifdef RECORD_LOCK_ACQUISITIONS
        virtual void RecordAcquisition(const char* pFile, int pLine);
        virtual void ResetAcquisition();
        virtual bool IsLockSelfAcquired();
    #endif
    
		virtual pthread_mutex_t* GetMutex() {return &mMutex;}

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

} // end namespace pm

#endif

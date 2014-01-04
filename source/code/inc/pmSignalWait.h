
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

#ifndef __PM_SIGNAL_WAIT__
#define __PM_SIGNAL_WAIT__

#include "pmBase.h"
#include "pmResourceLock.h"

namespace pm
{

/**
 * \brief An implementation of wait and signal mechanism.
*/

class pmSignalWait : public pmBase
{
	public:
		virtual pmStatus Wait() = 0;
		virtual pmStatus Signal() = 0;
        virtual bool WaitWithTimeOut(ulong pTriggerTime) = 0;    /* Returns true on timeout */

	private:
		virtual pmStatus WaitTillAllBlockedThreadsWakeup() = 0;
    
    protected:
        pmSignalWait(bool pOnceUse)
        : mOnceUse(pOnceUse)
        {}
    
        const bool mOnceUse;
};

class pmPThreadSignalWait : public pmSignalWait
{
	public:
		pmPThreadSignalWait(bool pOnceUse);
		virtual ~pmPThreadSignalWait();

		virtual pmStatus Wait();
		virtual pmStatus Signal();
        virtual bool WaitWithTimeOut(ulong pTriggerTime);

	private:
		virtual pmStatus WaitTillAllBlockedThreadsWakeup();

		bool mExiting;
		bool mCondEnforcer;
		uint mWaitingThreadCount;

		pthread_cond_t mCondVariable;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

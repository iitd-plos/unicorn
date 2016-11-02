
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

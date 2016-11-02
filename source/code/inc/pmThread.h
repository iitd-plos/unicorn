
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

#ifndef __PM_THREAD__
#define __PM_THREAD__

#include "pmBase.h"
#include "pmSignalWait.h"
#include "pmSafePriorityQueue.h"

#include THREADING_IMPLEMENTATION_HEADER

#ifdef VM_IMPLEMENTATION_HEADER2
#include VM_IMPLEMENTATION_HEADER2
#endif

#include <map>

namespace pm
{

template<typename T, typename P>
void* ThreadLoop(void* pThreadData);

/**
 * \brief The base thread class of PMLIB.
 * This class serves as base class to classes providing thread implementations.
 * The thread keeps on running indefinitely unless the implementing client object
 * destroys itself. The thread accepts commands, executes it and waits for next command.
 * This class is implemented based upon Abstract Factory Design Pattern so that
 * multiple thread implementations may be hooked up.
 * From their executing thread (could be the main application thread), the clients
 * call SwitchThread function which inturn calls the callback function ThreadSwitchCallback
 * (on pmThread) which all clients are required to implement. The data passed to
 * SwitchThread will be passed back to ThreadSwitchCallback and it's interpretation is client
 * specific. pmThread executes only one command at a time. The subsequent commands wait until
 * the first one returns.
*/

namespace thread
{

enum internalMessage
{
    TERMINATE,
    DISPATCH_COMMAND,
    MAX_INTERNAL_COMMANDS
};

}

typedef struct pmBasicThreadEvent : public pmNonCopyable
{
public:
    template<typename T, typename P>
    friend class pmPThread;

    bool BlocksSecondaryOperations()
    {
        return false;
    }

    // pSubmitted true means notification just before submitting to the queue
    // pSubmitted false means notification just after removal from the queue
    void EventNotification(void* pThreadQueue, bool pSubmitted)
    {
    }

private:
    thread::internalMessage msg;
} pmBasicThreadEvent;

typedef struct pmBasicBlockableThreadEvent : public pmBasicThreadEvent
{
    virtual bool BlocksSecondaryOperations()
    {
        return false;
    }
    
    virtual ~pmBasicBlockableThreadEvent() {}

    // pSubmitted true means notification just before submitting to the queue
    // pSubmitted false means notification just after removal from the queue
    virtual void EventNotification(void* pThreadQueue, bool pSubmitted) {}
    
} pmBasicBlockableThreadEvent;


template<typename T, typename P = ushort>
class pmThread : public pmBase
{
	public:
        pmThread()
        : mSafePQ(this)
        {}
    
		virtual ~pmThread() {}
        virtual void SwitchThread(const std::shared_ptr<T>& pCommand, P pPriority) = 0;

        virtual void InterruptThread() = 0;
		virtual void SetProcessorAffinity(int pProcesorId) = 0;
		
        virtual void WaitForQueuedCommands() = 0;
        virtual void WaitIfCurrentCommandMatches(typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, void* pMatchCriterion) = 0;

        void UnblockSecondaryCommands();
        void BlockSecondaryCommands();
    
        void CallWhenSecondaryCommandsUnblocked(const std::function<void ()>& pFunc);

        pmStatus DeleteAndGetFirstMatchingCommand(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::shared_ptr<T>& pCommand, bool pTemporarilyUnblockSecondaryCommands = false);
        void DeleteAndGetAllMatchingCommands(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::vector<std::shared_ptr<T>>& pCommands, bool pTemporarilyUnblockSecondaryCommands = false);
        void DeleteMatchingCommands(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, const void* pMatchCriterion);
        bool HasMatchingCommand(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, const void* pMatchCriterion);
    
		/* To be implemented by client */
        virtual void ThreadSwitchCallback(std::shared_ptr<T>& pCommand) = 0;
    
		pmSafePQ<T, P>& GetPriorityQueue() {return this->mSafePQ;}

	protected:
		pmSafePQ<T, P> mSafePQ;

	private:
		virtual void TerminateThread() = 0;
};

template<typename T, typename P = ushort>
class pmPThread : public pmThread<T, P>
{
	public:
		pmPThread();
		virtual ~pmPThread();

        virtual void SwitchThread(const std::shared_ptr<T>& pCommand, P pPriority);

		virtual void SetProcessorAffinity(int pProcesorId);

		/* To be implemented by client */
        virtual void ThreadSwitchCallback(std::shared_ptr<T>& pCommand) = 0;

        virtual void InterruptThread();
        virtual void WaitForQueuedCommands();
        virtual void WaitIfCurrentCommandMatches(typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, void* pMatchCriterion);
    
        friend void* ThreadLoop<T, P>(void* pThreadData);

	private:
        virtual void SubmitCommand(const std::shared_ptr<T>& pCommand, P pPriority);
		virtual void ThreadCommandLoop();
		virtual void TerminateThread();

        std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> mThreadStartSignalWaitPtr;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mSignalWait;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mReverseSignalWait;

		pthread_t mThread;
};

} // end namespace pm

#include "../src/pmThread.cpp"

#endif

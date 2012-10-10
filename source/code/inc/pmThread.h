
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

#ifndef __PM_THREAD__
#define __PM_THREAD__

#include "pmBase.h"
#include "pmCommand.h"
#include "pmSignalWait.h"
#include "pmSafePriorityQueue.h"

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
    
// T must be default constructible

template<typename T, typename P = ushort>
class pmThread : public pmBase
{
	public:
		typedef bool (*internalMatchFuncPtr)(T& pItem, void* pMatchCriterion);

		typedef struct internalMatchCriterion
		{
			internalMatchFuncPtr clientMatchFunc;
			void* clientMatchCriterion;
		} internalMatchCriterion;

		virtual ~pmThread() {}
		virtual pmStatus SwitchThread(T& pCommand, P pPriority) = 0;

		virtual pmStatus SetProcessorAffinity(int pProcesorId) = 0;
		
		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(T& pCommand) = 0;
    
        virtual pmStatus WaitForQueuedCommands() = 0;
    
        virtual pmStatus WaitIfCurrentCommandMatches(internalMatchFuncPtr pMatchFunc, void* pMatchCriterion) = 0;

        virtual pmStatus UnblockSecondaryCommands();
		virtual pmStatus DeleteAndGetFirstMatchingCommand(P pPriority, internalMatchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem, bool pTemporarilyUnblockSecondaryCommands = false);
        virtual pmStatus DeleteMatchingCommands(P pPriority, internalMatchFuncPtr pMatchFunc, void* pMatchCriterion);

		enum internalMessage
		{
			TERMINATE,
			DISPATCH_COMMAND
		};

		typedef struct internalType
		{
			internalMessage msg;
			T cmd;
			typedef pmThread<T, P> outerType;
        
            virtual bool BlocksSecondaryOperations()
            {
                return cmd.BlocksSecondaryCommands();
            }
		} internalType;
		
		pmSafePQ<typename pmThread<T, P>::internalType>& GetPriorityQueue() {return this->mSafePQ;}

	protected:
		pmSafePQ<typename pmThread<T, P>::internalType> mSafePQ;

	private:
		virtual pmStatus TerminateThread() = 0;
};

template<typename T, typename P = ushort>
class pmPThread : public pmThread<T, P>
{
	public:
		pmPThread();
		virtual ~pmPThread();

		virtual pmStatus SwitchThread(T& pCommand, P pPriority);

		virtual pmStatus SetProcessorAffinity(int pProcesorId);

		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(T& pCommand) = 0;
		
        virtual pmStatus WaitForQueuedCommands();

        virtual pmStatus WaitIfCurrentCommandMatches(typename pmThread<T, P>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion);

        friend void* ThreadLoop <T, P> (void* pThreadData);

	private:
		virtual pmStatus SubmitCommand(typename pmThread<T, P>::internalType& pInternalCommand, P pPriority);
		virtual pmStatus ThreadCommandLoop();
		virtual pmStatus TerminateThread();

        SIGNAL_WAIT_IMPLEMENTATION_CLASS mSignalWait;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mReverseSignalWait;

		pthread_t mThread;

        typename pmThread<T, P>::internalType mCurrentCommand;
};

template<typename T> 
bool internalMatchFunc(T& pInternalCommand, void* pCriterion);

} // end namespace pm

typedef struct pmBasicThreadEvent
{
    virtual bool BlocksSecondaryCommands()
    {
        return false;
    }
} pmBasicThreadEvent;

#include "../src/pmThread.cpp"


#endif

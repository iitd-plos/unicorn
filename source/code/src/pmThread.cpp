
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

#ifdef DUMP_THREADS
#include "pmLogger.h"
#endif

namespace pm
{

/* pmPThread Class */
template<typename T, typename P>
pmPThread<T, P>::pmPThread()
{
    mThreadStartSignalWaitPtr.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());

	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop<T, P>, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
    
    mThreadStartSignalWaitPtr->Wait();
    mThreadStartSignalWaitPtr.reset();
}

template<typename T, typename P>
pmPThread<T, P>::~pmPThread()
{
	TerminateThread();
}

template<typename T, typename P>
void pmPThread<T, P>::TerminateThread()
{
    std::shared_ptr<T> lSharedPtr(new T());
    lSharedPtr->msg = thread::TERMINATE;

	SubmitCommand(lSharedPtr, RESERVED_PRIORITY);

	THROW_ON_NON_ZERO_RET_VAL( pthread_join(mThread, NULL), pmThreadFailureException, pmThreadFailureException::THREAD_JOIN_ERROR );
}

template<typename T, typename P>
void pmPThread<T, P>::SwitchThread(const std::shared_ptr<T>& pCommand, P pPriority)
{
    pCommand->msg = thread::DISPATCH_COMMAND;

	SubmitCommand(pCommand, pPriority);
}

template<typename T, typename P>
void pmPThread<T, P>::ThreadCommandLoop()
{
    mThreadStartSignalWaitPtr->Signal();

	while(1)
	{
        mSignalWait.Wait();

        while(this->mSafePQ.GetSize() != 0)
        {
            std::shared_ptr<T> lCurrentCommand;
            if(this->mSafePQ.GetTopItem(lCurrentCommand) == pmSuccess)
            {
                switch(lCurrentCommand->msg)
                {
                    case thread::TERMINATE:
                    {
                    #ifdef DUMP_THREADS
                        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Thread Exiting");
                    #endif

                        return;
                    }
                    
                    case thread::DISPATCH_COMMAND:
                    {
                        ThreadSwitchCallback(lCurrentCommand);
                        break;
                    }
                        
                    default:
                        PMTHROW(pmFatalErrorException());
                }
                
                this->mSafePQ.MarkProcessingFinished();
            }
        }
        
        mReverseSignalWait.Signal();
	}
}

template<typename T, typename P>
void pmPThread<T, P>::InterruptThread()
{
    if(mThread == pthread_self())
        PMTHROW(pmFatalErrorException());

#ifdef MACOS
    THROW_ON_NON_ZERO_RET_VAL( pthread_kill(mThread, SIGBUS), pmThreadFailureException, pmThreadFailureException::SIGNAL_RAISE_ERROR );
#else
    THROW_ON_NON_ZERO_RET_VAL( pthread_kill(mThread, SIGSEGV), pmThreadFailureException, pmThreadFailureException::SIGNAL_RAISE_ERROR );
#endif
}

template<typename T, typename P>
void pmPThread<T, P>::WaitIfCurrentCommandMatches(typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
    this->mSafePQ.WaitIfMatchingItemBeingProcessed(pMatchFunc, pMatchCriterion);
}
    
template<typename T, typename P>
void pmPThread<T, P>::WaitForQueuedCommands()
{
    while(!this->mSafePQ.IsEmpty())
        mReverseSignalWait.Wait();
    
    this->mSafePQ.WaitForCurrentItem();
}

template<typename T, typename P>
void pmPThread<T, P>::SubmitCommand(const std::shared_ptr<T>& pInternalCommand, P pPriority)
{
	this->mSafePQ.InsertItem(pInternalCommand, pPriority);
    mSignalWait.Signal();
}

template<typename T, typename P>
pmStatus pmThread<T, P>::DeleteAndGetFirstMatchingCommand(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, void* pMatchCriterion, std::shared_ptr<T>& pCommand, bool pTemporarilyUnblockSecondaryCommands /* = false */)
{
	return this->mSafePQ.DeleteAndGetFirstMatchingItem(pPriority, pMatchFunc, pMatchCriterion, pCommand, pTemporarilyUnblockSecondaryCommands);
}

template<typename T, typename P>
void pmThread<T, P>::DeleteMatchingCommands(P pPriority, typename pmSafePQ<T, P>::matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	this->mSafePQ.DeleteMatchingItems(pPriority, pMatchFunc, pMatchCriterion);
}

template<typename T, typename P>
void pmThread<T, P>::UnblockSecondaryCommands()
{
    this->mSafePQ.UnblockSecondaryOperations();
}
    
template<typename T, typename P>
void pmPThread<T, P>::SetProcessorAffinity(int pProcessorId)
{
#ifdef LINUX
	pthread_t lThread = pthread_self();
	cpu_set_t lSetCPU;

	CPU_ZERO(&lSetCPU);
	CPU_SET(pProcessorId, &lSetCPU);

	THROW_ON_NON_ZERO_RET_VAL( pthread_setaffinity_np(lThread, sizeof(cpu_set_t), &lSetCPU), pmThreadFailureException, pmThreadFailureException::THREAD_AFFINITY_ERROR );

#ifdef _DEBUG
	THROW_ON_NON_ZERO_RET_VAL( pthread_getaffinity_np(lThread, sizeof(cpu_set_t), &lSetCPU), pmThreadFailureException, pmThreadFailureException::THREAD_AFFINITY_ERROR );
	if(!CPU_ISSET(pProcessorId, &lSetCPU))
		PMTHROW(pmFatalErrorException());
#endif
#endif
}

template<typename T, typename P>
void* ThreadLoop(void* pThreadData)
{
	pmPThread<T, P>* lObjectPtr = (pmPThread<T, P>*)pThreadData;
	lObjectPtr->ThreadCommandLoop();

	return NULL;
}

} // end namespace pm


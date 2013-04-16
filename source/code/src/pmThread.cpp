
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

namespace pm
{

/* pmPThread Class */
template<typename T, typename P>
pmPThread<T, P>::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop<T, P>, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
}

template<typename T, typename P>
pmPThread<T, P>::~pmPThread()
{
	TerminateThread();
}

template<typename T, typename P>
pmStatus pmPThread<T, P>::TerminateThread()
{
	typename pmThread<T, P>::internalType lInternalCommand;
	lInternalCommand.msg = pmThread<T, P>::TERMINATE;

	SubmitCommand(lInternalCommand, RESERVED_PRIORITY);

	THROW_ON_NON_ZERO_RET_VAL( pthread_join(mThread, NULL), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );
	//THROW_ON_NON_ZERO_RET_VAL( pthread_cancel(mThread), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );

	return pmSuccess;
}

template<typename T, typename P>
pmStatus pmPThread<T, P>::SwitchThread(T& pCommand, P pPriority)
{
	typename pmThread<T, P>::internalType lInternalCommand;
	lInternalCommand.msg = pmThread<T, P>::DISPATCH_COMMAND;
	lInternalCommand.cmd = pCommand;

	return SubmitCommand(lInternalCommand, pPriority);
}

template<typename T, typename P>
pmStatus pmPThread<T, P>::ThreadCommandLoop()
{
	while(1)
	{
        mSignalWait.Wait();

        while(this->mSafePQ.GetSize() != 0)
        {
            if(this->mSafePQ.GetTopItem(mCurrentCommand) == pmSuccess)
            {
                switch(mCurrentCommand.msg)
                {
                    case pmThread<T, P>::TERMINATE:
                    {
                        #ifdef DUMP_THREADS
                        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Thread Exiting");
                        #endif

                        return pmSuccess;
                    }
                    
                    case pmThread<T, P>::DISPATCH_COMMAND:
                    {
                        ThreadSwitchCallback(mCurrentCommand.cmd);
                        break;
                    }
                }
                
                this->mSafePQ.MarkProcessingFinished();
            }
        }
        
        mReverseSignalWait.Signal();
	}

	return pmSuccess;
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
pmStatus pmPThread<T, P>::WaitIfCurrentCommandMatches(typename pmThread<T, P>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	typename pmThread<T, P>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

    return this->mSafePQ.WaitIfMatchingItemBeingProcessed(mCurrentCommand, internalMatchFunc, (void*)(&lInternalMatchCriterion));
}
    
template<typename T, typename P>
pmStatus pmPThread<T, P>::WaitForQueuedCommands()
{
    while(!this->mSafePQ.IsEmpty())
        mReverseSignalWait.Wait();
    
    return pmSuccess;
}

template<typename T, typename P>
pmStatus pmPThread<T, P>::SubmitCommand(typename pmThread<T, P>::internalType& pInternalCommand, P pPriority)
{
	this->mSafePQ.InsertItem(pInternalCommand, pPriority);
    mSignalWait.Signal();

	return pmSuccess;
}

template<typename T, typename P>
pmStatus pmThread<T, P>::DeleteAndGetFirstMatchingCommand(P pPriority, typename pmThread<T, P>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem, bool pTemporarilyUnblockSecondaryCommands /* = false */)
{
	typename pmThread<T, P>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

	typename pmThread<T, P>::internalType lInternalItem;
	pmStatus lStatus = this->mSafePQ.DeleteAndGetFirstMatchingItem(pPriority, internalMatchFunc, (void*)(&lInternalMatchCriterion), lInternalItem, pTemporarilyUnblockSecondaryCommands);
	pItem = lInternalItem.cmd;

	return lStatus;
}

template<typename T, typename P>
pmStatus pmThread<T, P>::DeleteMatchingCommands(P pPriority, typename pmThread<T, P>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	typename pmThread<T, P>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

	return this->mSafePQ.DeleteMatchingItems(pPriority, internalMatchFunc, (void*)(&lInternalMatchCriterion));
}

template<typename T, typename P>
pmStatus pmThread<T, P>::UnblockSecondaryCommands()
{
    return this->mSafePQ.UnblockSecondaryOperations();
}
    
template<typename T, typename P>
pmStatus pmPThread<T, P>::SetProcessorAffinity(int pProcessorId)
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

	return pmSuccess;
}

template<typename T, typename P>
void* ThreadLoop(void* pThreadData)
{
	pmPThread<T, P>* lObjectPtr = (pmPThread<T, P>*)pThreadData;
	lObjectPtr->ThreadCommandLoop();

	return NULL;
}

template<typename T>
bool internalMatchFunc(T& pInternalCommand, void* pCriterion)
{
	typedef typename T::outerType::internalMatchCriterion matchCriterion;
    
    if(pInternalCommand.msg != T::outerType::DISPATCH_COMMAND)
        return false;

	matchCriterion* lInternalMatchCriterion = (matchCriterion*)(pCriterion);
	return (lInternalMatchCriterion->clientMatchFunc)(pInternalCommand.cmd, lInternalMatchCriterion->clientMatchCriterion);
}

} // end namespace pm



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
template<typename T>
pmPThread<T>::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop<T>, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
}

template<typename T>
pmPThread<T>::~pmPThread()
{
	TerminateThread();
}

template<typename T>
pmStatus pmPThread<T>::TerminateThread()
{
	typename pmThread<T>::internalType lInternalCommand;
	lInternalCommand.msg = pmThread<T>::TERMINATE;

	SubmitCommand(lInternalCommand, RESERVED_PRIORITY);

	THROW_ON_NON_ZERO_RET_VAL( pthread_join(mThread, NULL), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );
	//THROW_ON_NON_ZERO_RET_VAL( pthread_cancel(mThread), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );

	return pmSuccess;
}

template<typename T>
pmStatus pmPThread<T>::SwitchThread(T& pCommand, ushort pPriority)
{
	typename pmThread<T>::internalType lInternalCommand;
	lInternalCommand.msg = pmThread<T>::DISPATCH_COMMAND;
	lInternalCommand.cmd = pCommand;

	return SubmitCommand(lInternalCommand, pPriority);
}

template<typename T>
pmStatus pmPThread<T>::ThreadCommandLoop()
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
                    case pmThread<T>::TERMINATE:
                    {
                        #ifdef DUMP_THREADS
                        pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Thread Exiting");
                        #endif

                        return pmSuccess;
                    }
                    
                    case pmThread<T>::DISPATCH_COMMAND:
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

template<typename T>    
pmStatus pmPThread<T>::WaitIfCurrentCommandMatches(typename pmThread<T>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	typename pmThread<T>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

    return this->mSafePQ.WaitIfMatchingItemBeingProcessed(mCurrentCommand, internalMatchFunc, (void*)(&lInternalMatchCriterion));
}
    
template<typename T>
pmStatus pmPThread<T>::WaitForQueuedCommands()
{
    while(!this->mSafePQ.IsEmpty())
        mReverseSignalWait.Wait();
    
    return pmSuccess;
}

template<typename T>
pmStatus pmPThread<T>::SubmitCommand(typename pmThread<T>::internalType& pInternalCommand, ushort pPriority)
{
	this->mSafePQ.InsertItem(pInternalCommand, pPriority);
    mSignalWait.Signal();

	return pmSuccess;
}

template<typename T>
pmStatus pmThread<T>::DeleteAndGetFirstMatchingCommand(ushort pPriority, typename pmThread<T>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem)
{
	typename pmThread<T>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

	typename pmThread<T>::internalType lInternalItem;
	pmStatus lStatus = this->mSafePQ.DeleteAndGetFirstMatchingItem(pPriority, internalMatchFunc, (void*)(&lInternalMatchCriterion), lInternalItem);
	pItem = lInternalItem.cmd;

	return lStatus;
}

template<typename T>
pmStatus pmThread<T>::DeleteMatchingCommands(ushort pPriority, typename pmThread<T>::internalMatchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	typename pmThread<T>::internalMatchCriterion lInternalMatchCriterion;
	lInternalMatchCriterion.clientMatchFunc = pMatchFunc;
	lInternalMatchCriterion.clientMatchCriterion = pMatchCriterion;

	return this->mSafePQ.DeleteMatchingItems(pPriority, internalMatchFunc, (void*)(&lInternalMatchCriterion));
}

template<typename T>
pmStatus pmPThread<T>::SetProcessorAffinity(int pProcessorId)
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

template<typename T>
void* ThreadLoop(void* pThreadData)
{
	pmPThread<T>* lObjectPtr = (pmPThread<T>*)pThreadData;
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


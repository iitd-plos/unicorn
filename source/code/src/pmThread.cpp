
namespace pm
{

/* pmPThread Class */
template<typename T>
pmPThread<T>::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
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
			typename pmThread<T>::internalType lInternalCommand;
                       	this->mSafePQ.GetTopItem(lInternalCommand);

			switch(lInternalCommand.msg)
			{
				case pmThread<T>::TERMINATE:
					return pmSuccess;
				
				case pmThread<T>::DISPATCH_COMMAND:
                        		ThreadSwitchCallback(lInternalCommand.cmd);
					break;
			}
                }
	}

	return pmSuccess;
}

template<typename T>
pmStatus pmPThread<T>::SubmitCommand(typename pmThread<T>::internalType& pInternalCommand, ushort pPriority)
{
	this->mSafePQ.InsertItem(pInternalCommand, pPriority);

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

	matchCriterion* lInternalMatchCriterion = (matchCriterion*)(pCriterion);
	return (lInternalMatchCriterion->clientMatchFunc)(pInternalCommand.cmd, lInternalMatchCriterion->clientMatchCriterion);
}

} // end namespace pm


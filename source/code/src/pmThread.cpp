
#include "pmThread.h"
#include "pmCommand.h"

#include SYSTEM_CONFIGURATION_HEADER

namespace pm
{

/* pmPThread Class */
pmPThread::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailureException, pmThreadFailureException::COND_VAR_INIT_FAILURE );
	
	mCondEnforcer = false;

	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
}

pmPThread::~pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_cancel(mThread), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );

	// A locked mutex and in wait cond variable can not be deleted
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_signal(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_SIGNAL_FAILURE );

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_DESTROY_FAILURE );
}

pmStatus pmPThread::SwitchThread(pmThreadCommandPtr pCommand)
{
	return SubmitCommand(pCommand);
}

pmStatus pmPThread::ThreadCommandLoop()
{
	while(1)
	{
		THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
	
		while(!mCondEnforcer)
			THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, &mMutex), pmThreadFailureException, pmThreadFailureException::COND_VAR_WAIT_FAILURE );

		mCondEnforcer = false;

		pmStatus lStatus = ThreadSwitchCallback(mCommand);
		if(mCommand)
			mCommand->SetStatus(lStatus);

		THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );
	}

	return pmSuccess;
}

pmStatus pmPThread::SubmitCommand(pmThreadCommandPtr pCommand)
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
	
	mCondEnforcer = true;
	mCommand = pCommand;

	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_signal(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_SIGNAL_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

pmStatus pmPThread::SetProcessorAffinity(int pProcessorId)
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

void* ThreadLoop(void* pThreadData)
{
	pmPThread* lObjectPtr = (pmPThread*)pThreadData;
	lObjectPtr->ThreadCommandLoop();

	return NULL;
}

} // end namespace pm


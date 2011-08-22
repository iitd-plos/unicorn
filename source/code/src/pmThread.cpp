
#include "pmThread.h"
#include "pmCommand.h"

namespace pm
{

/* pmPThread Class */
pmPThread::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailureException, pmThreadFailureException::MUTEX_INIT_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailureException, pmThreadFailureException::COND_VAR_INIT_FAILURE );
	
	mCondEnforcer = false;
	mCommand = NULL;

	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop, this), pmThreadFailureException, pmThreadFailureException::THREAD_CREATE_ERROR );
}

pmPThread::~pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_cancel(mThread), pmThreadFailureException, pmThreadFailureException::THREAD_CANCEL_ERROR );

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_DESTROY_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_DESTROY_FAILURE );
}

pmStatus pmPThread::SwitchThread(pmThreadCommand* pCommand)
{
	return SubmitCommand(pCommand);
}

pmStatus pmPThread::ThreadCommandLoop()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );

	while(1)
	{
		while(!mCondEnforcer)
			THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, &mMutex), pmThreadFailureException, pmThreadFailureException::COND_VAR_WAIT_FAILURE );

		mCondEnforcer = false;

		if(mCommand)
			mCommand->SetStatus( ThreadSwitchCallback(mCommand) );
	}

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

pmStatus pmPThread::SubmitCommand(pmThreadCommand* pCommand)
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_LOCK_FAILURE );
	
	mCondEnforcer = true;
	mCommand = pCommand;

	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_signal(&mCondVariable), pmThreadFailureException, pmThreadFailureException::COND_VAR_SIGNAL_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailureException, pmThreadFailureException::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

void* ThreadLoop(void* pThreadData)
{
	pmPThread* lObjectPtr = (pmPThread*)pThreadData;
	lObjectPtr->ThreadCommandLoop();

	return NULL;
}

} // end namespace pm


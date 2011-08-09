
#include "pmThread.h"
#include "pmCommand.h"

namespace pm
{

/* pmPThread Class */
pmPThread::pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailure, pmThreadFailure::MUTEX_INIT_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_init(&mCondVariable, NULL), pmThreadFailure, pmThreadFailure::COND_VAR_INIT_FAILURE );
	
	mCondEnforcer = false;
	mCommand = NULL;

	THROW_ON_NON_ZERO_RET_VAL( pthread_create(&mThread, NULL, ThreadLoop, this), pmThreadFailure, pmThreadFailure::THREAD_CREATE_ERROR );
}

pmPThread::~pmPThread()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_cancel(mThread), pmThreadFailure, pmThreadFailure::THREAD_CANCEL_ERROR );

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_DESTROY_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_destroy(&mCondVariable), pmThreadFailure, pmThreadFailure::COND_VAR_DESTROY_FAILURE );
}

pmStatus pmPThread::SwitchThread(pmThreadCommand* pCommand)
{
	return SubmitCommand(pCommand);
}

pmStatus pmPThread::ThreadCommandLoop()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_LOCK_FAILURE );

	while(1)
	{
		while(!mCondEnforcer)
			THROW_ON_NON_ZERO_RET_VAL( pthread_cond_wait(&mCondVariable, &mMutex), pmThreadFailure, pmThreadFailure::COND_VAR_WAIT_FAILURE );

		mCondEnforcer = false;

		mCommand->SetStatus( ThreadSwitchCallback(mCommand) );
	}

	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

pmStatus pmPThread::SubmitCommand(pmThreadCommand* pCommand)
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_LOCK_FAILURE );
	
	mCondEnforcer = true;
	mCommand = pCommand;

	THROW_ON_NON_ZERO_RET_VAL( pthread_cond_signal(&mCondVariable), pmThreadFailure, pmThreadFailure::COND_VAR_SIGNAL_FAILURE );
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

void* ThreadLoop(void* pThreadData)
{
	pmPThread* lObjectPtr = (pmPThread*)pThreadData;
	lObjectPtr->ThreadCommandLoop();

	return NULL;
}

} // end namespace pm


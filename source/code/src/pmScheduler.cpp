
#include "pmScheduler.h"
#include "pmCommand.h"
#include "pmTask.h"
#include "pmSignalWait.h"

namespace pm
{

pmScheduler* pmScheduler::mScheduler = NULL;

pmScheduler::pmScheduler()
{
	SwitchThread(NULL);	// Create an infinite loop in a new thread
}

pmScheduler* pmScheduler::GetScheduler()
{
	if(!mScheduler)
		mScheduler = new pmScheduler();

	return mScheduler;
}

pmStatus pmScheduler::DestroyScheduler()
{
	delete mScheduler;
	mScheduler = NULL;

	return pmSuccess;
}

pmStatus pmScheduler::SubmitSubtasks(subtaskRange pRange)
{
	pmStatus lStatus = mPriorityQueue.InsertItem(pRange, pRange.task.GetPriority());

	mSignalWait.Signal();

	return lStatus;
}

pmStatus pmScheduler::ThreadSwitchCallback(pmThreadCommand* pCommand)
{
	while(1)
	{
		mSignalWait.Wait();

		while(mPriorityQueue.GetSize() != 0)
		{
			subtaskRange lRange;
			mPriorityQueue.GetTopItem(lRange);

			Execute(lRange);
		}
	}

	return pmSuccess;
}

pmStatus pmScheduler::Execute(subtaskRange pRange)
{


	return pmSuccess;
}

} // end namespace pm




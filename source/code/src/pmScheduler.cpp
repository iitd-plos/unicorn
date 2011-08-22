
#include "pmScheduler.h"
#include "pmCommand.h"

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

pmStatus pmScheduler::SubmitTask()
{
}

pmStatus pmScheduler::ThreadSwitchCallback(pmThreadCommand* pCommand)
{
	while(1)
	{
	}

	return pmSuccess;
}

} // end namespace pm




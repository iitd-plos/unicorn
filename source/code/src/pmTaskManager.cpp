
#include "pmTaskManager.h"

namespace pm
{

pmTaskManager* pmTaskManager::mTaskManager = NULL;

pmTaskManager::pmTaskManager()
{
}

pmTaskManager* pmTaskManager::GetTaskManager()
{
	if(!mTaskManager)
		mTaskManager = new pmTaskManager();

	return mTaskManager;
}

pmStatus pmTaskManager::DestroyTaskManager()
{
	delete mTaskManager;
	mTaskManager = NULL;

	return pmSuccess;
}

} // end namespace pm




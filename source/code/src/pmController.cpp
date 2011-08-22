
#include "pmController.h"
#include "pmCommunicator.h"
#include "pmDevicePool.h"
#include "pmNetwork.h"
#include "pmMemoryManager.h"
#include "pmTaskManager.h"
#include "pmScheduler.h"

namespace pm
{

#define SAFE_DESTROY(x, y) if(x) x->y();

pmController* pmController::mController = NULL;

pmController* pmController::GetController()
{
	if(!mController)
	{
		if(CreateAndInitializeController() == pmSuccess)
		{
			if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork() 
			|| !pmCommunicator::GetCommunicator()
			|| !pmDevicePool::GetDevicePool()
			|| !MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager()
			|| !pmTaskManager::GetTaskManager()
			|| !pmScheduler::GetScheduler()
			)
				throw pmFatalErrorException();
		}
	}

	return mController;
}

pmStatus pmController::DestroyController()
{
	SAFE_DESTROY(pmScheduler::GetScheduler(), DestroyScheduler);
	SAFE_DESTROY(pmTaskManager::GetTaskManager(), DestroyTaskManager);
	SAFE_DESTROY(MEMORY_MANAGER_IMPLEMENTATION_CLASS::GetMemoryManager(), DestroyMemoryManager);
	SAFE_DESTROY(pmDevicePool::GetDevicePool(), DestroyDevicePool);
	SAFE_DESTROY(pmCommunicator::GetCommunicator(), DestroyCommunicator);
	SAFE_DESTROY(NETWORK_IMPLEMENTATION_CLASS::GetNetwork(), DestroyNetwork);
	
	delete mController;
	mController = NULL;

	return pmSuccess;
}

pmStatus pmController::CreateAndInitializeController()
{
	mController = new pmController();

	if(mController)
		return pmSuccess;

	return pmFatalError;
}

pmStatus pmController::FetchMemoryRegion(void* pStartAddress, size_t pOffset, size_t pLength)
{
	return pmSuccess;
}

} // end namespace pm

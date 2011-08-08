
#include "pmController.h"
#include "pmCommunicator.h"
#include "pmDevicePool.h"
#include "pmNetwork.h"

namespace pm
{

#define SAFE_DESTROY(x, y) if(x) x->y();

pmController* pmController::GetController()
{
	if(!mController)
	{
		if(CreateAndInitializeController() == pmSuccess)
		{
			if(!NETWORK_IMPLEMENTATION_CLASS::GetNetwork() 
			|| !pmCommunicator::GetCommunicator()
			|| !pmDevicePool::GetDevicePool()
			)
				throw pmFatalErrorException();
		}
	}

	return mController;
}

pmStatus pmController::DestroyController()
{
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

	return pmSuccess;
}

} // end namespace pm
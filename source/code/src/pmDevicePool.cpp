
#include "pmDevicePool.h"

namespace pm
{

pmDevicePool* pmDevicePool::mDevicePool = NULL;

pmDevicePool* pmDevicePool::GetDevicePool()
{
	if(!mDevicePool)
		mDevicePool = new pmDevicePool();

	return mDevicePool;
}

pmStatus pmDevicePool::DestroyDevicePool()
{
	delete mDevicePool;
	mDevicePool = NULL;

	return pmSuccess;
}

pmStatus pmDevicePool::SetMachineCountAndDevicePool(uint pMachineCount)
{
	mMachineCount = pMachineCount;
	mPool.resize(mMachineCount);

	return pmSuccess;
}

pmStatus pmDevicePool::SetMachineData(uint pMachineId, pmMachineData& pMachineData)
{
	if(pMachineId >= mMachineCount)
		return pmInvalidIndex;

	mPool[pMachineId] = pMachineData;

	return pmSuccess;
}


}
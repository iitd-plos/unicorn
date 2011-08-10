
#include "pmDevicePool.h"
#include "pmResourceLock.h"

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

pmStatus pmDevicePool::InitializeDevicePool(uint pMachineCount)
{
	mMachineCount = pMachineCount;

	mResourceLock.Lock();

	mPool.resize(mMachineCount);

	for(uint i=0; i<pMachineCount; ++i)
		InitializeMachineData(i);

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmDevicePool::InitializeMachineData(uint pMachineId)
{
	if(pMachineId >= mMachineCount)
		throw pmUnknownMachineException(pMachineId);

	mPool[pMachineId].cpuCores = 0;
	mPool[pMachineId].gpuCards = 0;
	mPool[pMachineId].dataSent = 0;
	mPool[pMachineId].dataReceived = 0;
	mPool[pMachineId].sendTime = (double)0;
	mPool[pMachineId].receiveTime = (double)0;
	mPool[pMachineId].sendCount = 0;
	mPool[pMachineId].receiveCount = 0;

	return pmSuccess;
}

pmStatus pmDevicePool::SetMachineData(uint pMachineId, pmMachineData& pMachineData)
{
	if(pMachineId >= mMachineCount)
		throw pmUnknownMachineException(pMachineId);

	mResourceLock.Lock();
	mPool[pMachineId] = pMachineData;
	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmDevicePool::RegisterSendCompletion(uint pMachineId, ulong pDataSent, double pSendTime)
{
	if(pMachineId >= mMachineCount)
		throw pmUnknownMachineException(pMachineId);

	mResourceLock.Lock();

	mPool[pMachineId].dataSent += pDataSent;
	mPool[pMachineId].sendTime += pSendTime;
	++(mPool[pMachineId].sendCount);

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmDevicePool::RegisterReceiveCompletion(uint pMachineId, ulong pDataReceived, double pReceiveTime)
{
	if(pMachineId >= mMachineCount)
		throw pmUnknownMachineException(pMachineId);

	mResourceLock.Lock();

	mPool[pMachineId].dataReceived += pDataReceived;
	mPool[pMachineId].receiveTime += pReceiveTime;
	++(mPool[pMachineId].receiveCount);

	mResourceLock.Unlock();

	return pmSuccess;
}

}
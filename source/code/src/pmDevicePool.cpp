
#include "pmDevicePool.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"
#include "pmNetwork.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmCluster.h"

namespace pm
{

pmMachinePool* pmMachinePool::mMachinePool = NULL;
pmDevicePool* pmDevicePool::mDevicePool = NULL;
pmMachine* PM_LOCAL_MACHINE = NULL;

/* class pmMachinePool */
pmMachinePool* pmMachinePool::GetMachinePool()
{
	if(!mMachinePool)
		mMachinePool = new pmMachinePool(NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount());

	return mMachinePool;
}

pmStatus pmMachinePool::DestroyMachinePool()
{
	delete mMachinePool;
	mMachinePool = NULL;

	return pmSuccess;
}

pmMachinePool::pmMachinePool(uint pMachineCount)
{
	uint i=0;

	FINALIZE_PTR_ARRAY(fAll2AllBuffer, pmCommunicatorCommand::machinePool, new pmCommunicatorCommand::machinePool[pMachineCount]);

	{
		FINALIZE_RESOURCE(fMachinePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MACHINE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MACHINE_POOL_STRUCT)));

		All2AllMachineData(fAll2AllBuffer);
	}

	uint lLocalId = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId();

	for(i=0; i<pMachineCount; ++i)
	{
		pmMachine* lMachine = new pmMachine(i);

		if(i == lLocalId)
			PM_LOCAL_MACHINE = lMachine;
	
		pmMachineData lData;
		lData.cpuCores = fAll2AllBuffer[i].cpuCores;
		lData.gpuCards = fAll2AllBuffer[i].gpuCards;
		
		mMachinesVector.push_back(lMachine);
		mMachineDataVector.push_back(lData);
	}

	pmDevicePool* lDevicePool = pmDevicePool::GetDevicePool();

	FINALIZE_RESOURCE(fDevicePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::DEVICE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::DEVICE_POOL_STRUCT)));

	for(uint i=0; i<pMachineCount; ++i)
	{
		pmMachineData& lData = GetMachineData(i);
		pmMachine* lMachine = GetMachine(i);

		uint lGlobalIndex = lDevicePool->GetDeviceCount();
		mFirstDeviceIndexOnMachine.push_back(lGlobalIndex);

		uint lDevicesCount = lData.cpuCores + lData.gpuCards;

		FINALIZE_PTR_ARRAY(fDevicesBuffer, pmCommunicatorCommand::devicePool, new pmCommunicatorCommand::devicePool[lDevicesCount]);
		lDevicePool->BroadcastDeviceData(lMachine, fDevicesBuffer, lDevicesCount);

		lDevicePool->CreateMachineDevices(lMachine, fDevicesBuffer, lGlobalIndex, lDevicesCount);
	}
}

pmMachinePool::~pmMachinePool()
{
	size_t lSize = mMachinesVector.size();

	for(size_t i=0; i<lSize; ++i)
		delete mMachinesVector[i];
}

pmStatus pmMachinePool::All2AllMachineData(pmCommunicatorCommand::machinePool* pAll2AllBuffer)
{
	pmCommunicatorCommand::machinePool lSendBuffer;

	pmStubManager* lManager = pmStubManager::GetStubManager();
	lSendBuffer.cpuCores = (uint)(lManager->GetProcessingElementsCPU());
	lSendBuffer.gpuCards = (uint)(lManager->GetProcessingElementsGPU());

	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_PRIORITY_LEVEL, pmCommunicatorCommand::ALL2ALL, pmCommunicatorCommand::MACHINE_POOL_TRANSFER, NULL, pmCommunicatorCommand::MACHINE_POOL_STRUCT, &lSendBuffer, 1, pAll2AllBuffer, 1);

	pmCommunicator::GetCommunicator()->All2All(lCommand);

	lCommand->WaitForFinish();
	return lCommand->GetStatus();
}

pmMachinePool::pmMachineData& pmMachinePool::GetMachineData(uint pIndex)
{
	if(pIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException(pIndex));
	
	return mMachineDataVector[pIndex];
}

pmMachinePool::pmMachineData& pmMachinePool::GetMachineData(pmMachine* pMachine)
{
	return GetMachineData(*pMachine);
}

pmMachine* pmMachinePool::GetMachine(uint pIndex)
{
	if(pIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException(pIndex));

	return mMachinesVector[pIndex];
}

pmStatus pmMachinePool::GetAllDevicesOnMachine(uint pMachineIndex, std::vector<pmProcessingElement*>& pDevices)
{
	return GetAllDevicesOnMachine(GetMachine(pMachineIndex), pDevices);
}

pmStatus pmMachinePool::GetAllDevicesOnMachine(pmMachine* pMachine, std::vector<pmProcessingElement*>& pDevices)
{
	pDevices.clear();

	pmDevicePool* lDevicePool = pmDevicePool::GetDevicePool();
	uint lTotalDevices = lDevicePool->GetDeviceCount();
	uint lGlobalIndex = mFirstDeviceIndexOnMachine[*pMachine];

	for(uint i=lGlobalIndex; i<lTotalDevices; ++i)
	{
		pmProcessingElement* lDevice = lDevicePool->GetDeviceAtGlobalIndex(i);

		if(lDevice->GetMachine() == pMachine)
			pDevices.push_back(lDevice);
		else
			break;
	}

	return pmSuccess;
}

std::vector<pmMachine*>& pmMachinePool::GetAllMachines()
{
	return mMachinesVector;
}

uint pmMachinePool::GetFirstDeviceIndexOnMachine(uint pMachineIndex)
{
	if(pMachineIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException(pMachineIndex));

	return mFirstDeviceIndexOnMachine[pMachineIndex];
}

uint pmMachinePool::GetFirstDeviceIndexOnMachine(pmMachine* pMachine)
{
	return GetFirstDeviceIndexOnMachine(*pMachine);
}

pmStatus pmMachinePool::RegisterSendCompletion(pmMachine* pMachine, ulong pDataSent, double pSendTime)
{
	size_t lIndex = (size_t)(*pMachine);

	if(lIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException((uint)lIndex));

	mResourceLock.Lock();

	mMachineDataVector[lIndex].dataSent += pDataSent;
	mMachineDataVector[lIndex].sendTime += pSendTime;
	++(mMachineDataVector[lIndex].sendCount);

	mResourceLock.Unlock();

	return pmSuccess;
}

pmStatus pmMachinePool::RegisterReceiveCompletion(pmMachine* pMachine, ulong pDataReceived, double pReceiveTime)
{
	size_t lIndex = (size_t)(*pMachine);

	if(lIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException((uint)lIndex));

	mResourceLock.Lock();

	mMachineDataVector[lIndex].dataReceived += pDataReceived;
	mMachineDataVector[lIndex].receiveTime += pReceiveTime;
	++(mMachineDataVector[lIndex].receiveCount);

	mResourceLock.Unlock();

	return pmSuccess;
}


/* class pmDevicePool */
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

pmDevicePool::pmDevicePool()
{
	mDevicesVector.clear();
	mDeviceDataVector.clear();
}

pmDevicePool::~pmDevicePool()
{
	size_t lSize = mDevicesVector.size();
	for(size_t i=0; i<lSize; ++i)
		delete mDevicesVector[i];
}

pmStatus pmDevicePool::CreateMachineDevices(pmMachine* pMachine, pmCommunicatorCommand::devicePool* pDeviceData, uint pGlobalStartingDeviceIndex, uint pDeviceCount)
{
	for(uint i=0; i<pDeviceCount; ++i)
	{
		pmProcessingElement* lDevice = new pmProcessingElement(pMachine, i, pGlobalStartingDeviceIndex + i);

		pmDeviceData lData;
		lData.name = pDeviceData[i].name;
		lData.description = pDeviceData[i].description;

		mDevicesVector.push_back(lDevice);
		mDeviceDataVector.push_back(lData);
	}

	return pmSuccess;
}

pmStatus pmDevicePool::BroadcastDeviceData(pmMachine* pMachine, pmCommunicatorCommand::devicePool* pDeviceArray, uint pDeviceCount)
{
	pmStubManager* lManager = pmStubManager::GetStubManager();
	if(pMachine == PM_LOCAL_MACHINE)
	{
		for(uint i=0; i<pDeviceCount; ++i)
		{
			strncpy(pDeviceArray[i].name, lManager->GetStub(i)->GetDeviceName().c_str(), MAX_NAME_STR_LEN-1);
			strncpy(pDeviceArray[i].description, lManager->GetStub(i)->GetDeviceDescription().c_str(), MAX_DESC_STR_LEN-1);

			pDeviceArray[i].name[MAX_NAME_STR_LEN-1] = '\0';
			pDeviceArray[i].description[MAX_DESC_STR_LEN-1] = '\0';
		}
	}
	
	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_PRIORITY_LEVEL, pmCommunicatorCommand::BROADCAST, pmCommunicatorCommand::DEVICE_POOL_TRANSFER, pMachine,
		pmCommunicatorCommand::DEVICE_POOL_STRUCT, pDeviceArray, pDeviceCount, NULL, NULL);

	pmCommunicator::GetCommunicator()->Broadcast(lCommand);

	lCommand->WaitForFinish();
	return lCommand->GetStatus();
}

uint pmDevicePool::GetDeviceCount()
{
	return (uint)mDevicesVector.size();
}

pmDevicePool::pmDeviceData& pmDevicePool::GetDeviceData(pmProcessingElement* pDevice)
{
	uint lGlobalIndex = pDevice->GetGlobalDeviceIndex();
	if(lGlobalIndex >= (uint)mDevicesVector.size())
		PMTHROW(pmUnknownDeviceException(lGlobalIndex));

	return mDeviceDataVector[lGlobalIndex];
}

std::vector<pmProcessingElement*>& pmDevicePool::GetAllDevices()
{
	return mDevicesVector;
}

pmProcessingElement* pmDevicePool::GetDeviceAtMachineIndex(pmMachine* pMachine, uint pDeviceIndexOnMachine)
{
	uint lGlobalIndex = pmMachinePool::GetMachinePool()->GetFirstDeviceIndexOnMachine(pMachine) + pDeviceIndexOnMachine;
	if(lGlobalIndex >= (uint)mDevicesVector.size())
		PMTHROW(pmUnknownDeviceException(lGlobalIndex));

	pmProcessingElement* lDevice = GetDeviceAtGlobalIndex(lGlobalIndex);
	if(lDevice->GetMachine() != pMachine)
		PMTHROW(pmFatalErrorException());

	return lDevice;
}

pmProcessingElement* pmDevicePool::GetDeviceAtGlobalIndex(uint pGlobalDeviceIndex)
{
	if(pGlobalDeviceIndex >= (uint)mDevicesVector.size())
		PMTHROW(pmUnknownDeviceException(pGlobalDeviceIndex));

	return mDevicesVector[pGlobalDeviceIndex];
}

pmStatus pmDevicePool::GetAllDevicesOfTypeInCluster(pmDeviceTypes pType, pmCluster* pCluster, std::set<pmProcessingElement*>& pDevices)
{
	size_t lSize = mDevicesVector.size();
	for(size_t i=0; i<lSize; ++i)
	{
		if(mDevicesVector[i]->GetType() == pType && pCluster->ContainsMachine(mDevicesVector[i]->GetMachine()))
			pDevices.insert(mDevicesVector[i]);
	}

	return pmSuccess;
}

}



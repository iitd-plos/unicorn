
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

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
	return mMachinePool;
}

pmMachinePool::pmMachinePool()
{
    if(mMachinePool)
        PMTHROW(pmFatalErrorException());
    
    mMachinePool = this;

	uint i=0;
    uint lMachineCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

	FINALIZE_PTR_ARRAY(fAll2AllBuffer, pmCommunicatorCommand::machinePool, new pmCommunicatorCommand::machinePool[lMachineCount]);

	{
		FINALIZE_RESOURCE(fMachinePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::MACHINE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::MACHINE_POOL_STRUCT)));

		All2AllMachineData(fAll2AllBuffer);
	}

	uint lLocalId = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId();

	for(i=0; i<lMachineCount; ++i)
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

	pmDevicePool* lDevicePool = new pmDevicePool();

	FINALIZE_RESOURCE(fDevicePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(pmCommunicatorCommand::DEVICE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(pmCommunicatorCommand::DEVICE_POOL_STRUCT)));

	for(uint i=0; i<lMachineCount; ++i)
	{
		pmMachineData& lData = GetMachineData(i);
		pmMachine* lMachine = GetMachine(i);

		uint lGlobalIndex = lDevicePool->GetDeviceCount();
		mFirstDeviceIndexOnMachine.push_back(lGlobalIndex);

		uint lDevicesCount = lData.cpuCores + lData.gpuCards;

		FINALIZE_PTR_ARRAY(fDevicesBuffer, pmCommunicatorCommand::devicePool, new pmCommunicatorCommand::devicePool[lDevicesCount]);
		lDevicePool->BroadcastDeviceData(lMachine, fDevicesBuffer, lDevicesCount);

		lDevicePool->CreateMachineDevices(lMachine, lData.cpuCores, fDevicesBuffer, lGlobalIndex, lDevicesCount);
	}
}

pmMachinePool::~pmMachinePool()
{
    delete pmDevicePool::GetDevicePool();
    
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

	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_PRIORITY_LEVEL, pmCommunicatorCommand::ALL2ALL, pmCommunicatorCommand::MACHINE_POOL_TRANSFER_TAG, NULL, pmCommunicatorCommand::MACHINE_POOL_STRUCT, &lSendBuffer, 1, pAll2AllBuffer, 1);

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

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mMachineDataVector[lIndex].dataSent += pDataSent;
	mMachineDataVector[lIndex].sendTime += pSendTime;
	++(mMachineDataVector[lIndex].sendCount);

	return pmSuccess;
}

pmStatus pmMachinePool::RegisterReceiveCompletion(pmMachine* pMachine, ulong pDataReceived, double pReceiveTime)
{
	size_t lIndex = (size_t)(*pMachine);

	if(lIndex >= (uint)mMachinesVector.size())
		PMTHROW(pmUnknownMachineException((uint)lIndex));

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mMachineDataVector[lIndex].dataReceived += pDataReceived;
	mMachineDataVector[lIndex].receiveTime += pReceiveTime;
	++(mMachineDataVector[lIndex].receiveCount);

	return pmSuccess;
}


/* class pmDevicePool */
pmDevicePool* pmDevicePool::GetDevicePool()
{
	return mDevicePool;
}

pmDevicePool::pmDevicePool()
{
    if(mDevicePool)
        PMTHROW(pmFatalErrorException());
    
    mDevicePool = this;
    
	mDevicesVector.clear();
	mDeviceDataVector.clear();
}

pmDevicePool::~pmDevicePool()
{
	size_t lSize = mDevicesVector.size();
	for(size_t i=0; i<lSize; ++i)
		delete mDevicesVector[i];
}

pmStatus pmDevicePool::CreateMachineDevices(pmMachine* pMachine, uint pCpuDeviceCount, pmCommunicatorCommand::devicePool* pDeviceData, uint pGlobalStartingDeviceIndex, uint pDeviceCount)
{
	for(uint i=0; i<pDeviceCount; ++i)
	{
		pmDeviceType lDeviceType = CPU;
#ifdef SUPPORT_CUDA
		if(i >= pCpuDeviceCount)
			lDeviceType = GPU_CUDA;
#endif
		pmProcessingElement* lDevice = new pmProcessingElement(pMachine, lDeviceType, i, pGlobalStartingDeviceIndex + i);

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
	
	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand::CreateSharedPtr(MAX_PRIORITY_LEVEL, pmCommunicatorCommand::BROADCAST, pmCommunicatorCommand::DEVICE_POOL_TRANSFER_TAG, pMachine,
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

pmStatus pmDevicePool::GetAllDevicesOfTypeInCluster(pmDeviceType pType, pmCluster* pCluster, std::set<pmProcessingElement*>& pDevices)
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



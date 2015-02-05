
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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

using namespace communicator;
    
pmMachine* PM_LOCAL_MACHINE = NULL;

/* class pmMachinePool */
pmMachinePool* pmMachinePool::GetMachinePool()
{
    static pmMachinePool lMachinePool;
	return &lMachinePool;
}

pmMachinePool::pmMachinePool()
    : mResourceLock __LOCK_NAME__("pmMachinePool::mResourceLock")
{
    uint lMachineCount = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetTotalHostCount();

    mMachinesVector.reserve(lMachineCount);
    for(uint i = 0; i < lMachineCount; ++i)
        mMachinesVector.push_back(pmMachine(i));

    All2AllMachineData(lMachineCount);
    
	uint lLocalId = NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId();
    PM_LOCAL_MACHINE = &(mMachinesVector[lLocalId]);

	pmDevicePool* lDevicePool = pmDevicePool::GetDevicePool();

	FINALIZE_RESOURCE(fDevicePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(DEVICE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(DEVICE_POOL_STRUCT)));

    mFirstDeviceIndexOnMachine.reserve(lMachineCount);
	for(uint i = 0; i < lMachineCount; ++i)
	{
		pmMachineData& lData = GetMachineData(i);
		const pmMachine* lMachine = GetMachine(i);

		uint lGlobalIndex = lDevicePool->GetDeviceCount();
		mFirstDeviceIndexOnMachine.push_back(lGlobalIndex);

		uint lDevicesCount = lData.cpuCores + lData.gpuCards;

		lDevicePool->BroadcastAndCreateDeviceData(lMachine, lDevicesCount, lData.cpuCores, lGlobalIndex);
	}
}

void pmMachinePool::All2AllMachineData(size_t pMachineCount)
{
    FINALIZE_RESOURCE(fMachinePoolResource, (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->RegisterTransferDataType(MACHINE_POOL_STRUCT)), (NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->UnregisterTransferDataType(MACHINE_POOL_STRUCT)));

	pmStubManager* lManager = pmStubManager::GetStubManager();
    
	machinePool lSendBuffer((uint)(lManager->GetProcessingElementsCPU()), (uint)(lManager->GetProcessingElementsGPU()), (uint)(lManager->GetCpuNumaDomainsCount()));
    finalize_ptr<all2AllWrapper<machinePool> > lWrapper(new all2AllWrapper<machinePool>(lSendBuffer, pMachineCount));

    // lWrapper's ownership is now transferred to lCommand
	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<all2AllWrapper<machinePool> >::CreateSharedPtr(MAX_PRIORITY_LEVEL, ALL2ALL, MACHINE_POOL_TRANSFER_TAG, NULL, MACHINE_POOL_STRUCT, lWrapper, 1);

	pmCommunicator::GetCommunicator()->All2All(lCommand, true);

    all2AllWrapper<machinePool>* lData = ((all2AllWrapper<machinePool>*)(lCommand->GetData()));
    DEBUG_EXCEPTION_ASSERT(pMachineCount == lData->all2AllData.size());

    mMachineDataVector.reserve(lData->all2AllData.size());
    std::vector<machinePool>::const_iterator lIter = lData->all2AllData.begin(), lEndIter = lData->all2AllData.end();
	for(; lIter != lEndIter; ++lIter)
		mMachineDataVector.emplace_back((*lIter).cpuCores, (*lIter).gpuCards, (*lIter).cpuNumaDomains);
}

pmMachinePool::pmMachineData& pmMachinePool::GetMachineData(uint pIndex)
{
	DEBUG_EXCEPTION_ASSERT(pIndex < (uint)mMachinesVector.size());
	
	return mMachineDataVector[pIndex];
}

pmMachinePool::pmMachineData& pmMachinePool::GetMachineData(const pmMachine* pMachine)
{
	return GetMachineData(*pMachine);
}

const pmMachine* pmMachinePool::GetMachine(uint pIndex) const
{
	DEBUG_EXCEPTION_ASSERT(pIndex < (uint)mMachinesVector.size());

	return &(mMachinesVector[pIndex]);
}

void pmMachinePool::GetAllDevicesOnMachine(uint pMachineIndex, std::vector<const pmProcessingElement*>& pDevices) const
{
	GetAllDevicesOnMachine(GetMachine(pMachineIndex), pDevices);
}

void pmMachinePool::GetAllDevicesOnMachine(const pmMachine* pMachine, std::vector<const pmProcessingElement*>& pDevices) const
{
	pDevices.clear();

	pmDevicePool* lDevicePool = pmDevicePool::GetDevicePool();
	uint lTotalDevices = lDevicePool->GetDeviceCount();
	uint lGlobalIndex = mFirstDeviceIndexOnMachine[*pMachine];

	for(uint i = lGlobalIndex; i < lTotalDevices; ++i)
	{
		const pmProcessingElement* lDevice = lDevicePool->GetDeviceAtGlobalIndex(i);

		if(lDevice->GetMachine() == pMachine)
			pDevices.push_back(lDevice);
		else
			break;
	}
}

uint pmMachinePool::GetFirstDeviceIndexOnMachine(uint pMachineIndex) const
{
	DEBUG_EXCEPTION_ASSERT(pMachineIndex < (uint)mMachinesVector.size());

	return mFirstDeviceIndexOnMachine[pMachineIndex];
}

uint pmMachinePool::GetFirstDeviceIndexOnMachine(const pmMachine* pMachine) const
{
	return GetFirstDeviceIndexOnMachine(*pMachine);
}

void pmMachinePool::RegisterSendCompletion(const pmMachine* pMachine, ulong pDataSent, double pSendTime)
{
	size_t lIndex = (size_t)(*pMachine);

	DEBUG_EXCEPTION_ASSERT(lIndex < (uint)mMachinesVector.size());

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mMachineDataVector[lIndex].dataSent += pDataSent;
	mMachineDataVector[lIndex].sendTime += pSendTime;
	++(mMachineDataVector[lIndex].sendCount);
}

void pmMachinePool::RegisterReceiveCompletion(const pmMachine* pMachine, ulong pDataReceived, double pReceiveTime)
{
	size_t lIndex = (size_t)(*pMachine);

	DEBUG_EXCEPTION_ASSERT(lIndex < (uint)mMachinesVector.size());

	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mMachineDataVector[lIndex].dataReceived += pDataReceived;
	mMachineDataVector[lIndex].receiveTime += pReceiveTime;
	++(mMachineDataVector[lIndex].receiveCount);
}
    
uint pmMachinePool::GetCpuNumaDomainsOnMachine(uint pIndex)
{
    return GetMachineData(pIndex).cpuNumaDomains;
}


/* class pmDevicePool */
pmDevicePool* pmDevicePool::GetDevicePool()
{
	static pmDevicePool lDevicePool;
    return &lDevicePool;
}

void pmDevicePool::CreateMachineDevices(const pmMachine* pMachine, uint pCpuDeviceCount, const devicePool* pDeviceData, uint pGlobalStartingDeviceIndex, uint pDeviceCount)
{
	for(uint i = 0; i < pDeviceCount; ++i)
	{
		pmDeviceType lDeviceType = CPU;

    #ifdef SUPPORT_CUDA
		if(i >= pCpuDeviceCount)
			lDeviceType = GPU_CUDA;
    #endif

		mDevicesVector.push_back(pmProcessingElement(pMachine, lDeviceType, i, pGlobalStartingDeviceIndex + i, pDeviceData[i].numaDomain, pDeviceData));
		mDeviceDataVector.emplace_back(pDeviceData[i].name, pDeviceData[i].description);
	}
}

void pmDevicePool::BroadcastAndCreateDeviceData(const pmMachine* pMachine, uint pDeviceCount, uint pCpuDeviceCount, uint pGlobalStartingDeviceIndex)
{
    finalize_ptr<devicePool, deleteArrayDeallocator<devicePool> > lDevicePoolArray(new devicePool[pDeviceCount]);

	pmStubManager* lManager = pmStubManager::GetStubManager();
	if(pMachine == PM_LOCAL_MACHINE)
	{
		for(uint i = 0; i < pDeviceCount; ++i)
		{
            devicePool& lDevicePool = (lDevicePoolArray.get_ptr())[i];

            pmExecutionStub* lStub = lManager->GetStub(i);
			strncpy(lDevicePool.name, lStub->GetDeviceName().c_str(), MAX_NAME_STR_LEN - 1);
			strncpy(lDevicePool.description, lStub->GetDeviceDescription().c_str(), MAX_DESC_STR_LEN - 1);

			lDevicePool.name[MAX_NAME_STR_LEN - 1] = '\0';
			lDevicePool.description[MAX_DESC_STR_LEN - 1] = '\0';
            lDevicePool.numaDomain = ((lStub->GetType() == CPU) ? lManager->GetNumaDomainIdForCpuDevice(i) : std::numeric_limits<uint>::max());
		}
	}

    // Ownership of lDevicePoolArray is now transferred to lCommand
	pmCommunicatorCommandPtr lCommand = pmCommunicatorCommand<devicePool, deleteArrayDeallocator<devicePool> >::CreateSharedPtr(MAX_PRIORITY_LEVEL, BROADCAST, DEVICE_POOL_TRANSFER_TAG, pMachine, DEVICE_POOL_STRUCT, lDevicePoolArray, pDeviceCount);

	pmCommunicator::GetCommunicator()->Broadcast(lCommand, true);

    CreateMachineDevices(pMachine, pCpuDeviceCount, (devicePool*)(lCommand->GetData()), pGlobalStartingDeviceIndex, pDeviceCount);
}

uint pmDevicePool::GetDeviceCount() const
{
	return (uint)mDevicesVector.size();
}

const pmDevicePool::pmDeviceData& pmDevicePool::GetDeviceData(const pmProcessingElement* pDevice) const
{
	uint lGlobalIndex = pDevice->GetGlobalDeviceIndex();
    
    DEBUG_EXCEPTION_ASSERT(lGlobalIndex < (uint)mDevicesVector.size());

	return mDeviceDataVector[lGlobalIndex];
}

const pmProcessingElement* pmDevicePool::GetDeviceAtMachineIndex(const pmMachine* pMachine, uint pDeviceIndexOnMachine) const
{
	uint lGlobalIndex = pmMachinePool::GetMachinePool()->GetFirstDeviceIndexOnMachine(pMachine) + pDeviceIndexOnMachine;

    DEBUG_EXCEPTION_ASSERT(lGlobalIndex < (uint)mDevicesVector.size());

	const pmProcessingElement* lDevice = GetDeviceAtGlobalIndex(lGlobalIndex);

	DEBUG_EXCEPTION_ASSERT(lDevice->GetMachine() == pMachine);

	return lDevice;
}

const pmProcessingElement* pmDevicePool::GetDeviceAtGlobalIndex(uint pGlobalDeviceIndex) const
{
	DEBUG_EXCEPTION_ASSERT(pGlobalDeviceIndex < (uint)mDevicesVector.size());

	return &(mDevicesVector[pGlobalDeviceIndex]);
}

void pmDevicePool::GetAllDevicesOfTypeInCluster(pmDeviceType pType, const pmCluster* pCluster, std::vector<const pmProcessingElement*>& pDevices) const
{
	for(auto& lDevice: mDevicesVector)
	{
		if(lDevice.GetType() == pType && pCluster->ContainsMachine(lDevice.GetMachine()))
			pDevices.push_back(&lDevice);
	}
}

void pmDevicePool::GetAllDevicesOfTypeOnMachines(pmDeviceType pType, const std::set<const pmMachine*>& pMachines, std::vector<const pmProcessingElement*>& pDevices) const
{
	for(auto& lDevice: mDevicesVector)
	{
		if(lDevice.GetType() == pType && pMachines.find(lDevice.GetMachine()) != pMachines.end())
			pDevices.push_back(&lDevice);
	}
}
    
std::vector<const pmProcessingElement*> pmDevicePool::InterleaveDevicesFromDifferentMachines(const std::vector<const pmProcessingElement*>& pDevices) const
{
    std::map<const pmMachine*, std::vector<const pmProcessingElement*>> lMachineVersusDevicesMap;
    for_each(pDevices, [&] (const pmProcessingElement* pDevice)
    {
        const pmMachine* lMachine = pDevice->GetMachine();

        auto lIter = lMachineVersusDevicesMap.find(lMachine);
        if(lIter == lMachineVersusDevicesMap.end())
            lIter = lMachineVersusDevicesMap.emplace(lMachine, std::vector<const pmProcessingElement*>()).first;

        lIter->second.emplace_back(pDevice);
    });
    
    std::map<const pmMachine*, std::pair<std::map<const pmMachine*, std::vector<const pmProcessingElement*>>::iterator, std::vector<const pmProcessingElement*>::iterator>> lMachineVersusDevicesIterMap;
    auto lBeginIter = lMachineVersusDevicesMap.begin(), lEndIter = lMachineVersusDevicesMap.end(), lIter = lBeginIter;
    for(lIter = lBeginIter; lIter != lEndIter; ++lIter)
    {
        lMachineVersusDevicesIterMap.emplace(std::piecewise_construct, std::forward_as_tuple(lIter->first), std::forward_as_tuple(lIter, lIter->second.begin()));
    }

    size_t lDeviceCount = pDevices.size();

    std::vector<const pmProcessingElement*> lInterleavedDevices;
    lInterleavedDevices.reserve(lDeviceCount);

    ulong i = 0;
    auto lBeginIter2 = lMachineVersusDevicesIterMap.begin(), lEndIter2 = lMachineVersusDevicesIterMap.end(), lIter2 = lBeginIter2;
    while(i != lDeviceCount)
    {
        if(lIter2 == lEndIter2)
            lIter2 = lBeginIter2;
        
        if(lIter2->second.second != lIter2->second.first->second.end())
        {
            lInterleavedDevices.emplace_back(*lIter2->second.second);
            ++lIter2->second.second;

            ++i;
        }
        
        ++lIter2;
    }

    return lInterleavedDevices;
}
    
}



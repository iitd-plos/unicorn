
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

#include "pmHardware.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmDevicePool.h"

#include <string.h>

namespace pm
{

/* class pmMachine */
pmMachine::pmMachine(uint pMachineId)
{
	mMachineId = pMachineId;
}

pmMachine::operator uint() const
{
    return mMachineId;
}


/* class pmProcessingElement */
pmProcessingElement::pmProcessingElement(const pmMachine* pMachine, pmDeviceType pDeviceType, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex, const communicator::devicePool* pDevicePool)
	: mMachine(pMachine)
    , mDeviceIndexInMachine(pDeviceIndexInMachine)
	, mGlobalDeviceIndex(pGlobalDeviceIndex)
	, mDeviceType(pDeviceType)
{
    BuildDeviceInfo(pDevicePool);
}

const pmMachine* pmProcessingElement::GetMachine() const
{
	return mMachine;
}

uint pmProcessingElement::GetDeviceIndexInMachine() const
{
	return mDeviceIndexInMachine;
}

uint pmProcessingElement::GetGlobalDeviceIndex() const
{
	return mGlobalDeviceIndex;
}

pmDeviceType pmProcessingElement::GetType() const
{
	return mDeviceType;
}

pmExecutionStub* pmProcessingElement::GetLocalExecutionStub() const
{
	if(GetMachine() != PM_LOCAL_MACHINE)
		PMTHROW(pmFatalErrorException());

	return pmStubManager::GetStubManager()->GetStub(mDeviceIndexInMachine);
}

void pmProcessingElement::BuildDeviceInfo(const communicator::devicePool* pDevicePool)
{
    const char* lName = pDevicePool->name;
    const char* lDesc = pDevicePool->description;

    size_t lNameLength = std::min(strlen(lName), (size_t)(MAX_NAME_STR_LEN - 1));
    size_t lDescLength = std::min(strlen(lDesc), (size_t)(MAX_DESC_STR_LEN - 1));
    
    memcpy(mDeviceInfo.name, lName, lNameLength);
    memcpy(mDeviceInfo.description, lDesc, lDescLength);

    mDeviceInfo.deviceHandle = ((GetMachine() == PM_LOCAL_MACHINE) ? static_cast<void*>(GetLocalExecutionStub()) : this);
    mDeviceInfo.name[lNameLength] = '\0';
    mDeviceInfo.description[lDescLength] = '\0';

    mDeviceInfo.deviceType = GetType();
    mDeviceInfo.host = *(GetMachine());
}
    
const pmDeviceInfo& pmProcessingElement::GetDeviceInfo() const
{
    return mDeviceInfo;
}

void pmProcessingElement::GetMachines(std::set<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines)
{
	for(auto lDevice: pDevices)
		pMachines.insert(lDevice->GetMachine());
}

void pmProcessingElement::GetMachines(std::vector<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines)
{
	for(auto lDevice: pDevices)
		pMachines.insert(lDevice->GetMachine());
}

};

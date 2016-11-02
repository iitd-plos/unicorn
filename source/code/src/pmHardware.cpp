
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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
pmProcessingElement::pmProcessingElement(const pmMachine* pMachine, pmDeviceType pDeviceType, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex, ushort pNumaDomainId, const communicator::devicePool* pDevicePool)
	: mMachine(pMachine)
    , mDeviceIndexInMachine(pDeviceIndexInMachine)
	, mGlobalDeviceIndex(pGlobalDeviceIndex)
	, mDeviceType(pDeviceType)
    , mNumaDomainId(pNumaDomainId)
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

ushort pmProcessingElement::GetNumaDomainId() const
{
    return mNumaDomainId;
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

    PMLIB_MEMCPY(mDeviceInfo.name, lName, lNameLength, std::string("pmProcessingElement::BuildDeviceInfo1"));
    PMLIB_MEMCPY(mDeviceInfo.description, lDesc, lDescLength, std::string("pmProcessingElement::BuildDeviceInfo2"));

    mDeviceInfo.deviceHandle = ((GetMachine() == PM_LOCAL_MACHINE) ? static_cast<void*>(GetLocalExecutionStub()) : this);
    mDeviceInfo.name[lNameLength] = '\0';
    mDeviceInfo.description[lDescLength] = '\0';

    mDeviceInfo.deviceType = GetType();
    mDeviceInfo.host = *(GetMachine());
    
    mDeviceInfo.deviceIdOnHost = mDeviceIndexInMachine;
    mDeviceInfo.deviceIdInCluster = mGlobalDeviceIndex;
}
    
const pmDeviceInfo& pmProcessingElement::GetDeviceInfo() const
{
    return mDeviceInfo;
}

void pmProcessingElement::GetMachines(const std::set<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines)
{
	for(auto lDevice: pDevices)
		pMachines.insert(lDevice->GetMachine());
}

void pmProcessingElement::GetMachines(const std::vector<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines)
{
	for(auto lDevice: pDevices)
		pMachines.insert(lDevice->GetMachine());
}

void pmProcessingElement::GetMachinesInOrder(const std::vector<const pmProcessingElement*>& pDevices, std::vector<const pmMachine*>& pMachines)
{
    std::map<uint, const pmMachine*> lMap;

	for(auto lDevice: pDevices)
    {
        const pmMachine* lMachine = lDevice->GetMachine();
        lMap[(uint)(*lMachine)] = lMachine;
    }

    pMachines.reserve(lMap.size());
    for_each(lMap, [&] (const decltype(lMap)::value_type& pPair)
    {
        pMachines.emplace_back(pPair.second);
    });
}

};

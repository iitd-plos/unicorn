
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

#include "pmHardware.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmDevicePool.h"

#include <string.h>

namespace pm
{

/* class pmHardware */
pmHardware::pmHardware()
{
}

pmHardware::~pmHardware()
{
}


/* class pmMachine */
pmMachine::pmMachine(uint pMachineId)
{
	mMachineId = pMachineId;
}

pmMachine::~pmMachine()
{
}

pmMachine::operator uint()
{
	 return mMachineId;
}


/* class pmProcessingElement */
pmProcessingElement::pmProcessingElement(pmMachine* pMachine, pmDeviceTypes pDeviceType, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex) : mMachine(pMachine)
{
	mDeviceIndexInMachine = pDeviceIndexInMachine;
	mGlobalDeviceIndex = pGlobalDeviceIndex;
	mDeviceType = pDeviceType;
}

pmProcessingElement::~pmProcessingElement()
{
}

pmMachine* pmProcessingElement::GetMachine()
{
	return mMachine;
}

uint pmProcessingElement::GetDeviceIndexInMachine()
{
	return mDeviceIndexInMachine;
}

uint pmProcessingElement::GetGlobalDeviceIndex()
{
	return mGlobalDeviceIndex;
}

pmDeviceTypes pmProcessingElement::GetType()
{
	return mDeviceType;
}

pmExecutionStub* pmProcessingElement::GetLocalExecutionStub()
{
	if(GetMachine() != PM_LOCAL_MACHINE)
		PMTHROW(pmFatalErrorException());

	return pmStubManager::GetStubManager()->GetStub(mDeviceIndexInMachine);
}

pmDeviceInfo pmProcessingElement::GetDeviceInfo()
{
	pmDeviceInfo lDeviceInfo;

	pmDevicePool::pmDeviceData& lDeviceData = pmDevicePool::GetDevicePool()->GetDeviceData(this);

	const char* lName = lDeviceData.name.c_str();
	const char* lDesc = lDeviceData.description.c_str();

	size_t lNameLength = min(strlen(lName), (size_t)(MAX_NAME_STR_LEN-1));
	size_t lDescLength = min(strlen(lDesc), (size_t)(MAX_DESC_STR_LEN-1));
	
	memcpy(lDeviceInfo.name, lName, lNameLength);
	memcpy(lDeviceInfo.description, lDesc, lDescLength);

	lDeviceInfo.name[lNameLength] = '\0';
	lDeviceInfo.description[lDescLength] = '\0';

	lDeviceInfo.deviceTypeInfo = GetType();
	lDeviceInfo.host = *(GetMachine());

	return lDeviceInfo;
}

pmStatus pmProcessingElement::GetMachines(std::set<pmProcessingElement*>& pDevices, std::set<pmMachine*>& pMachines)
{
	std::set<pmProcessingElement*>::iterator lIter;
	for(lIter = pDevices.begin(); lIter != pDevices.end(); ++lIter)
		pMachines.insert((*lIter)->GetMachine());

	return pmSuccess;
}

pmStatus pmProcessingElement::GetMachines(std::vector<pmProcessingElement*>& pDevices, std::set<pmMachine*>& pMachines)
{
	std::vector<pmProcessingElement*>::iterator lIter;
	for(lIter = pDevices.begin(); lIter != pDevices.end(); ++lIter)
		pMachines.insert((*lIter)->GetMachine());

	return pmSuccess;
}

pmStatus pmProcessingElement::GetMachines(std::vector<pmProcessingElement*>& pDevices, std::vector<pmMachine*>& pMachines)
{
	std::vector<pmProcessingElement*>::iterator lIter;
	for(lIter = pDevices.begin(); lIter != pDevices.end(); ++lIter)
		pMachines.push_back((*lIter)->GetMachine());

	return pmSuccess;
}

};

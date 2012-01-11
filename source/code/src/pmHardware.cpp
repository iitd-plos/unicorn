
#include "pmHardware.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"

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
pmProcessingElement::pmProcessingElement(pmMachine* pMachine, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex) : mMachine(pMachine)
{
	mDeviceIndexInMachine = pDeviceIndexInMachine;
	mGlobalDeviceIndex = pGlobalDeviceIndex;
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
	return GetLocalExecutionStub()->GetType();
}

pmExecutionStub* pmProcessingElement::GetLocalExecutionStub()
{
	return pmStubManager::GetStubManager()->GetStub(mDeviceIndexInMachine);
}

pmDeviceInfo pmProcessingElement::GetDeviceInfo()
{
	pmDeviceInfo lDeviceInfo;

	pmExecutionStub* lStub = GetLocalExecutionStub();
	const char* lName = lStub->GetDeviceName().c_str();
	const char* lDesc = lStub->GetDeviceDescription().c_str();

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
		pMachines.insert(lIter._Mynode()->_Myval->GetMachine());

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

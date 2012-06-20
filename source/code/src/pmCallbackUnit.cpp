
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

#include "pmCallbackUnit.h"

namespace pm
{

std::map<std::string, pmCallbackUnit*> pmCallbackUnit::mKeyMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmCallbackUnit::mResourceLock;

pmCallbackUnit::pmCallbackUnit(char* pKey, pmDataDistributionCB* pDataDistributionCB, pmSubtaskCB* pSubtaskCB, pmDataReductionCB* pDataReductionCB, pmDeviceSelectionCB* pDeviceSelectionCB,
	pmDataRedistributionCB* pDataRedistributionCB, pmPreDataTransferCB* pPreDataTransferCB, pmPostDataTransferCB* pPostDataTransferCB)
{
	mDataDistributionCB = pDataDistributionCB;
	mSubtaskCB = pSubtaskCB;
	mDataReductionCB = pDataReductionCB;
	mDataRedistributionCB = pDataRedistributionCB;
	mDeviceSelectionCB = pDeviceSelectionCB;
	mPreDataTransferCB = pPreDataTransferCB;
	mPostDataTransferCB = pPostDataTransferCB;

	mKey = pKey;

	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());
	if(mKeyMap.find(mKey) != mKeyMap.end())
		PMTHROW(pmInvalidKeyException());

	mKeyMap[mKey] = this;
}

pmCallbackUnit::~pmCallbackUnit()
{
	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	mKeyMap.erase(mKey);
}

pmDataDistributionCB* pmCallbackUnit::GetDataDistributionCB()
{
	return mDataDistributionCB;
}

pmSubtaskCB* pmCallbackUnit::GetSubtaskCB()
{
	return mSubtaskCB;
}

pmDataReductionCB* pmCallbackUnit::GetDataReductionCB()
{
	return mDataReductionCB;
}

pmDataRedistributionCB* pmCallbackUnit::GetDataRedistributionCB()
{
	return mDataRedistributionCB;
}

pmDeviceSelectionCB* pmCallbackUnit::GetDeviceSelectionCB()
{
	return mDeviceSelectionCB;
}

pmPreDataTransferCB* pmCallbackUnit::GetPreDataTransferCB()
{
	return mPreDataTransferCB;
}

pmPostDataTransferCB* pmCallbackUnit::GetPostDataTransferCB()
{
	return mPostDataTransferCB;
}

const char* pmCallbackUnit::GetKey()
{
	return mKey.c_str();
}

pmCallbackUnit* pmCallbackUnit::FindCallbackUnit(char* pKey)
{
	std::string lStr(pKey);

	FINALIZE_RESOURCE(dResourceLock, mResourceLock.Lock(), mResourceLock.Unlock());

	std::map<std::string, pmCallbackUnit*>::iterator lIter = mKeyMap.find(lStr);
	if(lIter == mKeyMap.end())
		PMTHROW(pmInvalidKeyException());

	pmCallbackUnit* lCallbackUnit = mKeyMap[lStr];

	return lCallbackUnit;
}

};

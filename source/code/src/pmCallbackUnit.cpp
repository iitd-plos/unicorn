
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

#include "pmCallbackUnit.h"

namespace pm
{

STATIC_ACCESSOR(pmCallbackUnit::keyMapType, pmCallbackUnit, GetKeyMap)
STATIC_ACCESSOR_ARG(RESOURCE_LOCK_IMPLEMENTATION_CLASS, __STATIC_LOCK_NAME__("pmCallbackUnit::mResourceLock"), pmCallbackUnit, GetResourceLock)

pmCallbackUnit::pmCallbackUnit(const char* pKey, finalize_ptr<pmDataDistributionCB>&& pDataDistributionCB, finalize_ptr<pmSubtaskCB>&& pSubtaskCB, finalize_ptr<pmDataReductionCB>&& pDataReductionCB, finalize_ptr<pmDeviceSelectionCB>&& pDeviceSelectionCB, finalize_ptr<pmDataRedistributionCB>&& pDataRedistributionCB, finalize_ptr<pmPreDataTransferCB>&& pPreDataTransferCB, finalize_ptr<pmPostDataTransferCB>&& pPostDataTransferCB, finalize_ptr<pmTaskCompletionCB>&& pTaskCompletionCB)
    : mDataDistributionCB(std::move(pDataDistributionCB))
	, mSubtaskCB(std::move(pSubtaskCB))
	, mDataReductionCB(std::move(pDataReductionCB))
	, mDataRedistributionCB(std::move(pDataRedistributionCB))
	, mDeviceSelectionCB(std::move(pDeviceSelectionCB))
	, mPreDataTransferCB(std::move(pPreDataTransferCB))
	, mPostDataTransferCB(std::move(pPostDataTransferCB))
    , mTaskCompletionCB(std::move(pTaskCompletionCB))
	, mKey(pKey)
{
    keyMapType& lKeyMap = GetKeyMap();

	FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());
    
	if(lKeyMap.find(mKey) != lKeyMap.end())
		PMTHROW(pmInvalidKeyException());

	lKeyMap[mKey] = this;
}

pmCallbackUnit::~pmCallbackUnit()
{
	FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

	GetKeyMap().erase(mKey);
}

const pmDataDistributionCB* pmCallbackUnit::GetDataDistributionCB() const
{
	return mDataDistributionCB.get_ptr();
}

const pmSubtaskCB* pmCallbackUnit::GetSubtaskCB() const
{
	return mSubtaskCB.get_ptr();
}

const pmDataReductionCB* pmCallbackUnit::GetDataReductionCB() const
{
	return mDataReductionCB.get_ptr();
}

const pmDataRedistributionCB* pmCallbackUnit::GetDataRedistributionCB() const
{
	return mDataRedistributionCB.get_ptr();
}

const pmDeviceSelectionCB* pmCallbackUnit::GetDeviceSelectionCB() const
{
	return mDeviceSelectionCB.get_ptr();
}

const pmPreDataTransferCB* pmCallbackUnit::GetPreDataTransferCB() const
{
	return mPreDataTransferCB.get_ptr();
}

const pmPostDataTransferCB* pmCallbackUnit::GetPostDataTransferCB() const
{
	return mPostDataTransferCB.get_ptr();
}
    
const pmTaskCompletionCB* pmCallbackUnit::GetTaskCompletionCB() const
{
    return mTaskCompletionCB.get_ptr();
}
    
void pmCallbackUnit::SetTaskCompletionCB(finalize_ptr<pmTaskCompletionCB>&& pTaskCompletionCB)
{
    mTaskCompletionCB = std::move(pTaskCompletionCB);
}

const char* pmCallbackUnit::GetKey() const
{
	return mKey.c_str();
}

const pmCallbackUnit* pmCallbackUnit::FindCallbackUnit(char* pKey)
{
	std::string lStr(pKey);

	FINALIZE_RESOURCE(dResourceLock, GetResourceLock().Lock(), GetResourceLock().Unlock());

	DEBUG_EXCEPTION_ASSERT(GetKeyMap().find(lStr) != GetKeyMap().end());

	return GetKeyMap().find(lStr)->second;
}

};

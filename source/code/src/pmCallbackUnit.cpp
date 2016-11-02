
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

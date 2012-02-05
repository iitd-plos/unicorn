
#include "pmCallbackUnit.h"

namespace pm
{

std::map<std::string, pmCallbackUnit*> pmCallbackUnit::mKeyMap;
RESOURCE_LOCK_IMPLEMENTATION_CLASS pmCallbackUnit::mResourceLock;

pmCallbackUnit::pmCallbackUnit(char* pKey, pmDataDistributionCB* pDataDistributionCB, pmSubtaskCB* pSubtaskCB, pmDataReductionCB* pDataReductionCB, pmDeviceSelectionCB* pDeviceSelectionCB,
	pmPreDataTransferCB* pPreDataTransferCB, pmPostDataTransferCB* pPostDataTransferCB)
{
	mDataDistributionCB = pDataDistributionCB;
	mSubtaskCB = pSubtaskCB;
	mDataReductionCB = pDataReductionCB;
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

pmDataScatterCB* pmCallbackUnit::GetDataScatterCB()
{
	return mDataScatterCB;
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

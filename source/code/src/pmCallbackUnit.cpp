
#include "pmCallbackUnit.h"

namespace pm
{

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
		throw pmInvalidKeyException();

	mKeyMap[mKey] = this;
}

pmCallbackUnit::~pmCallbackUnit()
{
	mResourceLock.Lock();
	mKeyMap.erase(mKey);
	mResourceLock.Unlock();
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
		throw pmInvalidKeyException();

	pmCallbackUnit* lCallbackUnit = mKeyMap[lStr];

	return lCallbackUnit;
}

};

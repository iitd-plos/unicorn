
#include "pmCallbackUnit.h"

namespace pm
{

pmCallbackUnit::pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB, pmReductionCB pReductionCB, pmDeviceSelectionCB pDeviceSelectionCB,
	pmPreDataTransferCB pPreDataTransferCB, pmPostDataTransferCB pPostDataTransferCB, pmDataDistributionCB pDataDistributionCB)
{
	mPreSubtaskCB = pPreSubtaskCB;
	mSubtaskCB = pSubtaskCB;
	mReductionCB = pReductionCB;
	mDeviceSelectionCB = pDeviceSelectionCB;
	mPreDataTransferCB = pPreDataTransferCB;
	mPostDataTransferCB = pPostDataTransferCB;
	mDataDistributionCB = pDataDistributionCB;
}

pmCallbackUnit::pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB)
{
	mPreSubtaskCB = pPreSubtaskCB;
	mSubtaskCB = pSubtaskCB;
	mReductionCB = PM_CALLBACK_NOP;
	mDeviceSelectionCB = PM_CALLBACK_NOP;
	mPreDataTransferCB = PM_CALLBACK_NOP;
	mPostDataTransferCB = PM_CALLBACK_NOP;
	mDataDistributionCB = PM_CALLBACK_NOP;
}

pmCallbackUnit::pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB, pmReductionCB pReductionCB)
{
	mPreSubtaskCB = pPreSubtaskCB;
	mSubtaskCB = pSubtaskCB;
	mReductionCB = pReductionCB;
	mDeviceSelectionCB = PM_CALLBACK_NOP;
	mPreDataTransferCB = PM_CALLBACK_NOP;
	mPostDataTransferCB = PM_CALLBACK_NOP;
	mDataDistributionCB = PM_CALLBACK_NOP;
}

pmCallbackUnit::pmCallbackUnit(pmSubtaskCB pSubtaskCB, pmDataDistributionCB pDataDistributionCB)
{
	mPreSubtaskCB = PM_CALLBACK_NOP;
	mSubtaskCB = pSubtaskCB;
	mReductionCB = PM_CALLBACK_NOP;
	mDeviceSelectionCB = PM_CALLBACK_NOP;
	mPreDataTransferCB = PM_CALLBACK_NOP;
	mPostDataTransferCB = PM_CALLBACK_NOP;
	mDataDistributionCB = pDataDistributionCB;
}

pmCallbackUnit::~pmCallbackUnit()
{
}

pmCallback pmCallbackUnit::GetPreSubtaskCB()
{
	return mPreSubtaskCB;
}

pmCallback pmCallbackUnit::GetSubtaskCB()
{
	return mSubtaskCB;
}

pmCallback pmCallbackUnit::GetReductionCB()
{
	return mReductionCB;
}

pmCallback pmCallbackUnit::GetDeviceSelectionCB()
{
	return mDeviceSelectionCB;
}

pmCallback pmCallbackUnit::GetPreDataTransferCB()
{
	return mPreDataTransferCB;
}

pmCallback pmCallbackUnit::GetPostDataTransferCB()
{
	return mPostDataTransferCB;
}

pmCallback pmCallbackUnit::GetDataDistributionCB()
{
	return mDataDistributionCB;
}

};

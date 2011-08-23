
#ifndef __PM_CALLBACK_UNIT__
#define __PM_CALLBACK_UNIT__

#include "pmInternalDefinitions.h"
#include "pmCallback.h"

namespace pm
{

/**
* \brief The set defining all callbacks applicable to a task 
*/

class pmCallbackUnit
{
	public:
		pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB, pmReductionCB pReductionCB, pmDeviceSelectionCB pDeviceSelectionCB,
			pmPreDataTransferCB pPreDataTransferCB, pmPostDataTransferCB pPostDataTransferCB, pmDataDistributionCB pDataDistributionCB);

		pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB);
		pmCallbackUnit(pmPreSubtaskCB pPreSubtaskCB, pmSubtaskCB pSubtaskCB, pmReductionCB pReductionCB);
		pmCallbackUnit(pmSubtaskCB pSubtaskCB, pmDataDistributionCB pDataDistributionCB);

		~pmCallbackUnit();

		pmCallback GetPreSubtaskCB();
		pmCallback GetSubtaskCB();
		pmCallback GetReductionCB();
		pmCallback GetDeviceSelectionCB();
		pmCallback GetPreDataTransferCB();
		pmCallback GetPostDataTransferCB();
		pmCallback GetDataDistributionCB();

	private:
		/* Not defining callbacks to specific types so that PM_CALLBACK_NOP can be assigned to the undefined ones */
		pmCallback mPreSubtaskCB;
		pmCallback mSubtaskCB;
		pmCallback mReductionCB;
		pmCallback mDeviceSelectionCB;
		pmCallback mPreDataTransferCB;
		pmCallback mPostDataTransferCB;
		pmCallback mDataDistributionCB;
};

} // end namespace pm

#endif


#ifndef __PM_CALLBACK_UNIT__
#define __PM_CALLBACK_UNIT__

#include "pmCallback.h"
#include "pmResourceLock.h"

#include <string>
#include <map>

namespace pm
{

/**
* \brief The set defining all callbacks applicable to a task 
*/

class pmCallbackUnit : public pmBase
{
	public:
		pmCallbackUnit(char* pKey, pmDataDistributionCB* pDataDistributionCB, pmSubtaskCB* pSubtaskCB, pmDataReductionCB* pDataReductionCB, pmDeviceSelectionCB* pDeviceSelectionCB,
			pmDataRedistributionCB* pDataRedistributionCB, pmPreDataTransferCB* pPreDataTransferCB, pmPostDataTransferCB* pPostDataTransferCB);

		virtual ~pmCallbackUnit();

		pmDataDistributionCB* GetDataDistributionCB();
		pmSubtaskCB* GetSubtaskCB();
		pmDataReductionCB* GetDataReductionCB();
		pmDataRedistributionCB* GetDataRedistributionCB();
		pmDeviceSelectionCB* GetDeviceSelectionCB();
		pmPreDataTransferCB* GetPreDataTransferCB();
		pmPostDataTransferCB* GetPostDataTransferCB();

		const char* GetKey();

		static pmCallbackUnit* FindCallbackUnit(char* pKey);

	private:
		pmDataDistributionCB* mDataDistributionCB;
		pmSubtaskCB* mSubtaskCB;
		pmDataReductionCB* mDataReductionCB;
		pmDataRedistributionCB* mDataRedistributionCB;
		pmDeviceSelectionCB* mDeviceSelectionCB;
		pmPreDataTransferCB* mPreDataTransferCB;
		pmPostDataTransferCB* mPostDataTransferCB;

		std::string mKey;

		static std::map<std::string, pmCallbackUnit*> mKeyMap;
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

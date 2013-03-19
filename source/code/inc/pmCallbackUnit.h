
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
    typedef std::map<std::string, pmCallbackUnit*> keyMapType;

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

		static keyMapType& GetKeyMap();
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
};

} // end namespace pm

#endif

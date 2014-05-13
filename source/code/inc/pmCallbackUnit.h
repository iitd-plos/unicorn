
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
    typedef std::map<std::string, const pmCallbackUnit*> keyMapType;

	public:
		pmCallbackUnit(const char* pKey, finalize_ptr<pmDataDistributionCB>&& pDataDistributionCB, finalize_ptr<pmSubtaskCB>&& pSubtaskCB, finalize_ptr<pmDataReductionCB>&& pDataReductionCB, finalize_ptr<pmDeviceSelectionCB>&& pDeviceSelectionCB, finalize_ptr<pmDataRedistributionCB>&& pDataRedistributionCB, finalize_ptr<pmPreDataTransferCB>&& pPreDataTransferCB, finalize_ptr<pmPostDataTransferCB>&& pPostDataTransferCB);

		~pmCallbackUnit();

		const pmDataDistributionCB* GetDataDistributionCB() const;
		const pmSubtaskCB* GetSubtaskCB() const;
		const pmDataReductionCB* GetDataReductionCB() const;
		const pmDataRedistributionCB* GetDataRedistributionCB() const;
		const pmDeviceSelectionCB* GetDeviceSelectionCB() const;
		const pmPreDataTransferCB* GetPreDataTransferCB() const;
		const pmPostDataTransferCB* GetPostDataTransferCB() const;

		const char* GetKey() const;

		static const pmCallbackUnit* FindCallbackUnit(char* pKey);

	private:
		finalize_ptr<pmDataDistributionCB> mDataDistributionCB;
		finalize_ptr<pmSubtaskCB> mSubtaskCB;
		finalize_ptr<pmDataReductionCB> mDataReductionCB;
		finalize_ptr<pmDataRedistributionCB> mDataRedistributionCB;
		finalize_ptr<pmDeviceSelectionCB> mDeviceSelectionCB;
		finalize_ptr<pmPreDataTransferCB> mPreDataTransferCB;
		finalize_ptr<pmPostDataTransferCB> mPostDataTransferCB;

		const std::string mKey;

		static keyMapType& GetKeyMap();
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
};

} // end namespace pm

#endif

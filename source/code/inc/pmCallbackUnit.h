
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
		pmCallbackUnit(const char* pKey, finalize_ptr<pmDataDistributionCB>&& pDataDistributionCB, finalize_ptr<pmSubtaskCB>&& pSubtaskCB, finalize_ptr<pmDataReductionCB>&& pDataReductionCB, finalize_ptr<pmDeviceSelectionCB>&& pDeviceSelectionCB, finalize_ptr<pmDataRedistributionCB>&& pDataRedistributionCB, finalize_ptr<pmPreDataTransferCB>&& pPreDataTransferCB, finalize_ptr<pmPostDataTransferCB>&& pPostDataTransferCB, finalize_ptr<pmTaskCompletionCB>&& pTaskCompletionCB);

		~pmCallbackUnit();

		const pmDataDistributionCB* GetDataDistributionCB() const;
		const pmSubtaskCB* GetSubtaskCB() const;
		const pmDataReductionCB* GetDataReductionCB() const;
		const pmDataRedistributionCB* GetDataRedistributionCB() const;
		const pmDeviceSelectionCB* GetDeviceSelectionCB() const;
		const pmPreDataTransferCB* GetPreDataTransferCB() const;
		const pmPostDataTransferCB* GetPostDataTransferCB() const;
        const pmTaskCompletionCB* GetTaskCompletionCB() const;
    
        void SetTaskCompletionCB(finalize_ptr<pmTaskCompletionCB>&& pTaskCompletionCB);

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
        finalize_ptr<pmTaskCompletionCB> mTaskCompletionCB;

		const std::string mKey;

		static keyMapType& GetKeyMap();
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
};

} // end namespace pm

#endif

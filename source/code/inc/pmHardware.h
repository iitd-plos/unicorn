
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

#ifndef __PM_HARDWARE__
#define __PM_HARDWARE__

#include "pmBase.h"
#include "pmCommunicator.h"

#include <set>
#include <vector>

namespace pm
{

class pmExecutionStub;
class pmDevicePool;

/**
 * \brief The unique identifier for a machine, cluster/sub-cluster or a processing element in the entire cluster
 */

class pmHardware : public pmBase
{
};

class pmMachine : public pmHardware
{
	friend class pmMachinePool;

	public:
		operator uint() const;

	private:
		pmMachine(uint pMachineId);

		uint mMachineId;
};

extern pmMachine* PM_LOCAL_MACHINE;

class pmProcessingElement : public pmHardware
{
	friend class pmDevicePool;

	public:
		const pmMachine* GetMachine() const;
		uint GetDeviceIndexInMachine() const;
		uint GetGlobalDeviceIndex() const;
		pmDeviceType GetType() const;
        ushort GetNumaDomainId() const;

		pmExecutionStub* GetLocalExecutionStub() const;
		const pmDeviceInfo& GetDeviceInfo() const;

		static void GetMachines(const std::set<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines);
        static void GetMachines(const std::vector<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines);
        static void GetMachinesInOrder(const std::vector<const pmProcessingElement*>& pDevices, std::vector<const pmMachine*>& pMachines);

	private:
		pmProcessingElement(const pmMachine* pMachine, pmDeviceType pDeviceType, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex, ushort pNumaDomainId, const communicator::devicePool* pDevicePool);
    
        void BuildDeviceInfo(const communicator::devicePool* pDevicePool);

        pmDeviceInfo mDeviceInfo;
		const pmMachine* mMachine;
		uint mDeviceIndexInMachine;
		uint mGlobalDeviceIndex;
		pmDeviceType mDeviceType;
        ushort mNumaDomainId;   // Only valid for CPU processing elements
};

} // end namespace pm

#endif

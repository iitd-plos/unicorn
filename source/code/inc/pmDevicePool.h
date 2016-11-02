
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

#ifndef __PM_DEVICE_POOL__
#define __PM_DEVICE_POOL__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"

#include <vector>
#include <set>
#include <string>

namespace pm
{

class pmMachine;
class pmProcessingElement;
class pmCluster;

/**
 * \brief The record of all machines/devices and their last updated status
 * This is a per machine singleton class i.e. exactly one instance of pmMachinePool
 * and pmDevicePool exists per machine.
 * The machine pool records the following parameters for each machine -
 * 1. No. of CPU Cores
 * 2. No. of GPU Cards
 * 3. Data Sent (in bytes)
 * 4. Data Received (in bytes)
 * 5. Data Send Time (Total time spent in secs in sending data mentioned in 3)
 * 6. Data Receive Time (Total time spent in secs in receiving data mentioned in 4)
 * 7. Number of data send events from the machine
 * 8. Number of data receive events to the machine
*/
class pmMachinePool : public pmBase
{
    public:
		typedef struct pmMachineData
		{
			uint cpuCores;
			uint gpuCards;
            uint cpuNumaDomains;    // Top level domains only
			ulong dataSent;
			ulong dataReceived;
			double sendTime;
			double receiveTime;
			ulong sendCount;
			ulong receiveCount;
            
            pmMachineData(uint pCpuCores, uint pGpuCards, uint pCpuNumaDomains)
            : cpuCores(pCpuCores)
            , gpuCards(pGpuCards)
            , cpuNumaDomains(pCpuNumaDomains)
            , dataSent(0)
            , dataReceived(0)
            , sendTime(0)
            , receiveTime(0)
            , sendCount(0)
            , receiveCount(0)
            {}
		} pmMachineData;

		static pmMachinePool* GetMachinePool();

		const pmMachine* GetMachine(uint pIndex) const;
		pmMachineData& GetMachineData(uint pIndex);
		pmMachineData& GetMachineData(const pmMachine* pMachine);

		void GetAllDevicesOnMachine(uint pMachineIndex, std::vector<const pmProcessingElement*>& pDevices) const;
		void GetAllDevicesOnMachine(const pmMachine* pMachine, std::vector<const pmProcessingElement*>& pDevices) const;

		uint GetFirstDeviceIndexOnMachine(uint pMachineIndex) const;
		uint GetFirstDeviceIndexOnMachine(const pmMachine* pMachine) const;

		void RegisterSendCompletion(const pmMachine* pMachine, ulong pDataSent, double pSendTime);
		void RegisterReceiveCompletion(const pmMachine* pMachine, ulong pDataReceived, double pReceiveTime);
    
        uint GetCpuNumaDomainsOnMachine(uint pIndex);

	private:
		pmMachinePool();

		void All2AllMachineData(size_t pMachineCount);

		std::vector<uint> mFirstDeviceIndexOnMachine;
		std::vector<pmMachine> mMachinesVector;
		std::vector<pmMachineData> mMachineDataVector;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmDevicePool : public pmBase
{
    public:
		typedef struct pmDeviceData
		{
			std::string name;
			std::string description;	/* All other parameters in a comma separated string */
            
            pmDeviceData(const std::string& pName, const std::string& pDescription)
            : name(pName)
            , description(pDescription)
            {}
		} pmDeviceData;

		static pmDevicePool* GetDevicePool();

		uint GetDeviceCount() const;

		void CreateMachineDevices(const pmMachine* pMachine, uint pCpuDeviceCount, const communicator::devicePool* pDeviceData, uint pGlobalStartingDeviceIndex, uint pDeviceCount);

		const pmDeviceData& GetDeviceData(const pmProcessingElement* pDevice) const;

		const pmProcessingElement* GetDeviceAtMachineIndex(const pmMachine* pMachine, uint pDeviceIndexOnMachine) const;
		const pmProcessingElement* GetDeviceAtGlobalIndex(uint pGlobalDeviceIndex) const;

        void BroadcastAndCreateDeviceData(const pmMachine* pMachine, uint pDeviceCount, uint pCpuDeviceCount, uint pGlobalStartingDeviceIndex);

        void GetAllDevicesOfTypeInCluster(pmDeviceType pType, const pmCluster* pCluster, std::vector<const pmProcessingElement*>& pDevices) const;
        void GetAllDevicesOfTypeOnMachines(pmDeviceType pType, const std::set<const pmMachine*>& pMachines, std::vector<const pmProcessingElement*>& pDevices) const;
    
        std::vector<const pmProcessingElement*> InterleaveDevicesOfDifferentTypes(const std::vector<const pmProcessingElement*>& pDevices) const;
        std::vector<const pmProcessingElement*> InterleaveDevicesFromDifferentMachines(const std::vector<const pmProcessingElement*>& pDevices, bool pCycleDeviceTypes) const;
		
	private:
		pmDevicePool()
        {}

		std::vector<pmProcessingElement> mDevicesVector;
		std::vector<pmDeviceData> mDeviceDataVector;
};

} // end namespace pm

#endif

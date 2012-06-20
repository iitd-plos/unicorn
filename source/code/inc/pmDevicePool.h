
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

#ifndef __PM_DEVICE_POOL__
#define __PM_DEVICE_POOL__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmScheduler.h"
#include "pmCommand.h"

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
    friend class pmController;

    public:
		typedef struct pmMachineData
		{
			uint cpuCores;
			uint gpuCards;
			ulong  dataSent;
			ulong dataReceived;
			double sendTime;
			double receiveTime;
			ulong sendCount;
			ulong receiveCount;
		} pmMachineData;

		virtual ~pmMachinePool();

		static pmMachinePool* GetMachinePool();

		pmMachine* GetMachine(uint pIndex);
		pmMachineData& GetMachineData(uint pIndex);
		pmMachineData& GetMachineData(pmMachine* pMachine);

		pmStatus GetAllDevicesOnMachine(uint pMachineIndex, std::vector<pmProcessingElement*>& pDevices);
		pmStatus GetAllDevicesOnMachine(pmMachine* pMachine, std::vector<pmProcessingElement*>& pDevices);

		uint GetFirstDeviceIndexOnMachine(uint pMachineIndex);
		uint GetFirstDeviceIndexOnMachine(pmMachine* pMachine);

		std::vector<pmMachine*>& GetAllMachines();

		pmStatus RegisterSendCompletion(pmMachine* pMachine, ulong pDataSent, double pSendTime);
		pmStatus RegisterReceiveCompletion(pmMachine* pMachine, ulong pDataReceived, double pReceiveTime);

	private:
		pmMachinePool();

		pmStatus All2AllMachineData(pmCommunicatorCommand::machinePool* pAll2AllBuffer);

		static pmMachinePool* mMachinePool;
		std::vector<uint> mFirstDeviceIndexOnMachine;
		std::vector<pmMachine*> mMachinesVector;
		std::vector<pmMachineData> mMachineDataVector;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	/* For dynamically updating parameters */
};

class pmDevicePool : public pmBase
{
    friend class pmMachinePool;

    public:
		typedef struct pmDeviceData
		{
			std::string name;
			std::string description;	/* All other parameters in a comma separated string */
		} pmDeviceData;

		virtual ~pmDevicePool();

		static pmDevicePool* GetDevicePool();

		uint GetDeviceCount();

		pmStatus CreateMachineDevices(pmMachine* pMachine, uint pCpuDeviceCount, pmCommunicatorCommand::devicePool* pDeviceData, uint pGlobalStartingDeviceIndex, uint pDeviceCount);

		pmDeviceData& GetDeviceData(pmProcessingElement* pDevice);
		std::vector<pmProcessingElement*>& GetAllDevices();

		pmProcessingElement* GetDeviceAtMachineIndex(pmMachine* pMachine, uint pDeviceIndexOnMachine);
		pmProcessingElement* GetDeviceAtGlobalIndex(uint pGlobalDeviceIndex);

		pmStatus BroadcastDeviceData(pmMachine* pMachine, pmCommunicatorCommand::devicePool* pDeviceArray, uint pDeviceCount);

		pmStatus GetAllDevicesOfTypeInCluster(pmDeviceTypes pType, pmCluster* pCluster, std::set<pmProcessingElement*>& pDevices);
		
	private:
		pmDevicePool();

		static pmDevicePool* mDevicePool;
		std::vector<pmProcessingElement*> mDevicesVector;
		std::vector<pmDeviceData> mDeviceDataVector;
};

} // end namespace pm

#endif

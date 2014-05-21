
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

		static void GetMachines(std::set<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines);
		static void GetMachines(std::vector<const pmProcessingElement*>& pDevices, std::set<const pmMachine*>& pMachines);

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

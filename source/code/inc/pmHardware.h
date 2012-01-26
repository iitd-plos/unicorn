
#ifndef __PM_HARDWARE__
#define __PM_HARDWARE__

#include "pmBase.h"

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
	public:

	protected:
		pmHardware();
		virtual ~pmHardware();

	private:
};

class pmMachine : public pmHardware
{
	friend class pmMachinePool;

	public:
		virtual ~pmMachine();

		virtual operator uint();

	private:
		pmMachine(uint pMachineId);

		uint mMachineId;
};

extern pmMachine* PM_LOCAL_MACHINE;

class pmProcessingElement : public pmHardware
{
	friend class pmDevicePool;

	public:
		virtual ~pmProcessingElement();

		virtual pmMachine* GetMachine();
		virtual uint GetDeviceIndexInMachine();
		virtual uint GetGlobalDeviceIndex();
		virtual pmDeviceTypes GetType();

		virtual pmExecutionStub* GetLocalExecutionStub();
		virtual pmDeviceInfo GetDeviceInfo();

		static pmStatus GetMachines(std::set<pmProcessingElement*>& pDevices, std::set<pmMachine*>& pMachines);
		static pmStatus GetMachines(std::vector<pmProcessingElement*>& pDevices, std::set<pmMachine*>& pMachines);
		static pmStatus GetMachines(std::vector<pmProcessingElement*>& pDevices, std::vector<pmMachine*>& pMachines);

	private:
		pmProcessingElement(pmMachine* pMachine, uint pDeviceIndexInMachine, uint pGlobalDeviceIndex);

		pmMachine* mMachine;
		uint mDeviceIndexInMachine;
		uint mGlobalDeviceIndex;
};

} // end namespace pm

#endif

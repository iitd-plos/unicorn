
#ifndef __PM_DEVICE_POOL__
#define __PM_DEVICE_POOL__

#include "pmInternalDefinitions.h"
#include <vector>

namespace pm
{

/**
 * \brief The record of all devices and their last updated status
 * This is a per machine singleton class i.e. exactly one instance of pmDevicePool exists per machine.
 * Every machine maintains a list of resources in the entire MPI cluster.
 * The device pool is maintained as a vector each of which is a set of following
 * parameters for the machine -
 * 1. No. of CPU Cores
 * 2. No. of GPU Cards
 * 3. Data Transferred (in bytes)
 * 4. Transfer Time (Total time spent in secs in transferring data mentioned in 3)
*/
class pmDevicePool
{
	public:
		static pmDevicePool* GetDevicePool();
		pmStatus DestroyDevicePool();

	private:
		pmDevicePool() {mMachineCount = 0;}

		typedef struct pmMachineData
		{
			uint cpuCores;
			uint gpuCards;
			ulong  dataTransferred;
			ulong transferTime;
		} pmMachineData;

		pmStatus SetMachineCountAndDevicePool(uint pMachineCount);
		pmStatus SetMachineData(uint pMachineId, pmMachineData& pMachineData);

		uint mMachineCount;

		static pmDevicePool* mDevicePool;
		std::vector<pmMachineData> mPool;
};

} // end namespace pm

#endif

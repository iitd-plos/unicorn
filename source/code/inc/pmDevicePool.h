
#ifndef __PM_DEVICE_POOL__
#define __PM_DEVICE_POOL__

#include "pmInternalDefinitions.h"
#include <vector>

namespace pm
{

class RESOURCE_LOCK_IMPLEMENTATION_CLASS;

/**
 * \brief The record of all devices and their last updated status
 * This is a per machine singleton class i.e. exactly one instance of pmDevicePool exists per machine.
 * Every machine maintains a list of resources in the entire MPI cluster.
 * The device pool is maintained as a vector each of which is a set of following
 * parameters for the machine -
 * 1. No. of CPU Cores
 * 2. No. of GPU Cards
 * 3. Data Sent (in bytes)
 * 4. Data Received (in bytes)
 * 5. Data Send Time (Total time spent in secs in sending data mentioned in 3)
 * 6. Data Receive Time (Total time spent in secs in receiving data mentioned in 4)
 * 7. Number of data send events from the machine
 * 8. Number of data receive events to the machine
*/
class pmDevicePool
{
	public:
		typedef struct pmMachineData
		{
			/* All entries are initialized in InitializeMachineData */
			uint cpuCores;
			uint gpuCards;
			ulong  dataSent;
			ulong dataReceived;
			double sendTime;
			double receiveTime;
			ulong sendCount;
			ulong receiveCount;
		} pmMachineData;

		static pmDevicePool* GetDevicePool();
		pmStatus DestroyDevicePool();

		pmStatus InitializeDevicePool(uint pMachineCount);
		pmStatus SetMachineData(uint pMachineId, pmMachineData& pMachineData);
		
		pmStatus RegisterSendCompletion(uint pMachineId, ulong pDataSent, double pSendTime);
		pmStatus RegisterReceiveCompletion(uint pMachineId, ulong pDataReceived, double pReceiveTime);

	private:
		pmDevicePool() {mMachineCount = 0;}

		pmStatus InitializeMachineData(uint pMachineId);

		uint mMachineCount;

		static pmDevicePool* mDevicePool;
		std::vector<pmMachineData> mPool;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif


#ifndef __PM_REDISTRIBUTOR__
#define __PM_REDISTRIBUTOR__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"

#include <map>
#include <vector>

namespace pm
{

class pmTask;
class pmMachine;

class pmRedistributor : public pmBase
{
	public:
		pmRedistributor(pmTask* pTask);
		virtual ~pmRedistributor();

        pmStatus RedistributeData(ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder);
        pmStatus PerformRedistribution(pmMachine* pHost, ulong pBaseMemAddr, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector);
    
        pmStatus SendRedistributionInfo();
	
	private:
        typedef struct orderData
        {
            pmMachine* host;
            ulong hostMemBaseAddr;
            ulong offset;
            ulong length;
        } orderData;
    
        void SetRedistributedOwnership();
    
		pmTask* mTask;
        ulong mTotalLengthAccounted;
        ulong mSubtasksAccounted;
    
        std::map<uint, std::vector<orderData> > mGlobalRedistributionMap;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mGlobalRedistributionLock;

        std::vector<pmCommunicatorCommand::redistributionOrderStruct> mLocalRedistributionData;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mLocalRedistributionLock;
};

} // end namespace pm

#endif

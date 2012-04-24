
#ifndef __PM_REDISTRIBUTOR__
#define __PM_REDISTRIBUTOR__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <map>
#include <vector>

namespace pm
{

class pmTask;
class pmMachine;

class pmRedistributor : public pmBase
{
	public:
        typedef struct orderMetaData
        {
            ulong orderLength;
        } orderMetaData;
    
        typedef struct orderData
        {
            ulong offset;
            ulong length;
        } orderData;
    
		pmRedistributor(pmTask* pTask);
		virtual ~pmRedistributor();

        pmStatus RedistributeData(ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder);
    
        pmStatus PerformRedistribution();
	
	private:
		pmTask* mTask;

        ulong mTotalLength;
        std::map<ulong, std::pair<orderMetaData, std::vector<orderData> > > mRedistributionMap;     // Order number vs. redistribution data
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mRedistributionLock;

        ulong mSubtasksRedistributed;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mCountLock;

        std::vector<uint> mTransferMetaData;
};

} // end namespace pm

#endif

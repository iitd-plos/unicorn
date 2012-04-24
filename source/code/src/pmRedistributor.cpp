
#include "pmRedistributor.h"
#include "pmHardware.h"
#include "pmTask.h"

namespace pm
{

pmRedistributor::pmRedistributor(pmTask* pTask)
{
	mTask = pTask;
    mSubtasksRedistributed = 0;
    mTotalLength = 0;
}

pmRedistributor::~pmRedistributor()
{
}

pmStatus pmRedistributor::RedistributeData(ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder)
{
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mRedistributionLock, Lock(), Unlock());

    mTotalLength += pLength;
        
    orderData lOrderData;
    lOrderData.offset = pOffset;
    lOrderData.length = pLength;
    
    if(mRedistributionMap.find(pOrder) != mRedistributionMap.end())
    {
        mRedistributionMap[pOrder].first.orderLength += pLength;
        mRedistributionMap[pOrder].second.push_back(lOrderData);
    }
    else
    {
        orderMetaData lMetaData;
        lMetaData.orderLength = pLength;
        
        std::vector<orderData> lVector;
        lVector.push_back(lOrderData);

        mRedistributionMap[pOrder] = std::pair<orderMetaData, std::vector<orderData> >(lMetaData, lVector);
    }
        
    return pmSuccess;
}
    
pmStatus pmRedistributor::PerformRedistribution()
{
    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
    {
        uint lOrderCount = mRedistributionMap.size();
        
        std::map<ulong, std::pair<orderMetaData, std::vector<orderData> > >::iterator lStart, lEnd;
        lStart = mRedistributionMap.begin();
        lEnd = mRedistributionMap.end();
        
        for(; lStart != lEnd; ++lStart)
        {
            mTransferMetaData.push_back(lStart->first);
            mTransferMetaData.push_back(lStart->second.first.orderLength);
        }
        
        pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, lOrderCount, (void*)((uint*)(&mTransferMetaData[0])), mTransferMetaData.size()*sizeof(uint));
    }
    
    return pmSuccess;
}

}


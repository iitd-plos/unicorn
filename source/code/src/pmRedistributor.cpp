
#include "pmRedistributor.h"
#include "pmHardware.h"
#include "pmTask.h"
#include "pmMemSection.h"

namespace pm
{

pmRedistributor::pmRedistributor(pmTask* pTask)
{
	mTask = pTask;
    mTotalLengthAccounted = 0;
    mSubtasksAccounted = 0;
}

pmRedistributor::~pmRedistributor()
{
}

pmStatus pmRedistributor::RedistributeData(ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder)
{
    if(!pLength)
        return pmSuccess;
    
    pmSubscriptionInfo lOutputMemSubscriptionInfo;
    if(!mTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pSubtaskId, lOutputMemSubscriptionInfo))
        return pmInvalidOffset;

    ulong lGlobalOffset = lOutputMemSubscriptionInfo.offset + pOffset;
    if(lGlobalOffset >= mTask->GetMemSectionRW()->GetLength())
        return pmInvalidOffset;
        
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    pmCommunicatorCommand::redistributionOrderStruct lOrderData;
    lOrderData.order = pOrder;
    lOrderData.offset = lGlobalOffset;
    lOrderData.length = pLength;

    mLocalRedistributionData.push_back(lOrderData);
        
    return pmSuccess;
}
    
pmStatus pmRedistributor::SendRedistributionInfo()
{
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
    {
        pmOutputMemSection* lMemSection = static_cast<pmOutputMemSection*>(mTask->GetMemSectionRW());
        lMemSection->SetupPostRedistributionMemSection(false);
    }

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, &mLocalRedistributionData, (uint)mLocalRedistributionData.size());
    
    return pmSuccess;
}

pmStatus pmRedistributor::PerformRedistribution(pmMachine* pHost, ulong pBaseMemAddr, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector)
{
    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());
    
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mGlobalRedistributionLock, Lock(), Unlock());
    
    std::vector<pmCommunicatorCommand::redistributionOrderStruct>::const_iterator lStart = pVector.begin(), lEnd = pVector.end();
    for(; lStart != lEnd; ++lStart)
    {
        const pmCommunicatorCommand::redistributionOrderStruct& lData = *lStart;
        
        orderData lOrderData;
        lOrderData.host = pHost;
        lOrderData.hostMemBaseAddr = pBaseMemAddr;
        lOrderData.offset = lData.offset;
        lOrderData.length = lData.length;
        
        mGlobalRedistributionMap[lData.order].push_back(lOrderData);
    }
        
    mSubtasksAccounted += pSubtasksAccounted;

    if(mSubtasksAccounted == mTask->GetSubtaskCount())
        SetRedistributedOwnership();
    
    return pmSuccess;
}

// This method must be called with mGlobalRedistributionLock acquired
void pmRedistributor::SetRedistributedOwnership()
{
    pmOutputMemSection* lMemSection = static_cast<pmOutputMemSection*>(mTask->GetMemSectionRW());
    lMemSection->SetupPostRedistributionMemSection(true);

    pmOutputMemSection* lTempMemSection = lMemSection->GetPostRedistributionMemSection();
    char* lTempMemAddr = reinterpret_cast<char*>(lTempMemSection->GetMem());
    
    ulong lCurrentOffset = 0;
    std::map<uint, std::vector<orderData> >::iterator lStartIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
    for(; lStartIter != lEndIter; ++lStartIter)
    {
        std::vector<orderData>& lVector = lStartIter->second;
        
        std::vector<orderData>::iterator lInnerStartIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerStartIter != lInnerEndIter; ++lInnerStartIter)
        {
            orderData& lData = *lInnerStartIter;
            if(lData.host == PM_LOCAL_MACHINE)
            {
                lMemSection->AcquireOwnershipImmediate(lCurrentOffset, lData.length);
                lMemSection->Update(lCurrentOffset, lData.length, lTempMemAddr + lData.offset);
            }
            else
            {
                lMemSection->TransferOwnershipPostTaskCompletion(lData.host, lData.hostMemBaseAddr, lData.offset, lCurrentOffset, lData.length);
            }
            
            lCurrentOffset += lData.length;
        }
    }

    lMemSection->FlushOwnerships();

    dynamic_cast<pmLocalTask*>(mTask)->CompleteTask();
}
    
}






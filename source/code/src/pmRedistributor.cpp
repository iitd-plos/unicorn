
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

pmStatus pmRedistributor::RedistributeData(pmExecutionStub* pStub, ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder)
{
    if(!pLength)
        return pmSuccess;
    
#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDISTRIBUTION, true);
#endif

    pmSubscriptionInfo lOutputMemSubscriptionInfo;
    if(!mTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pStub, pSubtaskId, lOutputMemSubscriptionInfo))
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
        
#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDISTRIBUTION, false);
#endif
    
    return pmSuccess;
}
    
pmStatus pmRedistributor::SendRedistributionInfo()
{
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, &mLocalRedistributionData, (uint)mLocalRedistributionData.size());
    
    return pmSuccess;
}

pmStatus pmRedistributor::PerformRedistribution(pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector)
{    
    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());
    
#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDISTRIBUTION, true);
#endif

	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mGlobalRedistributionLock, Lock(), Unlock());
    
    std::vector<pmCommunicatorCommand::redistributionOrderStruct>::const_iterator lStart = pVector.begin(), lEnd = pVector.end();
    for(; lStart != lEnd; ++lStart)
    {
        const pmCommunicatorCommand::redistributionOrderStruct& lData = *lStart;
        
        orderData lOrderData;
        lOrderData.host = pHost;
        lOrderData.offset = lData.offset;
        lOrderData.length = lData.length;
        
        mGlobalRedistributionMap[lData.order].push_back(lOrderData);
    }
        
    mSubtasksAccounted += pSubtasksAccounted;

    if(mSubtasksAccounted == mTask->GetSubtaskCount())
        SetRedistributedOwnership();
    
#ifdef ENABLE_TASK_PROFILING
    mTask->GetTaskProfiler()->RecordProfileEvent(pmTaskProfiler::DATA_REDISTRIBUTION, false);
#endif

    return pmSuccess;
}

// This method must be called with mGlobalRedistributionLock acquired
void pmRedistributor::SetRedistributedOwnership()
{
    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    char* lMemAddr = reinterpret_cast<char*>(lMemSection->GetMem());

    pmMemSection::vmRangeOwner lRangeOwner;
    lRangeOwner.memIdentifier.memOwnerHost = *(lMemSection->GetMemOwnerHost());
    lRangeOwner.memIdentifier.generationNumber = lMemSection->GetGenerationNumber();

    pmMemSection* lRedistributedMemSection = pmMemSection::CreateMemSection(lMemSection->GetLength(), PM_LOCAL_MACHINE, lMemSection->GetMemInfo());
    lRedistributedMemSection->Lock(mTask);

    ulong lCurrentOffset = 0;
    std::map<uint, std::vector<orderData> >::iterator lIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
    for(; lIter != lEndIter; ++lIter)
    {
        std::vector<orderData>& lVector = lIter->second;
        
        std::vector<orderData>::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            orderData& lData = *lInnerIter;
            if(lData.host == PM_LOCAL_MACHINE)
            {
                lRedistributedMemSection->Update(lCurrentOffset, lData.length, lMemAddr + lData.offset);
            }
            else
            {
                lRangeOwner.host = lData.host;
                lRangeOwner.hostOffset = lData.offset;
                lRedistributedMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, lCurrentOffset, lData.length);
            }
            
            lCurrentOffset += lData.length;
        }
    }

    lMemSection->GetUserMemHandle()->Reset(lRedistributedMemSection);

    lMemSection->Unlock(mTask);
    lMemSection->UserDelete();
    dynamic_cast<pmLocalTask*>(mTask)->TaskRedistributionDone(lRedistributedMemSection);
}
    
}







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
#include "pmStubManager.h"
#include "pmExecutionStub.h"

namespace pm
{

pmRedistributor::pmRedistributor(pmTask* pTask)
	: mTask(pTask)
    , mTotalLengthAccounted(0)
    , mSubtasksAccounted(0)
    , mRedistributedMemSection(NULL)
    , mGlobalRedistributionLock __LOCK_NAME__("pmRedistributor::mGlobalRedistributionLock")
    , mLocalRedistributionLock __LOCK_NAME__("pmRedistributor::mLocalRedistributionLock")
    , mPendingBucketsCount(0)
    , mPendingBucketsCountLock __LOCK_NAME__("pmRedistributor::mPendingBucketCountLock")
{
}

pmRedistributor::~pmRedistributor()
{
}

pmStatus pmRedistributor::RedistributeData(pmExecutionStub* pStub, ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder)
{
    if(!pLength)
        return pmSuccess;
    
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    pmSubscriptionInfo lOutputMemSubscriptionInfo;
    if(!mTask->GetSubscriptionManager().GetOutputMemSubscriptionForSubtask(pStub, pSubtaskId, false, lOutputMemSubscriptionInfo))
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

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, &mLocalRedistributionData, (uint)mLocalRedistributionData.size());

    return pmSuccess;
}

pmStatus pmRedistributor::PerformRedistribution(pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector)
{
    if(pSubtasksAccounted == 0)
        return pmSuccess;
    
    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());
    
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
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

        globalRedistributionMapType::mapped_type& lPair = mGlobalRedistributionMap[lData.order];
        lPair.first.totalLength += lData.length;
        
        lPair.second.push_back(lOrderData);
    }

    mSubtasksAccounted += pSubtasksAccounted;

    if(mSubtasksAccounted == mTask->GetSubtaskCount())
    {
        ComputeRedistributionBuckets();
        DoParallelRedistribution();
    }

    return pmSuccess;
}

void pmRedistributor::ComputeRedistributionBuckets()
{
    size_t lDevices = pmStubManager::GetStubManager()->GetProcessingElementsCPU();
    size_t lOrders = mGlobalRedistributionMap.size();
    
    size_t lBuckets = ((lOrders > lDevices) ? lDevices : lOrders);
    mRedistributionBucketsVector.resize(lBuckets);
    
    size_t lOrdersPerBucket = ((lOrders + lBuckets - 1) / lBuckets);    
    size_t lRunningOffset = 0;
    
    globalRedistributionMapType::iterator lIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
    for(size_t i = 0; i < lBuckets; ++i)
    {
        mRedistributionBucketsVector[i].bucketOffset = lRunningOffset;
        mRedistributionBucketsVector[i].startIter = lIter;
        
        size_t j = 0;
        for(; j < lOrdersPerBucket && (lIter != lEndIter); ++lIter, ++j)
            lRunningOffset += lIter->second.first.totalLength;            
            
        mRedistributionBucketsVector[i].endIter = lIter;
    }
}

void pmRedistributor::DoParallelRedistribution()
{
    DoPreParallelRedistribution();
    
    mPendingBucketsCount = mRedistributionBucketsVector.size();
    
    pmStubManager* lStubManager = pmStubManager::GetStubManager();
    for(size_t i = 0; i < mPendingBucketsCount; ++i)
        lStubManager->GetStub((uint)i)->ProcessRedistributionBucket(mTask, i);
}
    
void pmRedistributor::DoPreParallelRedistribution()
{
    pmMemSection* lMemSection = mTask->GetMemSectionRW();

    mRedistributedMemSection = pmMemSection::CreateMemSection(lMemSection->GetLength(), PM_LOCAL_MACHINE);
    mRedistributedMemSection->Lock(mTask, lMemSection->GetMemInfo());
}

void pmRedistributor::ProcessRedistributionBucket(size_t pBucketIndex)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    pmMemSection* lMemSection = mTask->GetMemSectionRW();
    char* lMemAddr = reinterpret_cast<char*>(lMemSection->GetMem());

    pmMemSection::vmRangeOwner lRangeOwner;
    lRangeOwner.memIdentifier.memOwnerHost = *(lMemSection->GetMemOwnerHost());
    lRangeOwner.memIdentifier.generationNumber = lMemSection->GetGenerationNumber();

    redistributionBucket& lBucket = mRedistributionBucketsVector[pBucketIndex];
    ulong lCurrentOffset = lBucket.bucketOffset;

    globalRedistributionMapType::iterator lIter = lBucket.startIter;
    for(; lIter != lBucket.endIter; ++lIter)
    {
        std::vector<orderData>& lVector = lIter->second.second;
        
        std::vector<orderData>::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            orderData& lData = *lInnerIter;
            if(lData.host == PM_LOCAL_MACHINE)
            {
               mRedistributedMemSection->Update(lCurrentOffset, lData.length, lMemAddr + lData.offset);
            }
            else
            {
                lRangeOwner.host = lData.host;
                lRangeOwner.hostOffset = lData.offset;
                mRedistributedMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, lCurrentOffset, lData.length);
            }
        
            lCurrentOffset += lData.length;
        }
    }
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dPendingBucketsCountLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mPendingBucketsCountLock, Lock(), Unlock());
        
        --mPendingBucketsCount;
        if(mPendingBucketsCount == 0)
            DoPostParallelRedistribution();
    }
}

void pmRedistributor::DoPostParallelRedistribution()
{
    pmMemSection* lMemSection = mTask->GetMemSectionRW();

    lMemSection->GetUserMemHandle()->Reset(mRedistributedMemSection);
    lMemSection->Unlock(mTask);
    lMemSection->UserDelete();
    dynamic_cast<pmLocalTask*>(mTask)->TaskRedistributionDone(mRedistributedMemSection);
}
    
}






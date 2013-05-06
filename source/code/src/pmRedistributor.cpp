
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
#include "pmDevicePool.h"

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
    , mOrdersPerBucket(0)
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

    size_t lGlobalOffset = lOutputMemSubscriptionInfo.offset + pOffset;
    if(lGlobalOffset >= mTask->GetMemSectionRW()->GetLength())
        return pmInvalidOffset;
        
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    pmCommunicatorCommand::redistributionOrderStruct lOrderData;
    lOrderData.order = pOrder;
    lOrderData.length = pLength;

    mLocalRedistributionVector.push_back(lOrderData);
    mLocalRedistributionOffsets.push_back(lGlobalOffset);
    
    mLocalRedistributionMap[pOrder].push_back(mLocalRedistributionVector.size() - 1);

    return pmSuccess;
}
    
void pmRedistributor::SendRedistributionInfo()
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, &mLocalRedistributionVector, (uint)mLocalRedistributionVector.size());
std::cout << "SENDING" << std::endl;
    ComputeRedistributionBuckets();
}

pmStatus pmRedistributor::PerformRedistribution(pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector)
{
    if(pSubtasksAccounted == 0)
        return pmSuccess;
    
    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
        PMTHROW(pmFatalErrorException());
    
    uint lHostId = (uint)(*pHost);

#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mGlobalRedistributionLock, Lock(), Unlock());
    
    std::vector<pmCommunicatorCommand::redistributionOrderStruct>::const_iterator lIter = pVector.begin(), lEnd = pVector.end();
    for(; lIter != lEnd; ++lIter)
    {
        const pmCommunicatorCommand::redistributionOrderStruct& lData = *lIter;
        std::pair<uint, uint> lPair(lData.order, lHostId);
  
        globalRedistributionMapType::iterator lIter = mGlobalRedistributionMap.find(lPair);
        if(lIter == mGlobalRedistributionMap.end())
            mGlobalRedistributionMap[lPair] = lData.length;
        else
            lIter->second += lData.length;
    }

    mSubtasksAccounted += pSubtasksAccounted;
std::cout << "PERF" << std::endl;
    if(mSubtasksAccounted == mTask->GetSubtaskCount())
    {
        CreateRedistributedMemSection();

        ComputeGlobalOffsets();
        SendGlobalOffsets();
    }

    return pmSuccess;
}

void pmRedistributor::ComputeGlobalOffsets()
{
    size_t lRunningOffset = 0;
    uint lHostId = (uint)(*PM_LOCAL_MACHINE);
    
    pmMemSection::vmRangeOwner lRangeOwner;
    lRangeOwner.memIdentifier.memOwnerHost = *(mRedistributedMemSection->GetMemOwnerHost());
    lRangeOwner.memIdentifier.generationNumber = mRedistributedMemSection->GetGenerationNumber();

    globalRedistributionMapType::iterator lIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.second != lHostId)
        {
            lRangeOwner.host = pmMachinePool::GetMachinePool()->GetMachine(lIter->first.second);
            lRangeOwner.hostOffset = lRunningOffset;

            mRedistributedMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, lRunningOffset, lIter->second);
        }
        
        mGlobalOffsetsMap[lIter->first.second].push_back((ulong)lRunningOffset);

        lRunningOffset += lIter->second;
    }
}
    
void pmRedistributor::SendGlobalOffsets()
{
    std::map<uint, std::vector<ulong> >::iterator lIter = mGlobalOffsetsMap.begin(), lEndIter = mGlobalOffsetsMap.end();
    
    for(; lIter != lEndIter; ++lIter)
        pmScheduler::GetScheduler()->RedistributionOffsetsEvent(mTask, mRedistributedMemSection, lIter->first, &(lIter->second), (uint)(lIter->second.size()));
}
    
void pmRedistributor::ReceiveGlobalOffsets(const std::vector<ulong>& pGlobalOffsetsVector, ulong pGenerationNumber)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif
std::cout << "RGO" << std::endl;
    mGlobalOffsetsVector = pGlobalOffsetsVector;

    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
       CreateRedistributedMemSection(pGenerationNumber);
    
    DoParallelRedistribution();
}
    
void pmRedistributor::ComputeRedistributionBuckets()
{
    size_t lDevices = pmStubManager::GetStubManager()->GetProcessingElementsCPU();
    size_t lOrders = mLocalRedistributionMap.size();
    size_t lBuckets = ((lOrders > lDevices) ? lDevices : lOrders);
    mOrdersPerBucket = ((lOrders + lBuckets - 1) / lBuckets);
    
    mLocalRedistributionBucketsVector.resize(lBuckets);
    
    std::map<uint, std::vector<size_t> >::iterator lIter = mLocalRedistributionMap.begin(), lEndIter = mLocalRedistributionMap.end();
    for(size_t i = 0; i < lBuckets; ++i)
    {
        mLocalRedistributionBucketsVector[i].startIter = lIter;
        
        for(size_t j = 0; j < mOrdersPerBucket && (lIter != lEndIter); ++lIter, ++j);

        mLocalRedistributionBucketsVector[i].endIter = lIter;
    }
}
    
void pmRedistributor::DoParallelRedistribution()
{
    mPendingBucketsCount = mLocalRedistributionBucketsVector.size();

    pmStubManager* lStubManager = pmStubManager::GetStubManager();
    for(size_t i = 0; i < mPendingBucketsCount; ++i)
        lStubManager->GetStub((uint)i)->ProcessRedistributionBucket(mTask, i);
}

void pmRedistributor::CreateRedistributedMemSection(ulong pGenerationNumber /* = ((ulong)-1) */)
{
    pmMemSection* lMemSection = mTask->GetMemSectionRW();

    if(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
        mRedistributedMemSection = pmMemSection::CreateMemSection(lMemSection->GetLength(), PM_LOCAL_MACHINE);
    else
        mRedistributedMemSection = pmMemSection::CreateMemSection(lMemSection->GetLength(), lMemSection->GetMemOwnerHost(), pGenerationNumber);
    
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
    lRangeOwner.host = PM_LOCAL_MACHINE;
    lRangeOwner.memIdentifier.memOwnerHost = *(mRedistributedMemSection->GetMemOwnerHost());
    lRangeOwner.memIdentifier.generationNumber = mRedistributedMemSection->GetGenerationNumber();

    localRedistributionBucket& lBucket = mLocalRedistributionBucketsVector[pBucketIndex];

    size_t lGlobalOffsetsIndex = pBucketIndex * mOrdersPerBucket;

    localRedistributionMapType::iterator lIter = lBucket.startIter;
    for(; lIter != lBucket.endIter; ++lIter)
    {
        size_t lCurrentOffset = mGlobalOffsetsVector[lGlobalOffsetsIndex++];
        
        std::vector<size_t>& lVector = lIter->second;
        
        std::vector<size_t>::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            pmCommunicatorCommand::redistributionOrderStruct& lData = mLocalRedistributionVector[*lInnerIter];
            
        #ifdef _DEBUG
            if(lData.order != lIter->first)
                PMTHROW(pmFatalErrorException());
        #endif
            
            mRedistributedMemSection->Update(lCurrentOffset, lData.length, lMemAddr + mLocalRedistributionOffsets[*lInnerIter]);

            if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
            {
                lRangeOwner.hostOffset = lCurrentOffset;
                mRedistributedMemSection->TransferOwnershipPostTaskCompletion(lRangeOwner, lCurrentOffset, lData.length);
            }

            lCurrentOffset += lData.length;
        }
    }
    
    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dPendingBucketsCountLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mPendingBucketsCountLock, Lock(), Unlock());
        
        --mPendingBucketsCount;
std::cout << "PROCESS " << mPendingBucketsCount << std::endl;
        if(mPendingBucketsCount == 0)
            DoPostParallelRedistribution();
    }
}

void pmRedistributor::DoPostParallelRedistribution()
{
    pmMemSection* lMemSection = mTask->GetMemSectionRW();

    if(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
    {
std::cout << "DON" << std::endl;
        lMemSection->GetUserMemHandle()->Reset(mRedistributedMemSection);

        lMemSection->Unlock(mTask);
        lMemSection->UserDelete();
        dynamic_cast<pmLocalTask*>(mTask)->TaskRedistributionDone(mRedistributedMemSection);
    }
    else
    {
        lMemSection->Unlock(mTask);
        lMemSection->UserDelete();
        dynamic_cast<pmRemoteTask*>(mTask)->MarkRedistributionFinished(mRedistributedMemSection);
    }
}
    
}






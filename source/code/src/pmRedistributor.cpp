
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
#include "pmAddressSpace.h"
#include "pmStubManager.h"
#include "pmExecutionStub.h"
#include "pmDevicePool.h"

namespace pm
{

using namespace redistribution;
    
pmRedistributor::pmRedistributor(pmTask* pTask, uint pAddressSpaceIndex)
	: mTask(pTask)
    , mAddressSpaceIndex(pAddressSpaceIndex)
    , mSubtasksAccounted(0)
    , mRedistributedAddressSpace(NULL)
    , mGlobalRedistributionLock __LOCK_NAME__("pmRedistributor::mGlobalRedistributionLock")
    , mPendingBucketsCount(0)
    , mPendingBucketsCountLock __LOCK_NAME__("pmRedistributor::mPendingBucketCountLock")
    , mOrdersPerBucket(0)
{
}
    
uint pmRedistributor::GetAddressSpaceIndex() const
{
    return mAddressSpaceIndex;
}

void pmRedistributor::BuildRedistributionData()
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    mTask->GetSubscriptionManager().ConsolidateRedistributionRecords(*this, mLocalRedistributionData);
}
    
void pmRedistributor::SendRedistributionInfo()
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    BuildRedistributionData();

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, mAddressSpaceIndex, &mLocalRedistributionData.mLocalRedistributionVector);

    ComputeRedistributionBuckets();
}

void pmRedistributor::PerformRedistribution(const pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<communicator::redistributionOrderStruct>& pVector)
{
    if(pSubtasksAccounted == 0)
        return;
    
    DEBUG_EXCEPTION_ASSERT(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE);
    
    uint lHostId = (uint)(*pHost);

#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mGlobalRedistributionLock, Lock(), Unlock());
    
    std::vector<communicator::redistributionOrderStruct>::const_iterator lIter = pVector.begin(), lEnd = pVector.end();
    for(; lIter != lEnd; ++lIter)
    {
        const communicator::redistributionOrderStruct& lData = *lIter;
        std::pair<uint, uint> lPair(lData.order, lHostId);
  
        globalRedistributionMapType::iterator lInnerIter = mGlobalRedistributionMap.find(lPair);
        if(lInnerIter == mGlobalRedistributionMap.end())
            mGlobalRedistributionMap[lPair] = lData.length;
        else
            lInnerIter->second += lData.length;
    }

    mSubtasksAccounted += pSubtasksAccounted;

    if(mSubtasksAccounted == mTask->GetSubtaskCount())
    {
        CreateRedistributedAddressSpace();

        ComputeGlobalOffsets();
        SendGlobalOffsets();
    }
}

void pmRedistributor::ComputeGlobalOffsets()
{
    size_t lRunningOffset = 0;
    uint lHostId = (uint)(*PM_LOCAL_MACHINE);
    
    pmAddressSpace::vmRangeOwner lRangeOwner(NULL, 0, communicator::memoryIdentifierStruct(*mRedistributedAddressSpace->GetMemOwnerHost(), mRedistributedAddressSpace->GetGenerationNumber()));

    globalRedistributionMapType::iterator lIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.second != lHostId)
        {
            lRangeOwner.host = pmMachinePool::GetMachinePool()->GetMachine(lIter->first.second);
            lRangeOwner.hostOffset = lRunningOffset;

            mRedistributedAddressSpace->TransferOwnershipPostTaskCompletion(lRangeOwner, lRunningOffset, lIter->second);
        }
        
        mGlobalOffsetsMap[lIter->first.second].push_back((ulong)lRunningOffset);

        lRunningOffset += lIter->second;
    }
}

void pmRedistributor::SendGlobalOffsets()
{
    std::map<uint, std::vector<ulong> >::iterator lIter = mGlobalOffsetsMap.begin(), lEndIter = mGlobalOffsetsMap.end();
    
    for(; lIter != lEndIter; ++lIter)
        pmScheduler::GetScheduler()->RedistributionOffsetsEvent(mTask, mAddressSpaceIndex, mRedistributedAddressSpace, lIter->first, &(lIter->second));
}
    
void pmRedistributor::ReceiveGlobalOffsets(const std::vector<ulong>& pGlobalOffsetsVector, ulong pGenerationNumber)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    mGlobalOffsetsVector = pGlobalOffsetsVector;

    if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
       CreateRedistributedAddressSpace(pGenerationNumber);
    
    DoParallelRedistribution();
}
    
void pmRedistributor::ComputeRedistributionBuckets()
{
    size_t lDevices = pmStubManager::GetStubManager()->GetProcessingElementsCPU();
    size_t lOrders = mLocalRedistributionData.mLocalRedistributionMap.size();
    
    if(!lOrders)
        return;

    size_t lBuckets = ((lOrders > lDevices) ? lDevices : lOrders);
    mOrdersPerBucket = ((lOrders + lBuckets - 1) / lBuckets);
    
    mLocalRedistributionBucketsVector.resize(lBuckets);
    
    auto lIter = mLocalRedistributionData.mLocalRedistributionMap.begin(), lEndIter = mLocalRedistributionData.mLocalRedistributionMap.end();
    for(size_t i = 0; i < lBuckets - 1; ++i)
    {
        mLocalRedistributionBucketsVector[i].startIter = lIter;
        
        std::advance(lIter, mOrdersPerBucket);
        
        mLocalRedistributionBucketsVector[i].endIter = lIter;
    }

    mLocalRedistributionBucketsVector[lBuckets - 1].startIter = lIter;
    mLocalRedistributionBucketsVector[lBuckets - 1].endIter = lEndIter;
}
    
void pmRedistributor::DoParallelRedistribution()
{
    mPendingBucketsCount = mLocalRedistributionBucketsVector.size();

    if(mPendingBucketsCount)
    {
        pmStubManager* lStubManager = pmStubManager::GetStubManager();
        for(size_t i = 0; i < mPendingBucketsCount; ++i)
            lStubManager->GetStub((uint)i)->ProcessRedistributionBucket(mTask, mAddressSpaceIndex, i);
    }
    else
    {
        DoPostParallelRedistribution();
    }
}

void pmRedistributor::CreateRedistributedAddressSpace(ulong pGenerationNumber /* = std::numeric_limits<ulong>::max() */)
{
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(mAddressSpaceIndex);

    if(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
        mRedistributedAddressSpace = pmAddressSpace::CreateAddressSpace(lAddressSpace->GetLength(), PM_LOCAL_MACHINE);
    else
        mRedistributedAddressSpace = pmAddressSpace::CreateAddressSpace(lAddressSpace->GetLength(), lAddressSpace->GetMemOwnerHost(), pGenerationNumber);

    pmCommandPtr lCountDownCommand = pmCountDownCommand::CreateSharedPtr(1, mTask->GetPriority(), 0, NULL);
    lCountDownCommand->MarkExecutionStart();

    mRedistributedAddressSpace->EnqueueForLock(mTask, mTask->GetMemType(lAddressSpace), lCountDownCommand);
    
    lCountDownCommand->WaitForFinish();
}
    
void pmRedistributor::ProcessRedistributionBucket(size_t pBucketIndex)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(mAddressSpaceIndex);
    char* lMemAddr = reinterpret_cast<char*>(lAddressSpace->GetMem());

    pmAddressSpace::vmRangeOwner lRangeOwner(PM_LOCAL_MACHINE, 0, communicator::memoryIdentifierStruct(*mRedistributedAddressSpace->GetMemOwnerHost(), mRedistributedAddressSpace->GetGenerationNumber()));

    localRedistributionBucket& lBucket = mLocalRedistributionBucketsVector[pBucketIndex];

    size_t lGlobalOffsetsIndex = pBucketIndex * mOrdersPerBucket;

    localRedistributionMapType::iterator lIter = lBucket.startIter;
    for(; lIter != lBucket.endIter; ++lIter)
    {
        size_t lCurrentOffset = mGlobalOffsetsVector[lGlobalOffsetsIndex++];
        
        std::vector<std::pair<size_t, ulong>>& lVector = lIter->second;
        
        std::vector<std::pair<size_t, ulong>>::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            mRedistributedAddressSpace->Update(lCurrentOffset, lInnerIter->second, lMemAddr + lInnerIter->first);

            if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
            {
                lRangeOwner.hostOffset = lCurrentOffset;
                mRedistributedAddressSpace->TransferOwnershipPostTaskCompletion(lRangeOwner, lCurrentOffset, lInnerIter->second);
            }

            lCurrentOffset += lInnerIter->second;
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
    static_cast<pmRemoteTask*>(mTask)->MarkRedistributionFinished(mAddressSpaceIndex, mRedistributedAddressSpace);
}

pmRedistributionMetadata* pmRedistributor::GetRedistributionMetadata(ulong* pCount)
{
    if(mRedistributionMetaData.empty())
    {
        std::map<uint, uint> mMap;    // Order no. versus length
        globalRedistributionMapType::const_iterator lIter = mGlobalRedistributionMap.begin(), lEndIter = mGlobalRedistributionMap.end();
        for(; lIter != lEndIter; ++lIter)
        {
            auto lInnerIter = mMap.find(lIter->first.first);
            if(lInnerIter == mMap.end())
                lInnerIter = mMap.emplace(lIter->first.first, 0).first;
            
            lInnerIter->second += lIter->second;
        }
        
        *pCount = mMap.size();
        mRedistributionMetaData.reserve(mMap.size());
        
        for_each(mMap, [&] (typename decltype(mMap)::value_type& pPair)
                 {
                     mRedistributionMetaData.emplace_back(pPair.first, pPair.second);
                 });
    }
        
    return &mRedistributionMetaData[0];
}

}







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

pmRedistributor::pmRedistributor(pmTask* pTask, uint pAddressSpaceIndex)
	: mTask(pTask)
    , mAddressSpaceIndex(pAddressSpaceIndex)
    , mSubtasksAccounted(0)
    , mRedistributedAddressSpace(NULL)
    , mGlobalRedistributionLock __LOCK_NAME__("pmRedistributor::mGlobalRedistributionLock")
    , mLocalRedistributionLock __LOCK_NAME__("pmRedistributor::mLocalRedistributionLock")
    , mPendingBucketsCount(0)
    , mPendingBucketsCountLock __LOCK_NAME__("pmRedistributor::mPendingBucketCountLock")
    , mOrdersPerBucket(0)
{
}

void pmRedistributor::RedistributeData(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, ulong pOffset, ulong pLength, uint pOrder)
{
    if(!pLength)
        return;

#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

    pmSubscriptionInfo lSubscriptionInfo = mTask->GetSubscriptionManager().GetConsolidatedWriteSubscription(pStub, pSubtaskId, pSplitInfo, mAddressSpaceIndex);
    if(!lSubscriptionInfo.length)
        PMTHROW(pmFatalErrorException());
    
    size_t lGlobalOffset = lSubscriptionInfo.offset + pOffset;
    if(lGlobalOffset >= mTask->GetAddressSpace(mAddressSpaceIndex)->GetLength())
        PMTHROW(pmFatalErrorException());
        
	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    mLocalRedistributionVector.push_back(communicator::redistributionOrderStruct(pOrder, pLength));
    mLocalRedistributionOffsets.push_back(lGlobalOffset);
    
    mLocalRedistributionMap[pOrder].push_back(mLocalRedistributionVector.size() - 1);
}
    
void pmRedistributor::SendRedistributionInfo()
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mTask->GetTaskProfiler(), taskProfiler::DATA_REDISTRIBUTION);
#endif

	FINALIZE_RESOURCE_PTR(dRedistributionLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mLocalRedistributionLock, Lock(), Unlock());

    pmScheduler::GetScheduler()->RedistributionMetaDataEvent(mTask, mAddressSpaceIndex, &mLocalRedistributionVector);

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
  
        globalRedistributionMapType::iterator lIter = mGlobalRedistributionMap.find(lPair);
        if(lIter == mGlobalRedistributionMap.end())
            mGlobalRedistributionMap[lPair] = lData.length;
        else
            lIter->second += lData.length;
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
    size_t lOrders = mLocalRedistributionMap.size();
    size_t lBuckets = ((lOrders > lDevices) ? lDevices : lOrders);
    mOrdersPerBucket = ((lOrders + lBuckets - 1) / lBuckets);
    
    mLocalRedistributionBucketsVector.resize(lBuckets);
    
    std::map<ulong, std::vector<size_t> >::iterator lIter = mLocalRedistributionMap.begin(), lEndIter = mLocalRedistributionMap.end();
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
        lStubManager->GetStub((uint)i)->ProcessRedistributionBucket(mTask, mAddressSpaceIndex, i);
}

void pmRedistributor::CreateRedistributedAddressSpace(ulong pGenerationNumber /* = std::numeric_limits<ulong>::max() */)
{
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(mAddressSpaceIndex);

    if(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
        mRedistributedAddressSpace = pmAddressSpace::CreateAddressSpace(lAddressSpace->GetLength(), PM_LOCAL_MACHINE);
    else
        mRedistributedAddressSpace = pmAddressSpace::CreateAddressSpace(lAddressSpace->GetLength(), lAddressSpace->GetMemOwnerHost(), pGenerationNumber);

    mRedistributedAddressSpace->Lock(mTask, lAddressSpace->GetMemType());
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
        
        std::vector<size_t>& lVector = lIter->second;
        
        std::vector<size_t>::iterator lInnerIter = lVector.begin(), lInnerEndIter = lVector.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            communicator::redistributionOrderStruct& lData = mLocalRedistributionVector[*lInnerIter];
            
            DEBUG_EXCEPTION_ASSERT(lData.order == lIter->first);
            
            mRedistributedAddressSpace->Update(lCurrentOffset, lData.length, lMemAddr + mLocalRedistributionOffsets[*lInnerIter]);

            if(mTask->GetOriginatingHost() != PM_LOCAL_MACHINE)
            {
                lRangeOwner.hostOffset = lCurrentOffset;
                mRedistributedAddressSpace->TransferOwnershipPostTaskCompletion(lRangeOwner, lCurrentOffset, lData.length);
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
    pmAddressSpace* lAddressSpace = mTask->GetAddressSpace(mAddressSpaceIndex);

    if(mTask->GetOriginatingHost() == PM_LOCAL_MACHINE)
    {
        lAddressSpace->GetUserMemHandle()->Reset(mRedistributedAddressSpace);

        lAddressSpace->Unlock(mTask);
        lAddressSpace->UserDelete();
        static_cast<pmLocalTask*>(mTask)->TaskRedistributionDone(mAddressSpaceIndex, mRedistributedAddressSpace);
    }
    else
    {
        lAddressSpace->Unlock(mTask);
        lAddressSpace->UserDelete();
        static_cast<pmRemoteTask*>(mTask)->MarkRedistributionFinished(mAddressSpaceIndex, mRedistributedAddressSpace);
    }
}
    
}






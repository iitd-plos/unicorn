
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

#ifndef __PM_SUBSCRIPTION_MANAGER__
#define __PM_SUBSCRIPTION_MANAGER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"

#include <map>
#include <vector>
#include <limits>

namespace pm
{

class pmTask;
class pmExecutionStub;
class pmAddressSpace;

namespace subscription
{
	struct subscriptionData
	{
		std::vector<pmCommunicatorCommandPtr> receiveCommandVector;
	};

    typedef std::map<size_t, std::pair<size_t, subscriptionData> > subscriptionRecordType;
    
    bool operator==(const subscription::subscriptionRecordType& pRecord1, const subscription::subscriptionRecordType& pRecord2);
    bool operator!=(const subscription::subscriptionRecordType& pRecord1, const subscription::subscriptionRecordType& pRecord2);

    class shadowMemDeallocator
    {
        public:
            shadowMemDeallocator()
            : mTask(NULL)
            , mMemIndex(std::numeric_limits<uint>::max())
            , mExplicitAllocation(false)
            {}
    
            void SetTaskAndAddressSpaceIndex(pmTask* pTask, uint pMemIndex)
            {
                mTask = pTask;
                mMemIndex = pMemIndex;
            }
    
            void SetExplicitAllocation()
            {
                mExplicitAllocation = true;
            }
    
            void operator()(void* pMem);
    
        private:
            pmTask* mTask;
            uint mMemIndex;
            bool mExplicitAllocation;
    };
    
    struct pmSubtaskSubscriptionData
    {
        pmSubscriptionInfo mConsolidatedSubscriptions;
        subscriptionRecordType mSubscriptionRecords;
        
        bool operator==(const pmSubtaskSubscriptionData& pData) const
        {
            if(mConsolidatedSubscriptions != pData.mConsolidatedSubscriptions)
                return false;
            
            if(mSubscriptionRecords != pData.mSubscriptionRecords)
                return false;
            
            return true;
        }
        
        bool operator!=(const pmSubtaskSubscriptionData& pData) const
        {
            return !(this->operator==(pData));
        }
    };
    
    struct pmCompactViewData
    {
        pmSubscriptionInfo subscriptionInfo;
        std::vector<size_t> nonConsolidatedReadSubscriptionOffsets;
        std::vector<size_t> nonConsolidatedWriteSubscriptionOffsets;
        
        pmCompactViewData()
        : subscriptionInfo()
        {}
    };

    struct pmSubtaskAddressSpaceData
    {
        pmSubtaskSubscriptionData mReadSubscriptionData;
        pmSubtaskSubscriptionData mWriteSubscriptionData;

        finalize_ptr<pmSubscriptionInfo> mUnifiedSubscription;
        finalize_ptr<pmCompactViewData> mCompactedSubscription;

        finalize_ptr<void, shadowMemDeallocator> mShadowMem;

    #ifdef SUPPORT_LAZY_MEMORY
        std::vector<char> mWriteOnlyLazyDefaultValue;
        std::map<size_t, size_t> mWriteOnlyLazyUnprotectedPageRangesMap;
        size_t mWriteOnlyLazyUnprotectedPageCount;
    #endif
        
        pmSubtaskAddressSpaceData()
    #ifdef SUPPORT_LAZY_MEMORY
        : mWriteOnlyLazyUnprotectedPageCount(0)
    #endif
        {}
    };

	struct pmSubtask
	{
		pmCudaLaunchConf mCudaLaunchConf;

        finalize_ptr_array<char> mScratchBuffer;
        size_t mScratchBufferSize;
        pmScratchBufferType mScratchBufferType;

        finalize_ptr<pmSubtaskInfo> mSubtaskInfo;
        
        std::vector<pmMemInfo> mMemInfo;
        std::vector<pmSubtaskAddressSpaceData> mAddressSpacesData;  // The address space ordering is same as in class pmTask
    
        bool mReadyForExecution;    // a flag indicating that pmDataDistributionCB has already been executed (data may not be fetched)
        size_t mReservedCudaGlobalMemSize;
        
        pmSubtask(pmTask* pTask);
        
        pmSubtask(pmSubtask&& pSubtask) = default;
	};
    
    struct shadowMemDetails
    {
        pmSubscriptionInfo subscriptionInfo;
        pmAddressSpace* addressSpace;
        pmTask* task;

        shadowMemDetails(const pmSubscriptionInfo& pSubscriptionInfo, pmAddressSpace* pAddressSpace, pmTask* pTask)
        : subscriptionInfo(pSubscriptionInfo)
        , addressSpace(pAddressSpace)
        , task(pTask)
        {}
    };
    
    struct pmCompactPageInfo
    {
        ulong compactViewOffset;
        ulong addressSpaceOffset;
        ulong length;
    };
}
    
class pmSubscriptionManager : public pmBase
{
    friend class subscription::shadowMemDeallocator;
    typedef std::map<ulong, subscription::pmSubtask> subtaskMapType;

#ifdef SUPPORT_SPLIT_SUBTASKS
    typedef std::map<std::pair<ulong, pmSplitInfo>, subscription::pmSubtask> splitSubtaskMapType;
#endif

	public:
		pmSubscriptionManager(pmTask* pTask);
    
        void DropAllSubscriptions();
    
        void FindSubtaskMemDependencies(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);

        void EraseSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
		void InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
		void RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmSubscriptionType pSubscriptionType, const pmSubscriptionInfo& pSubscriptionInfo);
        void FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmDeviceType pDeviceType, bool pPrefetch);

		void SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf);
        void ReserveCudaGlobalMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pSize);

    #ifdef SUPPORT_LAZY_MEMORY
        void SetWriteOnlyLazyDefaultValue(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, char* pVal, size_t pLength);

        void AddWriteOnlyLazyUnprotection(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pPageNum);
        size_t GetWriteOnlyLazyUnprotectedPagesCount(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);
    
        const std::map<size_t, size_t>& GetWriteOnlyLazyUnprotectedPageRanges(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);
        void InitializeWriteOnlyLazyMemory(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, pmTask* pTask, pmAddressSpace* pAddressSpace, size_t pOffsetFromBase, void* pLazyPageAddr, size_t pLength);
    #endif

		pmCudaLaunchConf& GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        size_t GetReservedCudaGlobalMemSize(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        void DropScratchBufferIfNotRequiredPostSubtaskExec(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void* GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferType pScratchBufferType, size_t pBufferSize);
        void* CheckAndGetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t& pScratchBufferSize, pmScratchBufferType& pScratchBufferType);
    
        void GetNonConsolidatedReadSubscriptions(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);
        void GetNonConsolidatedWriteSubscriptions(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);
    
        const pmSubscriptionInfo& GetConsolidatedReadSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex);
        const pmSubscriptionInfo& GetConsolidatedWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex);
        const pmSubscriptionInfo& GetUnifiedReadWriteSubscription(const pmExecutionStub* pStub, ulong pSubtaskId, const pmSplitInfo* pSplitInfo, uint pMemIndex);

        bool SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);

        void CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pMem = NULL, size_t pMemLength = 0, size_t pWriteOnlyUnprotectedRanges = 0, uint* pUnprotectedRanges = NULL);
        void* GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);
    
        void DestroySubtaskRangeShadowMem(pmExecutionStub* pStub, ulong pStartSubtaskId, ulong pEndSubtaskId, uint pMemIndex);
        void DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);
        void CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);
    
		const pmSubtaskInfo& GetSubtaskInfo(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        const subscription::pmCompactViewData& GetCompactedSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);

        std::vector<subscription::pmCompactPageInfo> GetReadSubscriptionPagesForCompactViewPage(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, size_t pCompactViewPageOffset, size_t pPageSize);

    #ifdef SUPPORT_LAZY_MEMORY
        static pmAddressSpace* FindAddressSpaceContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr, pmTask*& pTask);
    #endif

	private:
        size_t GetAddressSpaceOffsetFromCompactViewOffsetInternal(subscription::pmSubtask& pSubtask, uint pMemIndex, size_t pCompactViewOffset);

        void InitializeSubtaskShadowMemNaturalView(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pShadowMem, void* pMem, size_t pMemLength, size_t pWriteOnlyUnprotectedRanges, uint* pUnprotectedRanges);
        void InitializeSubtaskShadowMemCompactView(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex, void* pShadowMem, void* pMem, size_t pMemLength, size_t pWriteOnlyUnprotectedRanges, uint* pUnprotectedRanges);

        void AddSubscriptionRecordToMap(const pmSubscriptionInfo& pSubscriptionInfo, subscription::subscriptionRecordType& pMap);
        void WaitForSubscriptions(subscription::pmSubtask& pSubtask, pmExecutionStub* pStub, pmDeviceType pDeviceType);

        void CheckAppropriateSubscription(pmAddressSpace* pAddressSpace, pmSubscriptionType pSubscriptionType) const;
        bool IsReadSubscription(pmSubscriptionType pSubscriptionType) const;
        bool IsWriteSubscription(pmSubscriptionType pSubscriptionType) const;

#ifdef SUPPORT_CUDA
    #ifdef SUPPORT_LAZY_MEMORY
        void ClearInputMemLazyProtectionForCuda(subscription::pmSubtask& pSubtask, pmAddressSpace* pAddressSpace, uint pAddressSpaceIndex, pmDeviceType pDeviceType);
    #endif
#endif

        void DestroySubtaskShadowMemInternal(subscription::pmSubtask& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, uint pMemIndex);

        void GetNonConsolidatedReadSubscriptionsInternal(subscription::pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);
        void GetNonConsolidatedWriteSubscriptionsInternal(subscription::pmSubtask& pSubtask, uint pMemIndex, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);

        const pmSubscriptionInfo& GetConsolidatedReadSubscriptionInternal(const pmExecutionStub* pStub, subscription::pmSubtask& pSubtask, uint pMemIndex);
        const pmSubscriptionInfo& GetConsolidatedWriteSubscriptionInternal(const pmExecutionStub* pStub, subscription::pmSubtask& pSubtask, uint pMemIndex);
        const pmSubscriptionInfo& GetUnifiedReadWriteSubscriptionInternal(const pmExecutionStub* pStub, subscription::pmSubtask& pSubtask, uint pMemIndex);
    
        bool SubtasksHaveMatchingSubscriptionsCommonStub(pmExecutionStub* pStub, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);
        bool SubtasksHaveMatchingSubscriptionsDifferentStubs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);
        bool SubtasksHaveMatchingSubscriptionsInternal(const subscription::pmSubtask& pSubtask1, const subscription::pmSubtask& pSubtask2, pmSubscriptionType pSubscriptionType) const;
        bool AddressSpacesHaveMatchingSubscriptionsInternal(const subscription::pmSubtaskAddressSpaceData& pAddressSpaceData1, const subscription::pmSubtaskAddressSpaceData& pAddressSpaceData2) const;

		pmTask* mTask;

        std::vector<std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS> > mSubtaskMapVector;

    #ifdef SUPPORT_SPLIT_SUBTASKS
        std::vector<std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS> > mSplitSubtaskMapVector;
    #endif
    
    #ifdef SUPPORT_LAZY_MEMORY
        typedef std::map<void*, subscription::shadowMemDetails> shadowMemMapType;

        static shadowMemMapType& GetShadowMemMap();	// Maps shadow memory regions to pmAddressSpace objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetShadowMemLock();
    #endif
};

} // end namespace pm

#endif

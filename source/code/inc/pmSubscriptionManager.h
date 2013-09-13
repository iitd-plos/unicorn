
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
#include "pmCommand.h"

#include <map>
#include <vector>

namespace pm
{

class pmTask;
class pmExecutionStub;
class pmMemSection;

namespace subscription
{
	typedef struct subscriptionData
	{
		std::vector<pmCommunicatorCommandPtr> receiveCommandVector;
	} subscriptionData;

    typedef std::map<size_t, std::pair<size_t, subscriptionData> > subscriptionRecordType;

    class shadowMemDeallocator
    {
        public:
            shadowMemDeallocator()
            : mTask(NULL)
            , mExplicitAllocation(false)
            {}
    
            void SetTask(pmTask* pTask)
            {
                mTask = pTask;
            }
    
            void SetExplicitAllocation()
            {
                mExplicitAllocation = true;
            }
    
            void operator()(void* pMem);
    
        private:
            pmTask* mTask;
            bool mExplicitAllocation;
    };
    
	typedef struct pmSubtask
	{
		pmCudaLaunchConf mCudaLaunchConf;
        finalize_ptr_array<char> mScratchBuffer;
        size_t mScratchBufferSize;
        pmScratchBufferInfo mScratchBufferInfo;

        pmSubscriptionInfo mConsolidatedInputMemSubscription;   // The contiguous range enclosing all ranges in mInputMemSubscriptions
        pmSubscriptionInfo mConsolidatedOutputMemReadSubscription;   // The contiguous range enclosing all ranges in mOutputMemReadSubscriptions
        pmSubscriptionInfo mConsolidatedOutputMemWriteSubscription;   // The contiguous range enclosing all ranges in mOutputMemWriteSubscriptions
    
		subscriptionRecordType mInputMemSubscriptions;
        subscriptionRecordType mOutputMemReadSubscriptions;
        subscriptionRecordType mOutputMemWriteSubscriptions;
    
        finalize_ptr<void, shadowMemDeallocator> mShadowMem;

    #ifdef SUPPORT_LAZY_MEMORY
        std::vector<char> mWriteOnlyLazyDefaultValue;
        std::map<size_t, size_t> mWriteOnlyLazyUnprotectedPageRangesMap;
        size_t mWriteOnlyLazyUnprotectedPageCount;
    #endif
    
        bool mReadyForExecution;    // a flag indicating that pmDataDistributionCB has already been executed (data may not be fetched)
        bool mExplicitInputMemSubscription; // to support zero length subscriptions
        bool mExplicitOutputMemReadSubscription; // to support zero length subscriptions
        bool mExplicitOutputMemWriteSubscription; // to support zero length subscriptions
        size_t mReservedCudaGlobalMemSize;
        
		pmStatus Initialize(pmTask* pTask);
	} pmSubtask;
    
    typedef struct shadowMemDetails
    {
        pmSubscriptionInfo subscriptionInfo;
        pmMemSection* memSection;
    } shadowMemDetails;
}
    
class pmSubscriptionManager : public pmBase
{
    friend class subscription::shadowMemDeallocator;
    typedef std::map<void*, subscription::shadowMemDetails> shadowMemMapType;
    typedef std::map<ulong, subscription::pmSubtask> subtaskMapType;

#ifdef SUPPORT_SPLIT_SUBTASKS
    typedef std::map<std::pair<ulong, pmSplitInfo>, subscription::pmSubtask> splitSubtaskMapType;
#endif

	public:
		pmSubscriptionManager(pmTask* pTask);
		virtual ~pmSubscriptionManager();
    
        void DropAllSubscriptions();
    
        void FindSubtaskMemDependencies(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);

        pmStatus EraseSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
		pmStatus InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
		pmStatus RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmSubscriptionType pSubscriptionType, pmSubscriptionInfo pSubscriptionInfo);
        pmStatus FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmDeviceType pDeviceType, bool pPrefetch);
		pmStatus SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmCudaLaunchConf& pCudaLaunchConf);
        pmStatus ReserveCudaGlobalMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pSize);

    #ifdef SUPPORT_LAZY_MEMORY
        pmStatus SetWriteOnlyLazyDefaultValue(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, char* pVal, size_t pLength);

        void AddWriteOnlyLazyUnprotection(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pPageNum);
        size_t GetWriteOnlyLazyUnprotectedPagesCount(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        const std::map<size_t, size_t>& GetWriteOnlyLazyUnprotectedPageRanges(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void InitializeWriteOnlyLazyMemory(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t pOffsetFromBase, void* pLazyPageAddr, size_t pLength);
    #endif

		pmCudaLaunchConf& GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        size_t GetReservedCudaGlobalMemSize(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        void DropScratchBufferIfNotRequiredPostSubtaskExec(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void* GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmScratchBufferInfo pScratchBufferInfo, size_t pBufferSize);
        void* CheckAndGetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, size_t& pScratchBufferSize, pmScratchBufferInfo& pScratchBufferInfo);

		bool GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmSubscriptionInfo& pSubscriptionInfo);
		bool GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo);
        bool GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, pmSubscriptionInfo& pSubscriptionInfo);
    
        bool GetNonConsolidatedInputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);
        bool GetNonConsolidatedOutputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pReadSubscriptions, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);

        bool SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);

        pmStatus CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, void* pMem = NULL, size_t pMemLength = 0, size_t pWriteOnlyUnprotectedRanges = 0, uint* pUnprotectedRanges = NULL);
        void* GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        void DestroySubtaskRangeShadowMem(pmExecutionStub* pStub, ulong pStartSubtaskId, ulong pEndSubtaskId);
        void DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, subscription::subscriptionRecordType::const_iterator& pBeginIter, subscription::subscriptionRecordType::const_iterator& pEndIter, ulong pShadowMemOffset);
    
        static pmMemSection* FindMemSectionContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr);

	private:
        pmStatus FreezeSubtaskSubscriptions(subscription::pmSubtask& pSubtask);
        pmStatus WaitForSubscriptions(subscription::pmSubtask& pSubtask, pmExecutionStub* pStub);
        pmStatus FetchInputMemSubscription(subscription::pmSubtask& pSubtask, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, ushort pPriority, subscription::subscriptionData& pData);
        pmStatus FetchOutputMemSubscription(subscription::pmSubtask& pSubtask, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, ushort pPriority, subscription::subscriptionData& pData);
    
#ifdef SUPPORT_CUDA
    #ifdef SUPPORT_LAZY_MEMORY
        void ClearInputMemLazyProtectionForCuda(subscription::pmSubtask& pSubtask, pmDeviceType pDeviceType);
    #endif
#endif

        pmStatus DestroySubtaskShadowMemInternal(subscription::pmSubtask& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        bool SubtasksHaveMatchingSubscriptionsCommonStub(pmExecutionStub* pStub, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);
        bool SubtasksHaveMatchingSubscriptionsDifferentStubs(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmSubscriptionType pSubscriptionType);
        bool SubtasksHaveMatchingSubscriptionsInternal(subscription::pmSubtask& pSubtask1, subscription::pmSubtask& pSubtask2, pmSubscriptionType pSubscriptionType);
    
        void GetSubtask(subscription::pmSubtask*& pSubtask, pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
        void GetSubtasks(subscription::pmSubtask*& pSubtask1, subscription::pmSubtask*& pSubtask2, pmExecutionStub* pStub, ulong pSubtaskId1, ulong pSubtaskId2, pmSplitInfo* pSplitInfo1, pmSplitInfo* pSplitInfo2);
    
        std::vector<std::pair<subtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS> > mSubtaskMapVector;

    #ifdef SUPPORT_SPLIT_SUBTASKS
        std::vector<std::pair<splitSubtaskMapType, RESOURCE_LOCK_IMPLEMENTATION_CLASS> > mSplitSubtaskMapVector;
    #endif

		pmTask* mTask;
    
        static shadowMemMapType& GetShadowMemMap();	// Maps shadow memory regions to pmMemSection objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetShadowMemLock();
};

bool operator==(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2);
bool operator!=(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2);

} // end namespace pm

#endif

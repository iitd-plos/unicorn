
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

namespace pm
{

class pmTask;
class pmExecutionStub;
class pmMemSection;

namespace subscription
{
    typedef class pmSubtaskTerminationCheckPointAutoPtr
    {
        public:
            pmSubtaskTerminationCheckPointAutoPtr(pmExecutionStub* pStub, ulong pSubtaskId);
            ~pmSubtaskTerminationCheckPointAutoPtr();
    
        private:
            pmExecutionStub* mStub;
            ulong mSubtaskId;
    } pmSubtaskTerminationCheckPointAutoPtr;
    
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

        pmSubscriptionInfo mConsolidatedInputMemSubscription;   // The contiguous range enclosing all ranges in mInputMemSubscriptions
        pmSubscriptionInfo mConsolidatedOutputMemReadSubscription;   // The contiguous range enclosing all ranges in mOutputMemReadSubscriptions
        pmSubscriptionInfo mConsolidatedOutputMemWriteSubscription;   // The contiguous range enclosing all ranges in mOutputMemWriteSubscriptions
    
		subscriptionRecordType mInputMemSubscriptions;
        subscriptionRecordType mOutputMemReadSubscriptions;
        subscriptionRecordType mOutputMemWriteSubscriptions;
    
        finalize_ptr<void, shadowMemDeallocator> mShadowMem;

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
    
	public:
		pmSubscriptionManager(pmTask* pTask);
		virtual ~pmSubscriptionManager();
    
        void DropAllSubscriptions();

		pmStatus InitializeSubtaskDefaults(pmExecutionStub* pStub, ulong pSubtaskId);
		pmStatus RegisterSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionType pSubscriptionType, pmSubscriptionInfo pSubscriptionInfo);
		pmStatus FetchSubtaskSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType);
		pmStatus SetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId, pmCudaLaunchConf& pCudaLaunchConf);
		pmCudaLaunchConf& GetCudaLaunchConf(pmExecutionStub* pStub, ulong pSubtaskId);
        void* GetScratchBuffer(pmExecutionStub* pStub, ulong pSubtaskId, size_t pBufferSize);

		bool GetInputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);
		bool GetOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, bool pReadSubscription, pmSubscriptionInfo& pSubscriptionInfo);
        bool GetUnifiedOutputMemSubscriptionForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSubscriptionInfo& pSubscriptionInfo);
    
        bool GetNonConsolidatedInputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);
        bool GetNonConsolidatedOutputMemSubscriptionsForSubtask(pmExecutionStub* pStub, ulong pSubtaskId, bool pReadSubscriptions, subscription::subscriptionRecordType::const_iterator& pBegin, subscription::subscriptionRecordType::const_iterator& pEnd);

        bool SubtasksHaveMatchingSubscriptions(pmExecutionStub* pStub1, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSubscriptionType pSubscriptionType);

        pmStatus CreateSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, char* pMem = NULL, size_t pMemLength = 0);
        void* GetSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId);
        pmStatus DestroySubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId);
        void CommitSubtaskShadowMem(pmExecutionStub* pStub, ulong pSubtaskId, subscription::subscriptionRecordType::const_iterator& pBeginIter, subscription::subscriptionRecordType::const_iterator& pEndIter, ulong pShadowMemOffset);

        static pmMemSection* FindMemSectionContainingShadowAddr(void* pAddr, size_t& pShadowMemOffset, void*& pShadowMemBaseAddr);

	private:
		pmStatus WaitForSubscriptions(pmExecutionStub* pStub, ulong pSubtaskId);
        pmStatus FetchInputMemSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, subscription::subscriptionData& pData);
        pmStatus FetchOutputMemSubscription(pmExecutionStub* pStub, ulong pSubtaskId, pmDeviceType pDeviceType, pmSubscriptionInfo pSubscriptionInfo, subscription::subscriptionData& pData);

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		std::map<std::pair<pmExecutionStub*, ulong>, subscription::pmSubtask> mSubtaskMap;

		pmTask* mTask;
    
        static std::map<void*, subscription::shadowMemDetails> mShadowMemMap;	// Maps shadow memory regions to pmMemSection objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS mShadowMemLock;
};

bool operator==(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2);
bool operator!=(pmSubscriptionInfo& pSubscription1, pmSubscriptionInfo& pSubscription2);

} // end namespace pm

#endif

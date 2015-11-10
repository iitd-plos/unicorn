
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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

#ifndef __PM_DATA_TYPES__
#define __PM_DATA_TYPES__

#include "pmPublicDefinitions.h"
#include "pmInternalDefinitions.h"
#include "pmErrorDefinitions.h"

#include <cstdlib>
#include <setjmp.h>

#include <algorithm>
#include <memory>
#include <vector>
#include <map>

namespace pm
{
#define EXCEPTION_ASSERT(x) \
    if(!(x)) \
        PMTHROW(pmFatalErrorException());
    
#ifdef _DEBUG
    #define DEBUG_EXCEPTION_ASSERT(x) EXCEPTION_ASSERT(x)
#else
    #define DEBUG_EXCEPTION_ASSERT(x)
#endif

	class pmTask;
    class pmProcessingElement;
    class pmExecutionStub;
    class pmAddressSpace;

    namespace communicator
    {
        enum communicatorCommandTypes
        {
            SEND,
            RECEIVE,
            BROADCAST,
            ALL2ALL,
            MAX_COMMUNICATOR_COMMAND_TYPES
        };

        enum communicatorCommandTags
        {
            MACHINE_POOL_TRANSFER_TAG,
            DEVICE_POOL_TRANSFER_TAG,
            REMOTE_TASK_ASSIGNMENT_TAG,
            REMOTE_SUBTASK_ASSIGNMENT_TAG,
            SEND_ACKNOWLEDGEMENT_TAG,
            SEND_ACKNOWLEDGEMENT_SCATTERED_TAG,
            TASK_EVENT_TAG,
            STEAL_REQUEST_TAG,
            STEAL_RESPONSE_TAG,
            OWNERSHIP_TRANSFER_TAG,
            SCATTERED_OWNERSHIP_TRANSFER_TAG,
            MEMORY_TRANSFER_REQUEST_TAG,
            MEMORY_RECEIVE_TAG,
            NO_REDUCTION_REQD_TAG,
            SUBTASK_REDUCE_TAG,
            UNKNOWN_LENGTH_TAG,
            HOST_FINALIZATION_TAG,
            DATA_REDISTRIBUTION_TAG,
            REDISTRIBUTION_OFFSETS_TAG,
            SUBTASK_RANGE_CANCEL_TAG,
            FILE_OPERATIONS_TAG,
            MULTI_FILE_OPERATIONS_TAG,
            AFFINITY_DATA_TRANSFER_TAG,
            SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_TAG,
        #ifdef SERIALIZE_DEFERRED_LOGS
            DEFERRED_LOG_LENGTH_TAG,
            DEFERRED_LOG_TAG,
        #endif
        #ifdef USE_AFFINITY_IN_STEAL
            STEAL_SUCCESS_DISCONTIGUOUS_TAG,
        #endif
            MAX_COMMUNICATOR_COMMAND_TAGS
        };

        enum communicatorDataTypes
        {
            BYTE,
            INT,
            UINT,
            MACHINE_POOL_STRUCT,
            DEVICE_POOL_STRUCT,
            MEMORY_IDENTIFIER_STRUCT,
            TASK_MEMORY_STRUCT,
            REMOTE_TASK_ASSIGN_STRUCT,
            REMOTE_TASK_ASSIGN_PACKED,
            REMOTE_SUBTASK_ASSIGN_STRUCT,
            OWNERSHIP_DATA_STRUCT,
            SCATTERED_OWNERSHIP_DATA_STRUCT,
            SEND_ACKNOWLEDGEMENT_STRUCT,
            SEND_ACKNOWLEDGEMENT_SCATTERED_STRUCT,
            SEND_ACKNOWLEDGEMENT_PACKED,
            SEND_ACKNOWLEDGEMENT_SCATTERED_PACKED,
            TASK_EVENT_STRUCT,
            STEAL_REQUEST_STRUCT,
            STEAL_RESPONSE_STRUCT,
            OWNERSHIP_CHANGE_STRUCT,
            SCATTERED_OWNERSHIP_CHANGE_STRUCT,
            OWNERSHIP_TRANSFER_PACKED,
            SCATTERED_OWNERSHIP_TRANSFER_PACKED,
            MEMORY_TRANSFER_REQUEST_STRUCT,
            SHADOW_MEM_TRANSFER_STRUCT,
            NO_REDUCTION_REQD_STRUCT,
            SUBTASK_REDUCE_STRUCT,
            SUBTASK_REDUCE_PACKED,
            MEMORY_RECEIVE_STRUCT,
            HOST_FINALIZATION_STRUCT,
            REDISTRIBUTION_ORDER_STRUCT,
            DATA_REDISTRIBUTION_STRUCT,
            DATA_REDISTRIBUTION_PACKED,
            REDISTRIBUTION_OFFSETS_STRUCT,
            REDISTRIBUTION_OFFSETS_PACKED,
            SUBTASK_RANGE_CANCEL_STRUCT,
            FILE_OPERATIONS_STRUCT,
            MULTI_FILE_OPERATIONS_STRUCT,
            MULTI_FILE_OPERATIONS_PACKED,
            AFFINITY_DATA_TRANSFER_PACKED,
            SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_STRUCT,
            SCATTERED_MEMORY_TRANSFER_REQUEST_COMBINED_PACKED,
        #ifdef USE_AFFINITY_IN_STEAL
            STEAL_SUCCESS_DISCONTIGUOUS_PACKED,
        #endif
            MAX_COMMUNICATOR_DATA_TYPES
        };
    }

    enum pmSubscriptionFormat
    {
        SUBSCRIPTION_CONTIGUOUS,
        SUBSCRIPTION_SCATTERED,
        SUBSCRIPTION_GENERAL,
        SUBSCRIPTION_FORMAT_MAX
    };

    enum pmAddressSpaceType
    {
        ADDRESS_SPACE_LINEAR,
        ADDRESS_SPACE_2D,
        MAX_ADDRESS_SPACE_TYPES
    };

    struct pmTaskMemory
    {
        pmAddressSpace* addressSpace;
        pmMemType memType;
        pmSubscriptionVisibilityType subscriptionVisibilityType;
        bool disjointReadWritesAcrossSubtasks;

        pmTaskMemory(pmAddressSpace* pAddressSpace, pmMemType pMemType, pmSubscriptionVisibilityType pVisibility, bool pDisjointReadWrites)
        : addressSpace(pAddressSpace)
        , memType(pMemType)
        , subscriptionVisibilityType(pVisibility)
        , disjointReadWritesAcrossSubtasks(pDisjointReadWrites)
        {}
    };

    class pmNonCopyable
    {
        public:
            pmNonCopyable() = default;
        
        private:
            pmNonCopyable(const pmNonCopyable&) = delete;
            pmNonCopyable& operator=(const pmNonCopyable&) = delete;
    };
    
    struct naturalSorter : std::binary_function<const std::string, const std::string, bool>
    {
        std::string GetNextBlock(const std::string& pStr, size_t& pIndex) const;
        bool operator() (const std::string& pStr1, const std::string& pStr2) const;
    };

    struct stubSorter : std::binary_function<const pmExecutionStub*, const pmExecutionStub*, bool>
    {
        bool operator() (const pmExecutionStub* pStub1, const pmExecutionStub* pStub2) const;
    };

    /* Comparison operators for pmSubscriptionInfo */
    bool operator==(const pmSubscriptionInfo& pSubscription1, const pmSubscriptionInfo& pSubscription2);
    bool operator!=(const pmSubscriptionInfo& pSubscription1, const pmSubscriptionInfo& pSubscription2);

    /* Comparison operators for pmScatteredSubscriptionInfo */
    bool operator==(const pmScatteredSubscriptionInfo& pScatteredSubscription1, const pmScatteredSubscriptionInfo& pScatteredSubscription2);
    bool operator!=(const pmScatteredSubscriptionInfo& pScatteredSubscription1, const pmScatteredSubscriptionInfo& pScatteredSubscription2);

    class pmSubtaskTerminationCheckPointAutoPtr
    {
        public:
            pmSubtaskTerminationCheckPointAutoPtr(pmExecutionStub* pStub);
            ~pmSubtaskTerminationCheckPointAutoPtr();
    
        private:
            pmExecutionStub* mStub;
    };
    
    class pmJmpBufAutoPtr
    {
        public:
            pmJmpBufAutoPtr();
            ~pmJmpBufAutoPtr();

            void Reset(sigjmp_buf* pJmpBuf, pmExecutionStub* pStub);
            void SetHasJumped();
        
        private:
            pmExecutionStub* mStub;
            bool mHasJumped;
    };
    
    struct pmSplitData
    {
        bool valid;
        uint splitId;
        uint splitCount;
        
        pmSplitData(bool pValid, uint pSplitId, uint pSplitCount)
        : valid(pValid)
        , splitId(pSplitId)
        , splitCount(pSplitCount)
        {}
        
        pmSplitData(pmSplitInfo* pSplitInfo)
        : valid(pSplitInfo != NULL)
        , splitId(pSplitInfo ? pSplitInfo->splitId : 0)
        , splitCount(pSplitInfo ? pSplitInfo->splitCount : 0)
        {}
        
        operator std::unique_ptr<pmSplitInfo>()
        {
            return std::unique_ptr<pmSplitInfo>(valid ? new pmSplitInfo(splitId, splitCount) : NULL);
        }
        
        /* Can't make a constructor because this class is added to unions and needs to be default constructible */
        static void ConvertSplitInfoToSplitData(pmSplitData& pSplitData, pmSplitInfo* pSplitInfo)
        {
            pSplitData.valid = (pSplitInfo != NULL);
            
            if(pSplitInfo)
            {
                pSplitData.splitId = pSplitInfo->splitId;
                pSplitData.splitCount = pSplitInfo->splitCount;
            }
        }
    };

#ifdef SUPPORT_SPLIT_SUBTASKS
    bool operator<(const pmSplitInfo& pInfo1, const pmSplitInfo& pInfo2);
#endif

#ifdef SUPPORT_CUDA
    class pmStubCUDA;
    class pmMemChunk;

    struct pmCudaCacheKey
    {
        const pmAddressSpace* addressSpace;
        pmSubscriptionVisibilityType visibility;
        pmSubscriptionFormat format;

        // Only one of these is meaningful
        pmSubscriptionInfo subscriptionInfo;    // Used for SUBSCRIPTION_CONTIGUOUS
        std::vector<pmScatteredSubscriptionInfo> scatteredSubscriptionVector;  // Used for SUBSCRIPTION_SCATTERED
        std::vector<pmSubscriptionInfo> subscriptionVector; // Used for SUBSCRIPTION_GENERAL

        pmCudaCacheKey(const pmAddressSpace* pAddressSpace, pmSubscriptionVisibilityType pVisibility, const pmSubscriptionInfo& pSubscriptionInfo)
        : addressSpace(pAddressSpace)
        , visibility(pVisibility)
        , format(SUBSCRIPTION_CONTIGUOUS)
        , subscriptionInfo(pSubscriptionInfo)
        {}

        pmCudaCacheKey(const pmAddressSpace* pAddressSpace, pmSubscriptionVisibilityType pVisibility, const std::vector<pmScatteredSubscriptionInfo>& pScatteredSubscriptionVector)
        : addressSpace(pAddressSpace)
        , visibility(pVisibility)
        , format(SUBSCRIPTION_SCATTERED)
        , scatteredSubscriptionVector(pScatteredSubscriptionVector)
        {}

        pmCudaCacheKey(const pmAddressSpace* pAddressSpace, pmSubscriptionVisibilityType pVisibility, const std::vector<pmSubscriptionInfo>& pSubscriptionVector)
        : addressSpace(pAddressSpace)
        , visibility(pVisibility)
        , format(SUBSCRIPTION_GENERAL)
        , subscriptionVector(pSubscriptionVector)
        {}
        
        pmCudaCacheKey(const pmCudaCacheKey& pKey)
        : addressSpace(pKey.addressSpace)
        , visibility(pKey.visibility)
        , format(pKey.format)
        , subscriptionInfo(pKey.subscriptionInfo)
        , scatteredSubscriptionVector(pKey.scatteredSubscriptionVector)
        , subscriptionVector(pKey.subscriptionVector)
        {}
        
        ~pmCudaCacheKey()
        {}

        bool operator== (const pmCudaCacheKey& pKey) const
        {
            if(addressSpace != pKey.addressSpace || visibility != pKey.visibility || format != pKey.format)
                return false;
            
            if(format == SUBSCRIPTION_CONTIGUOUS && subscriptionInfo == pKey.subscriptionInfo)
                return true;
            
            if(format == SUBSCRIPTION_SCATTERED && scatteredSubscriptionVector == pKey.scatteredSubscriptionVector)
                return true;

            if(format == SUBSCRIPTION_GENERAL && subscriptionVector == pKey.subscriptionVector)
                return true;
            
            return false;
        }
    };
    
    struct pmCudaCacheHasher
    {
        std::size_t operator() (const pmCudaCacheKey& pKey) const
        {
            size_t lVal = ((((size_t)pKey.visibility) << 8) | ((size_t)pKey.format));

            size_t lHash1 = (std::hash<size_t>()(reinterpret_cast<size_t>(pKey.addressSpace)) ^ std::hash<size_t>()(lVal));
            size_t lHash2 = 0;
            
            if(pKey.format == SUBSCRIPTION_CONTIGUOUS)
                lHash2 = (std::hash<size_t>()(pKey.subscriptionInfo.offset) ^ std::hash<size_t>()(pKey.subscriptionInfo.length));
            
            if(pKey.format == SUBSCRIPTION_SCATTERED)
            {
                std::for_each(pKey.scatteredSubscriptionVector.begin(), pKey.scatteredSubscriptionVector.end(), [&lHash2] (const pmScatteredSubscriptionInfo& pInfo)
                {
                    lHash2 ^= (std::hash<size_t>()(pInfo.offset) ^ std::hash<size_t>()(pInfo.size) ^ std::hash<size_t>()(pInfo.step) ^ std::hash<size_t>()(pInfo.count));
                });
            }

            if(pKey.format == SUBSCRIPTION_GENERAL)
            {
                std::for_each(pKey.subscriptionVector.begin(), pKey.subscriptionVector.end(), [&lHash2] (const pmSubscriptionInfo& pInfo)
                {
                    lHash2 ^= (std::hash<size_t>()(pInfo.offset) ^ std::hash<size_t>()(pInfo.length));
                });
            }
            
            return (lHash1 ^ (lHash2 << 1));
        }
    };
    
    struct pmCudaCacheValue
    {
        void* cudaPtr;
        size_t allocationLength;
        
        pmCudaCacheValue(void* pCudaPtr, size_t pAllocationLength)
        : cudaPtr(pCudaPtr)
        , allocationLength(pAllocationLength)
        {}
    };
    
    struct pmCudaCacheEvictor
    {
        pmCudaCacheEvictor(pmStubCUDA* pStub)
        : mStub(pStub)
        {}

        void operator() (const std::shared_ptr<pmCudaCacheValue>& pValue);
        
    private:
        pmStubCUDA* mStub;
    };
    
    struct pmCudaMemChunkTraits
    {
        typedef pmMemChunk allocator;
        static const bool alignedAllocations = true;

        struct creator
        {
            std::shared_ptr<pmMemChunk> operator()(size_t pSize);
        };
        
        struct destructor
        {
            void operator()(const std::shared_ptr<pmMemChunk>& pPtr);
        };
    };

#ifdef SUPPORT_CUDA_COMPUTE_MEM_TRANSFER_OVERLAP
    struct pmPinnedMemChunkTraits
    {
        typedef pmMemChunk allocator;
        static const bool alignedAllocations = true;

        struct creator
        {
            std::shared_ptr<pmMemChunk> operator()(size_t pSize);
        };
        
        struct destructor
        {
            void operator()(const std::shared_ptr<pmMemChunk>& pPtr);
        };
    };
#endif

#endif

#ifdef ENABLE_TASK_PROFILING
    class pmTaskProfiler;

    namespace taskProfiler
    {
        enum profileType
        {
            INPUT_MEMORY_TRANSFER,
            OUTPUT_MEMORY_TRANSFER,
            TOTAL_MEMORY_TRANSFER,    /* For internal use only */
            DATA_PARTITIONING,
            SUBTASK_EXECUTION,
            DATA_REDUCTION,
            DATA_REDISTRIBUTION,
            SHADOW_MEM_COMMIT,
            SUBTASK_STEAL_WAIT,
            SUBTASK_STEAL_SERVE,
            STUB_WAIT_ON_NETWORK,
            COPY_TO_PINNED_MEMORY,
            COPY_FROM_PINNED_MEMORY,
            CUDA_COMMAND_PREPARATION,
            PREPROCESSOR_TASK_EXECUTION,
            AFFINITY_SUBTASK_MAPPINGS,
            AFFINITY_USE_OVERHEAD,
            FLUSH_MEMORY_OWNERSHIPS,
            UNIVERSAL, /* For internal use only */
            MAX_PROFILE_TYPES
        };
    }

    class pmRecordProfileEventAutoPtr
    {
        public:
            pmRecordProfileEventAutoPtr(pmTaskProfiler* pTaskProfiler, taskProfiler::profileType pProfileType);
            ~pmRecordProfileEventAutoPtr();
        
        private:
            pmTaskProfiler* mTaskProfiler;
            taskProfiler::profileType mProfileType;
    };
#endif
    
#ifdef DUMP_EVENT_TIMELINE
    class pmEventTimeline;

#ifdef SUPPORT_SPLIT_SUBTASKS
    class pmSplitSubtaskExecutionTimelineAutoPtr
    {
        public:
            pmSplitSubtaskExecutionTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pSubtaskId, uint pSplitId, uint pSplitCount);
            ~pmSplitSubtaskExecutionTimelineAutoPtr();
        
            void SetGracefulCompletion();
        
            std::string GetEventName(ulong pSubtaskId, uint pSplitId, uint pSplitCount, pmTask* pTask);
            std::string GetCancelledEventName(ulong pSubtaskId, uint pSplitId, uint pSplitCount, pmTask* pTask);

        private:
            pmTask* mTask;
            pmEventTimeline* mEventTimeline;
            ulong mSubtaskId;
            uint mSplitId;
            uint mSplitCount;
            bool mCancelledOrException;
    };
#endif
    
    class pmSubtaskRangeExecutionTimelineAutoPtr
    {
        public:
            pmSubtaskRangeExecutionTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pStartSubtask, ulong pEndSubtask);
            ~pmSubtaskRangeExecutionTimelineAutoPtr();
        
            void ResetEndSubtask(ulong pEndSubtask);
        
            void InitializeNextSubtask();
            void SetGracefulCompletion();
        
            void FinishSubtask(ulong pSubtaskId);
        
            static std::string GetEventName(ulong pSubtaskId, pmTask* pTask);
            static std::string GetCancelledEventName(ulong pSubtaskId, pmTask* pTask);

        private:
            pmTask* mTask;
            pmEventTimeline* mEventTimeline;
            ulong mStartSubtask;
            ulong mEndSubtask;
            bool mRangeCancelledOrException;
            ulong mSubtasksInitialized;
    };
    
    class pmEventTimelineAutoPtr
    {
        public:
            pmEventTimelineAutoPtr(pmTask* pTask, pmEventTimeline* pEventTimeline, ulong pSubtaskId, const pmSplitData& pSplitData, uint pDeviceId, const std::string& pEventNameSuffix);
            ~pmEventTimelineAutoPtr();
        
        private:
            std::string GetEventName(ulong pSubtaskId, const pmSplitData& pSplitData, uint pDeviceId, const std::string& pEventNameSuffix);

            pmTask* mTask;
            pmEventTimeline* mEventTimeline;
            std::string mEventName;
    };
#endif

    typedef struct pmSubtaskRange
    {
        pmTask* task;
        const pmProcessingElement* originalAllottee;
        ulong startSubtask;
        ulong endSubtask;

        pmSubtaskRange(pmTask* pTask, const pmProcessingElement* pOriginalAllottee, ulong pStartSubtask, ulong pEndSubtask)
        : task(pTask)
        , originalAllottee(pOriginalAllottee)
        , startSubtask(pStartSubtask)
        , endSubtask(pEndSubtask)
        {}
        
        pmSubtaskRange(const pmSubtaskRange& pRange)
        : task(pRange.task)
        , originalAllottee(pRange.originalAllottee)
        , startSubtask(pRange.startSubtask)
        , endSubtask(pRange.endSubtask)
        {}
    } pmSubtaskRange;
    
#ifdef TRACK_MEM_COPIES
    class pmMemCopyTracker
    {
        struct pmMemCopyStatistics
        {
            ulong mBytesCopied;
            double mElapsedTime;
            double mStartTime;
            uint mOngoingMemCopies;

            pmMemCopyStatistics();
            
            void BeginRecord(size_t pBytes);
            void EndRecord();
        };
        
        public:
            pmMemCopyTracker();
            ~pmMemCopyTracker();

            void SetHostId(uint pHostId);
            void Begin(size_t pBytes, const std::string& pKey);
            void End(const std::string& pKey);

        private:
            uint mHostId;
            pmMemCopyStatistics mNodeTimer;
            std::map<std::string, pmMemCopyStatistics> mBifurcationMap;
    };
    
    extern pmMemCopyTracker gMemCopyTracker;

#define PMLIB_MEMCPY(a, b, c, key) { gMemCopyTracker.Begin(c, key); memcpy(a, b, c); gMemCopyTracker.End(key);}
#else
#define PMLIB_MEMCPY(a, b, c, key) memcpy(a, b, c)
#endif

#ifdef SUPPORT_SPLIT_SUBTASKS
    typedef struct pmSplitSubtask
    {
        pmTask* task;
        pmExecutionStub* sourceStub;
        ulong subtaskId;
        uint splitId;
        uint splitCount;
        
        pmSplitSubtask(pmTask* pTask, pmExecutionStub* pSourceStub, ulong pSubtaskId, uint pSplitId, uint pSplitCount)
        : task(pTask)
        , sourceStub(pSourceStub)
        , subtaskId(pSubtaskId)
        , splitId(pSplitId)
        , splitCount(pSplitCount)
        {}
    } pmSplitSubtask;
#endif
    
    #define STATIC_ACCESSOR(type, className, funcName) \
    type& className::funcName() \
    { \
        static type t; \
        return t; \
    }
    
    #define STATIC_ACCESSOR_INIT(type, className, funcName, initValue) \
    type& className::funcName() \
    { \
        static type t = initValue; \
        return t; \
    }
    
    #define STATIC_ACCESSOR_ARG(type, arg, className, funcName) \
    type& className::funcName() \
    { \
        static type t arg; \
        return t; \
    }
    
    #define STATIC_ACCESSOR_GLOBAL(type, arg, funcName) \
    type& funcName(); \
    type& funcName() \
    { \
        static type t arg; \
        return t; \
    }

    template<typename T>
    struct deleteDeallocator
    {
        void operator()(T* pMem)
        {
            delete pMem;
        }
    };
    
    template<typename T>
    struct deleteArrayDeallocator
    {
        void operator()(T* pMem)
        {
            delete[] pMem;
        }
    };
    
    template<typename T1, typename T2>
    struct select2nd
    {
        T2 operator() (const std::pair<T1, T2>& pPair)
        {
            return pPair.second;
        }
    };
    
	template<typename T, typename D = deleteDeallocator<T> >
	class finalize_ptr
	{
		public:
            typedef T ptrType;
        
			finalize_ptr(T* pMem = NULL, bool pHasOwnership = true)
            : mMem(pMem)
            , mHasOwnership(pHasOwnership)
			{
			}

			~finalize_ptr()
			{
                if(mMem && mHasOwnership)
                    mDeallocator.operator()(mMem);
			}

			T* get_ptr() const
			{
				return mMem;
			}

            T* release()
            {
                T* lMem = mMem;

                mMem = NULL;
                mHasOwnership = false;
                
                return lMem;
            }

            void reset(T* pMem, bool pHasOwnership = true)
            {
                if(mMem && mHasOwnership)
                    mDeallocator.operator()(mMem);
                
                mHasOwnership = pHasOwnership;
                mMem = pMem;
            }
        
            finalize_ptr(const finalize_ptr& pPtr) = delete;
            finalize_ptr& operator=(finalize_ptr& pPtr) = delete;

            finalize_ptr(finalize_ptr&& pPtr)
            : mMem(NULL)
            , mHasOwnership(false)
            {
                reset(pPtr.get_ptr(), pPtr.mHasOwnership);
                pPtr.release();
            }
            
            finalize_ptr& operator=(finalize_ptr&& pPtr)
            {
                reset(pPtr.get_ptr(), pPtr.mHasOwnership);
                pPtr.release();
            
                return *this;
            }
    
            T* operator->() const
            {
                return mMem;
            }
    
            D& GetDeallocator()
            {
                return mDeallocator;
            }
        
        private:
			T* mMem;
            D mDeallocator;
            bool mHasOwnership;
	};

	template<typename T>
	class finalize_ptr_array
	{
		public:
			finalize_ptr_array(T* pMem = NULL) : mMem(pMem)
			{
			}

			~finalize_ptr_array()
			{
				delete[] (T*)(mMem);
			}

			T* get_ptr() const
			{
				return mMem;
			}
        
            void release()
            {
                mMem = NULL;
            }

            void reset(T* pMem)
            {
                delete[] (T*)(mMem);
                mMem = pMem;
            }

            finalize_ptr_array(const finalize_ptr_array& pPtr)
            : mMem(NULL)
            {
                reset(pPtr.get_ptr());
                (const_cast<finalize_ptr_array&>(pPtr)).release();
            }
    
            const finalize_ptr_array& operator=(const finalize_ptr_array& pPtr)
            {
                reset(pPtr.get_ptr());
                (const_cast<finalize_ptr_array&>(pPtr)).release();                
                
                return *this;
            }
    
		private:
    
			T* mMem;
	};

	#define FINALIZE_PTR(name, datatype, ptr) \
	finalize_ptr<datatype> name##_obj(ptr); \
	datatype* name = name##_obj.get_ptr();
	
	#define FINALIZE_PTR_ARRAY(name, datatype, ptr) \
	finalize_ptr_array<datatype> name##_obj(ptr); \
	datatype* name = name##_obj.get_ptr();
    
	#define FINALIZE_RESOURCE(name, acquisition, destruction) \
	class name \
	{ \
		public:	\
			name() \
			{ \
				acquisition; \
			} \
			~name() \
			{ \
				destruction; \
			} \
	} name##_obj;

#ifdef RECORD_LOCK_ACQUISITIONS
	#define FINALIZE_RESOURCE_PTR(name, ptrType, ptr, acquisition, destruction) \
	class name \
	{ \
		public:	\
			name(ptrType* pPtr) \
			{ \
				mPtr = pPtr; \
				mPtr->acquisition; \
                mPtr->RecordAcquisition(__FILE__, __LINE__); \
			} \
			~name() \
			{ \
                mPtr->ResetAcquisition(); \
				mPtr->destruction; \
			} \
		private: \
			ptrType* mPtr; \
	} name##_obj(ptr);
#else
	#define FINALIZE_RESOURCE_PTR(name, ptrType, ptr, acquisition, destruction) \
	class name \
	{ \
		public:	\
			name(ptrType* pPtr) \
			{ \
				mPtr = pPtr; \
				mPtr->acquisition; \
			} \
			~name() \
			{ \
				mPtr->destruction; \
			} \
		private: \
			ptrType* mPtr; \
	} name##_obj(ptr);
#endif
    
    class pmScopeTimer
    {
        public:
            pmScopeTimer(const char* pStr);
            ~pmScopeTimer();

        private:
            const char* mStr;
            double mStartTime;
    };
    
    #define SCOPE_TIMER(name, str) pmScopeTimer name##_obj(str);

#ifdef ENABLE_ACCUMULATED_TIMINGS
    class TIMER_IMPLEMENTATION_CLASS;

    class pmAccumulatedTimesSorter
    {
        public:
            typedef struct accumulatedData
            {
                double minTime;
                double maxTime;
                double actualTime;
                uint execCount;
            } accumulatedData;
    
            ~pmAccumulatedTimesSorter();
    
            void Insert(std::string& pName, double pAccumulatedTime, double pMinTime, double pMaxTime, double pActualTime, uint pExecCount);
    
            void Lock();
            void Unlock();
    
            void FlushLogs();
    
            static pmAccumulatedTimesSorter* GetAccumulatedTimesSorter();
    
        private:
            pmAccumulatedTimesSorter();

            ushort mMaxNameLength;
            bool mLogsFlushed;
            std::map<std::pair<double, std::string>, accumulatedData> mAccumulatedTimesMap;
            pthread_mutex_t mMutex;
    };
    
    class pmAccumulationTimer
    {
        public:
            pmAccumulationTimer(const std::string& pStr);
    
            void RegisterExec();
            void DeregisterExec(double pTime);

            void Lock();
            void Unlock();
    
            ~pmAccumulationTimer();

        private:
            void RecordElapsedTime();

            std::string mStr;
            double mMinTime, mMaxTime, mAccumulatedTime, mActualTime;
            uint mExecCount, mThreadCount;
            finalize_ptr<TIMER_IMPLEMENTATION_CLASS> mTimer;
            pthread_mutex_t mMutex;
    };

    class pmAccumulationTimerHelper
    {
        public:
            pmAccumulationTimerHelper(pmAccumulationTimer* pAccumulationTimer);
            ~pmAccumulationTimerHelper();
        
        private:
            pmAccumulationTimer* mAccumulationTimer;
            double mStartTime;
    };
    
    #define ACCUMULATION_TIMER(name, str) \
    static pmAccumulationTimer name##_obj(str); \
    pmAccumulationTimerHelper name##_helperTimer(&name##_obj);
#else
    #define ACCUMULATION_TIMER(name, str)    
#endif
    
    #ifdef TRACK_MUTEX_TIMINGS
        #define __LOCK_NAME__(name) (name)
        #define __STATIC_LOCK_NAME__(name) (name)
    #else
        #define __LOCK_NAME__(name) ()
        #define __STATIC_LOCK_NAME__(name)
    #endif
    
    template<typename G, typename D, typename T>
	class guarded_scoped_ptr
	{
        public:
            guarded_scoped_ptr(G* pGuard, D* pTerminus, T** pPtr, T* pMem = NULL) : mGuard(pGuard), mTerminus(pTerminus), mPtr(pPtr), mLockAcquired(false)
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                
                if(mPtr)
                    *mPtr = pMem;
            }
    
            void SetLockAcquired()
            {
                mLockAcquired = true;
            }
    
            ~guarded_scoped_ptr()
            {
                if(mLockAcquired)
                {
                    if(mPtr)
                    {
                        if(mTerminus)
                            mTerminus->Terminating(*mPtr);
                            
                        delete (T*)(*mPtr);
                        *mPtr = NULL;
                    }
                
                    mGuard->Unlock();
                }
                else
                {
                    FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                                
                    if(mPtr)
                    {
                        if(mTerminus)
                            mTerminus->Terminating(*mPtr);
                            
                        delete (T*)(*mPtr);
                        *mPtr = NULL;
                    }
                }
            }
            
        private:
            guarded_scoped_ptr(const guarded_scoped_ptr& pPtr);
            const guarded_scoped_ptr& operator=(const guarded_scoped_ptr& pPtr);
    
            G* mGuard;
            D* mTerminus;
            T** mPtr;
            bool mLockAcquired;
	};

    template<typename G, typename T>
    class guarded_ptr
    {
        public:
            guarded_ptr(G* pGuard, T** pPtr, T* pMem = NULL)
            : mGuard(pGuard)
            , mPtr(pPtr)
            {
                if((*mPtr) != pMem)
                {
                    FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                    
                    if(mPtr)
                        *mPtr = pMem;
                }
            }

            ~guarded_ptr()
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                            
                if(mPtr)
                    *mPtr = NULL;
            }
            
        private:
            guarded_ptr(const guarded_ptr& pPtr);
            const guarded_ptr& operator=(const guarded_ptr& pPtr);

            G* mGuard;
            T** mPtr;
    };
    
    template<typename functor>
    class scope_exit
    {
        public:
            scope_exit(functor& pFunctor)
            : mFunctor(pFunctor)
            {}
            
            ~scope_exit()
            {
                mFunctor();
            }
            
        private:
            functor& mFunctor;
    };
    
    template<typename G, typename T>
    class guarded_swapper
    {
        public:
            guarded_swapper(G* pGuard, T* pPtr, T pNewValue, T pOldValue)
            : mGuard(pGuard)
            , mPtr(pPtr)
            , mOldValue(pOldValue)
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());

                *mPtr = pNewValue;
            }
        
            ~guarded_swapper()
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                
                *mPtr = mOldValue;
            }
        
        private:
            G* mGuard;
            T* mPtr;
            T mOldValue;
    };

    template<typename G, typename functor1, typename functor2>
    class guarded_scope_functors
    {
        public:
            guarded_scope_functors(G* pGuard, const functor1& pFunctor1, const functor2& pFunctor2)
            : mGuard(pGuard)
            , mFunctor1(pFunctor1)
            , mFunctor2(pFunctor2)
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());

                mFunctor1();
            }
        
            ~guarded_scope_functors()
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                
                mFunctor2();
            }
        
        private:
            G* mGuard;
            const functor1& mFunctor1;
            const functor2& mFunctor2;
    };

	class selective_finalize_base
	{
		public:
			virtual void SetDelete(bool pDelete) = 0;
            virtual ~selective_finalize_base() {}
	};

    template<typename T>
    class selective_finalize_ptr : public selective_finalize_base
    {
        public:
            selective_finalize_ptr<T>(T* pMem) : mMem(pMem), mDeleteMem(true)
            {
            }

            ~selective_finalize_ptr()
            {
                if(mDeleteMem)
                    delete (T*)(mMem);
            }
        
            virtual void SetDelete(bool pDelete)
            {
                mDeleteMem = pDelete;
            }

            T* get_ptr()
            {
                    return mMem;
            }

        private:
            selective_finalize_ptr(const selective_finalize_ptr& pPtr);
            const selective_finalize_ptr& operator=(const selective_finalize_ptr& pPtr);

            T* mMem;
            bool mDeleteMem;
    };

    template<typename T>
    class selective_finalize_ptr_array : public selective_finalize_base
    {
        public:
            selective_finalize_ptr_array<T>(T* pMem) : mMem(pMem), mDeleteMem(true)
            {
            }

            ~selective_finalize_ptr_array()
            {
                if(mDeleteMem)
                    delete[] (T*)(mMem);
            }

            virtual void SetDelete(bool pDelete)
            {
                mDeleteMem = pDelete;
            }

            T* get_ptr()
            {
                return mMem;
            }

        private:
            selective_finalize_ptr_array(const selective_finalize_ptr_array& pPtr);
            const selective_finalize_ptr_array& operator=(const selective_finalize_ptr_array& pPtr);
    
            T* mMem;
            bool mDeleteMem;
    };

	class pmDestroyOnException
	{
		public:
			pmDestroyOnException();
			virtual ~pmDestroyOnException();

            void AddFreePtr(void* pPtr);
			void AddDeletePtr(selective_finalize_base* pDeletePtr);
    
			void SetDestroy(bool pDestroy);
			bool ShouldDelete();

		private:
            pmDestroyOnException(const pmDestroyOnException& pPtr);
            const pmDestroyOnException& operator=(const pmDestroyOnException& pPtr);
    
			bool mDestroy;
			std::vector<selective_finalize_base*> mDeletePtrs;
			std::vector<void*> mFreePtrs;
	};

	#define START_DESTROY_ON_EXCEPTION(blockName) //pmDestroyOnException blockName; try {
	#define FREE_PTR_ON_EXCEPTION(blockName, name, ptr) name = ptr; //blockName.AddFreePtr(name);
	#define DESTROY_PTR_ON_EXCEPTION(blockName, name, dataType, ptr) name = ptr; //selective_finalize_ptr<dataType> name##_obj(name); blockName.AddDeletePtr(&(name##_obj));
	#define DESTROY_PTR_ARRAY_ON_EXCEPTION(blockName, name, dataType, ptr) name = ptr; //selective_finalize_ptr_array<dataType> name##_obj(name); blockName.AddDeletePtr(&(name##_obj));
	#define END_DESTROY_ON_EXCEPTION(blockName) //blockName.SetDestroy(false); } catch(...) {throw;}

	#define SAFE_FREE(ptr) if(ptr) free(ptr);


    template<typename _InputIterator, typename _Filter_Function, typename _Command_Function>
    _Command_Function filtered_for_each(_InputIterator __first, _InputIterator __last, _Filter_Function __f1, _Command_Function __f2)
    {
        for(; __first != __last; ++__first)
        {
            if(__f1(*__first))
                __f2(*__first);
        }

        return std::move(__f2);
    }

    template<typename _InputIterator, typename _Filter_Function, typename _Command_Function>
    _Command_Function filtered_for_each_with_index(_InputIterator __first, _InputIterator __last, _Filter_Function __f1, _Command_Function __f2)
    {
        for(size_t index = 0, filteredIndex = 0; __first != __last; ++__first, ++index)
        {
            if(__f1(*__first))
            {
                __f2(*__first, index, filteredIndex);
                ++filteredIndex;
            }
        }
        
        return std::move(__f2);
    }

    template<typename _InputIterator, typename _Command_Function>
    _Command_Function for_each_with_index(_InputIterator __first, _InputIterator __last, _Command_Function __f)
    {
        for(size_t index = 0; __first != __last; ++__first, ++index)
            __f(*__first, index);
        
        return std::move(__f);
    }
    
    template<typename _Container, typename _Filter_Function, typename _Command_Function>
    _Command_Function filtered_for_each(_Container& __container, _Filter_Function __f1, _Command_Function __f2)
    {
        return filtered_for_each(__container.begin(), __container.end(), __f1, __f2);
    }
    
    template<typename _Container, typename _Filter_Function, typename _Command_Function>
    _Command_Function filtered_for_each_with_index(_Container& __container, _Filter_Function __f1, _Command_Function __f2)
    {
        return filtered_for_each_with_index(__container.begin(), __container.end(), __f1, __f2);
    }

    template<typename _Container, typename _Command_Function>
    _Command_Function for_each_with_index(_Container& __container, _Command_Function __f)
    {
        return for_each_with_index(__container.begin(), __container.end(), __f);
    }

    template<typename _Container, typename _Command_Function>
    _Command_Function for_each(_Container& __container, _Command_Function __f)
    {
        return std::for_each(__container.begin(), __container.end(), __f);
    }

    template<typename _InputIterator, typename _Command_Function>
    _Command_Function for_each_with_lookahead(_InputIterator __first, _InputIterator __last, _Command_Function __f)
    {
        if(__first == __last)   // no lookahead available
            std::move(__f);
        
        for(; __first != __last; ++__first)
        {
            _InputIterator __next = __first + 1;
            if(__next == __last)
                break;

            __f(*__first, *__next);
        }

        return std::move(__f);
    }

    template<typename _Container, typename _Command_Function>
    _Command_Function for_each_with_lookahead(_Container& __container, _Command_Function __f)
    {
        return for_each_with_lookahead(__container.begin(), __container.end(), __f);
    }

    template<typename _LockType, typename _Container, typename _Command_Function>
    _Command_Function for_each_while_locked(_LockType& __l, _Container& __container, _Command_Function __f)
    {
        FINALIZE_RESOURCE_PTR(dLock, _LockType, &__l, Lock(), Unlock());

        return std::for_each(__container.begin(), __container.end(), __f);
    }

    template<typename _InputIterator, typename _Command_Function, typename _Arg_Type>
    _Command_Function for_each_with_arg(_InputIterator __first, _InputIterator __last, _Command_Function __f, _Arg_Type& __arg)
    {
        for(; __first != __last; ++__first)
            __f(*__first, __arg);
        
        return std::move(__f);
    }

    template<typename _InputIterator1, typename _InputIterator2, typename _Command_Function>
    _Command_Function multi_for_each(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Command_Function __f)
    {
        DEBUG_EXCEPTION_ASSERT(std::distance(__first1 , __last1) <= std::distance(__first2, __last2));

        for(; __first1 != __last1; ++__first1, ++__first2)
            __f(*__first1, *__first2);
        
        return std::move(__f);
    }

    template<typename _InputIterator1, typename _InputIterator2, typename _Command_Function>
    _Command_Function multi_for_each_with_index(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2, _Command_Function __f)
    {
        DEBUG_EXCEPTION_ASSERT(std::distance(__first1 , __last1) <= std::distance(__first2, __last2));

        for(size_t index = 0; __first1 != __last1; ++__first1, ++__first2, ++index)
            __f(*__first1, *__first2, index);
        
        return std::move(__f);
    }

    template<typename _Container1, typename _Container2, typename _Command_Function>
    _Command_Function multi_for_each(_Container1& __container1, _Container2& __container2, _Command_Function __f)
    {
        return multi_for_each(__container1.begin(), __container1.end(), __container2.begin(), __container2.end(), __f);
    }

    template<typename _Container1, typename _Container2, typename _Command_Function>
    _Command_Function multi_for_each_with_index(_Container1& __container1, _Container2& __container2, _Command_Function __f)
    {
        return multi_for_each_with_index(__container1.begin(), __container1.end(), __container2.begin(), __container2.end(), __f);
    }

} // end namespace pm

#endif

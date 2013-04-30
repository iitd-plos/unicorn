
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

#ifndef __PM_DATA_TYPES__
#define __PM_DATA_TYPES__

#include "pmInternalDefinitions.h"

#include <stdlib.h>
#include <setjmp.h>

#include <vector>
#include <map>

namespace pm
{
	typedef unsigned short int ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

	class pmTask;
    class pmProcessingElement;
    class pmExecutionStub;

    class pmSubtaskTerminationCheckPointAutoPtr
    {
        public:
            pmSubtaskTerminationCheckPointAutoPtr(pmExecutionStub* pStub, ulong pSubtaskId);
            ~pmSubtaskTerminationCheckPointAutoPtr();
    
        private:
            pmExecutionStub* mStub;
            ulong mSubtaskId;
    };
    
    class pmJmpBufAutoPtr
    {
        public:
            pmJmpBufAutoPtr();
            ~pmJmpBufAutoPtr();

            void Reset(sigjmp_buf* pJmpBuf, pmExecutionStub* pStub, ulong pSubtaskId);
            void SetHasJumped();
        
        private:
            pmExecutionStub* mStub;
            ulong mSubtaskId;
            bool mHasJumped;
    };
    
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
            PRE_SUBTASK_EXECUTION,
            SUBTASK_EXECUTION,
            DATA_REDUCTION,
            DATA_REDISTRIBUTION,
            SHADOW_MEM_COMMIT,
            SUBTASK_STEAL_WAIT,
            SUBTASK_STEAL_SERVE,
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

    typedef struct pmSubtaskRange
    {
        pmTask* task;
        pmProcessingElement* originalAllottee;
        ulong startSubtask;
        ulong endSubtask;
    } pmSubtaskRange;
    
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
    
    template<typename T>
    class deleteDeallocator
    {
        public:
            void operator()(T* pMem)
            {
                delete pMem;
            }
    };
    
    template<typename T>
    class deleteArrayDeallocator
    {
        public:
            void operator()(T* pMem)
            {
                delete[] pMem;
            }
    };
    
	template<typename T, typename D = deleteDeallocator<T> >
	class finalize_ptr
	{
		public:
			finalize_ptr(T* pMem = NULL) : mMem(pMem)
			{
			}

			~finalize_ptr()
			{
				mDeallocator.operator()(mMem);
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
                mDeallocator.operator()(mMem);
                mMem = pMem;
            }        

            finalize_ptr(const finalize_ptr& pPtr)
            : mMem(NULL)
            {
                reset(pPtr.get_ptr());
                (const_cast<finalize_ptr&>(pPtr)).release();
            }
            
            const finalize_ptr& operator=(const finalize_ptr& pPtr)
            {
                reset(pPtr.get_ptr());
                (const_cast<finalize_ptr&>(pPtr)).release();
            
                return *this;
            }
    
            T* operator->()
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
            TIMER_IMPLEMENTATION_CLASS* mTimer;
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
            guarded_ptr(G* pGuard, T** pPtr, T* pMem = NULL) : mGuard(pGuard), mPtr(pPtr)
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                
                if(mPtr)
                    *mPtr = pMem;
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

	class selective_finalize_base
	{
		public:
			virtual void SetDelete(bool pDelete) = 0;
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

} // end namespace pm

#endif

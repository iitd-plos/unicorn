
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

#include <stdlib.h>
#include <vector>

namespace pm
{
	typedef unsigned short int ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

	class pmTask;
    class pmProcessingElement;

    typedef struct pmSubtaskRange
    {
        pmTask* task;
        pmProcessingElement* originalAllottee;
        ulong startSubtask;
        ulong endSubtask;
    } pmSubtaskRange;
    
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
				D()(mMem);
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
                D()(mMem);
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
    
        private:
			T* mMem;
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
    
    template<typename G, typename D, typename T>
	class guarded_scoped_ptr
	{
        public:
            guarded_scoped_ptr(G* pGuard, D* pTerminus, T** pPtr, T* pMem = NULL) : mGuard(pGuard), mTerminus(pTerminus), mPtr(pPtr)
            {
                FINALIZE_RESOURCE_PTR(dGuard, G, mGuard, Lock(), Unlock());
                
                if(mPtr)
                    *mPtr = pMem;
            }
            
            ~guarded_scoped_ptr()
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
            
        private:
            guarded_scoped_ptr(const guarded_scoped_ptr& pPtr);
            const guarded_scoped_ptr& operator=(const guarded_scoped_ptr& pPtr);
    
            G* mGuard;
            D* mTerminus;
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

	typedef class pmDestroyOnException
	{
		public:
			pmDestroyOnException() {mDestroy = true;}
			virtual ~pmDestroyOnException()
			{
				if(mDestroy)
				{
					size_t i, lSize;
					lSize = mFreePtrs.size();
					for(i=0; i<lSize; ++i)
						free(mFreePtrs[i]);
				}
			}

			void AddFreePtr(void* pPtr) {mFreePtrs.push_back(pPtr);}
			void AddDeletePtr(selective_finalize_base* pDeletePtr) {mDeletePtrs.push_back(pDeletePtr);}
			void SetDestroy(bool pDestroy)
			{
				mDestroy = pDestroy;
				if(!mDestroy)
				{
					size_t i, lSize;
					lSize = mDeletePtrs.size();
					for(i=0; i<lSize; ++i)
						mDeletePtrs[i]->SetDelete(false);
				}
			}

			bool shouldDelete() {return mDestroy;}

		private:
            pmDestroyOnException(const pmDestroyOnException& pPtr);
            const pmDestroyOnException& operator=(const pmDestroyOnException& pPtr);
    
			bool mDestroy;
			std::vector<selective_finalize_base*> mDeletePtrs;
			std::vector<void*> mFreePtrs;
	} pmDestroyOnException;

	#define START_DESTROY_ON_EXCEPTION(blockName) //pmDestroyOnException blockName; try {
	#define FREE_PTR_ON_EXCEPTION(blockName, name, ptr) name = ptr; //blockName.AddFreePtr(name);
	#define DESTROY_PTR_ON_EXCEPTION(blockName, name, dataType, ptr) name = ptr; //selective_finalize_ptr<dataType> name##_obj(name); blockName.AddDeletePtr(&(name##_obj));
	#define DESTROY_PTR_ARRAY_ON_EXCEPTION(blockName, name, dataType, ptr) name = ptr; //selective_finalize_ptr_array<dataType> name##_obj(name); blockName.AddDeletePtr(&(name##_obj));
	#define END_DESTROY_ON_EXCEPTION(blockName) //blockName.SetDestroy(false); } catch(...) {throw;}

	#define SAFE_FREE(ptr) if(ptr) free(ptr);

} // end namespace pm

#endif

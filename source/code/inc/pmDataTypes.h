
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

    typedef struct pmSubtaskRange
    {
        pmTask* task;
        ulong startSubtask;
        ulong endSubtask;
    } pmSubtaskRange;

	template<typename T>
	class finalize_ptr
	{
		public:
			finalize_ptr<T>(T* pMem) : mMem(pMem)
			{
			}

			~finalize_ptr()
			{
				delete (T*)(mMem);
			}

			T* get_ptr()
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
			finalize_ptr_array<T>(T* pMem) : mMem(pMem)
			{
			}

			~finalize_ptr_array()
			{
				delete[] (T*)(mMem);
			}

			T* get_ptr()
			{
				return mMem;
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

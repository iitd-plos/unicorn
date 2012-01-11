
#ifndef __PM_DATA_TYPES__
#define __PM_DATA_TYPES__

#include <vector>

namespace pm
{
	typedef unsigned short int ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;

	template<typename T>
	class finalize_ptr
	{
		public:
			finalize_ptr<T>(T* pMem) : mMem(pMem)
			{
			}

			~finalize_ptr()
			{
				delete mMem;
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
				delete[] mMem;
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
	} name##_obj();

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

	typedef class pmDestroyOnException
	{
		public:
			pmDestroyOnException() {mDestroy = false;}
			virtual ~pmDestroyOnException()
			{
				if(mDestroy)
				{
					size_t i;

					size_t lSize = mPtrs.size();
					for(i=0; i<lSize; ++i)
						delete mPtrs[i];

					lSize = mPtrArrays.size();
					for(i=0; i<lSize; ++i)
						delete[] mPtrArrays[i];

					lSize = mFreePtrs.size();
					for(i=0; i<lSize; ++i)
						free(mPtrs[i]);
				}
			}

			pmStatus AddPtr(void* pPtr) {mPtrs.push_back(pPtr);}
			pmStatus AddPtrArray(void* pPtrArray) {mPtrArrays.push_back(pPtrArray);}
			pmStatus AddFreePtr(void* pPtr) {mFreePtrs.push_back(pPtr);}

			bool SetDestroy(bool pDestroy) {mDestroy = pDestroy;}

		private:
			bool mDestroy;
			std::vector<void*> mPtrs;
			std::vector<void*> mPtrArrays;
			std::vector<void*> mFreePtrs;
	} pmDestroyOnException;

	#define START_DESTROY_ON_EXCEPTION(blockName) pmDestroyOnException blockName; try {
	#define FREE_PTR_ON_EXCEPTION(blockName, name, ptr) name = ptr; blockName.AddFreePtr(name);
	#define DESTROY_PTR_ON_EXCEPTION(blockName, name, ptr) name = ptr; blockName.AddPtr(name);
	#define DESTROY_PTR_ARRAY_ON_EXCEPTION(blockName, name, ptr) name = ptr; blockName.AddPtrArray(name);
	#define END_DESTROY_ON_EXCEPTION(blockName) } catch(pmException dException) {blockName.SetDestroy(true); throw dException;}

	#define SAFE_FREE(ptr) if(ptr) free(ptr);

} // end namespace pm

#endif

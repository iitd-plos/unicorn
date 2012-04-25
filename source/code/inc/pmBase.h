
#ifndef __PM_BASE__
#define __PM_BASE__

#include "pmDataTypes.h"
#include "pmPublicDefinitions.h"
#include "pmErrorDefinitions.h"
#include "pmInternalDefinitions.h"
#include <assert.h>

#include <iostream>

namespace pm
{

/**
 * \brief The base class of all PMLIB classes. Currently used for memory allocation overrides and platform specific library open/close/execution routines.
 * This class throws pmOutofMemoryException on memory allocation failure.
 */

class pmBase
{
	public:
		pmBase();
		virtual ~pmBase();

//		void* operator new (size_t pSize);		//implicitly declared as a static member function
//		void operator delete (void *pPtr);		//implicitly declared as a static member function
//		void* operator new [] (size_t pSize);	//implicitly declared as a static member function
//		void operator delete [] (void* pPtr);	//implicitly declared as a static member function

		void* OpenLibrary(char* pPath);
		pmStatus CloseLibrary(void* pLibHandle);
		void* GetExportedSymbol(void* pLibHandle, char* pSymbol);

		uint GetRandomInt(uint pMaxLimit);

	private:
		static void* AllocateMemory(size_t pSize);
		static void DeallocateMemory(void* pPtr);
};

} // end namespace pm

#endif

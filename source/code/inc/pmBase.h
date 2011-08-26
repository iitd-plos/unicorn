
#ifndef __PM_BASE__
#define __PM_BASE__

#include "pmErrorDefinitions.h"

namespace pm
{

/**
 * \brief The base class of all PMLIB classes. Currently used for memory allocation overrides and platform specific library open/close/execution routines.
 * This class throws pmOutofMemoryException on memory allocation failure.
 */

class pmBase
{
	public:
		void* operator new(size_t pSize);		//implicitly declared as a static member function
		void operator delete(void *pPtr);		//implicitly declared as a static member function
		void* operator new [] (size_t pSize);	//implicitly declared as a static member function
		void operator delete [] (void* pPtr);	//implicitly declared as a static member function

		void* OpenLibrary(char* pPath);
		pmStatus CloseLibrary(void* pLibHandle);
		void* GetExportedSymbol(void* pLibHandle, char* pSymbol);

	private:
		static void* AllocateMemory(size_t pSize);
		static void DeallocateMemory(void* pPtr);

};

} // end namespace pm

#endif

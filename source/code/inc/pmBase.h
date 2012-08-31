
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
    
        ulong GetIntegralCurrentTimeInSecs();
        double GetCurrentTimeInSecs();

	private:
		static void* AllocateMemory(size_t pSize);
		static void DeallocateMemory(void* pPtr);
};

} // end namespace pm

#endif

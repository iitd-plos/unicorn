
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

#include <stdlib.h>
#include <time.h>
#include "pmBase.h"

#ifdef UNIX
#include "dlfcn.h"	// For dlopen/dlclose/dlsym
#endif

namespace pm
{

pmBase::pmBase()
{
	srand((int)time(NULL));
}

pmBase::~pmBase()
{
}

//void* pmBase::operator new (size_t  pSize)
//{
//	return AllocateMemory(pSize);
//}
//
//void pmBase::operator delete (void *pPtr)
//{	
//	DeallocateMemory(pPtr);	
//}
//
//void* pmBase::operator new [] (size_t pSize)
//{
//	return AllocateMemory(pSize);
//}
//
//void pmBase::operator delete [] (void* pPtr)
//{
//	DeallocateMemory(pPtr);	
//}

void* pmBase::AllocateMemory(size_t pSize)
{
	void* lPtr = ::malloc(pSize);
	if(!lPtr)
		PMTHROW(pmOutOfMemoryException());

	return lPtr;
}

void pmBase::DeallocateMemory(void* pPtr)
{
	::free(pPtr);
}

void* pmBase::OpenLibrary(char* pPath)
{
	return dlopen(pPath, RTLD_LAZY | RTLD_LOCAL);
}

pmStatus pmBase::CloseLibrary(void* pLibHandle)
{
	if(pLibHandle)
	{
		if(dlclose(pLibHandle) != 0)
			PMTHROW(pmIgnorableException(pmIgnorableException::LIBRARY_CLOSE_FAILURE));
	}

	return pmSuccess;
}

void* pmBase::GetExportedSymbol(void* pLibHandle, char* pSymbol)
{
	if(!pLibHandle)
		return NULL;

	return dlsym(pLibHandle, pSymbol);
}

uint pmBase::GetRandomInt(uint pMaxLimit)
{
	return rand() % pMaxLimit;
}

};

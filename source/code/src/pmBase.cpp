
#include <stdlib.h>
#include <time.h>
#include "pmBase.h"
#include "dlfcn.h"	// For dlopen/dlclose/dlsym

namespace pm
{

pmBase::pmBase()
{
	srand(time(NULL));
}

pmBase::~pmBase()
{
}

void* pmBase::operator new (size_t  pSize)
{
	return AllocateMemory(pSize);
}

void pmBase::operator delete (void *pPtr)
{	
	DeallocateMemory(pPtr);	
}

void* pmBase::operator new [] (size_t pSize)
{
	return AllocateMemory(pSize);
}

void pmBase::operator delete [] (void* pPtr)
{
	DeallocateMemory(pPtr);	
}

void* pmBase::AllocateMemory(size_t pSize)
{
	void* lPtr = ::malloc(pSize);
	if(!lPtr)
		throw pmOutOfMemoryException();

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
			throw pmIgnorableException(pmIgnorableException::LIBRARY_CLOSE_FAILURE);
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
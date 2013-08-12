
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
#include TIMER_IMPLEMENTATION_HEADER
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
//	return pmBase::AllocateMemory(pSize);
//}
//
//void pmBase::operator delete (void *pPtr)
//{	
//	pmBase::DeallocateMemory(pPtr);
//}
//
//void* pmBase::operator new [] (size_t pSize)
//{
//	return pmBase::AllocateMemory(pSize);
//}
//
//void pmBase::operator delete [] (void* pPtr)
//{
//	pmBase::DeallocateMemory(pPtr);
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

uint pmBase::GetRandomInt(uint pMaxLimit)
{
	return rand() % pMaxLimit;
}

ulong pmBase::GetIntegralCurrentTimeInSecs()
{
	struct timeval lTimeVal;
	struct timezone lTimeZone;
    
	::gettimeofday(&lTimeVal, &lTimeZone);
    
	return (ulong)lTimeVal.tv_sec;
}

double pmBase::GetCurrentTimeInSecs()
{
	struct timeval lTimeVal;
	struct timezone lTimeZone;
    
	::gettimeofday(&lTimeVal, &lTimeZone);
    
	double lCurrentTime = ((double)(lTimeVal.tv_sec * 1000000 + lTimeVal.tv_usec))/1000000;
    
	return lCurrentTime;
}

};

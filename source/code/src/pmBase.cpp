
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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

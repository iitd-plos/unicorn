
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

#ifndef __PM_BASE__
#define __PM_BASE__

#include "pmDataTypes.h"
#include "pmPublicUtilities.h"
#include <assert.h>

#include <string>
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

		uint GetRandomInt(uint pMaxLimit);
    
        static ulong GetIntegralCurrentTimeInSecs();
        static double GetCurrentTimeInSecs();
    
		static void* AllocateMemory(size_t pSize);
		static void DeallocateMemory(void* pPtr);
private:
};

} // end namespace pm

#endif

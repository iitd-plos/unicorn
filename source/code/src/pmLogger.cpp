
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

#include "pmLogger.h"
#include "pmNetwork.h"

#define LOG_LEVEL pmLogger::MINIMAL

namespace pm
{

pmLogger* pmLogger::GetLogger()
{
	static pmLogger lLogger(LOG_LEVEL);
    return &lLogger;
}

pmLogger::pmLogger(logLevel pLogLevel)
    : mLogLevel(pLogLevel)
    , mHostId((uint)-1)	// Unknown initially
    , mResourceLock __LOCK_NAME__("pmLogger::mResourceLock")
{
}

pmLogger::~pmLogger()
{
    if(!mDeferredStream.str().empty())
        std::cerr << mDeferredStream.str().c_str() << std::endl;
}

pmStatus pmLogger::SetHostId(uint pHostId)
{
	mHostId = pHostId;

	return pmSuccess;
}
    
void pmLogger::PrintDeferredLog()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(!mDeferredStream.str().empty())
        std::cerr << mDeferredStream.str().c_str() << std::endl;
    
    ClearDeferredLog();
}
    
const std::stringstream& pmLogger::GetDeferredLogStream()
{
    return mDeferredStream;
}
    
void pmLogger::ClearDeferredLog()
{
    mDeferredStream.str(std::string()); // clear stream
}

pmStatus pmLogger::LogDeferred(logLevel pMsgLevel, logType pMsgType, const char* pMsg, bool pPrependHostName /* = false */)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(pPrependHostName)
    {
        mDeferredStream << std::endl;
        mDeferredStream << "PMLIB [Host " << mHostId << "] " << pMsg << std::endl;
    }
    else
    {
        mDeferredStream << pMsg << std::endl;
    }

	return pmSuccess;
}
    
pmStatus pmLogger::Log(logLevel pMsgLevel, logType pMsgType, const char* pMsg, bool pLeadingBlankLine /* = false */)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(pMsgLevel <= mLogLevel)
	{
		if(pMsgType == INFORMATION || pMsgType == WARNING)
		{
            if(pLeadingBlankLine)
                fprintf(stdout, "\n");
        
			fprintf(stdout, "PMLIB [Host %d] %s\n", mHostId, pMsg);
			fflush(stdout);
		}
		else
		{
            if(pLeadingBlankLine)
                fprintf(stderr, "\n");

            fprintf(stderr, "PMLIB [Host %d] %s\n", mHostId, pMsg);
			fflush(stderr);
		}
	}

	return pmSuccess;
}

} // end namespace pm





/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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




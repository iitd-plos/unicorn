
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

#include "pmLogger.h"
#include "pmNetwork.h"

#define LOG_LEVEL pmLogger::MINIMAL

namespace pm
{

pmLogger* pmLogger::mLogger = NULL;

pmLogger* pmLogger::GetLogger()
{
	return mLogger;
}

pmLogger::pmLogger(logLevel pLogLevel)
{
    if(mLogger)
        PMTHROW(pmFatalErrorException());
    
    mLogger = this;
    
	mHostId = (uint)-1;	// Unknown initially
	mLogLevel = pLogLevel;
}

pmLogger::~pmLogger()
{
}

pmStatus pmLogger::SetHostId(uint pHostId)
{
	mHostId = pHostId;

	return pmSuccess;
}

pmStatus pmLogger::Log(logLevel pMsgLevel, logType pMsgType, const char* pMsg)
{
	if(pMsgLevel <= mLogLevel)
	{
		if(pMsgType == INFORMATION)
		{
			fprintf(stdout, "PMLIB [Host %d] %s\n", mHostId, pMsg);
			fflush(stdout);
		}
		else
		{
			fprintf(stderr, "PMLIB [Host %d] %s\n", mHostId, pMsg);
			fflush(stderr);
		}
	}

	return pmSuccess;
}

} // end namespace pm




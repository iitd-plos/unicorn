
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




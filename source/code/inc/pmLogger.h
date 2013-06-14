
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

#ifndef __PM_LOGGER__
#define __PM_LOGGER__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <sstream>

namespace pm
{

/**
 * \brief The output/error logger
 */

class pmLogger : public pmBase
{
public:
    typedef enum logLevel
    {
        MINIMAL,
        DEATILED,
        DEBUG_INTERNAL	/* Internal use Only; Not for production builds */
    } logLevel;

    typedef enum logType
    {
        INFORMATION,
        WARNING,
        ERROR
    } logType;

    static pmLogger* GetLogger();

    pmStatus SetHostId(uint pHostId);

    void PrintDeferredLog();
    pmStatus LogDeferred(logLevel pMsgLevel, logType pMsgType, const char* pMsg, bool pPrependHostName = false);
    pmStatus Log(logLevel pMsgLevel, logType pMsgType, const char* pMsg, bool pLeadingBlankLine = false);
    
    const std::stringstream& GetDeferredLogStream();
    void ClearDeferredLog();

private:
    pmLogger(logLevel pLogLevel);
    virtual ~pmLogger();

    ushort mLogLevel;
    uint mHostId;
    std::stringstream mDeferredStream;
    
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

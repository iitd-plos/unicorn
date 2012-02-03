
#ifndef __PM_LOGGER__
#define __PM_LOGGER__

#include "pmPublicDefinitions.h"

namespace pm
{

/**
 * \brief The output/error logger
 */

class pmLogger
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
		pmStatus DestroyLogger();

		pmStatus SetHostId(uint pHostId);

		pmStatus Log(logLevel pMsgLevel, logType pMsgType, const char* pMsg);

	private:
		pmLogger(logLevel pLogLevel);
		virtual ~pmLogger();

		ushort mLogLevel;
		uint mHostId;
		static pmLogger* mLogger;
};

} // end namespace pm

#endif

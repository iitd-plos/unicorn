
#ifndef __PM_LOGGER__
#define __PM_LOGGER__

#include "pmBase.h"

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
		pmStatus DestroyLogger();

		pmStatus Log(logLevel pMsgLevel, logType pMsgType, const char* pMsg);

	private:
		pmLogger(logLevel pLogLevel);
		virtual ~pmLogger();

		ushort mLogLevel;
		static pmLogger* mLogger;
};

} // end namespace pm

#endif

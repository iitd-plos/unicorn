
#ifndef __PM_ERROR_DEFINITIONS__
#define __PM_ERROR_DEFINITIONS__

#include "pmPublicDefinitions.h"
#include "pmDataTypes.h"

namespace pm
{
	/** 
	 * Error code to brief error description mappings
	 * Error codes are defined in pmPublicDefinitions.h (inside pmStatus enum)
    */
	const char* pmErrorMessages[] =
	{
		"No Error",
		"Execution status unknown or can't be determined.",
		"Fatal error inside library. Can't continue.",
		"Error in PMLIB initialization",
		"Error in network initialization",
		"Error in shutting down network communications",
		"Index out of bounds",
		"PMLIB internal command object decoding failure",
		"Internal failure in threading library"
	};

	/**
	  * Exceptions thrown internally by PMLIB
	  * These are mapped to pmStatus errors and are not sent to applications
	  * All exceptions are required to implement GetStatusCode() method
	*/
	class pmException
	{
		public:
			virtual pmStatus GetStatusCode() = 0;
	};

	class pmInitExceptionMPI : pmException
	{
		public:
			pmStatus GetStatusCode() {return pmNetworkInitError;}
	};

	class pmInvalidCommandIdException : pmException
	{
		public:
			pmInvalidCommandIdException(ushort pCommandId) {mCommandId = pCommandId;}
			pmStatus GetStatusCode() {return pmInvalidCommand;}
		
		private:
			ushort mCommandId;
	};

	class pmFatalErrorException : pmException
	{
		public:
			pmStatus GetStatusCode() {return pmFatalError;}
	};

	class pmThreadFailure : pmException
	{
		public:
			typedef enum failureTypes
			{
				MUTEX_INIT_FAILURE,
				MUTEX_LOCK_FAILURE,
				MUTEX_UNLOCK_FAILURE,
				MUTEX_DESTROY_FAILURE,
				COND_VAR_INIT_FAILURE,
				COND_VAR_SIGNAL_FAILURE,
				COND_VAR_WAIT_FAILURE,
				COND_VAR_DESTROY_FAILURE,
				THREAD_CREATE_ERROR,
				THREAD_CANCEL_ERROR
			} failureTypes;

			pmThreadFailure(failureTypes pFailureId, int pErrorCode) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmThreadingLibraryFailure;}

		private:
			ushort mFailureId;
			int mErrorCode;
	};

} // end namespace pm

#endif

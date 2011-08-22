
#ifndef __PM_ERROR_DEFINITIONS__
#define __PM_ERROR_DEFINITIONS__

#include "pmPublicDefinitions.h"
#include "pmDataTypes.h"

namespace pm
{
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

	class pmMpiInitException : public pmException
	{
		public:
			pmStatus GetStatusCode() {return pmNetworkInitError;}
	};

	class pmInvalidCommandIdException : public pmException
	{
		public:
			pmInvalidCommandIdException(ushort pCommandId) {mCommandId = pCommandId;}
			pmStatus GetStatusCode() {return pmInvalidCommand;}
		
		private:
			ushort mCommandId;
	};

	class pmFatalErrorException : public pmException
	{
		public:
			pmStatus GetStatusCode() {return pmFatalError;}
	};

	class pmThreadFailureException : public pmException
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

			pmThreadFailureException(failureTypes pFailureId, int pErrorCode) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmThreadingLibraryFailure;}

		private:
			failureTypes mFailureId;
			int mErrorCode;
	};

	class pmUnknownMachineException : public pmException
	{
		public:
			pmUnknownMachineException(uint pMachineId) {mMachineId = pMachineId;}
			pmStatus GetStatusCode() {return pmInvalidIndex;}

		private:
			uint mMachineId;
	};

	class pmTimerException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				ALREADY_RUNNING,
				ALREADY_STOPPED,
				ALREADY_PAUSED,
				NOT_STARTED,
				NOT_PAUSED,
				INVALID_STATE
			} failureTypes;

			pmTimerException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmTimerFailure;}

		private:
			failureTypes mFailureId;
	};

	class pmMemoryException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				ALLOCATION_FAILED,
				MEM_ALIGN_FAILED,
				MEM_PROT_NONE_FAILED,
				SEGFAULT_HANDLER_INSTALL_FAILED,
				SEGFAULT_HANDLER_UNINSTALL_FAILED
			} failureTypes;

			pmMemoryException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmMemoryError;}

		private:
			failureTypes mFailureId;
	};

	class pmNetworkException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				SEND_ERROR,
				RECEIVE_ERROR,
				STATUS_TEST_ERROR,
				DUMMY_REQUEST_CREATION_ERROR,
				WAIT_ERROR,
				CANCELLATION_TEST_ERROR,
				INVALID_DUMMY_REQUEST_STATUS_ERROR,
				DUMMY_REQUEST_CANCEL_ERROR
			} failureTypes;

			pmNetworkException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmNetworkError;}

		private:
			failureTypes mFailureId;
	};

} // end namespace pm

#endif

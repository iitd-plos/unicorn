
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

#ifndef __PM_ERROR_DEFINITIONS__
#define __PM_ERROR_DEFINITIONS__

#ifdef DEBUG
#include <iostream>
#include <stdio.h>	// For sprintf
#endif

namespace pm
{

#ifdef DEBUG
#define PMTHROW(x) { \
			char lInteger[64]; \
			sprintf(lInteger, " %d\n", __LINE__); \
			std::string lStr("Generating Exception "); \
			lStr += __FILE__; \
			lStr += lInteger; \
			std::cout << lStr.c_str() << std::flush; \
            throw x; \
		}
#define PMTHROW_NODUMP(x) throw x;
#else
#define PMTHROW(x) throw x;
#define PMTHROW_NODUMP(x) throw x;
#endif

	/**
	  * Exceptions thrown internally by PMLIB
	  * These are mapped to pmStatus errors and are not sent to applications
	  * All exceptions are required to implement GetStatusCode() method
	*/
	class pmException
	{
		public:
			virtual pmStatus GetStatusCode() {return pmFatalError;}
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
				THREAD_CANCEL_ERROR,
				THREAD_AFFINITY_ERROR,
                SIGNAL_RAISE_ERROR,
                TLS_KEY_ERROR
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
			pmUnknownMachineException(uint pIndex) {mIndex = pIndex;}
			pmStatus GetStatusCode() {return pmInvalidIndex;}

		private:
			uint mIndex;
	};

	class pmUnknownDeviceException : public pmException
	{
		public:
			pmUnknownDeviceException(uint pIndex) {mIndex = pIndex;}
			pmStatus GetStatusCode() {return pmInvalidIndex;}

		private:
			uint mIndex;
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

	class pmVirtualMemoryException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				ALLOCATION_FAILED,
				MEM_ALIGN_FAILED,
				MEM_PROT_NONE_FAILED,
				MEM_PROT_RW_FAILED,
                MMAP_FAILED,
                MUNMAP_FAILED,
                SHM_OPEN_FAILED,
                SHM_UNLINK_FAILED,
                FTRUNCATE_FAILED,
				SEGFAULT_HANDLER_INSTALL_FAILED,
				SEGFAULT_HANDLER_UNINSTALL_FAILED
			} failureTypes;

			pmVirtualMemoryException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmMemoryError;}

		private:
			failureTypes mFailureId;
	};

	class pmNetworkException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				REQUEST_CREATION_ERROR,
				REQUEST_FREE_ERROR,
				SEND_ERROR,
				RECEIVE_ERROR,
				BROADCAST_ERROR,
				GLOBAL_BARRIER_ERROR,
				ALL2ALL_ERROR,
				STATUS_TEST_ERROR,
				DUMMY_REQUEST_CREATION_ERROR,
				WAIT_ERROR,
				TEST_ERROR,
				CANCELLATION_TEST_ERROR,
				INVALID_DUMMY_REQUEST_STATUS_ERROR,
				DATA_TYPE_REGISTRATION,
				DATA_PACK_ERROR,
				DATA_UNPACK_ERROR,
				PROBE_ERROR,
				GET_COUNT_ERROR,
				DUMMY_REQUEST_CANCEL_ERROR
			} failureTypes;

			pmNetworkException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmNetworkError;}

		private:
			failureTypes mFailureId;
	};

	class pmCallbackException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				INVALID_CHAIN_INDEX
			} failureTypes;

			pmCallbackException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmInvalidIndex;}

		private:
			failureTypes mFailureId;
	};

	class pmStubException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				INVALID_STUB_INDEX
			} failureTypes;

			pmStubException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmInvalidIndex;}

		private:
			failureTypes mFailureId;
	};

	class pmOutOfMemoryException : public pmException
	{
		public:
			pmOutOfMemoryException() {}
			pmStatus GetStatusCode() {return pmMemoryError;}

		private:
	};

	class pmMemoryFetchException : public pmException
	{
		public:
			pmMemoryFetchException() {}
			pmStatus GetStatusCode() {return pmMemoryError;}

		private:
	};

	class pmIgnorableException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				LIBRARY_CLOSE_FAILURE
			} failureTypes;

			pmIgnorableException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmIgnorableError;}

		private:
			failureTypes mFailureId;
	};

	class pmExceptionGPU : public pmException
	{
		public:
			typedef enum gpuTypes
			{
				NVIDIA_CUDA
			} gpuTypes;

			typedef enum failureTypes
			{
				LIBRARY_OPEN_FAILURE,
				RUNTIME_ERROR,
				UNDEFINED_SYMBOL
			} failureTypes;

			pmExceptionGPU(gpuTypes pIdGPU, failureTypes pFailureId) {mIdGPU = pIdGPU; mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmGraphicsCardError;}

		private:
			gpuTypes mIdGPU;
			failureTypes mFailureId;
	};

	class pmBeyondComputationalLimitsException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				MPI_MAX_MACHINES,
				MPI_MAX_TRANSFER_LENGTH,
                ARITHMETIC_OVERFLOW
			} failureTypes;

			pmBeyondComputationalLimitsException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmBeyondComputationalLimits;}

		private:
			failureTypes mFailureId;
	};

	class pmUnrecognizedMemoryException : public pmException
	{
		public:
			pmStatus GetStatusCode() {return pmUnrecognizedMemory;}

		private:
	};

	class pmInvalidKeyException : public pmException
	{
		public:
			pmStatus GetStatusCode() {return pmInvalidKey;}

		private:
	};

	class pmDataProcessingException : public pmException
	{
		public:
			typedef enum failureTypes
			{
				DATA_PACKING_FAILED,
				DATA_UNPACKING_FAILED
			} failureTypes;

			pmDataProcessingException(failureTypes pFailureId) {mFailureId = pFailureId;}
			pmStatus GetStatusCode() {return pmDataProcessingFailure;}

		private:
			failureTypes mFailureId;
	};

	class pmConfFileNotFoundException : public pmException
	{
    public:
        pmStatus GetStatusCode() {return pmConfFileNotFound;}
        
    private:
	};
    
    class pmPrematureExitException : public pmException
    {
    public:
        
    private:
    };

} // end namespace pm

#endif

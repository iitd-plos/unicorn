
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

#ifdef DUMP_EXCEPTION_BACKTRACE
#include <iostream>
#include <stdio.h>	// For sprintf
#include <execinfo.h>
#endif

namespace pm
{

typedef unsigned short int ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#ifdef EXIT_ON_EXCEPTION
#define EXCEPTION_ACTION(x) exit(1);
#else
#define EXCEPTION_ACTION(x) throw x;
#endif
    
#ifdef DUMP_EXCEPTION_BACKTRACE
#define PMTHROW(x) { \
			char dInteger[64], dHost[32]; \
            sprintf(dInteger, " %d", __LINE__); \
            sprintf(dHost, " [Host %d]\n", pmGetHostId()); \
			std::string dStr("Generating Exception "); \
			dStr += __FILE__; \
            dStr += dInteger; \
            dStr += dHost; \
			std::cout << dStr.c_str() << std::flush; \
            \
            void* dCallstack[128]; \
            int dBacktraceFrames = backtrace(dCallstack, 128); \
            char** dBacktraceStrs = backtrace_symbols(dCallstack, dBacktraceFrames); \
            for(int backtraceIndex = 0; backtraceIndex < dBacktraceFrames; ++backtraceIndex) \
                std::cout << dBacktraceStrs[backtraceIndex] << std::endl; \
            free(dBacktraceStrs); \
            EXCEPTION_ACTION(x); \
		}
#else
#define PMTHROW(x) EXCEPTION_ACTION(x);
#endif

#define PMTHROW_NODUMP(x) throw x;

	/**
	  * Exceptions thrown internally by PMLIB
	  * These are mapped to pmStatus errors and are not sent to applications
	  * All exceptions are required to implement GetStatusCode() method
	*/
	class pmException
	{
		public:
			virtual pmStatus GetStatusCode() {return pmFatalError;}
            virtual ~pmException() {}
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
				THREAD_JOIN_ERROR,
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
                INIT_ERROR,
                FINALIZE_ERROR,
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
				UNDEFINED_SYMBOL,
                DRIVER_VERSION_UNSUPPORTED
			} failureTypes;

            pmExceptionGPU(gpuTypes pIdGPU, failureTypes pFailureId, int cudaError = 0) {mIdGPU = pIdGPU; mFailureId = pFailureId; mCudaError = cudaError;}
			pmStatus GetStatusCode() {return pmGraphicsCardError;}
        
            failureTypes GetFailureId() {return mFailureId;}

		private:
			gpuTypes mIdGPU;
			failureTypes mFailureId;
            int mCudaError;
	};

	class pmExceptionOpenCL : public pmException
	{
		public:
			typedef enum failureTypes
			{
				LIBRARY_OPEN_FAILURE,
				RUNTIME_ERROR,
				UNDEFINED_SYMBOL,
                NO_OPENCL_DEVICES,
                DEVICE_FISSION_UNAVAILABLE,
                DEVICE_COUNT_MISMATCH
			} failureTypes;

            pmExceptionOpenCL(failureTypes pFailureId, int openCLError = 0) {mFailureId = pFailureId; mOpenCLError = openCLError;}
			pmStatus GetStatusCode() {return pmGraphicsCardError;}
        
            failureTypes GetFailureId() {return mFailureId;}

		private:
			failureTypes mFailureId;
            int mOpenCLError;
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
        pmPrematureExitException(bool pSubtaskLockAcquired)
        : mSubtaskLockAcquired(pSubtaskLockAcquired)
        {}
        
        bool IsSubtaskLockAcquired()
        {
            return mSubtaskLockAcquired;
        }
        
    private:
        bool mSubtaskLockAcquired;
    };

	class pmUserErrorException : public pmException
	{
		public:
			pmStatus GetStatusCode() {return pmUserError;}

		private:
	};

} // end namespace pm

#endif

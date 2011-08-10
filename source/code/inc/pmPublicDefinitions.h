
#ifndef __PM_PUBLIC_DEFINITIONS__
#define __PM_PUBLIC_DEFINITIONS__

/** 
 *						PARTITIONED MEMORY LIBRARY
 * PMLIB namespace. All PMLIB definitions are present in this namespace.
*/
namespace pm
{
	/** 
	 * This enumeration defines success and all error conditions for the PMLIB Application Programming Interface (API).
	 * Applications can depend upon these status flags to know the outcome of PMLIB functions.
	 * Applications may use pmGetLastError for a brief description of the error.
	*/
	typedef enum pmStatus
	{
		pmSuccess = 0,
		pmStatusUnavailable,
		pmFatalError,
		pmInitializationFailure,
		pmNetworkInitError,
		pmNetworkTerminationError,
		pmInvalidIndex,
		pmInvalidCommand,
		pmThreadingLibraryFailure,
		pmTimerFailure,
		pmMaxStatusValues
	} pmStatus;

	/** This function returns a brief description of the last error (if any) caused by execution of any PMLIB function */
	const char* pmGetLastError();

	/** This function initializes the PMLIB library. It must be the first PMLIB API called on all machines under MPI cluster. */
	pmStatus pmInitialize();

	/** This function marks the termination of use of PMLIB in an application. This must be the application's last call to PMLIB. */
	pmStatus pmFinalize();

} // end namespace pm

#endif
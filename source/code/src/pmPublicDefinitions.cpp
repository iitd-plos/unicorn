
#include "pmInternalDefinitions.h"
#include "pmErrorDefinitions.h"

#include "pmController.h"

namespace pm
{

/** 
 * This file defines all functions exported to applications.
 * All functions in this file must be wrapped inside try/catch blocks
 * and converted to pmStatus errors while reporting to the application.
 * No exception is ever sent to the applications (for C compatibility)
*/

#define SAFE_GET_CONTROLLER(x) { x = pmController::GetController(); if(!x) return pmInitializationFailure; }

const char* pmGetLastError()
{
	uint lErrorCode = pmSuccess;

	try
	{
		pmController* lController = pmController::GetController();
		if(!lController)
			return pmErrorMessages[pmInitializationFailure];

		lErrorCode = lController->GetLastErrorCode();

		if(lErrorCode >= pmMaxStatusValues)
			return pmErrorMessages[pmSuccess];
	}
	catch(pmException&)
	{}

	return pmErrorMessages[lErrorCode];
}

pmStatus pmInitialize()
{
	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController)		// Initializes the variable lController; If no controller returns error
	}
	catch(pmException& e)
	{
		return e.GetStatusCode();
	}

	return pmSuccess;
}

pmStatus pmFinalize()
{
	pmStatus lStatus = pmSuccess;

	try
	{
		pmController* lController;
		SAFE_GET_CONTROLLER(lController)		// Initializes the variable lController; If no controller returns error

		lStatus = lController->DestroyController();
	}
	catch(pmException& e)
	{
		return e.GetStatusCode();
	}

	return lStatus;
}

} // end namespace pm

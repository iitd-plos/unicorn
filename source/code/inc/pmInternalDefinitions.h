
#ifndef __PM_INTERNAL_DEFINITIONS__
#define __PM_INTERNAL_DEFINITIONS__

#include "pmErrorDefinitions.h"
#include "pmDataTypes.h"

#include <stdio.h>
#include <iostream>

/** 
 * The actual implementations to be used in the build for abstract factory based classes
*/
#define NETWORK_IMPLEMENTATION_CLASS pmMPI
#define CLUSTER_IMPLEMENTATION_CLASS pmMPICluster
#define THREADING_IMPLEMENTATION_CLASS pmPThread	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SAFE_PRIORITY_QUEUE_IMPLEMENTATION_CLASS pmPThreadPQ	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SIGNAL_WAIT_IMPLEMENTATION_CLASS pmPThreadSignalWait	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER

#define THREADING_IMPLEMENTATION_HEADER <pthread.h>	// Used alongwith THREADING_IMPLEMENTATION_CLASS, SAFE_PRIORITY_QUEUE_IMPLEMENTATION_CLASS & SIGNAL_WAIT_IMPLEMENTATION_CLASS

#define THROW_ON_NON_ZERO_RET_VAL(x, y, z) { int __ret_val__ = x; if(!__ret_val__) throw y(z, __ret_val__); }

#endif
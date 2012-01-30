
#ifndef __PM_INTERNAL_DEFINITIONS__
#define __PM_INTERNAL_DEFINITIONS__

#include <stdio.h>
#include <iostream>

#define PMLIB_VERSION "1.0.0" /* Format MajorVersion_MinorVersion_Update */

#define SYSTEM_CONFIGURATION_HEADER <unistd.h>

/** 
 * The actual implementations to be used in the build for abstract factory based classes
*/
#define NETWORK_IMPLEMENTATION_CLASS pmMPI
#define CLUSTER_IMPLEMENTATION_CLASS pmClusterMPI

#ifdef UNIX

#define TIMER_IMPLEMENTATION_CLASS pmLinuxTimer
#define TIMER_IMPLEMENTATION_HEADER <sys/time.h>

#define VM_IMPLEMENTATION_HEADER1 <sys/mman.h>
#define VM_IMPLEMENTATION_HEADER2 <signal.h>
#define VM_IMPLEMENTATION_HEADER3 <unistd.h>

#define MEMORY_MANAGER_IMPLEMENTATION_CLASS pmLinuxMemoryManager

#define THREADING_IMPLEMENTATION_CLASS pmPThread	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SIGNAL_WAIT_IMPLEMENTATION_CLASS pmPThreadSignalWait	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define RESOURCE_LOCK_IMPLEMENTATION_CLASS pmPThreadResourceLock	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define THREADING_IMPLEMENTATION_HEADER <pthread.h>	// Used alongwith THREADING_IMPLEMENTATION_CLASS, SIGNAL_WAIT_IMPLEMENTATION_CLASS & RESOURCE_LOCK_IMPLEMENTATION_CLASS

#else

#define TIMER_IMPLEMENTATION_CLASS pmWinTimer
#define TIMER_IMPLEMENTATION_HEADER 

#define VM_IMPLEMENTATION_HEADER1 
#define VM_IMPLEMENTATION_HEADER2 
#define VM_IMPLEMENTATION_HEADER3 

#define MEMORY_MANAGER_IMPLEMENTATION_CLASS pmWinMemoryManager

#define THREADING_IMPLEMENTATION_CLASS pmWinThread	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SIGNAL_WAIT_IMPLEMENTATION_CLASS pmWinThreadSignalWait	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define RESOURCE_LOCK_IMPLEMENTATION_CLASS pmWinThreadResourceLock	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define THREADING_IMPLEMENTATION_HEADER 	// Used alongwith THREADING_IMPLEMENTATION_CLASS, SIGNAL_WAIT_IMPLEMENTATION_CLASS & RESOURCE_LOCK_IMPLEMENTATION_CLASS

#endif

#define THROW_ON_NON_ZERO_RET_VAL(x, y, z) { int __ret_val__ = x; if(__ret_val__) throw y(z, __ret_val__); }

/* This code is taken from http://www.fefe.de/intof.html */
#define __HALF_MAX_SIGNED(type) ((type)1 << (sizeof(type)*8-2))
#define __MAX_SIGNED(type) (__HALF_MAX_SIGNED(type) - 1 + __HALF_MAX_SIGNED(type))
#define __MIN_SIGNED(type) (-1 - __MAX_SIGNED(type))

#define __MIN(type) ((type)-1 < 1?__MIN_SIGNED(type):(type)0)
#define __MAX(type) ((type)~__MIN(type))
///////////////////////////////////////////////////////////

#define MPI_TRANSFER_MAX_LIMIT __MAX(int)

#define MAX_CONTROL_PRIORITY 0
#define MAX_PRIORITY_LEVEL MAX_CONTROL_PRIORITY+1	// 0 is used for control messages
#define MIN_PRIORITY_LEVEL __MAX(ushort)
#define CONTROL_EVENT_PRIORITY 0
#define DEFAULT_PRIORITY_LEVEL 5

#define DEFAULT_SCHEDULING_MODEL pmScheduler::PUSH

#define TRACK_MEMORY_ALLOCATIONS

#define SLOW_START_SCHEDULING_INITIAL_SUBTASK_COUNT 1	 // must be a power of 2
#define SLOW_START_SCHEDULING_UPPER_LIMIT_EXEC_TIME_PER_ALLOCATION 15	// in seconds
#define SLOW_START_SCHEDULING_LOWER_LIMIT_EXEC_TIME_PER_ALLOCATION 8	// in seconds

#define MAX_STEAL_ATTEMPTS 25

#define PROPAGATE_FAILURE_RET_STATUS(x) {pmStatus dRetStatus = x; if(dRetStatus != pmSuccess) return dRetStatus;}

#define GET_VM_PAGE_START_ADDRESS(memAddr, pageSize) (memAddr - (reinterpret_cast<size_t>(memAddr) % pageSize))

#define USE_LAZY_MEMORY

#endif


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
#define STANDARD_ERROR_HEADER <sys/errno.h>

#define VM_IMPLEMENTATION_HEADER1 <sys/mman.h>
#define VM_IMPLEMENTATION_HEADER2 <signal.h>
#define VM_IMPLEMENTATION_HEADER3 <unistd.h>

#define MEMORY_MANAGER_IMPLEMENTATION_CLASS pmLinuxMemoryManager

#define THREADING_IMPLEMENTATION_CLASS pmPThread	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define TLS_IMPLEMENTATION_CLASS pmPThreadTls	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SIGNAL_WAIT_IMPLEMENTATION_CLASS pmPThreadSignalWait	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define RESOURCE_LOCK_IMPLEMENTATION_CLASS pmPThreadResourceLock	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define THREADING_IMPLEMENTATION_HEADER <pthread.h>	// Used alongwith THREADING_IMPLEMENTATION_CLASS, SIGNAL_WAIT_IMPLEMENTATION_CLASS & RESOURCE_LOCK_IMPLEMENTATION_CLASS

#else

#define TIMER_IMPLEMENTATION_CLASS pmWinTimer
#define TIMER_IMPLEMENTATION_HEADER 
#define STANDARD_ERROR_HEADER

#define VM_IMPLEMENTATION_HEADER1 
#define VM_IMPLEMENTATION_HEADER2 
#define VM_IMPLEMENTATION_HEADER3 

#define MEMORY_MANAGER_IMPLEMENTATION_CLASS pmWinMemoryManager

#define THREADING_IMPLEMENTATION_CLASS pmWinThread	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define SIGNAL_WAIT_IMPLEMENTATION_CLASS pmWinThreadSignalWait	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define RESOURCE_LOCK_IMPLEMENTATION_CLASS pmWinThreadResourceLock	// Define the implementation header in THREADING_IMPLEMENTATION_HEADER
#define THREADING_IMPLEMENTATION_HEADER 	// Used alongwith THREADING_IMPLEMENTATION_CLASS, SIGNAL_WAIT_IMPLEMENTATION_CLASS & RESOURCE_LOCK_IMPLEMENTATION_CLASS

#endif

#define THROW_ON_NON_ZERO_RET_VAL(x, y, z) { int __ret_val__ = x; if(__ret_val__) PMTHROW(y(z, __ret_val__)); }

/* This code is taken from http://www.fefe.de/intof.html */
#define __HALF_MAX_SIGNED(type) ((type)1 << (sizeof(type)*8-2))
#define __MAX_SIGNED(type) (__HALF_MAX_SIGNED(type) - 1 + __HALF_MAX_SIGNED(type))
#define __MIN_SIGNED(type) (-1 - __MAX_SIGNED(type))

#define __MIN(type) ((type)-1 < 1?__MIN_SIGNED(type):(type)0)
#define __MAX(type) ((type)~__MIN(type))
///////////////////////////////////////////////////////////

#define MPI_TRANSFER_MAX_LIMIT __MAX(int)

const unsigned short RESERVED_PRIORITY = 0;
const unsigned short MAX_CONTROL_PRIORITY = RESERVED_PRIORITY+1;
const unsigned short MAX_PRIORITY_LEVEL = MAX_CONTROL_PRIORITY+1;	// 0 is used for control messages
const unsigned short MIN_PRIORITY_LEVEL = __MAX(unsigned short);
const unsigned short DEFAULT_PRIORITY_LEVEL = MAX_PRIORITY_LEVEL;

const unsigned short TASK_MULTI_ASSIGN_FLAG_VAL = 0x0001;   // LSB
const unsigned long MEMORY_TRANSFER_TIMEOUT = 3;    // in secs

#define DEFAULT_SCHEDULING_MODEL scheduler::PUSH

#define SLOW_START_SCHEDULING_INITIAL_SUBTASK_COUNT 1	 // must be a power of 2
#define SLOW_START_SCHEDULING_UPPER_LIMIT_EXEC_TIME_PER_ALLOCATION 15	// in seconds
#define SLOW_START_SCHEDULING_LOWER_LIMIT_EXEC_TIME_PER_ALLOCATION 8	// in seconds

#define MAX_SUBTASK_MULTI_ASSIGN_COUNT 3    // Max no. of devices to which a subtask may be assigned at any given time
#define MAX_STEAL_CYCLES_PER_DEVICE 5   // Max no. of steal attempts from a device to any other device

#define PROPAGATE_FAILURE_RET_STATUS(x) {pmStatus dRetStatus = x; if(dRetStatus != pmSuccess) return dRetStatus;}

#define GET_VM_PAGE_START_ADDRESS(memAddr, pageSize) (memAddr - (memAddr % pageSize))

#define SUPPORT_LAZY_MEMORY

#ifdef SUPPORT_LAZY_MEMORY
    #define LAZY_FORWARD_PREFETCH_PAGE_COUNT 5
#endif

//#define PROGESSIVE_SLEEP_NETWORK_THREAD

#ifdef PROGRESSIVE_SLEEP_NETWORK_THREAD
#define MIN_PROGRESSIVE_SLEEP_TIME_MILLI_SECS 0
#define PROGRESSIVE_SLEEP_TIME_INCREMENT_MILLI_SECS 500
#define MAX_PROGRESSIVE_SLEEP_TIME_MILLI_SECS 5000
#endif

#define PROPORTIONAL_SCHEDULING_CONF_FILE "propSchedConf.txt"

/* Diagnostics */
//#define TRACK_MEMORY_ALLOCATIONS
//#define TRACK_MEMORY_REQUESTS
//#define TRACK_SUBTASK_EXECUTION
//#define TRACK_SUBTASK_EXECUTION_VERBOSE
//#define TRACK_SUBTASK_STEALS
//#define TRACK_MULTI_ASSIGN
//#define DUMP_THREADS
//#define ENABLE_TASK_PROFILING
//#define ENABLE_MEM_PROFILING
//#define DUMP_SHADOW_MEM
//#define DUMP_NETWORK_STATS
//#define DUMP_TASK_EXEC_STATS
//#define DUMP_SCHEDULER_EVENT
//#define DUMP_EVENT_TIMELINE
#define DUMP_SUBTASK_EXECUTION_PROFILE
//#define DUMP_MPI_CALLS

#ifdef DUMP_EVENT_TIMELINE
    #define SERIALIZE_DEFERRED_LOGS
#endif

#endif




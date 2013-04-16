
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

#ifndef __PM_PUBLIC_UTILITIES__
#define __PM_PUBLIC_UTILITIES__

/** 
 *						PARTITIONED MEMORY LIBRARY
 * PMLIB namespace. All PMLIB definitions are present in this namespace.
 * ***************  Public include file for PMLIB utils  ***************
*/

namespace pm
{
    typedef enum pmReductionType
    {
        REDUCE_ADD,
        MAX_REDUCTION_TYPES
    } pmReductionType;

    /** The following functions can be used for optimal inbuilt reduction of two subtasks. Various flavors differ in the data type they process */
    pmStatus pmReduceInts(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType);

    pmStatus pmReduceUInts(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType);

    pmStatus pmReduceLongs(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType);

    pmStatus pmReduceULongs(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType);

    pmStatus pmReduceFloats(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmReductionType pReductionType);

    const size_t MAX_FILE_SIZE_LEN = 2048;

    /** This function memory maps an entire file specified by pPath on all machines in the cluster.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmMapFile(const char* pPath);

    /** This function returns the starting address of the file specified by pPath and memory mapped by the call pmMapFile.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    void* pmGetMappedFile(const char* pPath);

    /** This function unmaps the file specified by pPath and mapped by the call pmMapFile from all machines in the cluster.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmUnmapFile(const char* pPath);

}   // end namespace pm

#endif
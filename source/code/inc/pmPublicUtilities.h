
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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
    typedef enum pmReductionOpType
    {
        REDUCE_ADD,
        REDUCE_MIN,
        REDUCE_MAX,
        REDUCE_PRODUCT,
        REDUCE_LOGICAL_AND,
        REDUCE_BITWISE_AND,
        REDUCE_LOGICAL_OR,
        REDUCE_BITWISE_OR,
        REDUCE_LOGICAL_XOR,
        REDUCE_BITWISE_XOR,
        MAX_REDUCTION_OP_TYPES
    } pmReductionOpType;

    typedef enum pmReductionDataType
    {
        REDUCE_INTS,
        REDUCE_UNSIGNED_INTS,
        REDUCE_LONGS,
        REDUCE_UNSIGNED_LONGS,
        REDUCE_FLOATS,
        REDUCE_DOUBLES,
        MAX_REDUCTION_DATA_TYPES
    } pmReductionDataType;

    /** The following function can be used for optimal inbuilt reduction of two subtasks. */
    pmDataReductionCallback pmGetSubtaskReductionCallbackImpl(pmReductionOpType pOperation, pmReductionDataType pDataType);
    
    /**  The following function can be called from within a custom implementation of pmDataReductionCallback. */
    pmStatus pmReduceSubtasks(pmTaskHandle pTaskHandle, pmDeviceHandle pDevice1Handle, unsigned long pSubtask1Id, pmSplitInfo& pSplitInfo1, pmDeviceHandle pDevice2Handle, unsigned long pSubtask2Id, pmSplitInfo& pSplitInfo2, pmReductionOpType pOperation, pmReductionDataType pDataType);

    const size_t MAX_FILE_SIZE_LEN = 2048;

    /** This function returns the starting address of the file specified by pPath and memory mapped by the call pmMapFile.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    void* pmGetMappedFile(const char* pPath);

    /** This function memory maps an entire file specified by pPath on all machines in the cluster.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmMapFile(const char* pPath);

    /** This function unmaps the file specified by pPath and mapped by the call pmMapFile(s) from all machines in the cluster.
     The number of bytes in pPath must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmUnmapFile(const char* pPath);

    /** This function memory maps pFileCount files specified by pPaths on all machines in the cluster.
     The number of bytes in each pPaths entry must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmMapFiles(const char* const* pPaths, uint pFileCount);

    /** This function unmaps pFileCount files specified by pPaths and mapped by the call pmMapFile(s) from all machines in the cluster.
     The number of bytes in each pPaths entry must be less than MAX_FILE_SIZE_LEN. */
    pmStatus pmUnmapFiles(const char* const* pPaths, uint pFileCount);

}   // end namespace pm

#endif
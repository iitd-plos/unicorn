
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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

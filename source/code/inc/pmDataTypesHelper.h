
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

#ifndef __PM_DATA_TYPES_HELPER__
#define __PM_DATA_TYPES_HELPER__

namespace pm
{

#ifdef ENABLE_TASK_PROFILING
class pmTaskProfiler;

namespace taskProfiler
{
    enum profileType
    {
        INPUT_MEMORY_TRANSFER,
        OUTPUT_MEMORY_TRANSFER,
        TOTAL_MEMORY_TRANSFER,    /* For internal use only */
        DATA_PARTITIONING,
        SUBTASK_EXECUTION,
        LOCAL_DATA_REDUCTION,
        REMOTE_DATA_REDUCTION,
        DATA_REDISTRIBUTION,
        SHADOW_MEM_COMMIT,
        SUBTASK_STEAL_WAIT,
        SUBTASK_STEAL_SERVE,
        STUB_WAIT_ON_NETWORK,
        COPY_TO_PINNED_MEMORY,
        COPY_FROM_PINNED_MEMORY,
        CUDA_COMMAND_PREPARATION,
        PREPROCESSOR_TASK_EXECUTION,
        AFFINITY_SUBTASK_MAPPINGS,
        AFFINITY_USE_OVERHEAD,
        FLUSH_MEMORY_OWNERSHIPS,
        NETWORK_DATA_COMPRESSION,
        GPU_DATA_COMPRESSION,
        UNIVERSAL, /* For internal use only */
        MAX_PROFILE_TYPES
    };
}

class pmRecordProfileEventAutoPtr
{
public:
    pmRecordProfileEventAutoPtr(pmTaskProfiler* pTaskProfiler, taskProfiler::profileType pProfileType);
    ~pmRecordProfileEventAutoPtr();
    
private:
    pmTaskProfiler* mTaskProfiler;
    taskProfiler::profileType mProfileType;
};
#endif
    
#ifdef DUMP_DATA_COMPRESSION_STATISTICS
class pmCompressionDataRecorder
{
public:
    static void RecordCompressionData(ulong pUncompresedSize, ulong pCompressedSize, bool pIsDataForNetworkTransfer);
};
#endif


} // end namespace pm

#endif
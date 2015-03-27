
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

#ifndef __PM_AFFINITY_TABLE__
#define __PM_AFFINITY_TABLE__

#include "pmBase.h"
#include "pmTable.h"

#include <set>

namespace pm
{

class pmLocalTask;
class pmMachine;
class pmAddressSpace;

class pmAffinityTable : public pmBase
{
public:
    pmAffinityTable(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion);
    
    void PopulateAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector);
    void CreateSubtaskMappings();
    
#ifdef USE_AFFINITY_IN_STEAL
    static std::vector<ulong> FindSubtasksWithBestAffinity(pmTask* pTask, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, const pmMachine* pMachine);
    static std::vector<ulong> FindSubtasksWithMaxDifferenceInAffinities(pmTask* pTask, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, const pmMachine* pMachine1, const pmMachine* pMachine2);
#endif
    
#ifdef USE_DYNAMIC_AFFINITY
    static ulong GetSubtaskWithBestAffinity(pmTask* pTask, pmExecutionStub* pStub, const std::vector<ulong>& pSubtasks, std::shared_ptr<void>& pSharedPtr, bool pUpdate);
#endif

private:
    template<typename T, typename S>
    void MakeAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, T pSentinelValue);

#ifdef USE_AFFINITY_IN_STEAL
    template<typename T, typename S>
    static std::vector<ulong> FindSubtasksWithBestAffinity(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex, ulong pSubtaskCount);

    template<typename T, typename S, typename D>
    static std::vector<ulong> FindSubtasksWithMaxDifferenceInAffinities(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex1, size_t pMachineIndex2, ulong pSubtaskCount);
#endif

    pmLocalTask* mLocalTask;
    pmAffinityCriterion mAffinityCriterion;
    
#ifdef MACHINES_PICK_BEST_SUBTASKS
    pmTable<uint, std::vector<ulong>> mTable; // machine index versus subtasks in order of preference
#else
    pmTable<ulong, std::vector<const pmMachine*>> mTable; // subtask id versus machines in order of preference
#endif
};

} // end namespace pm

#endif

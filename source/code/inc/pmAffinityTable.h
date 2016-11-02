
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

#ifndef __PM_AFFINITY_TABLE__
#define __PM_AFFINITY_TABLE__

#include "pmBase.h"
#include "pmTable.h"

#include <set>
#include <cmath>

namespace pm
{

class pmLocalTask;
class pmMachine;
class pmAddressSpace;
    
struct derivedAffinityData
{
//    ulong localBytes;
//    uint remoteNodes;
    ulong remoteEvents;
    float estimatedTime;
    
    derivedAffinityData operator- (const derivedAffinityData& pData)
    {
        derivedAffinityData lData;
//        lData.localBytes = localBytes - pData.localBytes;
//        lData.remoteNodes = remoteNodes - pData.remoteNodes;
        lData.remoteEvents = remoteEvents - pData.remoteEvents;
        lData.estimatedTime = estimatedTime - pData.estimatedTime;
        
        return lData;
    }
};

std::ostream& operator<< (std::ostream& pOStream, const derivedAffinityData& pData);

struct derivedAffinityDataSorter : std::binary_function<derivedAffinityData, derivedAffinityData, bool>
{
    bool operator()(const derivedAffinityData& pData1, const derivedAffinityData& pData2) const
    {
        const float lPercent = 10.0/100.0;

    #if 0
        if(std::abs(pData1.estimatedTime - pData2.estimatedTime) < lPercent * std::max(pData1.estimatedTime, pData2.estimatedTime))
        {
            if(std::abs((float)pData1.remoteEvents - (float)pData2.remoteEvents) < lPercent * std::max(pData1.remoteEvents, pData2.remoteEvents))
                return pData1.remoteNodes < pData2.remoteNodes;

            return pData1.remoteEvents < pData2.remoteEvents;
        }
        
        return pData1.estimatedTime < pData2.estimatedTime;
    #else
        // If the subtask is better on both criterion, select it
        if(pData1.remoteEvents < pData2.remoteEvents && pData1.estimatedTime < pData2.estimatedTime)
            return true;
        
        if(pData2.remoteEvents < pData1.remoteEvents && pData2.estimatedTime < pData1.estimatedTime)
            return false;

        // If the subtask is better on one criterion and within a percentage on other, select it
        bool lBetter1 = (pData1.remoteEvents < pData2.remoteEvents && (pData2.estimatedTime - pData1.estimatedTime) < lPercent * pData2.estimatedTime);
        bool lBetter2 = (pData2.estimatedTime < pData1.estimatedTime && (pData1.remoteEvents - pData2.remoteEvents) < lPercent * pData1.remoteEvents);
        bool lBetter3 = (pData1.estimatedTime < pData2.estimatedTime && (pData2.remoteEvents - pData1.remoteEvents) < lPercent * pData2.remoteEvents);
        bool lBetter4 = (pData2.remoteEvents < pData1.remoteEvents && (pData1.estimatedTime - pData1.estimatedTime) < lPercent * pData1.estimatedTime);
        
        if(lBetter1 && !lBetter2)
            return true;
        
        if(lBetter3 && !lBetter4)
            return true;
        
        return false;
    #endif
    }
};

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
    template<typename TS>
    void MakeAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector);

#ifdef MACHINES_PICK_BEST_SUBTASKS
#ifdef GENERALIZED_RESIDUAL_PROFIT_ASSIGNMENT
    template<typename T>
    double GetProfitValue(T& pData);
#endif
#endif
    
#ifdef USE_AFFINITY_IN_STEAL
    template<typename T, typename S>
    static std::vector<ulong> FindSubtasksWithBestAffinity(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex, ulong pSubtaskCount);

    template<typename T, typename S, typename D>
    static std::vector<ulong> FindSubtasksWithMaxDifferenceInAffinities(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex1, size_t pMachineIndex2, ulong pSubtaskCount);
#endif

    pmLocalTask* mLocalTask;
    pmAffinityCriterion mAffinityCriterion;
    
#ifdef MACHINES_PICK_BEST_SUBTASKS
    #ifdef GENERALIZED_RESIDUAL_PROFIT_ASSIGNMENT
        pmTable<uint, std::vector<double>> mTable; // machine index versus vector of profit (vector index is subtask id)
    #else
        pmTable<uint, std::vector<ulong>> mTable; // machine index versus subtasks in order of preference
    #endif
#else
    pmTable<ulong, std::vector<std::pair<const pmMachine*, double>>> mTable; // subtask id versus (machine, estimated completion time) pair in order of preference
#endif
};

} // end namespace pm

#endif


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

#include "pmAffinityTable.h"
#include "pmAddressSpace.h"
#include "pmSubtaskManager.h"
#include "pmDevicePool.h"
#include "pmHardware.h"
#include "pmTask.h"

namespace pm
{

pmAffinityTable::pmAffinityTable(pmLocalTask* pLocalTask)
    : mLocalTask(pLocalTask)
{
}

void pmAffinityTable::PopulateAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::set<const pmMachine*>& pMachinesSet)
{
    ulong* lAffinityMem = (ulong*)pAffinityAddressSpace->GetMem();

    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
    EXCEPTION_ASSERT(pMachinesSet.size() * lSubtaskCount * sizeof(ulong) == pAffinityAddressSpace->GetLength());

    std::vector<rowType> lRowVectors;
    lRowVectors.resize(lSubtaskCount);

    ulong index = 0;
    for_each(pMachinesSet, [&] (const pmMachine* pMachine)
    {
        for(ulong i = 0; i < lSubtaskCount; ++i)
        {
            lRowVectors[i].emplace(pMachine, lAffinityMem[index++]);
        }
    });
    
    for_each_with_index(lRowVectors, [&] (rowType& pRow, size_t pSubtask)
    {
        mTable.AddRow(pSubtask, std::move(pRow));
    });
}

void pmAffinityTable::CreateSubtaskMappings()
{
    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();

    std::vector<ulong> lLogicalToPhysicalSubtaskMapping(lSubtaskCount);
    std::vector<ulong> lPhysicalToLogicalSubtaskMapping(lSubtaskCount);
    
    pmPullSchedulingManager* lManager = dynamic_cast<pmPullSchedulingManager*>(mLocalTask->GetSubtaskManager());
    EXCEPTION_ASSERT(lManager);
    
    std::map<uint, std::pair<ulong, ulong>> lMap = lManager->ComputeMachineVersusInitialSubtaskCountMap();

    for(ulong i = 0; i < lSubtaskCount; ++i)
    {
        const rowType& lSubtaskRow = mTable.GetRow(i);
        
        // Most preferred machine for this subtask is at front of the row
        for(auto lSubtaskRowIter = lSubtaskRow.begin(), lSubtaskRowEndIter = lSubtaskRow.end(); lSubtaskRowIter != lSubtaskRowEndIter; ++lSubtaskRowIter)
        {
            uint lMachine = *lSubtaskRowIter->machine;
            
            auto lMapIter = lMap.find(lMachine);
            if(lMapIter != lMap.end() && lMapIter->second.second)
            {
                lLogicalToPhysicalSubtaskMapping[lMapIter->second.first] = i;
                lPhysicalToLogicalSubtaskMapping[i] = lMapIter->second.first;

                ++lMapIter->second.first;
                --lMapIter->second.second;
                
                break;
            }
        }
    }
    
    mLocalTask->SetAffinityMappings(std::move(lLogicalToPhysicalSubtaskMapping), std::move(lPhysicalToLogicalSubtaskMapping));
}

}




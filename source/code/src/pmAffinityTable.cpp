
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

#include <map>

namespace pm
{

pmAffinityTable::pmAffinityTable(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion)
    : mLocalTask(pLocalTask)
    , mAffinityCriterion(pAffinityCriterion)
{
}

void pmAffinityTable::PopulateAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector)
{
    void* lAffinityMem = pAffinityAddressSpace->GetMem();

    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();

    size_t lMachines = pMachinesVector.size();

    switch(mAffinityCriterion)
    {
        case MAXIMIZE_LOCAL_DATA:
        {
            EXCEPTION_ASSERT(pMachinesVector.size() * lSubtaskCount * sizeof(ulong) == pAffinityAddressSpace->GetLength());
            ulong* lAffinityData = (ulong*)lAffinityMem;

            std::vector<std::multimap<ulong, const pmMachine*, std::greater<ulong>>> lVector;    // local bytes versus machines for each subtask
            lVector.resize(lSubtaskCount);

            ulong index = 0;
            for_each(pMachinesVector, [&] (const pmMachine* pMachine)
            {
                for(ulong i = 0; i < lSubtaskCount; ++i)
                {
                    lVector[i].emplace(lAffinityData[index++], pMachine);
                }
            });
            
            for_each_with_index(lVector, [&] (const decltype(lVector)::value_type& pEntry, size_t pSubtask)
            {
                std::vector<const pmMachine*> lTableRow;
                lTableRow.reserve(lMachines);
                
                std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<ulong, const pmMachine*>());
                mTable.AddRow(pSubtask, std::move(lTableRow));
            });
            
            break;
        }
            
        case MINIMIZE_REMOTE_SOURCES:
        {
            EXCEPTION_ASSERT(pMachinesVector.size() * lSubtaskCount * sizeof(uint) == pAffinityAddressSpace->GetLength());
            uint* lAffinityData = (uint*)lAffinityMem;

            std::vector<std::multimap<uint, const pmMachine*, std::less<uint>>> lVector;    // remote sources versus machines for each subtask
            lVector.resize(lSubtaskCount);

            ulong index = 0;
            for_each(pMachinesVector, [&] (const pmMachine* pMachine)
            {
                for(ulong i = 0; i < lSubtaskCount; ++i)
                {
                    lVector[i].emplace(lAffinityData[index++], pMachine);
                }
            });
            
            for_each_with_index(lVector, [&] (const decltype(lVector)::value_type& pEntry, size_t pSubtask)
            {
                std::vector<const pmMachine*> lTableRow;
                lTableRow.reserve(lMachines);
                
                std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<uint, const pmMachine*>());
                mTable.AddRow(pSubtask, std::move(lTableRow));
            });
            
            break;
        }
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
        {
            EXCEPTION_ASSERT(pMachinesVector.size() * lSubtaskCount * sizeof(ulong) == pAffinityAddressSpace->GetLength());
            ulong* lAffinityData = (ulong*)lAffinityMem;

            std::vector<std::multimap<ulong, const pmMachine*, std::less<ulong>>> lVector;    // remote fetch time versus machines for each subtask
            lVector.resize(lSubtaskCount);

            ulong index = 0;
            for_each(pMachinesVector, [&] (const pmMachine* pMachine)
            {
                for(ulong i = 0; i < lSubtaskCount; ++i)
                {
                    lVector[i].emplace(lAffinityData[index++], pMachine);
                }
            });
            
            for_each_with_index(lVector, [&] (const decltype(lVector)::value_type& pEntry, size_t pSubtask)
            {
                std::vector<const pmMachine*> lTableRow;
                lTableRow.reserve(lMachines);
                
                std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<ulong, const pmMachine*>());
                mTable.AddRow(pSubtask, std::move(lTableRow));
            });
            
            break;
        }

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
        {
            EXCEPTION_ASSERT(pMachinesVector.size() * lSubtaskCount * sizeof(float) == pAffinityAddressSpace->GetLength());
            float* lAffinityData = (float*)lAffinityMem;

            std::vector<std::multimap<float, const pmMachine*, std::less<float>>> lVector;    // remote fetch time versus machines for each subtask
            lVector.resize(lSubtaskCount);

            ulong index = 0;
            for_each(pMachinesVector, [&] (const pmMachine* pMachine)
            {
                for(ulong i = 0; i < lSubtaskCount; ++i)
                {
                    lVector[i].emplace(lAffinityData[index++], pMachine);
                }
            });
            
            for_each_with_index(lVector, [&] (const decltype(lVector)::value_type& pEntry, size_t pSubtask)
            {
                std::vector<const pmMachine*> lTableRow;
                lTableRow.reserve(lMachines);
                
                std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<float, const pmMachine*>());
                mTable.AddRow(pSubtask, std::move(lTableRow));
            });
            
            break;
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    };
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
        const std::vector<const pmMachine*>& lSubtaskRow = mTable.GetRow(i);
        
        bool lAssigned = false;
        
        // Most preferred machine for this subtask is at front of the row
        for(auto lSubtaskRowIter = lSubtaskRow.begin(), lSubtaskRowEndIter = lSubtaskRow.end(); lSubtaskRowIter != lSubtaskRowEndIter; ++lSubtaskRowIter)
        {
            const pmMachine* lMachinePtr = *lSubtaskRowIter;
            uint lMachine = *lMachinePtr;
            
            auto lMapIter = lMap.find(lMachine);
            if(lMapIter != lMap.end() && lMapIter->second.second)
            {
                lLogicalToPhysicalSubtaskMapping[lMapIter->second.first] = i;
                lPhysicalToLogicalSubtaskMapping[i] = lMapIter->second.first;

                ++lMapIter->second.first;
                --lMapIter->second.second;
                
                lAssigned = true;
                break;
            }
        }

        EXCEPTION_ASSERT(lAssigned);
    }
    
    mLocalTask->SetAffinityMappings(std::move(lLogicalToPhysicalSubtaskMapping), std::move(lPhysicalToLogicalSubtaskMapping));
}

}




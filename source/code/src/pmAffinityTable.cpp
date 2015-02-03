
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
#include "pmLogger.h"

#include <map>
#include <sstream>

namespace pm
{
    
template<pmAffinityCriterion>
struct GetAffinityDataType
{};
    
template<>
struct GetAffinityDataType<MAXIMIZE_LOCAL_DATA>
{
    typedef ulong type;
    typedef std::greater<type> sorter;
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>
{
    typedef uint type;
    typedef std::less<type> sorter;
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>
{
    typedef ulong type;
    typedef std::less<type> sorter;
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>
{
    typedef float type;
    typedef std::less<type> sorter;
};


pmAffinityTable::pmAffinityTable(pmLocalTask* pLocalTask, pmAffinityCriterion pAffinityCriterion)
    : mLocalTask(pLocalTask)
    , mAffinityCriterion(pAffinityCriterion)
{
}

void pmAffinityTable::PopulateAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector)
{
    switch(mAffinityCriterion)
    {
        case MAXIMIZE_LOCAL_DATA:
        {
            MakeAffinityTable<GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::type, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sorter>(pAffinityAddressSpace, pMachinesVector);
            break;
        }
            
        case MINIMIZE_REMOTE_SOURCES:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::type, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sorter>(pAffinityAddressSpace, pMachinesVector);
            break;
        }
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sorter>(pAffinityAddressSpace, pMachinesVector);
            break;
        }

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sorter>(pAffinityAddressSpace, pMachinesVector);
            break;
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
}

template<typename T, typename S>
void pmAffinityTable::MakeAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector)
{
#ifdef DUMP_AFFINITY_DATA
    std::stringstream lStream;

    lStream << std::endl;
    lStream << "Affinity data (Host: subtask => data; ...) for task [" << (uint)(*mLocalTask->GetOriginatingHost()) << ", " << mLocalTask->GetSequenceNumber() << "] ..." << std::endl;
#endif

    T* lAffinityData = static_cast<T*>(pAffinityAddressSpace->GetMem());

    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
    uint lMachines = (uint)pMachinesVector.size();

    EXCEPTION_ASSERT(pMachinesVector.size() * lSubtaskCount * sizeof(T) == pAffinityAddressSpace->GetLength());

#ifdef MACHINES_PICK_BEST_SUBTASKS
    std::vector<std::multimap<T, ulong, S>> lVector;    // data versus subtasks for each machine
    lVector.resize(lMachines);
#else
    std::vector<std::multimap<T, const pmMachine*, S>> lVector;    // data versus machines for each subtask
    lVector.resize(lSubtaskCount);
#endif

    ulong index = 0;
    for_each_with_index(pMachinesVector, [&] (const pmMachine* pMachine, size_t pMachineIndex)
    {
    #ifdef DUMP_AFFINITY_DATA
        lStream << "Host " << (uint)(*pMachine) << ": ";
    #endif

        for(ulong i = 0; i < lSubtaskCount; ++i)
        {
            const T& lData = lAffinityData[index++];

        #ifdef DUMP_AFFINITY_DATA
            lStream << i << " => " << lData;
            
            if(i != lSubtaskCount - 1)
                lStream << "; ";
        #endif

        #ifdef MACHINES_PICK_BEST_SUBTASKS
            lVector[pMachineIndex].emplace(lData, i);
        #else
            lVector[i].emplace(lData, pMachine);
        #endif
        }

    #ifdef DUMP_AFFINITY_DATA
        lStream << std::endl;
    #endif
    });
    
#ifdef DUMP_AFFINITY_DATA
    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif

    for_each_with_index(lVector, [&] (const typename decltype(lVector)::value_type& pEntry, size_t pIndex)
    {
    #ifdef MACHINES_PICK_BEST_SUBTASKS
        std::vector<ulong> lTableRow;
        lTableRow.reserve(lSubtaskCount);
        
        std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<T, ulong>());
        mTable.AddRow((uint)(*pMachinesVector[pIndex]), std::move(lTableRow));
    #else
        std::vector<const pmMachine*> lTableRow;
        lTableRow.reserve(lMachines);
        
        std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<T, const pmMachine*>());
        mTable.AddRow(pIndex, std::move(lTableRow));
    #endif
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

#ifdef MACHINES_PICK_BEST_SUBTASKS
    std::set<ulong> lSubtasksAllotted;
    auto lSubtasksAllottedEndIter = lSubtasksAllotted.end();

    std::map<uint, std::vector<ulong>::const_iterator> lSubtasksIterMap;
    for_each(lMap, [&] (const decltype(lMap)::value_type& pPair)
    {
        lSubtasksIterMap.emplace(pPair.first, mTable.GetRow(pPair.first).begin());
    });

    auto lIter = lMap.begin(), lEndIter = lMap.end();
    while(lSubtasksAllotted.size() != lSubtaskCount)
    {
        // Get next available machine
        bool lMachineFound = false;
        
        while(!lMachineFound)
        {
            if(lIter == lEndIter)
                lIter = lMap.begin();

            if(lIter->second.second)
                lMachineFound = true;
            else
                ++lIter;
        }
        
        uint lMachine = lIter->first;
        
        // Find next best subtask for this machine
        auto lSubtasksIter = lSubtasksIterMap.find(lMachine)->second;
        while(lSubtasksAllotted.find(*lSubtasksIter) != lSubtasksAllottedEndIter)
            ++lSubtasksIter;

        // Assign the selected subtask to the selected machine
        lLogicalToPhysicalSubtaskMapping[lIter->second.first] = *lSubtasksIter;
        lPhysicalToLogicalSubtaskMapping[*lSubtasksIter] = lIter->second.first;

        lSubtasksAllotted.emplace(*lSubtasksIter);
        ++lIter->second.first;
        --lIter->second.second;
        ++lIter;    // change machine every iteration
    }
#else
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
#endif
    
#ifdef _DEBUG
    for_each_with_index(lLogicalToPhysicalSubtaskMapping, [&] (ulong lPhysicalSubtask, size_t lLogicalSubtask)
    {
        EXCEPTION_ASSERT(lPhysicalToLogicalSubtaskMapping[lPhysicalSubtask] == lLogicalSubtask);
        std::cout << "Logical subtask = " << lLogicalSubtask << "; Physical subtask = " << lPhysicalSubtask << "; Reverse mapped logical subtask = " << lPhysicalToLogicalSubtaskMapping[lPhysicalSubtask] << std::endl;
    });
#endif

#ifdef DUMP_AFFINITY_DATA
    std::stringstream lStream;

    lStream << "Affinity table (physical subtask => logical subtask) for task [" << (uint)(*mLocalTask->GetOriginatingHost()) << ", " << mLocalTask->GetSequenceNumber() << "] ..." << std::endl;
    for_each_with_index(lPhysicalToLogicalSubtaskMapping, [&] (ulong lLogicalSubtask, size_t lPhysicalSubtask)
    {
        lStream << lPhysicalSubtask << " => " << lLogicalSubtask;
        
        if(lPhysicalSubtask != lSubtaskCount - 1)
            lStream << ", ";
    });
    
    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif

    mLocalTask->SetAffinityMappings(std::move(lLogicalToPhysicalSubtaskMapping), std::move(lPhysicalToLogicalSubtaskMapping));
}

#ifdef USE_AFFINITY_IN_STEAL
std::vector<ulong> pmAffinityTable::FindSubtasksWithBestAffinity(pmTask* pTask, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, const pmMachine* pMachine)
{
    std::vector<const pmMachine*> lMachinesVector;
    pmProcessingElement::GetMachinesInOrder(((dynamic_cast<pmLocalTask*>(pTask) != NULL) ? ((pmLocalTask*)pTask)->GetAssignedDevices() : ((pmRemoteTask*)pTask)->GetAssignedDevices()), lMachinesVector);

    pmAddressSpace* lAffinityAddressSpace = ((dynamic_cast<pmLocalTask*>(pTask) != NULL) ? ((pmLocalTask*)pTask)->GetAffinityAddressSpace() : ((pmRemoteTask*)pTask)->GetAffinityAddressSpace());
    EXCEPTION_ASSERT(lAffinityAddressSpace != NULL);
    
    auto lIter = std::find(lMachinesVector.begin(), lMachinesVector.end(), pMachine);
    EXCEPTION_ASSERT(lIter != lMachinesVector.end());
    
    size_t lMachineIndex = lIter - lMachinesVector.begin();
    
    switch(pTask->GetAffinityCriterion())
    {
        case MAXIMIZE_LOCAL_DATA:
            return FindSubtasksWithBestAffinity<GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::type, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sorter>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex, pTask->GetSubtaskCount());
            
        case MINIMIZE_REMOTE_SOURCES:
            return FindSubtasksWithBestAffinity<GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::type, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sorter>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex, pTask->GetSubtaskCount());
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
            return FindSubtasksWithBestAffinity<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sorter>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex, pTask->GetSubtaskCount());

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
            return FindSubtasksWithBestAffinity<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sorter>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex, pTask->GetSubtaskCount());
            
        default:
            PMTHROW(pmFatalErrorException());
    }
    
    return std::vector<ulong>();
}
    
template<typename T, typename S>
std::vector<ulong> pmAffinityTable::FindSubtasksWithBestAffinity(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex, ulong pSubtaskCount)
{
    T* lAffinityData = static_cast<T*>(pAffinityAddressSpace->GetMem());
    
    lAffinityData += pMachineIndex * pSubtaskCount;
    
    std::multimap<T, ulong, S> lArrangedSubtasks;
    for(ulong i = pStartSubtask; i <= pEndSubtask; ++i)
        lArrangedSubtasks.emplace(lAffinityData[i], i);
    
    EXCEPTION_ASSERT(lArrangedSubtasks.size() >= pCount);
    
    auto lIter = lArrangedSubtasks.begin();

    std::vector<ulong> lSubtasksVector;
    lSubtasksVector.reserve(pCount);

    for(size_t i = 0; i < pCount; ++i, ++lIter)
        lSubtasksVector.emplace_back(lIter->second);
    
    std::sort(lSubtasksVector.begin(), lSubtasksVector.end());
    
    return lSubtasksVector;
}
#endif

}




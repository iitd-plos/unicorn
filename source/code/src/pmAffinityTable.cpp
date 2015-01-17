
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
    };
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

    std::vector<std::multimap<T, const pmMachine*, S>> lVector;    // data versus machines for each subtask
    lVector.resize(lSubtaskCount);

    ulong index = 0;
    for_each(pMachinesVector, [&] (const pmMachine* pMachine)
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
            
            lVector[i].emplace(lData, pMachine);
        }

    #ifdef DUMP_AFFINITY_DATA
        lStream << std::endl;
    #endif
    });
    
#ifdef DUMP_AFFINITY_DATA
    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str());
#endif

    for_each_with_index(lVector, [&] (const typename decltype(lVector)::value_type& pEntry, size_t pSubtask)
    {
        std::vector<const pmMachine*> lTableRow;
        lTableRow.reserve(lMachines);
        
        std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<T, const pmMachine*>());
        mTable.AddRow(pSubtask, std::move(lTableRow));
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

}





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
    typedef double difference_type;
    
    static constexpr type sentinel_value = std::numeric_limits<type>::min();
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>
{
    typedef uint type;
    typedef std::less<type> sorter;
    typedef long difference_type;

    static constexpr type sentinel_value = std::numeric_limits<type>::max();
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>
{
    typedef ulong type;
    typedef std::less<type> sorter;
    typedef double difference_type;

    static constexpr type sentinel_value = std::numeric_limits<type>::max();
};

template<>
struct GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>
{
    typedef float type;
    typedef std::less<type> sorter;
    typedef double difference_type;

    static constexpr type sentinel_value = std::numeric_limits<type>::max();
};

template<>
struct GetAffinityDataType<DERIVED_AFFINITY>
{
    typedef derivedAffinityData type;
    typedef derivedAffinityDataSorter sorter;
    typedef derivedAffinityData difference_type;

    static type sentinel_value;
};
    
derivedAffinityData GetAffinityDataType<DERIVED_AFFINITY>::sentinel_value = {std::numeric_limits<ulong>::max(), std::numeric_limits<float>::max()};

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
            MakeAffinityTable<GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::type, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sorter>(pAffinityAddressSpace, pMachinesVector, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sentinel_value);
            break;
        }
            
        case MINIMIZE_REMOTE_SOURCES:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::type, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sorter>(pAffinityAddressSpace, pMachinesVector, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sentinel_value);
            break;
        }
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sorter>(pAffinityAddressSpace, pMachinesVector, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sentinel_value);
            break;
        }

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
        {
            MakeAffinityTable<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sorter>(pAffinityAddressSpace, pMachinesVector, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sentinel_value);
            break;
        }
            
        case DERIVED_AFFINITY:
        {
            MakeAffinityTable<GetAffinityDataType<DERIVED_AFFINITY>::type, GetAffinityDataType<DERIVED_AFFINITY>::sorter>(pAffinityAddressSpace, pMachinesVector, GetAffinityDataType<DERIVED_AFFINITY>::sentinel_value);
            break;
        }

        default:
            PMTHROW(pmFatalErrorException());
    }
}

template<typename T, typename S>
void pmAffinityTable::MakeAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, T pSentinelValue)
{
#ifdef DUMP_AFFINITY_DATA
    std::stringstream lStream;

    lStream << std::endl;
    lStream << "Affinity data (Host: subtask => data; ...) for task [" << (uint)(*mLocalTask->GetOriginatingHost()) << ", " << mLocalTask->GetSequenceNumber() << "] ..." << std::endl;
#endif

    void* lAffinityData = pAffinityAddressSpace->GetMem();

    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();
    uint lMachines = (uint)pMachinesVector.size();
    
    bool lSubtaskIdEmbedded = (lSubtaskCount * lMachines != mLocalTask->GetPreprocessorTask()->GetSubtaskCount());

    EXCEPTION_ASSERT((lSubtaskIdEmbedded ? (2 * mLocalTask->GetPreprocessorTask()->GetSubtaskCount()) : (pMachinesVector.size() * lSubtaskCount)) * sizeof(T) == pAffinityAddressSpace->GetLength());

#ifdef MACHINES_PICK_BEST_SUBTASKS
    std::vector<std::multimap<T, ulong, S>> lVector;    // data versus subtasks for each machine
    lVector.resize(lMachines);
#else
    std::vector<std::multimap<T, const pmMachine*, S>> lVector;    // data versus machines for each subtask
    lVector.resize(lSubtaskCount);
#endif
    
    for_each_with_index(pMachinesVector, [&] (const pmMachine* pMachine, size_t pMachineIndex)
    {
    #ifdef DUMP_AFFINITY_DATA
        lStream << "Host " << (uint)(*pMachine) << ": ";
    #endif

        T lData;
        for(ulong i = 0; i < lSubtaskCount; ++i)
        {
            if(lSubtaskIdEmbedded)
            {
                ulong lEmbeddedSubtaskId = *((ulong*)lAffinityData);
                if(i == lEmbeddedSubtaskId)
                {
                    lAffinityData = (void*)((ulong*)lAffinityData + 1);

                    lData = *((T*)lAffinityData);
                    lAffinityData = (void*)((T*)lAffinityData + 1);
                }
                else
                {
                    lData = pSentinelValue;
                }
            }
            else
            {
                lData = *((T*)lAffinityData);
                lAffinityData = (void*)((T*)lAffinityData + 1);
            }

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
        #ifdef GENERALIZED_RESIDUAL_PROFIT_ASSIGNMENT
            std::vector<double> lTableRow;
            lTableRow.resize(lSubtaskCount);

            for_each(pEntry, [&] (const std::pair<T, ulong>& pMapEntry)
            {
                lTableRow[pMapEntry.second] = GetProfitValue(pMapEntry.first);
            });

            mTable.AddRow((uint)(*pMachinesVector[pIndex]), std::move(lTableRow));
        #else
            std::vector<ulong> lTableRow;
            lTableRow.reserve(lSubtaskCount);
            
            std::transform(pEntry.begin(), pEntry.end(), std::back_inserter(lTableRow), select2nd<T, ulong>());
            mTable.AddRow((uint)(*pMachinesVector[pIndex]), std::move(lTableRow));
        #endif
    #else
        std::vector<std::pair<const pmMachine*, double>> lTableRow;
        lTableRow.resize(lMachines);
        
        for_each(pEntry, [&] (const std::pair<T, const pmMachine*>& pMapEntry)
        {
            lTableRow.emplace_back(pMapEntry.second, GetEstimatedCompletionTime(pMapEntry.first));
        });

        mTable.AddRow(pIndex, std::move(lTableRow));
    #endif
    });
}

void pmAffinityTable::CreateSubtaskMappings()
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(mLocalTask->GetTaskProfiler(), taskProfiler::AFFINITY_SUBTASK_MAPPINGS);
#endif

    ulong lSubtaskCount = mLocalTask->GetSubtaskCount();

    std::vector<ulong> lLogicalToPhysicalSubtaskMapping(lSubtaskCount);
    std::vector<ulong> lPhysicalToLogicalSubtaskMapping(lSubtaskCount);
    
    pmPullSchedulingManager* lManager = dynamic_cast<pmPullSchedulingManager*>(mLocalTask->GetSubtaskManager());
    EXCEPTION_ASSERT(lManager);
    
    // Logical subtask ids are already assigned by scheduler; this code maps them to the physical ones
    std::vector<ulong> lLogicalSubtaskIdsVector;
    std::map<uint, std::pair<ulong, ulong>> lMap = lManager->ComputeMachineVersusInitialSubtaskCountMap(lLogicalSubtaskIdsVector);
    
#ifdef MACHINES_PICK_BEST_SUBTASKS
    #ifdef GENERALIZED_RESIDUAL_PROFIT_ASSIGNMENT
        std::vector<long> lSubtaskAssignment;
        lSubtaskAssignment.resize(lSubtaskCount, -1);
    
        uint lMachines = (uint)mTable.GetRowCount();
        for(uint i = 0; i < lMachines; ++i)
        {
            std::multimap<double, ulong, std::greater<double>> lResidualProfitsMap;
            const std::vector<double>& lRow = mTable.GetRow(i);
            
            for(ulong j = 0; j < lSubtaskCount; ++j)
            {
                long lCurrentMachine = lSubtaskAssignment[j];
                double lResidualProfitValue = lRow[j] - ((lCurrentMachine == -1) ? 0 : mTable.GetRow((uint)lCurrentMachine)[j]);
                
                lResidualProfitsMap.emplace(lResidualProfitValue, j);
            }

            ulong lMachineAssignments = lMap[i].second;
            auto lIter = lResidualProfitsMap.begin();
            for(ulong k = 0; k < lMachineAssignments; ++lIter, ++k)
                lSubtaskAssignment[lIter->second] = i;
        }
            
        std::vector<ulong> lUnassignedSubtasks;
        lUnassignedSubtasks.reserve(lSubtaskCount);

        for_each_with_index(lSubtaskAssignment, [&] (long pSubtaskMachine, size_t pSubtaskId)
        {
            if(pSubtaskMachine == -1)
            {
                lUnassignedSubtasks.emplace_back(pSubtaskId);
            }
            else
            {
                auto lMapIter = lMap.find((uint)pSubtaskMachine);
                EXCEPTION_ASSERT(lMapIter != lMap.end());
                EXCEPTION_ASSERT(lMapIter->second.second);

                ulong lLogicalSubtaskId = lLogicalSubtaskIdsVector[lMapIter->second.first];

                // Assign the selected subtask to the selected machine
                lLogicalToPhysicalSubtaskMapping[lLogicalSubtaskId] = pSubtaskId;
                lPhysicalToLogicalSubtaskMapping[pSubtaskId] = lLogicalSubtaskId;

                ++lMapIter->second.first;
                --lMapIter->second.second;
            }
        });
        
        for_each(lMap, [&] (decltype(lMap)::value_type& pPair)
        {
            const std::vector<double>& lProfitsRow = mTable.GetRow(pPair.first);

            while(pPair.second.second)
            {
                std::vector<ulong>::iterator lSelectedSubtaskIter = lUnassignedSubtasks.end();
                double lSelectedSubtaskProfit = std::numeric_limits<double>::lowest();

                // Find best subtask from the unassigned ones
                auto lSubtaskIter = lUnassignedSubtasks.begin(), lSubtaskEndIter = lUnassignedSubtasks.end();
                for(; lSubtaskIter != lSubtaskEndIter; ++lSubtaskIter)
                {
                    if(lProfitsRow[*lSubtaskIter] > lSelectedSubtaskProfit)
                    {
                        lSelectedSubtaskProfit = lProfitsRow[*lSubtaskIter];
                        lSelectedSubtaskIter = lSubtaskIter;
                    }
                }
                
                EXCEPTION_ASSERT(lSelectedSubtaskIter != lSubtaskEndIter);
                
                ulong lSelectedSubtask = *lSelectedSubtaskIter;
                ulong lLogicalSubtaskId = lLogicalSubtaskIdsVector[pPair.second.first];

                // Assign the selected subtask to the selected machine
                lLogicalToPhysicalSubtaskMapping[lLogicalSubtaskId] = lSelectedSubtask;
                lPhysicalToLogicalSubtaskMapping[lSelectedSubtask] = lLogicalSubtaskId;
                
                lUnassignedSubtasks.erase(lSelectedSubtaskIter);
                
                ++pPair.second.first;
                --pPair.second.second;
            }
        });
            
        EXCEPTION_ASSERT(lUnassignedSubtasks.empty());
    #else
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

            ulong lLogicalSubtaskId = lLogicalSubtaskIdsVector[lIter->second.first];

            // Assign the selected subtask to the selected machine
            lLogicalToPhysicalSubtaskMapping[lLogicalSubtaskId] = *lSubtasksIter;
            lPhysicalToLogicalSubtaskMapping[*lSubtasksIter] = lLogicalSubtaskId;

            lSubtasksAllotted.emplace(*lSubtasksIter);
            ++lIter->second.first;
            --lIter->second.second;
            ++lIter;    // change machine every iteration
        }
    #endif
#else
    std::vector<double> mEstimatedCompletionTimeOnMachineVector(lMap.size(), 0);
    double mCurrentMaximumCompletionTime = 0;
    
    for(ulong i = 0; i < lSubtaskCount; ++i)
    {
        const std::vector<std::pair<const pmMachine*, double>>& lSubtaskRow = mTable.GetRow(i);
        
        const pmMachine* lTargetMachinePtr = NULL;
        double lTargettedCompletionTime = 0;
        
        double lCompletionTimeOfProbableMachine = std::numeric_limits<double>::max();
        const pmMachine* lProbableMachinePtr = NULL;

        // Most preferred machine for this subtask is at front of the row
        // Assign that machine to the subtask which is high on preference and does not increase the total completion time beyond the current global maximum.
        // If there is no such machine, assign subtask to the machine that increases the global completion time by minimum.
        for(auto lSubtaskRowIter = lSubtaskRow.begin(), lSubtaskRowEndIter = lSubtaskRow.end(); lSubtaskRowIter != lSubtaskRowEndIter; ++lSubtaskRowIter)
        {
            const pmMachine* lMachinePtr = lSubtaskRowIter->first;
            uint lMachine = *lMachinePtr;

            double lCompletionTimeOfSubtask = lSubtaskRowIter->second;
            double lCompletionTime = mEstimatedCompletionTimeOnMachineVector[lMachine] + lCompletionTimeOfSubtask;
            
            auto lMapIter = lMap.find(lMachine);
            if(lMapIter != lMap.end() && lMapIter->second.second)
            {
                if(lCompletionTime < mCurrentMaximumCompletionTime)
                {
                    lTargetMachinePtr = lMachinePtr;
                    lTargettedCompletionTime = lCompletionTime;

                    break;
                }
                else
                {
                    if(lCompletionTimeOfProbableMachine > lCompletionTime)
                    {
                        lCompletionTimeOfProbableMachine = lCompletionTime;
                        lProbableMachinePtr = lMachinePtr;
                    }
                }
            }
            
        }
        
        if(!lTargetMachinePtr)
        {
            lTargetMachinePtr = lProbableMachinePtr;
            lTargettedCompletionTime = lCompletionTimeOfProbableMachine;
        }

        EXCEPTION_ASSERT(lTargetMachinePtr && lTargettedCompletionTime);

        uint lMachine = *lTargetMachinePtr;

        auto lMapIter = lMap.find(lMachine);
        ulong lLogicalSubtaskId = lLogicalSubtaskIdsVector[lMapIter->second.first];

        lLogicalToPhysicalSubtaskMapping[lLogicalSubtaskId] = i;
        lPhysicalToLogicalSubtaskMapping[i] = lLogicalSubtaskId;

        ++lMapIter->second.first;
        --lMapIter->second.second;

        mEstimatedCompletionTimeOnMachineVector[lMachine] = lTargettedCompletionTime;
        mCurrentMaximumCompletionTime = std::max(mCurrentMaximumCompletionTime, lTargettedCompletionTime);

    }
#endif

#if 0   //def _DEBUG
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

#ifdef MACHINES_PICK_BEST_SUBTASKS

#ifdef GENERALIZED_RESIDUAL_PROFIT_ASSIGNMENT
template<typename T>
double pmAffinityTable::GetProfitValue(T& pData)
{
    return 0;
}
#endif
    
#else

template<typename T>
double pmAffinityTable::GetEstimatedCompletionTime(T& pData)
{
    return 0;
}

#endif

    
#ifdef USE_AFFINITY_IN_STEAL
std::vector<ulong> pmAffinityTable::FindSubtasksWithBestAffinity(pmTask* pTask, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, const pmMachine* pMachine)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::AFFINITY_USE_OVERHEAD);
#endif

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
            
        case DERIVED_AFFINITY:
            return FindSubtasksWithBestAffinity<GetAffinityDataType<DERIVED_AFFINITY>::type, GetAffinityDataType<DERIVED_AFFINITY>::sorter>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex, pTask->GetSubtaskCount());

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

std::vector<ulong> pmAffinityTable::FindSubtasksWithMaxDifferenceInAffinities(pmTask* pTask, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, const pmMachine* pMachine1, const pmMachine* pMachine2)
{
#ifdef ENABLE_TASK_PROFILING
    pmRecordProfileEventAutoPtr lRecordProfileEventAutoPtr(pTask->GetTaskProfiler(), taskProfiler::AFFINITY_USE_OVERHEAD);
#endif

    std::vector<const pmMachine*> lMachinesVector;
    pmProcessingElement::GetMachinesInOrder(((dynamic_cast<pmLocalTask*>(pTask) != NULL) ? ((pmLocalTask*)pTask)->GetAssignedDevices() : ((pmRemoteTask*)pTask)->GetAssignedDevices()), lMachinesVector);

    pmAddressSpace* lAffinityAddressSpace = ((dynamic_cast<pmLocalTask*>(pTask) != NULL) ? ((pmLocalTask*)pTask)->GetAffinityAddressSpace() : ((pmRemoteTask*)pTask)->GetAffinityAddressSpace());
    EXCEPTION_ASSERT(lAffinityAddressSpace != NULL);
    
    auto lIter1 = std::find(lMachinesVector.begin(), lMachinesVector.end(), pMachine1), lIter2 = std::find(lMachinesVector.begin(), lMachinesVector.end(), pMachine2);
    EXCEPTION_ASSERT(lIter1 != lMachinesVector.end());
    
    size_t lMachineIndex1 = lIter1 - lMachinesVector.begin();
    size_t lMachineIndex2 = lIter2 - lMachinesVector.begin();
    
    switch(pTask->GetAffinityCriterion())
    {
        case MAXIMIZE_LOCAL_DATA:
            return FindSubtasksWithMaxDifferenceInAffinities<GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::type, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sorter, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::difference_type>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex1, lMachineIndex2, pTask->GetSubtaskCount());
            
        case MINIMIZE_REMOTE_SOURCES:
            return FindSubtasksWithMaxDifferenceInAffinities<GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::type, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sorter, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::difference_type>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex1, lMachineIndex2, pTask->GetSubtaskCount());
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
            return FindSubtasksWithMaxDifferenceInAffinities<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sorter, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::difference_type>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex1, lMachineIndex2, pTask->GetSubtaskCount());

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
            return FindSubtasksWithMaxDifferenceInAffinities<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::type, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sorter, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::difference_type>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex1, lMachineIndex2, pTask->GetSubtaskCount());
            
        case DERIVED_AFFINITY:
            return FindSubtasksWithMaxDifferenceInAffinities<GetAffinityDataType<DERIVED_AFFINITY>::type, GetAffinityDataType<DERIVED_AFFINITY>::sorter, GetAffinityDataType<DERIVED_AFFINITY>::difference_type>(lAffinityAddressSpace, lMachinesVector, pStartSubtask, pEndSubtask, pCount, lMachineIndex1, lMachineIndex2, pTask->GetSubtaskCount());

        default:
            PMTHROW(pmFatalErrorException());
    }
    
    return std::vector<ulong>();
}
    
template<typename T, typename S, typename D>
std::vector<ulong> pmAffinityTable::FindSubtasksWithMaxDifferenceInAffinities(pmAddressSpace* pAffinityAddressSpace, const std::vector<const pmMachine*>& pMachinesVector, ulong pStartSubtask, ulong pEndSubtask, ulong pCount, size_t pMachineIndex1, size_t pMachineIndex2, ulong pSubtaskCount)
{
    T* lAffinityData = static_cast<T*>(pAffinityAddressSpace->GetMem());

    T* lAffinityData1 = lAffinityData + pMachineIndex1 * pSubtaskCount;
    T* lAffinityData2 = lAffinityData + pMachineIndex2 * pSubtaskCount;
    
    std::multimap<D, ulong, S> lArrangedSubtasks;
    for(ulong i = pStartSubtask; i <= pEndSubtask; ++i)
    {
        D val = (D)(lAffinityData2[i]) - (D)(lAffinityData1[i]);
        lArrangedSubtasks.emplace(val, i);
    }
    
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
    
#ifdef USE_DYNAMIC_AFFINITY
ulong pmAffinityTable::GetSubtaskWithBestAffinity(pmTask* pTask, pmExecutionStub* pStub, const std::vector<ulong>& pSubtasks, std::shared_ptr<void>& pSharedPtr, bool pUpdate)
{
    switch(pTask->GetAffinityCriterion())
    {
        case MAXIMIZE_LOCAL_DATA:
        {
            typedef typename std::multimap<GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::type, ulong, GetAffinityDataType<MAXIMIZE_LOCAL_DATA>::sorter> map_type;

            map_type* lMultiMap = NULL;

            if(pUpdate)
            {
                pSharedPtr.reset(new map_type());
                map_type* lMultiMap = (map_type*)(pSharedPtr.get());

                pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

                for_each(pSubtasks, [&] (ulong pSubtaskId)
                {
                    lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL, true);

                    lMultiMap->emplace(lSubscriptionManager.FindLocalInputDataSizeForSubtask(pStub, pSubtaskId), pSubtaskId);
                });
            }
            
            lMultiMap = (map_type*)(pSharedPtr.get());
            EXCEPTION_ASSERT(lMultiMap);
            
            return lMultiMap->begin()->first;
        }

        case MINIMIZE_REMOTE_SOURCES:
        {
            typedef typename std::multimap<GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::type, ulong, GetAffinityDataType<MINIMIZE_REMOTE_SOURCES>::sorter> map_type;

            map_type* lMultiMap = NULL;

            if(pUpdate)
            {
                pSharedPtr.reset(new map_type());
                map_type* lMultiMap = (map_type*)(pSharedPtr.get());

                pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

                for_each(pSubtasks, [&] (ulong pSubtaskId)
                {
                    lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL, true);

                    lMultiMap->emplace(lSubscriptionManager.FindRemoteDataSourcesForSubtask(pStub, pSubtaskId), pSubtaskId);
                });
            }
            
            lMultiMap = (map_type*)(pSharedPtr.get());
            EXCEPTION_ASSERT(lMultiMap);
            
            return lMultiMap->begin()->first;
        }
            
        case MINIMIZE_REMOTE_TRANSFER_EVENTS:
        {
            typedef typename std::multimap<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::type, ulong, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFER_EVENTS>::sorter> map_type;

            map_type* lMultiMap = NULL;

            if(pUpdate)
            {
                pSharedPtr.reset(new map_type());
                map_type* lMultiMap = (map_type*)(pSharedPtr.get());

                pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

                for_each(pSubtasks, [&] (ulong pSubtaskId)
                {
                    lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL, true);

                    lMultiMap->emplace(lSubscriptionManager.FindRemoteTransferEventsForSubtask(pStub, pSubtaskId), pSubtaskId);
                });
            }
            
            lMultiMap = (map_type*)(pSharedPtr.get());
            EXCEPTION_ASSERT(lMultiMap);
            
            return lMultiMap->begin()->first;
        }

        case MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME:
        {
            typedef typename std::multimap<GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::type, ulong, GetAffinityDataType<MINIMIZE_REMOTE_TRANSFERS_ESTIMATED_TIME>::sorter> map_type;

            map_type* lMultiMap = NULL;

            if(pUpdate)
            {
                pSharedPtr.reset(new map_type());
                map_type* lMultiMap = (map_type*)(pSharedPtr.get());

                pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

                for_each(pSubtasks, [&] (ulong pSubtaskId)
                {
                    lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL, true);

                    lMultiMap->emplace(lSubscriptionManager.FindRemoteTransferEstimateForSubtask(pStub, pSubtaskId), pSubtaskId);
                });
            }
            
            lMultiMap = (map_type*)(pSharedPtr.get());
            EXCEPTION_ASSERT(lMultiMap);
            
            return lMultiMap->begin()->first;
        }

        case DERIVED_AFFINITY:
        {
            typedef typename std::multimap<GetAffinityDataType<DERIVED_AFFINITY>::type, ulong, GetAffinityDataType<DERIVED_AFFINITY>::sorter> map_type;

            map_type* lMultiMap = NULL;

            if(pUpdate)
            {
                pSharedPtr.reset(new map_type());
                map_type* lMultiMap = (map_type*)(pSharedPtr.get());

                pmSubscriptionManager& lSubscriptionManager = pTask->GetSubscriptionManager();

                for_each(pSubtasks, [&] (ulong pSubtaskId)
                {
                    lSubscriptionManager.FindSubtaskMemDependencies(pStub, pSubtaskId, NULL, true);

                    derivedAffinityData data;
                    //data.localBytes = lSubscriptionManager.FindLocalInputDataSizeForSubtask(pStub, pSubtaskId);
                    //data.remoteNodes = lSubscriptionManager.FindRemoteDataSourcesForSubtask(pStub, pSubtaskId);
                    data.remoteEvents = lSubscriptionManager.FindRemoteTransferEventsForSubtask(pStub, pSubtaskId);
                    data.estimatedTime = lSubscriptionManager.FindRemoteTransferEstimateForSubtask(pStub, pSubtaskId);
                    lMultiMap->emplace(data, pSubtaskId);
                });
            }
            
            lMultiMap = (map_type*)(pSharedPtr.get());
            EXCEPTION_ASSERT(lMultiMap);
            
            return lMultiMap->begin()->first;
        }

        default:
            PMTHROW(pmFatalErrorException());
    }

    return std::numeric_limits<ulong>::max();
}
#endif

std::ostream& operator<< (std::ostream& pOStream, const derivedAffinityData& pData)
{
    //return pOStream << "[" << pData.localBytes << ", " << pData.remoteNodes << ", " << pData.remoteEvents << ", " << pData.estimatedTime << "]";
    //return pOStream << "[" << pData.remoteNodes << ", " << pData.remoteEvents << ", " << pData.estimatedTime << "]";
    return pOStream << "[" << pData.remoteEvents << ", " << pData.estimatedTime << "]";
}

}




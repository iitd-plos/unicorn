
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
    
class pmAffinityTable
{
    struct subtaskData
    {
        const pmMachine* machine;
        ulong localBytes;
        
        subtaskData(const pmMachine* pMachine, ulong pLocalBytes)
        : machine(pMachine)
        , localBytes(pLocalBytes)
        {}
        
        friend bool operator< (const subtaskData& pFirst, const subtaskData& pSecond)
        {
            return (pFirst.localBytes > pSecond.localBytes);
        }
    };

    typedef std::set<subtaskData> rowType;
    
public:
    pmAffinityTable(pmLocalTask* pLocalTask);
    
    void PopulateAffinityTable(pmAddressSpace* pAffinityAddressSpace, const std::set<const pmMachine*>& pMachinesSet);
    void CreateSubtaskMappings();

private:
    pmLocalTask* mLocalTask;
    pmTable<ulong, rowType> mTable; // subtask id versus machines
};

} // end namespace pm

#endif

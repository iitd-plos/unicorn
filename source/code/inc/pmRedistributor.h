
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
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

#ifndef __PM_REDISTRIBUTOR__
#define __PM_REDISTRIBUTOR__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommand.h"

#include <map>
#include <vector>

namespace pm
{

class pmTask;
class pmMachine;

class pmRedistributor : public pmBase
{
	public:
		pmRedistributor(pmTask* pTask);
		virtual ~pmRedistributor();

        pmStatus RedistributeData(pmExecutionStub* pStub, ulong pSubtaskId, ulong pOffset, ulong pLength, uint pOrder);
        pmStatus PerformRedistribution(pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<pmCommunicatorCommand::redistributionOrderStruct>& pVector);
    
        pmStatus SendRedistributionInfo();
	
	private:
        typedef struct orderData
        {
            pmMachine* host;
            ulong offset;
            ulong length;
        } orderData;
    
        void SetRedistributedOwnership();
    
		pmTask* mTask;
        ulong mTotalLengthAccounted;
        ulong mSubtasksAccounted;
    
        std::map<uint, std::vector<orderData> > mGlobalRedistributionMap;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mGlobalRedistributionLock;

        std::vector<pmCommunicatorCommand::redistributionOrderStruct> mLocalRedistributionData;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mLocalRedistributionLock;
};

} // end namespace pm

#endif

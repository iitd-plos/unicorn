
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

#ifndef __PM_REDUCER__
#define __PM_REDUCER__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <vector>

namespace pm
{

class pmTask;
class pmMachine;
class pmExecutionStub;

class pmReducer : public pmBase
{
	public:
		pmReducer(pmTask* pTask);
		virtual ~pmReducer();

		pmStatus CheckReductionFinish();
		pmStatus AddSubtask(pmExecutionStub* pStub, ulong pSubtaskId);
	
	private:
		pmStatus PopulateExternalMachineList();
		ulong GetMaxPossibleExternalReductionReceives(uint pFollowingMachineCount);
        pmStatus CheckReductionFinishInternal();

		std::pair<pmExecutionStub*, ulong> mLastSubtask;

		ulong mReductionsDone;
		ulong mExternalReductionsRequired;
		bool mReduceState;

		pmMachine* mSendToMachine;			// Machine to which this machine will send
		pmTask* mTask;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

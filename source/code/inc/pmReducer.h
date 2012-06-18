
#ifndef __PM_REDUCER__
#define __PM_REDUCER__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <vector>

namespace pm
{

class pmTask;
class pmMachine;

class pmReducer : public pmBase
{
	public:
		pmReducer(pmTask* pTask);
		virtual ~pmReducer();

		pmStatus CheckReductionFinish();
		pmStatus AddSubtask(ulong pSubtaskId);
	
	private:
		pmStatus PopulateExternalMachineList();
		ulong GetMaxPossibleExternalReductionReceives(uint pFollowingMachineCount);
        pmStatus CheckReductionFinishInternal();

		ulong mLastSubtaskId;
		uint mCurrentStubId;

		ulong mReductionsDone;
		ulong mExternalReductionsRequired;
		bool mReduceState;

		pmMachine* mSendToMachine;			// Machine to which this machine will send
		pmTask* mTask;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

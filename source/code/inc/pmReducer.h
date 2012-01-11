
#ifndef __PM_REDUCER__
#define __PM_REDUCER__

#include "pmInternalDefinitions.h"
#include "pmResourceLock.h"
#include "pmTask.h"
#include "pmCommand.h"

#include <vector>

namespace pm
{

class pmReducer : public pmBase
{
	public:
		pmReducer(pmTask* pTask);
		virtual ~pmReducer();

		pmStatus CheckReductionFinish(bool pAlreadyLocked = false);
		pmStatus AddSubtask(ulong pSubtaskId);
	
	private:
		pmStatus PopulateExternalMachineList();
		ulong GetMaxPossibleExternalReductionReceives(uint pFollowingMachineCount);

		ulong mLastSubtaskId;
		uint mCurrentStubId;

		ulong mReductionsDone;
		ulong mExternalReductionsRequired;
		bool mReduceState;

		pmMachine* mSendToMachine;			// Index of machine to which this machine will send
		pmTask* mTask;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

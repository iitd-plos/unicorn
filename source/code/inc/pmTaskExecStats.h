
#ifndef __PM_TASK_EXEC_STATS__
#define __PM_TASK_EXEC_STATS__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <map>

namespace pm
{

class pmExecutionStub;

class pmTaskExecStats : public pmBase
{
	public:
		typedef struct stubStats
		{
			ulong subtasksExecuted;
			double executionTime;	  // in secs
			uint stealAttempts;

			stubStats();
		} stubStats;

		pmTaskExecStats();
		virtual ~pmTaskExecStats();

		pmStatus RecordSubtaskExecutionStats(pmExecutionStub* pStub, ulong pSubtasksExecuted, double pExecutionTimeInSecs);
		double GetStubExecutionRate(pmExecutionStub* pStub);

		uint GetStealAttempts(pmExecutionStub* pStub);
		pmStatus RecordStealAttempt(pmExecutionStub* pStub);
		pmStatus ClearStealAttempts(pmExecutionStub* pStub);

	private:
		std::map<pmExecutionStub*, stubStats> mStats;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

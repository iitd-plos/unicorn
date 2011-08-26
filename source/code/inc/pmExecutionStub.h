
#ifndef __PM_EXECUTION_STUB__
#define __PM_EXECUTION_STUB__

#include "pmInternalDefinitions.h"
#include "pmThread.h"
#include "pmScheduler.h"

namespace pm
{

class pmSignalWait;
class pmThreadCommand;

/**
 * \brief The controlling thread of each processing element.
 */

class pmExecutionStub : public THREADING_IMPLEMENTATION_CLASS
{
	public:
		pmExecutionStub();
		~pmExecutionStub();

		virtual pmStatus Execute(pmScheduler::subtaskRange pRange);

		virtual pmStatus ThreadSwitchCallback(pmThreadCommand* pCommand);

	private:
		pmSignalWait mSignalWait;
};

} // end namespace pm

#endif


#ifndef __PM_STUB_MANAGER__
#define __PM_STUB_MANAGER__

#include "pmInternalDefinitions.h"

#include <vector>

namespace pm
{

class pmExecutionStub;

/**
 * \brief The representation of a parallel task.
 */

class pmStubManager
{
	public:
		pmStubManager();
		~pmStubManager();

		ulong GetStubCount();
		pmExecutionStub* GetStubAtIndex(ulong Index);

	private:
		pmStatus CreateExecutionStubs();
		pmStatus DestroyExecutionStubs();

		std::vector<pmExecutionStub*> mStubVector;
};

} // end namespace pm

#endif

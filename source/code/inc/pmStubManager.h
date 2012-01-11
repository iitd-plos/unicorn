
#ifndef __PM_STUB_MANAGER__
#define __PM_STUB_MANAGER__

#include "pmInternalDefinitions.h"
#include "pmHardware.h"

#include <vector>

namespace pm
{

class pmExecutionStub;

/**
 * \brief The representation of a parallel task.
 */

class pmStubManager : public pmBase
{
	public:
		static pmStubManager* GetStubManager();
		pmStatus DestroyStubManager();

		size_t GetProcessingElementsCPU();
		size_t GetProcessingElementsGPU();
		size_t GetStubCount();

		pmExecutionStub* GetStub(pmProcessingElement* pDevice);
		pmExecutionStub* GetStub(uint pIndex);

	private:
		pmStubManager();
		virtual ~pmStubManager();

		pmStatus CreateExecutionStubs();
		pmStatus DestroyExecutionStubs();

		pmStatus CountAndProbeProcessingElements();

		std::vector<pmExecutionStub*> mStubVector;
		
		size_t mProcessingElementsCPU;
		size_t mProcessingElementsGPU;
		size_t mStubCount;

		static pmStubManager* mStubManager;
};

} // end namespace pm

#endif

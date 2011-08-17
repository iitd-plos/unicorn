
#ifndef __PM_SCHEDULER__
#define __PM_SCHEDULER__

#include "pmInternalDefinitions.h"

namespace pm
{

/**
 * \brief The task scheduler
 * An instance of this scheduler runs on all machines
*/
class pmScheduler
{
	public:
		static pmScheduler* GetScheduler
			();
		pmStatus DestroyController();

		pmStatus FetchMemoryRegion(void* pStartAddress, size_t pOffset, size_t pLength);

		pmStatus SetLastErrorCode(uint pErrorCode) {mLastErrorCode = pErrorCode; return pmSuccess;}
		uint GetLastErrorCode() {return mLastErrorCode;}

	private:
		pmController() {mLastErrorCode = 0;}
	
		static pmStatus CreateAndInitializeController();

		static pmController* mController;
		uint mLastErrorCode;
};

} // end namespace pm

#endif


#ifndef __PM_CALLBACK_CHAIN__
#define __PM_CALLBACK_CHAIN__

#include "pmInternalDefinitions.h"

namespace pm
{

class pmCallbackUnit;

/**
 * \brief The list of callbacks to be called for the task.
 */

class pmCallbackChain : public pmBase
{
	public:
		pmCallbackChain(uint pChainLength, pmCallbackUnit* pCallbackUnits);
		~pmCallbackChain();

		pmStatus AddCallbackUnit(pmCallbackUnit& pCallbackUnit);
		uint GetCallbackChainLength() {return mChainLength;}
		pmCallbackUnit GetCallbackUnitAtIndex(uint pIndex);

	private:
		uint mChainLength;
		pmCallbackUnit* mCallbackUnits;
};

} // end namespace pm

#endif

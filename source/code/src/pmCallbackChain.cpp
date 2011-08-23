
#include "pmCallbackChain.h"
#include "pmCallback.h"

namespace pm
{

pmCallbackChain::pmCallbackChain(uint pChainLength, pmCallbackUnit* pCallbackUnits)
{
	mChainLength = pChainLength;
	mCallbackUnits = pCallbackUnits;
}

pmCallbackChain::~pmCallbackChain()
{
}

pmStatus pmCallbackChain::AddCallbackUnit(pmCallbackUnit& pCallbackUnit)
{
}

pmCallbackUnit pmCallbackChain::GetCallbackUnitAtIndex(uint pIndex)
{
	if(pIndex >= mChainLength)
		throw pmCallbackException(pmCallbackException::INVALID_CHAIN_INDEX);

	return mCallbackUnits[pIndex];
}

};
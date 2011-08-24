
#include "pmTask.h"
#include "pmMemSection.h"
#include "pmCallbackChain.h"

namespace pm
{

pmTask::pmTask(pmMemSection pMemRO, pmMemSection pMemRW, ulong pSubtaskCount, pmCallbackChain pChain, pmHardware pOriginatingHost /* = pmNetwork::GetNetwork()->GetHostId() */,
	pmCluster pCluster /* = PM_GLOBAL_CLUSTER */, ushort pPriority /* = MAX_PRIORITY_LEVEL */)
{
	mMemRO = pMemRO;
	mMemRW = pMemRW;
	mSubtaskCount = pSubtaskCount;
	mChain = pChain;
	mOriginatingHost = pOriginatingHost;
	mCluster = pCluster;
	mPriority = pPriority;
}

pmTask::~pmTask()
{
}

};
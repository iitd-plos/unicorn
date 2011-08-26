
#include "pmExecutionStub.h"
#include "pmSignalWait.h"
#include "pmCommand.h"

namespace pm
{

pmExecutionStub::pmExecutionStub()
{
}

pmExecutionStub::~pmExecutionStub()
{
}

pmStatus pmExecutionStub::Execute(pmScheduler::subtaskRange pRange)
{
	pmScheduler::subtaskRange* lRange = new pmScheduler::subtaskRange(pRange);
	pmThreadCommand* lCommand = new pmThreadCommand(pmThreadCommand::STUB_COMMAND_WRAPPER, lRange, sizeof(lRange));

	return SwitchThread(lCommand);
}

pmStatus pmExecutionStub::ThreadSwitchCallback(pmThreadCommand* pCommand)
{
	pmScheduler::subtaskRange* lRange = (pmScheduler::subtaskRange*)(pCommand->GetData());

	delete lRange;
	delete pCommand;

	return pmSuccess;
}

};
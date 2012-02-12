#include <iostream>
#include "pmPublicDefinitions.h"

using namespace pm;

#define ARRAY_LEN 1024

void loadData(void* pMem)
{
	int* lMem = (int*)pMem;
	for(int i=0; i<ARRAY_LEN; ++i)
		lMem[i] = i;
}

bool checkMem(void* pMem)
{
	int* lMem = (int*)pMem;
	for(int i=0; i<ARRAY_LEN; ++i)
	{
		if(lMem[i] != i)
			return false;
	}

	return true;
}

pmStatus sampleCB(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo)
{
	if(!checkMem(pSubtaskInfo.inputMem))
		std::cout << "Data Error ... Subtask " << pSubtaskInfo.subtaskId << std::endl;

	return pmSuccess;
}

int main()
{
	if(pmInitialize() != pmSuccess)
	{
		std::cout << "Initialization ...	Fail" << std::endl;
		exit(1);
	}

	pmCallbacks lCallbacks;
	pmCallbackHandle lCallbackHandle;

        lCallbacks.dataDistribution = NULL;
        lCallbacks.subtask_cpu = sampleCB;
        lCallbacks.subtask_gpu_cuda = NULL;

	if(pmRegisterCallbacks("CB", lCallbacks, &lCallbackHandle) == pmSuccess)
		std::cout << "Callback Registration ...	Pass" << std::endl;
	else
		std::cout << "Callback Registration ...	Fail" << std::endl;

	if(pmGetHostId() == 0)
	{
		pmMemHandle lInputMem;
		pmMemHandle lOutputMem;

		if(pmCreateMemory(INPUT_MEM_READ_ONLY, ARRAY_LEN * sizeof(int), &lInputMem) == pmSuccess)
			std::cout << "Input Mem Creation ...	Pass" << std::endl;
		else
			std::cout << "Input Mem Creation ...	Fail" << std::endl;

		loadData(lInputMem);

		if(pmCreateMemory(OUTPUT_MEM_WRITE_ONLY, ARRAY_LEN * sizeof(int), &lOutputMem) == pmSuccess)
			std::cout << "Output Mem Creation ...	Pass" << std::endl;
		else
			std::cout << "Output Mem Creation ...	Fail" << std::endl;

		pmTaskDetails lTaskDetails;
		pmTaskHandle lTaskHandle;

		lTaskDetails.taskConf = NULL;
		lTaskDetails.taskConfLength = 0;
		lTaskDetails.inputMem = lInputMem;
		lTaskDetails.outputMem = lOutputMem;
		lTaskDetails.callbackHandle = lCallbackHandle;
		lTaskDetails.subtaskCount = ARRAY_LEN;

		if(pmSubmitTask(lTaskDetails, &lTaskHandle) == pmSuccess)
			std::cout << "Task Submission ...	Pass" << std::endl;
		else
			std::cout << "Task Submission ...	Fail" << std::endl;

		if(pmReleaseTask(lTaskHandle) == pmSuccess)
			std::cout << "Task Release ...	Pass" << std::endl;
		else
			std::cout << "Task Release ...	Fail" << std::endl;

		if(pmReleaseMemory(lInputMem) == pmSuccess)
			std::cout << "Input Mem Release ...	Pass" << std::endl;
		else
			std::cout << "Input Mem Release ...	Fail" << std::endl;
		
		if(pmReleaseMemory(lOutputMem) == pmSuccess)
			std::cout << "Output Mem Release ...	Pass" << std::endl;
		else
			std::cout << "Output Mem Release ...	Fail" << std::endl;
	}
	
	if(pmReleaseCallbacks(lCallbackHandle) == pmSuccess)
		std::cout << "Callback Release ...	Pass" << std::endl;
	else
		std::cout << "Callback Release ...	Fail" << std::endl;

	if(pmFinalize() != pmSuccess)
		std::cout << "Finalization ...		Fail" << std::endl;
}




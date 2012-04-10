
#ifndef __PM_CALLBACK__
#define __PM_CALLBACK__

#include "pmBase.h"

namespace pm
{

/**
 * \brief The user callback for a task (wrapper for DataDistribution (PreSubtask, PostSubtask), Subtask, DeviceSelection, DataTransfer, etc. callbacks)
 */

class pmTask;
class pmProcessingElement;

class pmCallback : public pmBase
{
	public:

	protected:
		pmCallback();
		virtual ~pmCallback();

	private:
};

class pmDataDistributionCB : public pmCallback
{
	public:
		pmDataDistributionCB(pmDataDistributionCallback pCallback);
		virtual ~pmDataDistributionCB();

		virtual pmStatus Invoke(pmTask* pTask, ulong pSubtaskId, pmDeviceTypes pDeviceType);

	private:
		pmDataDistributionCallback mCallback;
};

class pmSubtaskCB : public pmCallback
{
	public:
		pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA);
		virtual ~pmSubtaskCB();

		virtual pmStatus Invoke(pmDeviceTypes pDeviceType, pmTask* pTask, ulong pSubtaskId, size_t pBoundHardwareDeviceIndex);

		virtual bool IsCallbackDefinedForDevice(pmDeviceTypes pDeviceType);

	private:
		pmSubtaskCallback_CPU mCallback_CPU;
		pmSubtaskCallback_GPU_CUDA mCallback_GPU_CUDA;
};

class pmDataReductionCB : public pmCallback
{
	public:
		pmDataReductionCB(pmDataReductionCallback pCallback);
		virtual ~pmDataReductionCB();

		virtual pmStatus Invoke(pmTask* pTask, ulong pSubtaskId1, ulong pSubtaskId2);

	private:
		pmDataReductionCallback mCallback;
};

class pmDataScatterCB : public pmCallback
{
	public:
		pmDataScatterCB(pmDataScatterCallback pCallback);
		virtual ~pmDataScatterCB();

		virtual pmStatus Invoke(pmTask* pTask);

	private:
		pmDataScatterCallback mCallback;
};

class pmDeviceSelectionCB : public pmCallback
{
	public:
		pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback);
		virtual ~pmDeviceSelectionCB();

		virtual bool Invoke(pmTask* pTask, pmProcessingElement* pProcessingElement);

	private:
		pmDeviceSelectionCallback mCallback;
};

class pmPreDataTransferCB : public pmCallback
{
	public:
		pmPreDataTransferCB(pmPreDataTransferCallback pCallback);
		virtual ~pmPreDataTransferCB();

		virtual pmStatus Invoke();

	private:
		pmPreDataTransferCallback mCallback;
};

class pmPostDataTransferCB : public pmCallback
{
	public:
		pmPostDataTransferCB(pmPostDataTransferCallback pCallback);
		virtual ~pmPostDataTransferCB();

		virtual pmStatus Invoke();

	private:
		pmPostDataTransferCallback mCallback;
};

} // end namespace pm

#endif


/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 */

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
class pmExecutionStub;

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

		virtual pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId);

	private:
		pmDataDistributionCallback mCallback;
};

class pmSubtaskCB : public pmCallback
{
	public:
		pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom);
		virtual ~pmSubtaskCB();

		virtual pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, size_t pBoundHardwareDeviceIndex);

		virtual bool IsCallbackDefinedForDevice(pmDeviceType pDeviceType);

	private:
		pmSubtaskCallback_CPU mCallback_CPU;
        pmSubtaskCallback_GPU_CUDA mCallback_GPU_CUDA;
        pmSubtaskCallback_GPU_Custom mCallback_GPU_Custom;
};

class pmDataReductionCB : public pmCallback
{
	public:
		pmDataReductionCB(pmDataReductionCallback pCallback);
		virtual ~pmDataReductionCB();

		virtual pmStatus Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmExecutionStub* pStub2, ulong pSubtaskId2);

	private:
		pmDataReductionCallback mCallback;
};

class pmDataRedistributionCB : public pmCallback
{
	public:
		pmDataRedistributionCB(pmDataRedistributionCallback pCallback);
		virtual ~pmDataRedistributionCB();

		virtual pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId);

	private:
		pmDataRedistributionCallback mCallback;
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


/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institute of Technology, New Delhi. Redistribution, 
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

	private:
};

class pmDataDistributionCB : public pmCallback
{
	public:
		pmDataDistributionCB(pmDataDistributionCallback pCallback);

		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) const;

	private:
		pmDataDistributionCallback mCallback;
};

class pmSubtaskCB : public pmCallback
{
	public:
    #ifdef SUPPORT_OPENCL
        pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom, std::string pOpenCLImpplementation);
    #else
		pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom);
    #endif

		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo, void* pStreamPtr = NULL) const;

		bool IsCallbackDefinedForDevice(pmDeviceType pDeviceType) const;
        bool HasCustomGpuCallback() const;
        bool HasBothCpuAndGpuCallbacks() const;
        bool HasOpenCLCallback() const;

	private:
		pmSubtaskCallback_CPU mCallback_CPU;
        pmSubtaskCallback_GPU_CUDA mCallback_GPU_CUDA;
        pmSubtaskCallback_GPU_Custom mCallback_GPU_Custom;
        std::string mOpenCLImplementation;
};

class pmDataReductionCB : public pmCallback
{
	public:
		pmDataReductionCB(pmDataReductionCallback pCallback);

		pmStatus Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, bool pMultiAssign1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, bool pMultiAssign2) const;

	private:
		pmDataReductionCallback mCallback;
};

class pmDataRedistributionCB : public pmCallback
{
	public:
		pmDataRedistributionCB(pmDataRedistributionCallback pCallback);

		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign) const;

	private:
		pmDataRedistributionCallback mCallback;
};

class pmDeviceSelectionCB : public pmCallback
{
	public:
		pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback);

		bool Invoke(pmTask* pTask, const pmProcessingElement* pProcessingElement) const;

	private:
		pmDeviceSelectionCallback mCallback;
};

class pmPreDataTransferCB : public pmCallback
{
	public:
		pmPreDataTransferCB(pmPreDataTransferCallback pCallback);

		pmStatus Invoke() const;

	private:
		pmPreDataTransferCallback mCallback;
};

class pmPostDataTransferCB : public pmCallback
{
	public:
		pmPostDataTransferCB(pmPostDataTransferCallback pCallback);

		pmStatus Invoke() const;

	private:
		pmPostDataTransferCallback mCallback;
};

} // end namespace pm

#endif

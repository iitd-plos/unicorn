
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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
struct pmCudaMemcpyCommand;

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
        pmStatus InvokeDirect(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo) const;

	private:
		const pmDataDistributionCallback mCallback;
};

class pmSubtaskCB : public pmCallback
{
	public:
    #ifdef SUPPORT_OPENCL
        pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom, std::string pOpenCLImpplementation);
    #else
		pmSubtaskCB(pmSubtaskCallback_CPU pCallback_CPU, pmSubtaskCallback_GPU_CUDA pCallback_GPU_CUDA, pmSubtaskCallback_GPU_Custom pCallback_GPU_Custom);
    #endif

    #ifdef SUPPORT_CUDA
		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo, std::vector<pmCudaMemcpyCommand>* pHostToDeviceCommands = NULL, std::vector<pmCudaMemcpyCommand>* pDeviceToHostCommands = NULL, void* pStreamPtr = NULL, pmReductionDataType pSentinelCompressionReductionDataType = MAX_REDUCTION_DATA_TYPES) const;
    #else
		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, pmSplitInfo* pSplitInfo, bool pMultiAssign, const pmTaskInfo& pTaskInfo, const pmSubtaskInfo& pSubtaskInfo) const;
    #endif

		bool IsCallbackDefinedForDevice(pmDeviceType pDeviceType) const;
        bool HasCustomGpuCallback() const;
        bool HasBothCpuAndGpuCallbacks() const;
        bool HasOpenCLCallback() const;
    
        pmSubtaskCallback_CPU GetCpuCallback() const;

	private:
		const pmSubtaskCallback_CPU mCallback_CPU;
        const pmSubtaskCallback_GPU_CUDA mCallback_GPU_CUDA;
        const pmSubtaskCallback_GPU_Custom mCallback_GPU_Custom;
        const std::string mOpenCLImplementation;
};

class pmDataReductionCB : public pmCallback
{
	public:
		pmDataReductionCB(pmDataReductionCallback pCallback);

		pmStatus Invoke(pmTask* pTask, pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, bool pMultiAssign1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, bool pMultiAssign2) const;

        const pmDataReductionCallback GetCallback() const;
    
	private:
		const pmDataReductionCallback mCallback;
};

class pmDataRedistributionCB : public pmCallback
{
	public:
		pmDataRedistributionCB(pmDataRedistributionCallback pCallback);

		pmStatus Invoke(pmExecutionStub* pStub, pmTask* pTask, ulong pSubtaskId, pmSplitInfo* pSplitInfo, bool pMultiAssign) const;

	private:
		const pmDataRedistributionCallback mCallback;
};

class pmDeviceSelectionCB : public pmCallback
{
	public:
		pmDeviceSelectionCB(pmDeviceSelectionCallback pCallback);

		bool Invoke(pmTask* pTask, const pmProcessingElement* pProcessingElement) const;

	private:
		const pmDeviceSelectionCallback mCallback;
};

class pmPreDataTransferCB : public pmCallback
{
	public:
		pmPreDataTransferCB(pmPreDataTransferCallback pCallback);

		pmStatus Invoke() const;

	private:
		const pmPreDataTransferCallback mCallback;
};

class pmPostDataTransferCB : public pmCallback
{
	public:
		pmPostDataTransferCB(pmPostDataTransferCallback pCallback);

		pmStatus Invoke() const;

	private:
		const pmPostDataTransferCallback mCallback;
};

class pmTaskCompletionCB : public pmCallback
{
	public:
		pmTaskCompletionCB(pmTaskCompletionCallback pCallback);

		pmStatus Invoke(pmTask* pTask) const;
    
        pmTaskCompletionCallback GetCallback() const;

        void* GetUserData() const;
        void SetUserData(void* pUserData);

	private:
		const pmTaskCompletionCallback mCallback;
        void* mUserData;
};

} // end namespace pm

#endif

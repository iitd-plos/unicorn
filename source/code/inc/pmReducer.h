
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

#ifndef __PM_REDUCER__
#define __PM_REDUCER__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"

#include <vector>
#include <limits>

namespace pm
{

class pmTask;
class pmMachine;
class pmExecutionStub;

namespace reducer
{

struct lastSubtaskData
{
    pmExecutionStub* stub;
    ulong subtaskId;
    finalize_ptr<pmSplitInfo> splitInfo;
    
    lastSubtaskData()
    : stub(NULL)
    , subtaskId(std::numeric_limits<ulong>::max())
    {}

    lastSubtaskData(const lastSubtaskData& pLastSubtaskData)
    : stub(pLastSubtaskData.stub)
    , subtaskId(pLastSubtaskData.subtaskId)
    , splitInfo(pLastSubtaskData.splitInfo.get_ptr() ? new pmSplitInfo(*pLastSubtaskData.splitInfo.get_ptr()) : NULL)
    {}
};

}
    
class pmReducer : public pmBase
{
    friend void PostMpiReduceCommandCompletionCallback(const pmCommandPtr& pCommand);
    friend void PostExternalReduceCommandCompletionCallback(const pmCommandPtr& pCommand);

    public:
		pmReducer(pmTask* pTask);

		void CheckReductionFinish();
        void HandleReductionFinish();
		void AddSubtask(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo);
    
        void SignalSendToMachineAboutNoLocalReduction();
        void RegisterNoReductionReqdResponse();
    
        void PrepareForExternalReceive(communicator::subtaskMemoryReduceStruct& pStruct);

        void ReduceSubtasks(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionOpType pReductionOperation, pmReductionDataType pReductionDataType);
    
        void ReduceExternalMemory(pmExecutionStub* pStub, const pmCommandPtr& pCommand);

        void PerformDirectExternalReductions();
        void RegisterExternalReductionFinish();
    
	private:
		void PopulateExternalMachineList();
		ulong GetMaxPossibleExternalReductionReceives(uint pFollowingMachineCount);
        void CheckReductionFinishInternal();
        void AddReductionFinishEvent();

        void SignalSendToMachineAboutNoLocalReductionInternal();

        template<typename datatype>
        void ReduceSubtasks(pmExecutionStub* pStub1, ulong pSubtaskId1, pmSplitInfo* pSplitInfo1, pmExecutionStub* pStub2, ulong pSubtaskId2, pmSplitInfo* pSplitInfo2, pmReductionOpType pReductionType);

        template<typename datatype>
        void ReduceMemories(datatype* pShadowMem1, datatype* pShadowMem2, size_t pDataCount, pmReductionOpType pReductionType);

        template<typename datatype>
        void ReduceMemoriesCompressed(datatype* pShadowMem1, datatype* pShadowMem2, size_t pDataCount, pmReductionOpType pReductionType, datatype pSentinel);

        reducer::lastSubtaskData mLastSubtask;

		ulong mReductionsDone;
		ulong mExternalReductionsRequired;
		bool mReduceState;

		const pmMachine* mSendToMachine;			// Machine to which this machine will send
		pmTask* mTask;
    
        std::vector<communicator::subtaskMemoryReduceStruct> mSubtaskMemoryReduceStructVector;

        bool mReductionTerminated;
    
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif

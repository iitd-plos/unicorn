
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


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

#ifndef __PM_SUBTASK_SPLITTER__
#define __PM_SUBTASK_SPLITTER__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <limits>
#include <set>
#include <list>

#ifdef SUPPORT_SPLIT_SUBTASKS

namespace pm
{

class pmTask;
class pmExecutionStub;

namespace splitter
{

struct splitRecord
{
    pmExecutionStub* sourceStub; // Stub to which the original unsplitted subtask was assigned
    ulong subtaskId;
    uint splitCount;
    uint splitId;
    uint pendingCompletions;
    std::vector<std::pair<pmExecutionStub*, bool> > assignedStubs;
    bool reassigned;
    bool prefetched;
    
    splitRecord()
    : sourceStub(NULL)
    , subtaskId(std::numeric_limits<ulong>::max())
    , splitCount(0)
    , splitId(0)
    , pendingCompletions(0)
    , reassigned(false)
    , prefetched(false)
    {}
    
    splitRecord(pmExecutionStub* pStub, ulong pSubtaskId, uint pSplitCount)
    : sourceStub(pStub)
    , subtaskId(pSubtaskId)
    , splitCount(pSplitCount)
    , splitId(0)
    , pendingCompletions(pSplitCount)
    , reassigned(false)
    , prefetched(false)
    {
        assignedStubs.reserve(splitCount);
    }
};

}
    
class pmSubtaskSplitter;
    
class pmSplitGroup
{
    friend class pmSubtaskSplitter;

public:
    pmSplitGroup(const pmSubtaskSplitter* pSubtaskSplitter)
    : mDummyEventsFreezed(false)
    , mSubtaskSplitter(pSubtaskSplitter)
    {}
    
    std::unique_ptr<pmSplitSubtask> GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub);
    void FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination);

    bool Negotiate(ulong pSubtaskId);
    void StubHasProcessedDummyEvent(pmExecutionStub* pStub);

    void FreezeDummyEvents();
    void PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId);

private:
    void AddDummyEventToRequiredStubs();
    void AddDummyEventToStub(pmExecutionStub* pStub);

    std::vector<pmExecutionStub*> mConcernedStubs;    // Stubs in each split group

    bool mDummyEventsFreezed;
    std::set<pmExecutionStub*> mStubsWithDummyEvent;    // SPLIT_SUBTASK_CHECK
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mDummyEventLock;

    std::list<splitter::splitRecord> mSplitRecordList;
    std::map<ulong, typename std::list<splitter::splitRecord>::iterator> mSplitRecordMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mSplitRecordListLock;
    
    const pmSubtaskSplitter* mSubtaskSplitter;
};
    
class pmSubtaskSplitter : public pmBase
{
    friend class pmSplitGroup;

public:
    pmSubtaskSplitter(pmTask* pTask);

    pmDeviceType GetSplittingType();
    bool IsSplitting(pmDeviceType pDeviceType);
    size_t GetSplitFactor();
    
    bool IsSplitGroupLeader(pmExecutionStub* pStub);
    
    std::unique_ptr<pmSplitSubtask> GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub);
    void FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination);
    
    bool Negotiate(pmExecutionStub* pStub, ulong pSubtaskId);
    void StubHasProcessedDummyEvent(pmExecutionStub* pStub);
    
    void FreezeDummyEvents();
    void PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId);

private:
    void FindConcernedStubs(pmDeviceType pDeviceType);

    pmTask* mTask;
    uint mSplitFactor;
    uint mSplitGroups;
    pmDeviceType mSplittingType;

    std::vector<pmSplitGroup> mSplitGroupVector;
    std::map<pmExecutionStub*, uint> mSplitGroupMap;
};

} // end namespace pm

#endif

#endif


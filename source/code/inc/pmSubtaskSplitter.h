
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
class pmLocalTask;
class pmExecutionStub;

namespace splitter
{

struct splitRecord : public pmNonCopyable
{
    pmExecutionStub* sourceStub; // Stub to which the original unsplitted subtask was assigned
    ulong subtaskId;
    uint splitCount;
    uint splitId;
    uint pendingCompletions;
    std::vector<std::pair<pmExecutionStub*, bool> > assignedStubs;
    bool reassigned;
    bool prefetched;
    double subtaskExecTime;
    
    splitRecord(pmExecutionStub* pStub, ulong pSubtaskId, uint pSplitCount)
    : sourceStub(pStub)
    , subtaskId(pSubtaskId)
    , splitCount(pSplitCount)
    , splitId(0)
    , pendingCompletions(pSplitCount)
    , reassigned(false)
    , prefetched(false)
    , subtaskExecTime(0)
    {
        assignedStubs.reserve(splitCount);
    }
};

}
    
class pmSubtaskSplitter;
    
class pmSplitFactorCalculator
{
public:
    pmSplitFactorCalculator(uint pMaxSplitFactor)
    : mMaxSplitFactor(pMaxSplitFactor)
#ifdef ENABLE_DYNAMIC_SPLITTING
    , mFirstSplitFactor(pMaxSplitFactor)
    , mSecondSplitFactor(std::max((uint)1, pMaxSplitFactor/2))
    , mThirdSplitFactor(0)
#endif
    {}

    void RegisterSubtaskCompletion(uint pSplitFactor, double pTime);
    uint GetNextSplitFactor();
    
private:
#ifdef ENABLE_DYNAMIC_SPLITTING
    double GetAverageTimeForSplitFactor(uint pSplitFactor);
#endif

    uint mMaxSplitFactor;
    
#ifdef ENABLE_DYNAMIC_SPLITTING
    uint mFirstSplitFactor;
    uint mSecondSplitFactor;
    uint mThirdSplitFactor;
    
    std::map<uint, std::pair<double, uint>> mMap;    // splitFactor versus pair of total time across all instances and number of instances
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
#endif
};
    
class pmSplitGroup : public pmBase
{
    friend class pmSubtaskSplitter;

public:
    pmSplitGroup(const pmSubtaskSplitter* pSubtaskSplitter, std::vector<pmExecutionStub*>&& pConcernedStubs)
    : mConcernedStubs(std::move(pConcernedStubs))
    , mDummyEventsFreezed(false)
    , mSplitRecordListLock __LOCK_NAME__("pmSplitGroup::mSplitRecordListLock")
    , mSubtaskSplitter(pSubtaskSplitter)
    , mSplitFactorCalculator((uint)mConcernedStubs.size())
    {}
    
    std::unique_ptr<pmSplitSubtask> GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub);
    void FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination, double pExecTime);

    bool Negotiate(ulong pSubtaskId, std::vector<pmExecutionStub*>& pStubsToBeCancelled, pmExecutionStub*& pSourceStub);
    void StubHasProcessedDummyEvent(pmExecutionStub* pStub);

    void FreezeDummyEvents();
    void PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId);

    float GetProgress(ulong pSubtaskId);

private:
    void AddDummyEventToRequiredStubs();
    void AddDummyEventToStub(pmExecutionStub* pStub);
    
    std::vector<pmExecutionStub*> mConcernedStubs;    // Stubs in each split group

    bool mDummyEventsFreezed;
    std::set<pmExecutionStub*> mStubsWithDummyEvent;    // SPLIT_SUBTASK_CHECK
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mDummyEventLock;

    std::list<std::shared_ptr<splitter::splitRecord>> mSplitRecordList;
    std::map<ulong, typename std::list<std::shared_ptr<splitter::splitRecord>>::iterator> mSplitRecordMap;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mSplitRecordListLock;
    
    const pmSubtaskSplitter* mSubtaskSplitter;
    pmSplitFactorCalculator mSplitFactorCalculator;
};

class pmSubtaskSplitter : public pmBase
{
    friend class pmSplitGroup;

public:
    pmSubtaskSplitter(pmTask* pTask);

    pmDeviceType GetSplittingType();
    bool IsSplitting(pmDeviceType pDeviceType);
    
    std::unique_ptr<pmSplitSubtask> GetPendingSplit(ulong* pSubtaskId, pmExecutionStub* pSourceStub);
    void FinishedSplitExecution(ulong pSubtaskId, uint pSplitId, pmExecutionStub* pStub, bool pPrematureTermination, double pExecTime);
    
    bool Negotiate(pmExecutionStub* pStub, ulong pSubtaskId, std::vector<pmExecutionStub*>& pStubsToBeCancelled, pmExecutionStub*& pSourceStub);
    void StubHasProcessedDummyEvent(pmExecutionStub* pStub);
    
    void FreezeDummyEvents();
    void PrefetchSubscriptionsForUnsplittedSubtask(pmExecutionStub* pStub, ulong pSubtaskId);
    
    void MakeDeviceGroups(const std::vector<const pmProcessingElement*>& pDevices, std::vector<std::vector<const pmProcessingElement*>>& pDeviceGroups, std::map<const pmProcessingElement*, std::vector<const pmProcessingElement*>*>& pQueryMap, ulong& pUnsplittedDevices);

    std::vector<std::pair<std::vector<const pmProcessingElement*>, std::pair<ulong, ulong>>> MakeInitialSchedulingAllotments(pmLocalTask* pLocalTask);
    
    float GetProgress(ulong pSubtaskId, pmExecutionStub* pStub);

private:
    pmTask* mTask;
    pmDeviceType mSplittingType;

    std::vector<std::shared_ptr<pmSplitGroup>> mSplitGroupVector;
    std::map<pmExecutionStub*, uint> mSplitGroupMap;
};

} // end namespace pm

#endif

#endif


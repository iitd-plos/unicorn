
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

#ifndef __PM_STEAL_AGENT__
#define __PM_STEAL_AGENT__

#include "pmBase.h"
#include "pmResourceLock.h"

#ifdef USE_STEAL_AGENT_PER_NODE

namespace pm
{
    
class pmTask;
class pmExecutionStub;

namespace stealAgent
{
    struct stubData
    {
        ulong subtasksPendingInStubQueue;
        ulong subtasksPendingInPipeline;
        bool isMultiAssigning;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS stubLock;
        
        stubData()
        : subtasksPendingInStubQueue(0)
        , subtasksPendingInPipeline(0)
        , isMultiAssigning(false)
        , stubLock __LOCK_NAME__("stealAgent::stubData::stubLock")
        {}
    };
    
    struct dynamicAggressionStubData
    {
        double stealStartTime;
        double totalStealWaitTime;
        double totalSteals;
        double totalSubscriptionFetchTime;
        double totalSubscriptionsFetched;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS stubLock;
        
        dynamicAggressionStubData()
        : stealStartTime(0)
        , totalStealWaitTime(0)
        , totalSteals(0)
        , totalSubscriptionFetchTime(0)
        , totalSubscriptionsFetched(0)
        {}
    };
}
    
// Most methods in this class are not synchronized. All stubs have exclusive data locations.
// If USE_DYNAMIC_AFFINITY is defined, mHibernatedSubtasks is synchronized
class pmStealAgent : public pmBase
{
public:
    pmStealAgent(pmTask* pTask);
    
    void RegisterPendingSubtasks(pmExecutionStub* pStub, ulong pPendingSubtasks);   // subtasks in stub queue
    void DeregisterPendingSubtasks(pmExecutionStub* pStub, ulong pFinishedSubtasks); // subtasks removed from stub queue
    
    void RegisterExecutingSubtasks(pmExecutionStub* pStub, ulong pExecutingSubtasks);   // subtasks in execution but stealable
    void DeregisterExecutingSubtasks(pmExecutionStub* pStub, ulong pFinishedSubtasks);
    void ClearExecutingSubtasks(pmExecutionStub* pStub);
    
    void SetStubMultiAssignment(pmExecutionStub* pStub, bool pCanMultiAssign);
    
    bool HasAnotherStubToStealFrom(pmExecutionStub* pStub, bool pCanMultiAssign);

    pmExecutionStub* GetStubWithMaxStealLikelihood(bool pConsiderMultiAssign, pmExecutionStub* pIgnoreStub = NULL);

#ifdef USE_DYNAMIC_AFFINITY
    void HibernateSubtasks(const std::vector<ulong>& pSubtasksVector);
    ulong GetNextHibernatedSubtask(pmExecutionStub* pStub);
#endif
    
#ifdef ENABLE_DYNAMIC_AGGRESSION
    void RecordStealRequestIssue(pmExecutionStub* pStub);
    void RecordSuccessfulSteal(pmExecutionStub* pStub);
    double GetAverageStealWaitTime(pmExecutionStub* pStub);
    void RecordSubtaskSubscriptionFetchTime(pmExecutionStub* pStub, double pTime);
    double GetAverageSubtaskSubscriptionFetchTime(pmExecutionStub* pStub);
#endif

private:
    pmTask* mTask;
    std::vector<stealAgent::stubData> mStubSink;
    
#ifdef ENABLE_DYNAMIC_AGGRESSION
    std::vector<stealAgent::dynamicAggressionStubData> mDynamicAggressionStubSink;
#endif
    
#ifdef USE_DYNAMIC_AFFINITY
    std::vector<ulong> mHibernatedSubtasks;
    std::shared_ptr<void> mAffinityData;
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mHibernationLock;
#endif
};
    
} // end namespace pm

#endif

#endif

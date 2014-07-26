
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
}
    
// There is no synchronization inside this class. All stubs have exclusive data locations.
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

private:
    pmTask* mTask;
    std::vector<stealAgent::stubData> mStubSink;
};
    
} // end namespace pm

#endif

#endif

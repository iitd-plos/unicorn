
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

#ifndef __PM_TIMED_EVENT__
#define __PM_TIMED_EVENT__

#include "pmBase.h"
#include "pmThread.h"
#include "pmSignalWait.h"
#include <tr1/memory>

namespace pm
{

namespace timed
{
    enum eventIdentifier
    {
        TASK_TIME_OUT,
        CLEAR_TASK_TIME_OUT
    };
    
    struct taskTimeOut
    {
        pmLocalTask* mLocalTask;
    };    

    struct clearTaskTimeOut
    {
        pmLocalTask* mLocalTask;
        ulong mTaskTimeOutTriggerTime;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS* mClearTimeOutSignalWait;
    };    

    struct timedEvent
    {
        eventIdentifier eventId;
        ulong triggerTime;
        union
        {
            taskTimeOut taskTimeOutDetails;
            clearTaskTimeOut clearTaskTimeOutDetails;
        };
    };
};

class pmTimedEventManager : public THREADING_IMPLEMENTATION_CLASS<timed::timedEvent, ulong>
{
    friend class pmController;
    
public:
    static pmTimedEventManager* GetTimedEventManager();
    virtual ~pmTimedEventManager();

    void AddTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime);
    void ClearTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime);
    
private:
    pmTimedEventManager();
    virtual pmStatus ThreadSwitchCallback(timed::timedEvent& pEvent);

    SIGNAL_WAIT_IMPLEMENTATION_CLASS mSignalWait;
    static pmTimedEventManager* mTimedEventManager;
};
    
bool timeOutClearMatchFunc(timed::timedEvent& pEvent, void* pCriterion);
    
} // end namespace pm

#endif

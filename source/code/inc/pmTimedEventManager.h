
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

#ifndef __PM_TIMED_EVENT__
#define __PM_TIMED_EVENT__

#include "pmBase.h"
#include "pmThread.h"
#include "pmSignalWait.h"
#include "pmTask.h"
#include <memory>

namespace pm
{

namespace timed
{
    enum eventIdentifier
    {
        TASK_TIME_OUT,
        CLEAR_TASK_TIME_OUT,
        MAX_TIMED_EVENTS
    };
    
    struct timedEvent : public pmBasicThreadEvent
    {
        eventIdentifier eventId;
        ulong triggerTime;
        
        timedEvent(eventIdentifier pEventId = MAX_TIMED_EVENTS, ulong pTriggerTime = 0)
        : eventId(pEventId)
        , triggerTime(pTriggerTime)
        {}
    };

    struct taskTimeOutEvent : public timedEvent
    {
        pmLocalTask* mLocalTask;
        
        taskTimeOutEvent(eventIdentifier pEventId, ulong pTriggerTime, pmLocalTask* pLocalTask)
        : timedEvent(pEventId, pTriggerTime)
        , mLocalTask(pLocalTask)
        {}
    };    

    struct clearTaskTimeOutEvent : public timedEvent
    {
        pmLocalTask* mLocalTask;
        ulong mTaskTimeOutTriggerTime;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS* mClearTimeOutSignalWait;
        
        clearTaskTimeOutEvent(eventIdentifier pEventId, ulong pTriggerTime, pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime, SIGNAL_WAIT_IMPLEMENTATION_CLASS* pClearTimeOutSignalWait)
        : timedEvent(pEventId, pTriggerTime)
        , mLocalTask(pLocalTask)
        , mTaskTimeOutTriggerTime(pTaskTimeOutTriggerTime)
        , mClearTimeOutSignalWait(pClearTimeOutSignalWait)
        {}
    };    

};

class pmTimedEventManager : public THREADING_IMPLEMENTATION_CLASS<timed::timedEvent, ulong>
{
public:
    static pmTimedEventManager* GetTimedEventManager();
    virtual ~pmTimedEventManager();

    void AddTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime);
    void ClearTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime);
    
private:
    pmTimedEventManager();
    virtual void ThreadSwitchCallback(std::shared_ptr<timed::timedEvent>& pEvent);

    SIGNAL_WAIT_IMPLEMENTATION_CLASS mSignalWait;
};
    
bool timeOutClearMatchFunc(const timed::timedEvent& pEvent, const void* pCriterion);
    
} // end namespace pm

#endif

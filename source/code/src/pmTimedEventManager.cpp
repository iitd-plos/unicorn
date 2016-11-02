
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

#include "pmTimedEventManager.h"
#include "pmTask.h"

namespace pm
{

using namespace timed;

pmTimedEventManager* pmTimedEventManager::GetTimedEventManager()
{
    static pmTimedEventManager lTimedEventManager;
    return &lTimedEventManager;
}

pmTimedEventManager::pmTimedEventManager()
    : mSignalWait(false)
{
}
    
pmTimedEventManager::~pmTimedEventManager()
{
#ifdef DUMP_THREADS
	pmLogger::GetLogger()->Log(pmLogger::MINIMAL, pmLogger::INFORMATION, "Shutting down timed event thread");
#endif
}

void pmTimedEventManager::AddTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime)
{
    SwitchThread(std::shared_ptr<timedEvent>(new taskTimeOutEvent(TASK_TIME_OUT, pTaskTimeOutTriggerTime, pLocalTask)), pTaskTimeOutTriggerTime);

    mSignalWait.Signal();
}
    
void pmTimedEventManager::ClearTaskTimeOutEvent(pmLocalTask* pLocalTask, ulong pTaskTimeOutTriggerTime)
{
    SIGNAL_WAIT_IMPLEMENTATION_CLASS lSignalWait(true);

    ulong lTriggerTime = GetIntegralCurrentTimeInSecs();
    SwitchThread(std::shared_ptr<timedEvent>(new clearTaskTimeOutEvent(CLEAR_TASK_TIME_OUT, lTriggerTime, pLocalTask, pTaskTimeOutTriggerTime, &lSignalWait)), lTriggerTime);

    mSignalWait.Signal();
    lSignalWait.Wait();
}
    
void pmTimedEventManager::ThreadSwitchCallback(std::shared_ptr<timedEvent>& pEvent)
{
    ulong lTime = GetIntegralCurrentTimeInSecs();
    if(lTime < pEvent->triggerTime)
    {
        if(!mSignalWait.WaitWithTimeOut(pEvent->triggerTime))
        {
            SwitchThread(pEvent, pEvent->triggerTime);
            return;
        }
    }
        
    switch(pEvent->eventId)
    {
        case TASK_TIME_OUT:
        {
            taskTimeOutEvent& lEvent = static_cast<taskTimeOutEvent&>(*pEvent);
            pmScheduler::GetScheduler()->TaskCancelEvent(lEvent.mLocalTask);

            break;
        }

        case CLEAR_TASK_TIME_OUT:
        {
            clearTaskTimeOutEvent& lEvent = static_cast<clearTaskTimeOutEvent&>(*pEvent);
            
            DeleteMatchingCommands(lEvent.mTaskTimeOutTriggerTime, timeOutClearMatchFunc, lEvent.mLocalTask);
            lEvent.mClearTimeOutSignalWait->Signal();
         
            break;
        }
            
        default:
            PMTHROW(pmFatalErrorException());
    }
}

bool timeOutClearMatchFunc(const timed::timedEvent& pEvent, const void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case timed::TASK_TIME_OUT:
        {
            const taskTimeOutEvent& lEvent = static_cast<const taskTimeOutEvent&>(pEvent);

            if(lEvent.mLocalTask == static_cast<const pmTask*>(pCriterion))
                return true;

            break;
        }
        
        default:
            return false;
    }
    
    return false;
}
    
}

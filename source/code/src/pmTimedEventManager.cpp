
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
    SIGNAL_WAIT_IMPLEMENTATION_CLASS lSignalWait;

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

bool timeOutClearMatchFunc(const timed::timedEvent& pEvent, void* pCriterion)
{
    switch(pEvent.eventId)
    {
        case timed::TASK_TIME_OUT:
        {
            const taskTimeOutEvent& lEvent = static_cast<const taskTimeOutEvent&>(pEvent);

            if(lEvent.mLocalTask == (pmTask*)pCriterion)
                return true;

            break;
        }
        
        default:
            return false;
    }
    
    return false;
}
    
}

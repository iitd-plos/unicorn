
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

#include "pmEventTimeline.h"
#include "pmNetwork.h"
#include "pmLogger.h"
#include "pmTask.h"

#include <sstream>

#ifdef DUMP_EVENT_TIMELINE

namespace pm
{

pmEventTimeline::pmEventTimeline(const std::string& pName)
    : mName(pName)
    , mHostId(NETWORK_IMPLEMENTATION_CLASS::GetNetwork()->GetHostId())
    , mZeroTime(GetCurrentTimeInSecs())
{
}
    
pmEventTimeline::~pmEventTimeline()
{
    std::stringstream lStream;
    lStream << "Event Timeline " << mName;
    
    auto lIter = mEventMap.begin(), lEnd = mEventMap.end();
    for(; lIter != lEnd; ++lIter)
        lStream << std::endl << lIter->first << " " << lIter->second.first - mZeroTime << " " << lIter->second.second - mZeroTime;

    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str(), true);
}

void pmEventTimeline::RecordEvent(pmTask* pTask, const std::string& pEventName, bool pStart)
{
    if(pTask->ShouldSuppressTaskLogs())
        return;
    
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(pStart && mEventMap.find(pEventName) != mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(!pStart && (mEventMap.find(pEventName) == mEventMap.end() || mEventMap[pEventName].second != -1.0))
        PMTHROW(pmFatalErrorException());
    
    if(pStart)
        mEventMap[pEventName] = std::make_pair(GetCurrentTimeInSecs(), -1.0);
    else
        mEventMap[pEventName] = std::make_pair(mEventMap[pEventName].first, GetCurrentTimeInSecs());
}
    
void pmEventTimeline::RenameEvent(pmTask* pTask, const std::string& pEventName, const std::string& pNewName)
{
    if(pTask->ShouldSuppressTaskLogs())
        return;

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    auto lIter = mEventMap.find(pEventName);
    if(lIter == mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(mEventMap.find(pNewName) != mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    std::pair<double, double> lPair = lIter->second;
    mEventMap.erase(lIter);

    mEventMap[pNewName] = lPair;
}

void pmEventTimeline::StopEventIfRequired(pmTask* pTask, const std::string& pEventName)
{
    if(pTask->ShouldSuppressTaskLogs())
        return;
    
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    auto lIter = mEventMap.find(pEventName);
    EXCEPTION_ASSERT(lIter != mEventMap.end());
    
    if(lIter->second.second == -1.0)
        lIter->second.second = GetCurrentTimeInSecs();
}
    
void pmEventTimeline::DropEvent(pmTask* pTask, const std::string& pEventName)
{
    if(pTask->ShouldSuppressTaskLogs())
        return;
    
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    EXCEPTION_ASSERT(mEventMap.find(pEventName) != mEventMap.end());

    mEventMap.erase(pEventName);
}

} // end namespace pm

#endif

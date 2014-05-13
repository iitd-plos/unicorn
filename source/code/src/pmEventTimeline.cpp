
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

#include "pmEventTimeline.h"
#include "pmNetwork.h"
#include "pmLogger.h"

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
    
    std::map<std::string, std::pair<double, double> >::iterator lIter = mEventMap.begin(), lEnd = mEventMap.end();
    for(; lIter != lEnd; ++lIter)
        lStream << std::endl << lIter->first << " " << lIter->second.first - mZeroTime << " " << lIter->second.second - mZeroTime;

    pmLogger::GetLogger()->LogDeferred(pmLogger::DEBUG_INTERNAL, pmLogger::INFORMATION, lStream.str().c_str(), true);
}

void pmEventTimeline::RecordEvent(const std::string& pEventName, bool pStart)
{
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
    
void pmEventTimeline::RenameEvent(const std::string& pEventName, const std::string& pNewName)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::map<std::string, std::pair<double, double> >::iterator lIter = mEventMap.find(pEventName);
    if(lIter == mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(mEventMap.find(pNewName) != mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    std::pair<double, double> lPair = lIter->second;
    mEventMap.erase(lIter);

    mEventMap[pNewName] = lPair;
}

} // end namespace pm

#endif

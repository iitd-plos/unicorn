
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

#include "pmEventTimeline.h"
#include "pmNetwork.h"

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
    lStream << "Event Timeline " << mName << " [Host " << mHostId << "]" << std::endl;
    
    std::map<std::string, std::pair<double, double> >::iterator lIter = mEventMap.begin(), lEnd = mEventMap.end();
    for(; lIter != lEnd; ++lIter)
        lStream << lIter->first << " " << lIter->second.first - mZeroTime << " " << lIter->second.second - mZeroTime << std::endl;
    
    lStream << std::endl;
    
    std::cout << lStream.str() << std::flush;
}

void pmEventTimeline::RecordEvent(const std::string& pEventName, bool pStart)
{
    if(pStart && mEventMap.find(pEventName) != mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(!pStart && mEventMap.find(pEventName) == mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(pStart)
        mEventMap[pEventName] = std::make_pair(GetCurrentTimeInSecs(), -1.0);
    else
        mEventMap[pEventName] = std::make_pair(mEventMap[pEventName].first, GetCurrentTimeInSecs());
}
    
void pmEventTimeline::RenameEvent(const std::string& pEventName, const std::string& pNewName)
{
    if(mEventMap.find(pEventName) == mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    if(mEventMap.find(pNewName) != mEventMap.end())
        PMTHROW(pmFatalErrorException());
    
    mEventMap[pNewName] = mEventMap[pEventName];
    mEventMap.erase(pEventName);
}

} // end namespace pm

#endif

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

#ifndef __PM_EVENT_TIMELINE__
#define __PM_EVENT_TIMELINE__

#include "pmBase.h"
#include "pmTimer.h"
#include "pmResourceLock.h"

#include <map>

#ifdef DUMP_EVENT_TIMELINE

namespace pm
{
/**
 * \brief The event timeline
 */
    
class pmEventTimeline : public pmBase
{
public:
    pmEventTimeline(const std::string& pName);
    ~pmEventTimeline();

    void RecordEvent(const std::string& pEventName, bool pStart);
    void RenameEvent(const std::string& pEventName, const std::string& pNewName);
    
private:
    std::string mName;
    uint mHostId;
    double mZeroTime;
    std::map<std::string, std::pair<double, double> > mEventMap;   // Event name versus start and end time
    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};
    
} // end namespace pm

#endif

#endif

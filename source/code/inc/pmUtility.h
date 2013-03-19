
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

#ifndef __PM_UTILITY__
#define __PM_UTILITY__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <string>
#include <map>

namespace pm
{

class pmUtility : public pmBase
{
    typedef std::map<std::string, std::pair<void*, size_t> > fileMappingsMapType;

public:
    static void* MapFile(const char* pPath);
    static void* GetMappedFile(const char* pPath);
    static void UnmapFile(const char* pPath);
    
private:
    static fileMappingsMapType& GetFileMappingsMap();
    static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
};

} // end namespace pm

#endif

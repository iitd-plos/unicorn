
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

#ifndef __PM_CACHE__
#define __PM_CACHE__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <list>
#include <unordered_map>

namespace pm
{

/* An LRU cache with ref counted values */
template<typename __key, typename __value, typename __hasher, typename __evictor>
class pmCache : public pmBase, pmNonCopyable
{
public:
    typedef std::pair<const __key, std::shared_ptr<__value>> mappedType;

    pmCache() = default;
    
    pmCache(const __evictor& pEvictor)
    : mEvictor(pEvictor)
    {}

    void Insert(const __key& pKey, std::shared_ptr<__value>& pValue);
    std::shared_ptr<__value>& Get(const __key& pKey);

    bool Purge();
    
private:
    std::shared_ptr<__value> mEmptyValue;
    __evictor mEvictor;

    std::list<mappedType> mCacheList;
    std::unordered_map<const __key, typename std::list<mappedType>::iterator, __hasher> mCacheHash;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#include "../src/pmCache.cpp"

#endif


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

#ifndef __PM_CACHE__
#define __PM_CACHE__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <list>
#include <random>
#include <unordered_map>

namespace pm
{

enum pmCacheEvictionPolicy
{
    LEAST_RECENTLY_USED,
    MOST_RECENTLY_USED,
    LEAST_FREQUENTLY_USED,
    MOST_FREQUENTLY_USED,
    RANDOM_EVICTION
};

namespace cacheInternals
{

template<typename mappedType>
struct listContainer
{
    typedef std::list<mappedType> containerType;
    typedef typename containerType::const_iterator constIteratorType;
    typedef typename containerType::iterator iteratorType;

    containerType mCacheList;
    
    iteratorType emplace(const typename mappedType::first_type& pKey, const typename mappedType::second_type& pValue)
    {
        mCacheList.emplace_front(pKey, pValue);
        return mCacheList.begin();
    }

    iteratorType onAccess(iteratorType pIter, bool& pIterInvalidated)
    {
        pIterInvalidated = false;

        mCacheList.splice(mCacheList.begin(), mCacheList, pIter);
        return mCacheList.begin();
    }
    
    void erase(iteratorType pIter)
    {
        mCacheList.erase(pIter);
    }
    
    bool empty()
    {
        return mCacheList.empty();
    }
    
    const typename mappedType::second_type& getValue(constIteratorType pIter)
    {
        return pIter->second;
    }
    
    typename mappedType::second_type& getValue(iteratorType pIter)
    {
        return pIter->second;
    }

    void purge(pmCacheEvictionPolicy pEvictionPolicy, typename mappedType::second_type& pValue, const std::function<void(const typename mappedType::first_type&)>& pLambda)
    {
        if(pEvictionPolicy == LEAST_RECENTLY_USED)
        {
            typename decltype(mCacheList)::reverse_iterator lIter = mCacheList.rbegin(), lEndIter = mCacheList.rend();
            for(; lIter != lEndIter; ++lIter)
            {
                if(lIter->second.use_count() == 1)
                {
                    pLambda(lIter->first);
                    pValue = lIter->second;
                    mCacheList.erase(--lIter.base()); // convert reverse iter to forward
                    
                    break;
                }
            }
        }
        else if(pEvictionPolicy == MOST_RECENTLY_USED || pEvictionPolicy == RANDOM_EVICTION)
        {
            typename decltype(mCacheList)::iterator lIter = mCacheList.begin(), lEndIter = mCacheList.end();
            for(; lIter != lEndIter; ++lIter)
            {
                if(lIter->second.use_count() == 1)
                {
                    pLambda(lIter->first);
                    pValue = lIter->second;
                    mCacheList.erase(lIter);
                    
                    break;
                }
            }
        }
    }
};
    
template<typename mappedType>
struct listContainerWithRandomEviction : public listContainer<mappedType>
{
    typedef typename listContainer<mappedType>::iteratorType iteratorType;
    using listContainer<mappedType>::mCacheList;

    iteratorType onAccess(iteratorType pIter, bool& pIterInvalidated)
    {
        pIterInvalidated = false;

        iteratorType lIter = mCacheList.begin();
        
        std::random_device lRandomDevice;
        std::mt19937 lGenerator(lRandomDevice());

        size_t lAdvanceAmount = lGenerator() % mCacheList.size();
        std::advance(lIter, lAdvanceAmount);
        
        mCacheList.splice(mCacheList.begin(), mCacheList, lIter);
        return mCacheList.begin();
    }
};

template<typename mappedType>
struct mapContainer
{
    typedef uint frequencyType;
    typedef std::multimap<frequencyType, mappedType> containerType;   // access frequency versus mapped value
    typedef typename containerType::const_iterator constIteratorType;
    typedef typename containerType::iterator iteratorType;

    containerType mCacheMap;
    
    iteratorType emplace(const typename mappedType::first_type& pKey, const typename mappedType::second_type& pValue)
    {
        return mCacheMap.emplace(std::piecewise_construct, std::forward_as_tuple(0), std::forward_as_tuple(pKey, pValue));
    }
    
    iteratorType onAccess(iteratorType pIter, bool& pIterInvalidated)
    {
        pIterInvalidated = true;

        frequencyType lFrequency = pIter->first + 1;
        mappedType lMappedValue = pIter->second;
        
        mCacheMap.erase(pIter);
        
        return mCacheMap.emplace(std::piecewise_construct, std::forward_as_tuple(lFrequency), std::forward_as_tuple(lMappedValue));
    }

    void erase(iteratorType pIter)
    {
        mCacheMap.erase(pIter);
    }

    bool empty()
    {
        return mCacheMap.empty();
    }

    const typename mappedType::second_type& getValue(constIteratorType pIter)
    {
        return pIter->second.second;
    }

    typename mappedType::second_type& getValue(iteratorType pIter)
    {
        return pIter->second.second;
    }

    void purge(pmCacheEvictionPolicy pEvictionPolicy, typename mappedType::second_type& pValue, const std::function<void(const typename mappedType::first_type&)>& pLambda)
    {
        if(pEvictionPolicy == MOST_FREQUENTLY_USED)
        {
            typename decltype(mCacheMap)::reverse_iterator lIter = mCacheMap.rbegin(), lEndIter = mCacheMap.rend();
            for(; lIter != lEndIter; ++lIter)
            {
                if(lIter->second.second.use_count() == 1)
                {
                    pLambda(lIter->second.first);
                    pValue = lIter->second.second;
                    mCacheMap.erase(--lIter.base()); // convert reverse iter to forward
                    
                    break;
                }
            }
        }
        else if(pEvictionPolicy == LEAST_FREQUENTLY_USED)
        {
            typename decltype(mCacheMap)::iterator lIter = mCacheMap.begin(), lEndIter = mCacheMap.end();
            for(; lIter != lEndIter; ++lIter)
            {
                if(lIter->second.second.use_count() == 1)
                {
                    pLambda(lIter->second.first);
                    pValue = lIter->second.second;
                    mCacheMap.erase(lIter);
                    
                    break;
                }
            }
        }
    }
};
    
template<typename mappedType, pmCacheEvictionPolicy __evictionPolicy>
struct containerSelector
{
    typedef listContainer<mappedType> abstractContainerType;
};
    
template<typename mappedType>
struct containerSelector<mappedType, RANDOM_EVICTION>
{
    typedef listContainerWithRandomEviction<mappedType> abstractContainerType;
};

template<typename mappedType>
struct containerSelector<mappedType, LEAST_FREQUENTLY_USED>
{
    typedef mapContainer<mappedType> abstractContainerType;
};

template<typename mappedType>
struct containerSelector<mappedType, MOST_FREQUENTLY_USED>
{
    typedef mapContainer<mappedType> abstractContainerType;
};

}

/* Cache with ref counted values */
template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
class pmCache : public pmBase, pmNonCopyable
{
public:
    typedef std::pair<const __key, std::shared_ptr<__value>> mappedType;
    typedef typename cacheInternals::containerSelector<mappedType, __evictionPolicy>::abstractContainerType abstractContainerType;
    typedef std::unordered_map<const __key, typename abstractContainerType::containerType::iterator, __hasher> hashType;
    
    pmCache(const __evictor& pEvictor)
    : mEvictor(pEvictor)
    {}

    void Insert(const __key& pKey, std::shared_ptr<__value>& pValue);
    std::shared_ptr<__value>& Get(const __key& pKey);
    void RemoveKey(const __key& pKey);
    
    void RemoveKeys(const std::function<bool (const __key&)>& pFunction);

    bool Purge();
    
private:
    pmCache() = default;

    typename hashType::iterator RemoveKeyInternal(typename hashType::iterator pIter);
    
    std::shared_ptr<__value> mEmptyValue;
    __evictor mEvictor;

    abstractContainerType mContainer;
    hashType mCacheHash;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#include "../src/pmCache.cpp"

#endif

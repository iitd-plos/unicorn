
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

namespace pm
{

template<typename __key, typename __value, typename __hasher, typename __evictor>
inline void pmCache<__key, __value, __hasher, __evictor>::Insert(const __key& pKey, std::shared_ptr<__value>& pValue)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    EXCEPTION_ASSERT(mCacheHash.find(pKey) == mCacheHash.end());
    
    mCacheList.emplace_front(pKey, pValue);
    mCacheHash.emplace(pKey, mCacheList.begin());
}

template<typename __key, typename __value, typename __hasher, typename __evictor>
inline std::shared_ptr<__value>& pmCache<__key, __value, __hasher, __evictor>::Get(const __key& pKey)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename decltype(mCacheHash)::iterator lIter = mCacheHash.find(pKey);
    if(lIter == mCacheHash.end())
        return mEmptyValue;
    
    mCacheList.splice(mCacheList.begin(), mCacheList, lIter->second);
    
    return lIter->second->second;
}

template<typename __key, typename __value, typename __hasher, typename __evictor>
inline void pmCache<__key, __value, __hasher, __evictor>::RemoveKey(const __key& pKey)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename decltype(mCacheHash)::iterator lIter = mCacheHash.find(pKey);
    EXCEPTION_ASSERT(lIter != mCacheHash.end());
    
    EXCEPTION_ASSERT(lIter->second->second.unique());

    if(lIter->second->second.get())
        mEvictor(lIter->second->second);
    
    mCacheHash.erase(lIter);
    mCacheList.erase(lIter->second);
}
    
template<typename __key, typename __value, typename __hasher, typename __evictor>
inline bool pmCache<__key, __value, __hasher, __evictor>::Purge()
{
    std::shared_ptr<__value> lValue;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if(mCacheList.empty())
            return false;

        typename decltype(mCacheList)::reverse_iterator lIter = mCacheList.rbegin(), lEndIter = mCacheList.rend();
        for(; lIter != lEndIter; ++lIter)
        {
            if(lIter->second.use_count() == 1)
            {
                lValue = lIter->second;

                mCacheHash.erase(lIter->first);
                mCacheList.erase(--lIter.base()); // convert reverse iter to forward
                
                break;
            }
        }
    }
    
    if(!lValue.get())
        return false;
    
    mEvictor(lValue);
    
    return true;
}

};


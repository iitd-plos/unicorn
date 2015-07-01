
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

namespace pm
{
    
template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline void pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::Insert(const __key& pKey, std::shared_ptr<__value>& pValue)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    EXCEPTION_ASSERT(mCacheHash.find(pKey) == mCacheHash.end());

    auto lIter = mContainer.emplace(pKey, pValue);
    mCacheHash.emplace(pKey, lIter);
}

template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline std::shared_ptr<__value>& pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::Get(const __key& pKey)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename decltype(mCacheHash)::iterator lIter = mCacheHash.find(pKey);
    if(lIter == mCacheHash.end())
        return mEmptyValue;

    bool lIterInvalidated = false;
    auto lReceivedIter = mContainer.onAccess(lIter->second, lIterInvalidated);

    if(lIterInvalidated)
    {
        mCacheHash.erase(lIter);
        mCacheHash.emplace(pKey, lReceivedIter);

        lIter = mCacheHash.find(pKey);
        EXCEPTION_ASSERT(lIter != mCacheHash.end());
    }
    
    return lIter->second->second;
}

template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline void pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::RemoveKey(const __key& pKey)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename decltype(mCacheHash)::iterator lIter = mCacheHash.find(pKey);
    EXCEPTION_ASSERT(lIter != mCacheHash.end());

    RemoveKeyInternal(lIter);
}
    
template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline void pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::RemoveKeys(const std::function<bool (const __key&)>& pFunction)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    auto lIter = mCacheHash.begin(), lEndIter = mCacheHash.end();

    while(lIter != lEndIter)
    {
        if(pFunction(lIter->first))
            lIter = RemoveKeyInternal(lIter);
        else
            ++lIter;
    }
}

// Must be called with mResourceLock acquired
template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline typename pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::hashType::iterator pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::RemoveKeyInternal(typename pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::hashType::iterator pIter)
{
    const std::shared_ptr<__value>& lValue = mContainer.getValue(pIter->second);
    EXCEPTION_ASSERT(lValue.unique());
    
    if(lValue.get())
        mEvictor(lValue);
    
    mContainer.erase(pIter->second);

    return mCacheHash.erase(pIter);
}
    
template<typename __key, typename __value, typename __hasher, typename __evictor, pmCacheEvictionPolicy __evictionPolicy>
inline bool pmCache<__key, __value, __hasher, __evictor, __evictionPolicy>::Purge()
{
    std::shared_ptr<__value> lValue;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if(mContainer.empty())
            return false;
        
        mContainer.purge(__evictionPolicy, lValue, [&] (const __key& pKey) -> void
                                                   {
                                                       mCacheHash.erase(pKey);
                                                   });
    }

    if(!lValue.get())
        return false;
    
    mEvictor(lValue);

    return true;
}

};


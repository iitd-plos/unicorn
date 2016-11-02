
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
    
    return mContainer.getValue(lIter->second);
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


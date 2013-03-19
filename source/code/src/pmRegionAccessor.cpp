
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

#include <deque>

namespace pm
{
    
template<typename valueType>
pmRegionAccessor<valueType>::pmRegionAccessor()
{
}

template<typename valueType>
pmRegionAccessor<valueType>::~pmRegionAccessor()
{
}
    
template<typename valueType>
void pmRegionAccessor<valueType>::Insert(size_t pOffset, size_t pLength, const valueType& pValue)
{
    pointType lPoint1(pOffset), lPoint2(pOffset + pLength - 1);
    boxType lBox(lPoint1, lPoint2);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!pLength || !mRTree.find(lBox).empty())
        PMTHROW(pmFatalErrorException());
#endif
    
    mRTree.insert(lBox, pValue);
}

template<typename valueType>
bool pmRegionAccessor<valueType>::FindExact(size_t pOffset, size_t pLength, valueType& pValue)
{
    pointType lPoint1(pOffset), lPoint2(pOffset + pLength - 1);
    boxType lBox(lPoint1, lPoint2);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    std::deque<valueType>& lDeque = mRTree.FindExact(lBox);
#ifdef _DEBUG
    if(!pLength || lDeque.size() > 1)
        PMTHROW(pmFatalErrorException());
#endif
    
    if(lDeque.empty())
        return false;

    pValue = lDeque.front();
    return true;
}
    
template<typename valueType>
void pmRegionAccessor<valueType>::Erase(size_t pOffset, size_t pLength)
{
    pointType lPoint1(pOffset), lPoint2(pOffset + pLength - 1);
    boxType lBox(lPoint1, lPoint2);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

#ifdef _DEBUG
    if(!pLength || mRTree.find(lBox).size() > 1)
        PMTHROW(pmFatalErrorException());
#endif

    mRTree.remove(lBox);
}

} // end namespace pm


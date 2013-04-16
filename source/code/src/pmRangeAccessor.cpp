
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

template<typename valueType>
pmRangeAccessor<valueType>::pmRangeAccessor()
{
}

template<typename valueType>
pmRangeAccessor<valueType>::~pmRangeAccessor()
{
}
    
template<typename valueType>
void pmRangeAccessor<valueType>::Insert(size_t pOffset, size_t pLength, const valueType& pValue)
{
    std::pair<size_t, size_t> lPair(pOffset, pLength);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(mHash.find(lPair) != mHash.end())
        PMTHROW(pmFatalErrorException());
    
    mHash[lPair] = pValue;
}

template<typename valueType>
bool pmRangeAccessor<valueType>::Find(size_t pOffset, size_t pLength, valueType& pValue)
{
    std::pair<size_t, size_t> lPair(pOffset, pLength);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    typename hashType::iterator lIter = mHash.find(lPair);
    if(lIter == mHash.end())
        return false;

    pValue = lIter->second;
    
    return true;
}
    
template<typename valueType>
void pmRangeAccessor<valueType>::Erase(size_t pOffset, size_t pLength)
{
    std::pair<size_t, size_t> lPair(pOffset, pLength);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    mHash.erase(lPair);
}

} // end namespace pm


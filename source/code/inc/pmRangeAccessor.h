
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

#ifndef __PM_RANGE_ACCESSOR__
#define __PM_RANGE_ACCESSOR__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <tr1/unordered_map>

namespace pm
{

struct rangeComparator
{
    bool operator() (std::pair<size_t, size_t>& pRange1, std::pair<size_t, size_t>& pRange2)
    {
        return ((pRange1.first <= pRange2.first && pRange1.second >= pRange2.second) || (pRange2.first <= pRange1.first && pRange2.second >= pRange1.second));
    }
};
    
template<typename valueType>
class pmRangeAccessor : public pmBase
{
    typedef std::tr1::unordered_map<std::pair<size_t, size_t>, valueType, rangeComparator> hashType;
    
    public:
        pmRangeAccessor();
        ~pmRangeAccessor();
    
        void Insert(size_t pOffset, size_t pLength, const valueType& pValue);
        bool Find(size_t pOffset, size_t pLength, valueType& pValue);
        void Erase(size_t pOffset, size_t pLength);
    
    private:
        hashType mHash;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};
    
} // end namespace pm

#include "../src/pmRangeAccessor.cpp"

#endif

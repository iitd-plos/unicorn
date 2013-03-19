
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

#ifndef __PM_REGION_ACCESSOR__
#define __PM_REGION_ACCESSOR__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <boost/geometry.hpp>
#include <boost/geometry/extensions/index/rtree/rtree.hpp>

#include <deque>

namespace pm
{

template <typename boxType, typename valueType>
class pmRTree : public boost::geometry::index::rtree<boxType, valueType>
{
    public:
        pmRTree()
        : boost::geometry::index::rtree<boxType, valueType>(16, 4)
        {
        }
    
        inline std::deque<valueType> FindExact(boxType const& box) const
        {
            std::deque<valueType> result;
            boost::geometry::index::rtree<boxType, valueType>::m_root->find(box, result, true);
            return result;
        }
    
    private:
};
    
template<typename valueType>
class pmRegionAccessor
{
    typedef boost::geometry::model::point<size_t, 1, boost::geometry::cs::cartesian> pointType;
    typedef boost::geometry::model::box<pointType> boxType;
    typedef pmRTree<boxType, valueType> rtreeType;

    public:
        pmRegionAccessor();
        ~pmRegionAccessor();
    
        void Insert(size_t pOffset, size_t pLength, const valueType& pValue);
        bool FindExact(size_t pOffset, size_t pLength, valueType& pValue);
        void Erase(size_t pOffset, size_t pLength);
    
    private:
        rtreeType mRTree;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};
    
} // end namespace pm

#include "../src/pmRegionAccessor.cpp"

#endif

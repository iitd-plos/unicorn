
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

#include "pmTable.h"

namespace pm
{

template<typename rowHeaderType, typename rowType>
pmTable<rowHeaderType, rowType>::pmTable()
{
}

template<typename rowHeaderType, typename rowType>
void pmTable<rowHeaderType, rowType>::AddRow(const rowHeaderType& pRowHeader, rowType&& pRow)
{
    mTable[pRowHeader] = std::move(pRow);
}

template<typename rowHeaderType, typename rowType>
const rowType& pmTable<rowHeaderType, rowType>::GetRow(const rowHeaderType& pRowHeader)
{
    auto lRowIter = mTable.find(pRowHeader);
    
    EXCEPTION_ASSERT(lRowIter != mTable.end());
    
    return lRowIter->second;
}

template<typename rowHeaderType, typename rowType>
size_t pmTable<rowHeaderType, rowType>::GetRowCount() const
{
    return mTable.size();
}
    
}





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

#ifndef __PM_TABLE__
#define __PM_TABLE__

#include "pmBase.h"

#include <map>

namespace pm
{
    
template<typename rowHeaderType, typename rowType>
class pmTable
{
public:
    pmTable();

    void AddRow(const rowHeaderType& pRowHeader, rowType&& pRow);    
    const rowType& GetRow(const rowHeaderType& pRowHeader);

    size_t GetRowCount() const;
private:
    std::map<rowHeaderType, rowType> mTable;
};

} // end namespace pm

#include "../src/pmTable.cpp"

#endif


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

#ifndef __PM_MEMORY_DIRECTORY__
#define __PM_MEMORY_DIRECTORY__

#include "pmBase.h"
#include "pmCommunicator.h"
#include "pmResourceLock.h"

#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

#define FIND_FLOOR_ELEM(mapType, mapVar, searchKey, iterAddr) \
{ \
    if(mapVar.empty()) \
    { \
        iterAddr = NULL; \
    } \
    else \
    { \
        mapType::iterator dUpper = mapVar.lower_bound(searchKey); \
        if(dUpper == mapVar.begin() && (ulong)(dUpper->first) > (ulong)searchKey) \
            iterAddr = NULL; \
        else if(dUpper == mapVar.end() || (ulong)(dUpper->first) > (ulong)searchKey) \
            *iterAddr = (--dUpper); \
        else \
            *iterAddr = dUpper; \
    } \
}

namespace pm
{

struct vmRangeOwner
{
    const pmMachine* host;                                  // Host where memory page lives
    ulong hostOffset;                                       // Offset on host (in case of data redistribution offsets at source and destination hosts are different)
    communicator::memoryIdentifierStruct memIdentifier;     // a different memory might be holding the required data (e.g. redistribution)
    
    vmRangeOwner(const pmMachine* pHost, ulong pHostOffset, const communicator::memoryIdentifierStruct& pMemIdentifier)
    : host(pHost)
    , hostOffset(pHostOffset)
    , memIdentifier(pMemIdentifier)
    {}
    
    vmRangeOwner(const vmRangeOwner& pRangeOwner)
    : host(pRangeOwner.host)
    , hostOffset(pRangeOwner.hostOffset)
    , memIdentifier(pRangeOwner.memIdentifier)
    {}
    
    bool operator==(const vmRangeOwner& pRangeOwner) const
    {
        return (host == pRangeOwner.host && hostOffset == pRangeOwner.hostOffset && memIdentifier == pRangeOwner.memIdentifier);
    }
};

struct pmMemTransferData
{
    vmRangeOwner rangeOwner;
    ulong offset;
    ulong length;
    
    pmMemTransferData(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
    : rangeOwner(pRangeOwner)
    , offset(pOffset)
    , length(pLength)
    {}
};

struct pmScatteredMemTransferData
{
    vmRangeOwner rangeOwner;
    ulong offset;
    ulong size;
    ulong step;
    ulong count;
    
    pmScatteredMemTransferData(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pSize, ulong pStep, ulong pCount)
    : rangeOwner(pRangeOwner)
    , offset(pOffset)
    , size(pSize)
    , step(pStep)
    , count(pCount)
    {}
};

typedef std::map<size_t, std::pair<size_t, vmRangeOwner>> pmMemOwnership;
typedef std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>> pmScatteredMemOwnership;

class pmMemoryDirectory : public pmBase
{
public:
    pmMemoryDirectory(const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct);

    virtual void Reset(const pmMachine* pOwner) = 0;

    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength) = 0;
    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount) = 0;

    virtual void GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships) = 0;
    virtual void GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships) = 0;
    
    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships) = 0;
    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships) = 0;

    virtual void Clear() = 0;
    virtual bool IsEmpty() = 0;
    
    virtual void CloneFrom(pmMemoryDirectory* pDirectory) = 0;
    
protected:
    communicator::memoryIdentifierStruct mMemoryIdentifierStruct;

    RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;
};
    
class pmMemoryDirectoryLinear : public pmMemoryDirectory
{
public:
    pmMemoryDirectoryLinear(ulong pAddressSpaceLength, const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct);

    virtual void Reset(const pmMachine* pOwner);

    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

    virtual void GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);
    virtual void GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);

    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);
    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);

    virtual void Clear();
    virtual bool IsEmpty();

    virtual void CloneFrom(pmMemoryDirectory* pDirectory);

private:
    void SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength);
    void GetOwnersInternal(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);

#ifdef _DEBUG
    void CheckMergability(const pmMemOwnership::iterator& pRange1, const pmMemOwnership::iterator& pRange2) const;
    void SanitizeOwnerships() const;
    void PrintOwnerships() const;
#endif

    ulong mAddressSpaceLength;
    
    pmMemOwnership mOwnershipMap;       // offset versus pair of length of region and vmRangeOwner
};
    
class pmMemoryDirectory2D : public pmMemoryDirectory
{
    typedef typename boost::geometry::model::point<ulong, 2, boost::geometry::cs::cartesian> boost_point_type;
    typedef typename boost::geometry::model::box<boost_point_type> boost_box_type;
    typedef typename boost::geometry::index::rtree<std::pair<boost_box_type, vmRangeOwner>, boost::geometry::index::quadratic<16>> boost_rtree_type;

public:
    pmMemoryDirectory2D(ulong pAddressSpaceRows, ulong pAddressSpaceCols, const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct);

    virtual void Reset(const pmMachine* pOwner);

    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
    virtual void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

    virtual void GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);
    virtual void GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);

    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);
    virtual void GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);

    virtual void Clear();
    virtual bool IsEmpty();

    virtual void CloneFrom(pmMemoryDirectory* pDirectory);
    
private:
    void GetOwnersInternal(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);
    void GetDifferenceOfBoxes(const boost_box_type& pBox1, const boost_box_type& pBox2, std::vector<boost_box_type>& pRemainingBoxes);
    
#ifdef _DEBUG
    void PrintOwnerships() const;
    void SanitizeOwnerships() const;
    void PrintBox(const boost_box_type& pBox) const;
    bool AreBoxesEqual(const boost_box_type& pBox1, const boost_box_type& pBox2) const;
#endif

    boost_box_type GetBox(ulong pOffset, ulong pLength, ulong pStep, ulong pCount) const;
    pmScatteredSubscriptionInfo GetReverseBoxMapping(const boost_box_type& pBox) const;
    ulong GetReverseBoxOffset(const boost_box_type& pBox) const;
    
    void CombineAndInsertBox(boost_box_type pBox, vmRangeOwner pRangeOwner);

    ulong mAddressSpaceRows, mAddressSpaceCols;
    
    boost_rtree_type mOwnershipRTree;       // offset versus pair of length of region and vmRangeOwner
};
    
} // end namespace pm

#endif

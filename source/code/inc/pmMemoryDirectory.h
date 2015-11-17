
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
#include "pmHardware.h"

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
    
typedef std::vector<std::tuple<pmSubscriptionInfo, vmRangeOwner, pmCommandPtr>> pmLinearTransferVectorType;
typedef std::map<const pmMachine*, std::vector<std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>>> pmScatteredTransferMapType;
typedef std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>> pmRemoteRegionsInfoMapType;

typedef std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>> pmScatteredMemOwnership;

class pmMemoryDirectory : public pmBase
{
    friend class pmMemoryDirectoryLinear;
    friend class pmMemoryDirectory2D;
    
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

    virtual void CancelUnreferencedRequests() = 0;

    virtual pmScatteredTransferMapType SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet) = 0;
    virtual pmLinearTransferVectorType SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector) = 0;
    virtual pmRemoteRegionsInfoMapType GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr) = 0;
    
    virtual void CopyOrUpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource) = 0;
    virtual void UpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount) = 0;
    
protected:
    communicator::memoryIdentifierStruct mMemoryIdentifierStruct;

    RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;
};
    
class pmMemoryDirectoryLinear : public pmMemoryDirectory
{
    struct regionFetchData
    {
        pmCommandPtr receiveCommand;
        
        std::map<size_t, size_t> partialReceiveRecordMap;
        size_t accumulatedPartialReceivesLength;
        
        regionFetchData()
        : accumulatedPartialReceivesLength(0)
        {}
        
        regionFetchData(pmCommandPtr& pCommand)
        : receiveCommand(pCommand)
        , accumulatedPartialReceivesLength(0)
        {}
    };

    typedef std::map<void*, std::pair<size_t, regionFetchData>> pmInFlightRegions;

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

    virtual void CancelUnreferencedRequests();
    
    virtual pmScatteredTransferMapType SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet);
    virtual pmLinearTransferVectorType SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector);
    virtual pmRemoteRegionsInfoMapType GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr);

    virtual void CopyOrUpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource);
    virtual void UpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

private:
    void SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength);
    void GetOwnersInternal(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);

    template<typename consumer_type>
    void FindRegionsNotInFlight(pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, consumer_type& pRegionsToBeFetched, std::vector<pmCommandPtr>& pCommandVector);

    void AcquireOwnershipImmediateInternal(ulong pOffset, ulong pLength);
    bool CopyOrUpdateReceivedMemoryInternal(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmInFlightRegions& pInFlightMap, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource = NULL);

#ifdef _DEBUG
    void CheckMergability(const pmMemOwnership::iterator& pRange1, const pmMemOwnership::iterator& pRange2) const;
    void SanitizeOwnerships() const;
    void PrintOwnerships() const;
#endif

    ulong mAddressSpaceLength;
    
    pmMemOwnership mOwnershipMap;       // offset versus pair of length of region and vmRangeOwner
    pmInFlightRegions mInFlightLinearMap;	// Map for regions being fetched; pair is length of region and regionFetchData
};
    
class pmMemoryDirectory2D : public pmMemoryDirectory
{
    typedef typename boost::geometry::model::point<ulong, 2, boost::geometry::cs::cartesian> boost_point_type;
    typedef typename boost::geometry::model::box<boost_point_type> boost_box_type;

    struct regionFetchData2D
    {
        pmCommandPtr receiveCommand;
        
        std::vector<boost_box_type> partialReceiveRecordVector;
        size_t accumulatedPartialReceivesLength;
        
        regionFetchData2D()
        : accumulatedPartialReceivesLength(0)
        {}
        
        regionFetchData2D(pmCommandPtr& pCommand)
        : receiveCommand(pCommand)
        , accumulatedPartialReceivesLength(0)
        {}
        
        bool operator==(const regionFetchData2D& pData) const
        {
            return (receiveCommand == pData.receiveCommand);
        }
    };

    typedef typename boost::geometry::index::rtree<std::pair<boost_box_type, vmRangeOwner>, boost::geometry::index::quadratic<16>> pmScatteredMemOwnershipRTree;
    typedef typename boost::geometry::index::rtree<std::pair<boost_box_type, regionFetchData2D>, boost::geometry::index::quadratic<16>> pmInFlightScatteredRegions;

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
    
    virtual void CancelUnreferencedRequests();

    virtual pmScatteredTransferMapType SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet);
    virtual pmLinearTransferVectorType SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector);
    virtual pmRemoteRegionsInfoMapType GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr);

    virtual void CopyOrUpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource);
    virtual void UpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

private:
    void GetOwnersInternal(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);

    void GetDifferenceOfBoxes(const boost_box_type& pBox1, const boost_box_type& pBox2, std::vector<boost_box_type>& pRemainingBoxes);
    bool UpdateReceivedMemoryInternal(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

    void SetRangeOwnerInternal(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

#ifdef _DEBUG
    void PrintOwnerships() const;
    void SanitizeOwnerships() const;
    void PrintBox(const boost_box_type& pBox) const;
#endif

    bool AreBoxesEqual(const boost_box_type& pBox1, const boost_box_type& pBox2) const;

    boost_box_type GetBox(ulong pOffset, ulong pLength, ulong pStep, ulong pCount) const;
    pmScatteredSubscriptionInfo GetReverseBoxMapping(const boost_box_type& pBox) const;
    ulong GetReverseBoxOffset(const boost_box_type& pBox) const;
    
    void CombineAndInsertBox(boost_box_type pBox, vmRangeOwner pRangeOwner);

    ulong mAddressSpaceRows, mAddressSpaceCols;

    pmScatteredMemOwnershipRTree mOwnershipRTree;
    pmInFlightScatteredRegions mInFlightRTree;
};

class pmScatteredSubscriptionFilter
{
public:
    pmScatteredSubscriptionFilter(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);

    // pRowFunctor should call AddNextSubRow for every range to be kept
    const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& FilterBlocks(const std::function<void (size_t)>& pRowFunctor);

    void AddNextSubRow(ulong pOffset, ulong pLength, vmRangeOwner& pRangeOwner);
    
private:
    const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& GetLeftoverBlocks();
    void PromoteCurrentBlocks();

    struct blockData
    {
        ulong startCol;
        ulong colCount;
        pmScatteredSubscriptionInfo subscriptionInfo;
        vmRangeOwner rangeOwner;
        
        blockData(ulong pStartCol, ulong pColCount, const pmScatteredSubscriptionInfo& pSubscriptionInfo, vmRangeOwner& pRangeOwner)
        : startCol(pStartCol)
        , colCount(pColCount)
        , subscriptionInfo(pSubscriptionInfo)
        , rangeOwner(pRangeOwner)
        {}
    };

    const pmScatteredSubscriptionInfo& mScatteredSubscriptionInfo;
    
    std::list<blockData> mCurrentBlocks;   // computed till last row processed
    std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>> mBlocksToBeFetched;
};
    
struct pmScatteredSubscriptionFilterHelper
{
public:
    pmScatteredSubscriptionFilterHelper(pmScatteredSubscriptionFilter& pGlobalFilter, void* pBaseAddr, pmMemoryDirectoryLinear& pMemoryDirectoryLinear, bool pUnprotected, const pmMachine* pFilteredMachine = PM_LOCAL_MACHINE);

    void emplace_back(ulong pStartAddr, ulong pLastAddr);
    
private:
    pmScatteredSubscriptionFilter& mGlobalFilter;
    pmMemoryDirectoryLinear& mMemoryDirectoryLinear;
    ulong mMem;
    bool mUnprotected;
    const pmMachine* mFilteredMachine;
};
    
} // end namespace pm

#endif

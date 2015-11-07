
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

#include "pmMemoryDirectory.h"
#include "pmHardware.h"

namespace pm
{
    
/* class pmMemoryDirectory */
pmMemoryDirectory::pmMemoryDirectory(const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct)
    : mMemoryIdentifierStruct(pMemoryIdentifierStruct)
    , mOwnershipLock __LOCK_NAME__("pmMemoryDirectory::mOwnershipLock")
{}
    
/* class pmMemoryDirectoryLinear */
pmMemoryDirectoryLinear::pmMemoryDirectoryLinear(ulong pAddressSpaceLength, const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct)
    : pmMemoryDirectory(pMemoryIdentifierStruct)
    , mAddressSpaceLength(pAddressSpaceLength)
{}

void pmMemoryDirectoryLinear::Reset(const pmMachine* pOwner)
{
	mOwnershipMap.emplace(std::piecewise_construct, std::forward_as_tuple(0), std::forward_as_tuple(std::piecewise_construct, std::forward_as_tuple(mAddressSpaceLength), std::forward_as_tuple(pOwner, 0, mMemoryIdentifierStruct)));
}

void pmMemoryDirectoryLinear::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    SetRangeOwnerInternal(pRangeOwner, pOffset, pLength);
}

// Must be called with mOwnershipLock acquired
void pmMemoryDirectoryLinear::SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength)
{
#ifdef _DEBUG
#if 0
    PrintOwnerships();
    if(pRangeOwner.memIdentifier.operator==(mMemoryIdentifierStruct))
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner, Owner Offset): (" << pOffset << ", " << pLength << ", " << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
    else
        std::cout << "Host " << pmGetHostId() << " Set Range Owner: (Offset, Length, Owner address space (Host, Generation Number), Owner, Owner Offset): (" << pOffset << ", " << pLength << ", (" << pRangeOwner.memIdentifier.memOwnerHost << ", " << pRangeOwner.memIdentifier.generationNumber << ")," << pRangeOwner.host << ", " << pRangeOwner.hostOffset << ")" << std::endl;
#endif
#endif
    
	// Remove present ownership
	size_t lLastAddr = pOffset + pLength - 1;
	size_t lOwnerLastAddr = pRangeOwner.hostOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		PMTHROW(pmFatalErrorException());
    
	assert(lStartIter->first <= pOffset);
	assert(lEndIter->first <= lLastAddr);
	assert(lStartIter->first + lStartIter->second.first > pOffset);
	assert(lEndIter->first + lEndIter->second.first > lLastAddr);

	size_t lStartOffset = lStartIter->first;
	//size_t lStartLength = lStartIter->second.first;
	vmRangeOwner lStartOwner = lStartIter->second.second;

	size_t lEndOffset = lEndIter->first;
	size_t lEndLength = lEndIter->second.first;
	vmRangeOwner lEndOwner = lEndIter->second.second;

	mOwnershipMap.erase(lStartIter, lEndIter);
    mOwnershipMap.erase(lEndIter);

	if(lStartOffset < pOffset)
	{
		if(lStartOwner.host == pRangeOwner.host && lStartOwner.memIdentifier == pRangeOwner.memIdentifier && lStartOwner.hostOffset == (pRangeOwner.hostOffset - (pOffset - lStartOffset)))
        {
            pRangeOwner.hostOffset -= (pOffset - lStartOffset);
			pOffset = lStartOffset;		// Combine with previous range
        }
		else
        {
			mOwnershipMap.insert(std::make_pair(lStartOffset, std::make_pair(pOffset-lStartOffset, lStartOwner)));
        }
	}
    else
    {
        if(lStartOffset != pOffset)
            PMTHROW(pmFatalErrorException());
        
        pmMemOwnership::iterator lPrevIter;
        pmMemOwnership::iterator* lPrevIterAddr = &lPrevIter;

        if(pOffset)
        {
            size_t lPrevAddr = pOffset - 1;
            FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, lPrevAddr, lPrevIterAddr);
            if(lPrevIterAddr)
            {
                size_t lPrevOffset = lPrevIter->first;
                size_t lPrevLength = lPrevIter->second.first;
                vmRangeOwner lPrevOwner = lPrevIter->second.second;
                
                if(lPrevOwner.host == pRangeOwner.host && lPrevOwner.memIdentifier == pRangeOwner.memIdentifier && lPrevOwner.hostOffset + lPrevLength == pRangeOwner.hostOffset)
                {
                    pRangeOwner.hostOffset -= (lStartOffset - lPrevOffset);
                    pOffset = lPrevOffset;		// Combine with previous range                

                    mOwnershipMap.erase(lPrevIter);
                }
            }
        }
    }

	if(lEndOffset + lEndLength - 1 > lLastAddr)
	{
		if(lEndOwner.host == pRangeOwner.host && lEndOwner.memIdentifier == pRangeOwner.memIdentifier && (lEndOwner.hostOffset + (lLastAddr - lEndOffset)) == lOwnerLastAddr)
        {
			lLastAddr = lEndOffset + lEndLength - 1;	// Combine with following range
        }
		else
        {
            vmRangeOwner lEndRangeOwner = lEndOwner;
            lEndRangeOwner.hostOffset += (lLastAddr - lEndOffset + 1);
			mOwnershipMap.insert(std::make_pair(lLastAddr + 1, std::make_pair(lEndOffset + lEndLength - 1 - lLastAddr, lEndRangeOwner)));
        }
	}
    else
    {
        if(lEndOffset + lEndLength - 1 != lLastAddr)
            PMTHROW(pmFatalErrorException());

        pmMemOwnership::iterator lNextIter;
        pmMemOwnership::iterator* lNextIterAddr = &lNextIter;
    
        if(lLastAddr + 1 < mAddressSpaceLength)
        {
            size_t lNextAddr = lLastAddr + 1;
            FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, lNextAddr, lNextIterAddr);
            if(lNextIterAddr)
            {
                size_t lNextOffset = lNextIter->first;
                size_t lNextLength = lNextIter->second.first;
                vmRangeOwner lNextOwner = lNextIter->second.second;
                
                if(lNextOwner.host == pRangeOwner.host && lNextOwner.memIdentifier == pRangeOwner.memIdentifier && lNextOwner.hostOffset == lOwnerLastAddr + 1)
                {
                    lLastAddr = lNextOffset + lNextLength - 1;	// Combine with following range

                    mOwnershipMap.erase(lNextIter);
                }
            }
        }
    }

	mOwnershipMap.insert(std::make_pair(pOffset, std::make_pair(lLastAddr - pOffset + 1, pRangeOwner)));

#ifdef _DEBUG
    SanitizeOwnerships();
#endif
}

void pmMemoryDirectoryLinear::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    for(ulong i = 0; i < pCount; ++i)
        SetRangeOwnerInternal(pRangeOwner, pOffset + i * pStep, pLength);
}

// Must be called with mOwnershipLock acquired
void pmMemoryDirectoryLinear::GetOwnersInternal(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
	ulong lLastAddr = pOffset + pLength - 1;

	pmMemOwnership::iterator lStartIter, lEndIter;
	pmMemOwnership::iterator* lStartIterAddr = &lStartIter;
	pmMemOwnership::iterator* lEndIterAddr = &lEndIter;

	FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, pOffset, lStartIterAddr);
	FIND_FLOOR_ELEM(pmMemOwnership, mOwnershipMap, lLastAddr, lEndIterAddr);

	if(!lStartIterAddr || !lEndIterAddr)
		PMTHROW(pmFatalErrorException());

    size_t lSpan = lStartIter->first + lStartIter->second.first - 1;
    if(lLastAddr < lSpan)
    {
        lSpan = lLastAddr;
        if(lStartIter != lEndIter)
            PMTHROW(pmFatalErrorException());
    }
    
    vmRangeOwner lRangeOwner = lStartIter->second.second;
    lRangeOwner.hostOffset += (pOffset - lStartIter->first);
	pOwnerships.insert(std::make_pair(pOffset, std::make_pair(lSpan - pOffset + 1, lRangeOwner)));
	
	pmMemOwnership::iterator lIter = lStartIter;
	++lIter;

	if(lStartIter != lEndIter)
	{
		for(; lIter != lEndIter; ++lIter)
        {
            lSpan = lIter->first + lIter->second.first - 1;
            if(lLastAddr < lSpan)
            {
                lSpan = lLastAddr;
                if(lIter != lEndIter)
                    PMTHROW(pmFatalErrorException());
            }
            
			pOwnerships.insert(std::make_pair(lIter->first, std::make_pair(lSpan - lIter->first + 1, lIter->second.second)));
        }
        
        lSpan = lEndIter->first + lEndIter->second.first - 1;
        if(lLastAddr < lSpan)
            lSpan = lLastAddr;
        
        pOwnerships.insert(std::make_pair(lEndIter->first, std::make_pair(lSpan - lEndIter->first + 1, lEndIter->second.second)));
	}
}

void pmMemoryDirectoryLinear::GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    GetOwnersInternal(pOffset, pLength, pOwnerships);
}
    
void pmMemoryDirectoryLinear::GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmMemOwnership& pOwnerships)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    for(ulong i = 0; i < pCount; ++i)
        GetOwnersInternal(pOffset + i * pStep, pLength, pOwnerships);
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectoryLinear::GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    GetOwnersInternal(pOffset, pLength, pOwnerships);
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectoryLinear::GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmMemOwnership& pOwnerships)
{
    for(ulong i = 0; i < pCount; ++i)
        GetOwnersInternal(pOffset + i * pStep, pLength, pOwnerships);
}
    
void pmMemoryDirectoryLinear::Clear()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    mOwnershipMap.clear();
}

bool pmMemoryDirectoryLinear::IsEmpty()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    return mOwnershipMap.empty();
}
    
void pmMemoryDirectoryLinear::CloneFrom(pmMemoryDirectory* pDirectory)
{
    EXCEPTION_ASSERT(dynamic_cast<pmMemoryDirectoryLinear*>(pDirectory) != NULL);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    mOwnershipMap = static_cast<pmMemoryDirectoryLinear*>(pDirectory)->mOwnershipMap;
}


#ifdef _DEBUG
void pmMemoryDirectoryLinear::CheckMergability(pmMemOwnership::iterator& pRange1, pmMemOwnership::iterator& pRange2)
{
    size_t lOffset1 = pRange1->first;
    size_t lOffset2 = pRange2->first;
    size_t lLength1 = pRange1->second.first;
    size_t lLength2 = pRange2->second.first;
    vmRangeOwner& lRangeOwner1 = pRange1->second.second;
    vmRangeOwner& lRangeOwner2 = pRange2->second.second;
    
    if(lOffset1 + lLength1 != lOffset2)
        std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Range end points don't match. Range 1: Offset = " << lOffset1 << " Length = " << lLength1 << " Range 2: Offset = " << lOffset2 << std::endl;
    
    if(lRangeOwner1.host == lRangeOwner2.host && lRangeOwner1.memIdentifier == lRangeOwner2.memIdentifier && lRangeOwner1.hostOffset + lLength1 == lRangeOwner2.hostOffset)
        std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Mergable Ranges Found (" << lOffset1 << ", " << lLength1 << ") - (" << lOffset2 << ", " << lLength2 << ") map to (" << lRangeOwner1.hostOffset << ", " << lLength1 << ") - (" << lRangeOwner2.hostOffset << ", " << lLength2 << ") on host " << (uint)(*lRangeOwner1.host) << std::endl;
}
    
void pmMemoryDirectoryLinear::SanitizeOwnerships()
{
    if(mOwnershipMap.size() == 1)
        return;
    
    pmMemOwnership::iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end(), lPenultimate = lEnd;
    --lPenultimate;
    
    for(lIter = lBegin; lIter != lPenultimate; ++lIter)
    {
        pmMemOwnership::iterator lNext = lIter;
        ++lNext;
        
        CheckMergability(lIter, lNext);
    }
}

void pmMemoryDirectoryLinear::PrintOwnerships()
{
    std::cout << "Host " << pmGetHostId() << " Ownership Dump " << std::endl;
    pmMemOwnership::iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end();
    for(lIter = lBegin; lIter != lEnd; ++lIter)
        std::cout << "Range (" << lIter->first << " , " << lIter->second.first << ") is owned by host " << (uint)(*(lIter->second.second.host)) << " (" << lIter->second.second.hostOffset << ", " << lIter->second.first << ")" << std::endl;
        
    std::cout << std::endl;
}
#endif


/* class pmMemoryDirectory2D */
pmMemoryDirectory2D::pmMemoryDirectory2D(ulong pAddressSpaceRows, ulong pAddressSpaceCols, const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct)
    : pmMemoryDirectory(pMemoryIdentifierStruct)
    , mAddressSpaceRows(pAddressSpaceRows)
    , mAddressSpaceCols(pAddressSpaceCols)
{}

pmMemoryDirectory2D::boost_box_type pmMemoryDirectory2D::GetBox(ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    boost_point_type lPoint1(pOffset % pStep, pOffset / pStep);
    boost_point_type lPoint2(lPoint1.get<0>() + pLength - 1, lPoint1.get<1>() + pCount - 1);
    
    return boost_box_type(lPoint1, lPoint2);
}

pmScatteredSubscriptionInfo pmMemoryDirectory2D::GetReverseBoxMapping(const boost_box_type& pBox)
{
    const boost_point_type& lMinCorner = pBox.min_corner();
    const boost_point_type& lMaxCorner = pBox.max_corner();
    
    return pmScatteredSubscriptionInfo(lMinCorner.get<0>() + lMinCorner.get<1>() * mAddressSpaceCols, lMaxCorner.get<0>() - lMinCorner.get<0>() + 1, mAddressSpaceCols, lMaxCorner.get<1>() - lMinCorner.get<1>() + 1);
}
    
void pmMemoryDirectory2D::Reset(const pmMachine* pOwner)
{
    mOwnershipRTree.insert(std::make_pair(GetBox(0, mAddressSpaceCols, mAddressSpaceCols, mAddressSpaceRows), vmRangeOwner(pOwner, 0, mMemoryIdentifierStruct)));
}

void pmMemoryDirectory2D::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    SetRangeOwner(pRangeOwner, pOffset, pLength, 0, 1);
}

void pmMemoryDirectory2D::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    boost_box_type lBox = GetBox(pOffset, pLength, pStep, pCount);
    
    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    std::vector<boost_box_type> lRemainingBoxes;
    lRemainingBoxes.reserve(4);
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        lRemainingBoxes.clear();
        GetDifferenceOfBoxes(pPair.first, lBox, lRemainingBoxes);
        
        mOwnershipRTree.remove(pPair);
        
        for_each(lRemainingBoxes, [&] (const boost_box_type& pBox)
        {
            mOwnershipRTree.insert(std::make_pair(pBox, pPair.second));
        });
    });

//#ifdef _DEBUG
    lOverlappingBoxes.clear();
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    EXCEPTION_ASSERT(lOverlappingBoxes.empty());
//#endif

    mOwnershipRTree.insert(std::make_pair(lBox, pRangeOwner));
}

void pmMemoryDirectory2D::GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    GetOwners(pOffset, pLength, 0, 1, pOwnerships);
}
    
void pmMemoryDirectory2D::GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmMemOwnership& pOwnerships)
{
    EXCEPTION_ASSERT(((pStep == 0 && pCount == 1) || (mAddressSpaceCols == pStep)) && pOffset + pStep * (pCount - 1) + pLength <= mAddressSpaceRows * mAddressSpaceCols);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    GetOwnersInternal(pOffset, pLength, pStep, pCount, pOwnerships);
}

// Must be called with mOwnershipLock acquired
void pmMemoryDirectory2D::GetOwnersInternal(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmMemOwnership& pOwnerships)
{
    boost_box_type lBox = GetBox(pOffset, pLength, pStep, pCount);
    
    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        boost_box_type lOutBox;
        boost::geometry::intersection(lBox, pPair.first, lOutBox);

        pmScatteredSubscriptionInfo lScatteredSubscriptinInfo = GetReverseBoxMapping(lOutBox);
        
        for(ulong i = 0; i < lScatteredSubscriptinInfo.count; ++i)
            pOwnerships.insert(std::make_pair(lScatteredSubscriptinInfo.offset + i * lScatteredSubscriptinInfo.step, std::make_pair(lScatteredSubscriptinInfo.size, pPair.second)));
    });
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectory2D::GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    GetOwnersInternal(pOffset, pLength, 0, 1, pOwnerships);
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectory2D::GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmMemOwnership& pOwnerships)
{
    GetOwnersInternal(pOffset, pLength, pStep, pCount, pOwnerships);
}

void pmMemoryDirectory2D::Clear()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    mOwnershipRTree.clear();
}

bool pmMemoryDirectory2D::IsEmpty()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());

    return mOwnershipRTree.empty();
}

void pmMemoryDirectory2D::CloneFrom(pmMemoryDirectory* pDirectory)
{
    EXCEPTION_ASSERT(dynamic_cast<pmMemoryDirectory2D*>(pDirectory) != NULL);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, Lock(), Unlock());
    
    mOwnershipRTree = static_cast<pmMemoryDirectory2D*>(pDirectory)->mOwnershipRTree;
}

void pmMemoryDirectory2D::GetDifferenceOfBoxes(const boost_box_type& pBox1, const boost_box_type& pBox2, std::vector<boost_box_type>& pRemainingBoxes)
{
    const boost_point_type& lMinCorner1 = pBox1.min_corner();
    const boost_point_type& lMaxCorner1 = pBox1.max_corner();
    const boost_point_type& lMinCorner2 = pBox2.min_corner();
    const boost_point_type& lMaxCorner2 = pBox2.max_corner();
    
    ulong lXMin1 = lMinCorner1.get<0>();
    ulong lYMin1 = lMinCorner1.get<1>();
    ulong lXMin2 = lMinCorner2.get<0>();
    ulong lYMin2 = lMinCorner2.get<1>();

    ulong lXMax1 = lMaxCorner1.get<0>();
    ulong lYMax1 = lMaxCorner1.get<1>();
    ulong lXMax2 = lMaxCorner2.get<0>();
    ulong lYMax2 = lMaxCorner2.get<1>();

    // Create a rectangle at top if required
    if(lYMin1 < lYMin2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMin1, lYMin1), boost_point_type(lXMax1, lYMin2 - 1));
    
    // Create a rectangle at bottom if required
    if(lYMax1 > lYMax2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMin1, lYMax2 + 1), boost_point_type(lXMax1, lYMax2));

    // Create a rectangle at left if required
    if(lXMin1 < lXMin2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMin1, std::max(lYMin1, lYMin2)), boost_point_type(lXMin2 - 1, std::min(lYMax1, lYMax2)));

    // Create a rectangle at right if required
    if(lXMax1 > lXMax2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMax2 + 1, std::max(lYMin1, lYMin2)), boost_point_type(lXMax1, std::min(lYMax1, lYMax2)));
}

}

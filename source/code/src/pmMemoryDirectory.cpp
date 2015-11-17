
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
#include "pmTask.h"

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
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    SetRangeOwnerInternal(pRangeOwner, pOffset, pLength);
}
    
void pmMemoryDirectoryLinear::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    vmRangeOwner lRangeOwner(pRangeOwner);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    for(ulong i = 0; i < pCount; ++i)
    {
        lRangeOwner.hostOffset = pRangeOwner.hostOffset + i * pStep;
        SetRangeOwnerInternal(lRangeOwner, pOffset + i * pStep, pLength);
    }
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
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());

    GetOwnersInternal(pOffset, pLength, pOwnerships);
}

void pmMemoryDirectoryLinear::GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());

    pmMemOwnership lOwnerships;
    for(ulong i = 0; i < pCount; ++i)
        GetOwnersInternal(pOffset + i * pStep, pLength, lOwnerships);
    
    pScatteredOwnerships.reserve(lOwnerships.size());
    for_each(lOwnerships, [&] (const pmMemOwnership::value_type& pPair)
    {
        pScatteredOwnerships.emplace_back(std::piecewise_construct, std::forward_as_tuple(pPair.first, pPair.second.first, 0, 1), std::forward_as_tuple(pPair.second.second));
    });
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectoryLinear::GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    GetOwnersInternal(pOffset, pLength, pOwnerships);
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectoryLinear::GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships)
{
    pmMemOwnership lOwnerships;
    for(ulong i = 0; i < pCount; ++i)
        GetOwnersInternal(pOffset + i * pStep, pLength, lOwnerships);

    pScatteredOwnerships.reserve(lOwnerships.size());
    for_each(lOwnerships, [&] (const pmMemOwnership::value_type& pPair)
    {
        pScatteredOwnerships.emplace_back(std::piecewise_construct, std::forward_as_tuple(pPair.first, pPair.second.first, 0, 1), std::forward_as_tuple(pPair.second.second));
    });
}
    
void pmMemoryDirectoryLinear::Clear()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    mOwnershipMap.clear();
}

bool pmMemoryDirectoryLinear::IsEmpty()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());

    return mOwnershipMap.empty();
}
    
void pmMemoryDirectoryLinear::CloneFrom(pmMemoryDirectory* pDirectory)
{
    EXCEPTION_ASSERT(dynamic_cast<pmMemoryDirectoryLinear*>(pDirectory) != NULL);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    FINALIZE_RESOURCE_PTR(dSrcDirectoryOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pDirectory->mOwnershipLock, ReadLock(), Unlock());
    
    mOwnershipMap = static_cast<pmMemoryDirectoryLinear*>(pDirectory)->mOwnershipMap;
}


#ifdef _DEBUG
void pmMemoryDirectoryLinear::CheckMergability(const pmMemOwnership::iterator& pRange1, const pmMemOwnership::iterator& pRange2) const
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
    
void pmMemoryDirectoryLinear::SanitizeOwnerships() const
{
    if(mOwnershipMap.size() == 1)
        return;
    
    pmMemOwnership::const_iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end(), lPenultimate = lEnd;
    --lPenultimate;
    
    for(lIter = lBegin; lIter != lPenultimate; ++lIter)
    {
        pmMemOwnership::iterator lNext = lIter;
        ++lNext;
        
        CheckMergability(lIter, lNext);
    }
}

void pmMemoryDirectoryLinear::PrintOwnerships() const
{
    std::cout << "Host " << pmGetHostId() << " Ownership Dump " << std::endl;
    pmMemOwnership::const_iterator lIter, lBegin = mOwnershipMap.begin(), lEnd = mOwnershipMap.end();
    for(lIter = lBegin; lIter != lEnd; ++lIter)
        std::cout << "Range (" << lIter->first << " , " << lIter->second.first << ") is owned by host " << (uint)(*(lIter->second.second.host)) << " (" << lIter->second.second.hostOffset << ", " << lIter->second.first << ")" << std::endl;
        
    std::cout << std::endl;
}
#endif
    
template<typename consumer_type>
void pmMemoryDirectoryLinear::FindRegionsNotInFlight(pmInFlightRegions& pInFlightMap, void* pMem, size_t pOffset, size_t pLength, consumer_type& pRegionsToBeFetched, std::vector<pmCommandPtr>& pCommandVector)
{
    DEBUG_EXCEPTION_ASSERT(pLength);

	pmInFlightRegions::iterator lStartIter, lEndIter;
	pmInFlightRegions::iterator* lStartIterAddr = &lStartIter;
	pmInFlightRegions::iterator* lEndIterAddr = &lEndIter;

	char* lFetchAddress = (char*)pMem + pOffset;
	char* lLastFetchAddress = lFetchAddress + pLength - 1;

    FIND_FLOOR_ELEM(pmInFlightRegions, pInFlightMap, lFetchAddress, lStartIterAddr);	// Find range in flight just previous to the start of new range
    FIND_FLOOR_ELEM(pmInFlightRegions, pInFlightMap, lLastFetchAddress, lEndIterAddr);	// Find range in flight just previous to the end of new range
    
    // Both start and end of new range fall prior to all ranges in flight or there is no range in flight
    if(!lStartIterAddr && !lEndIterAddr)
    {
        pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, (ulong)lLastFetchAddress);
    }
    else
    {
        // If start of new range falls prior to all ranges in flight but end of new range does not
        if(!lStartIterAddr)
        {
            char* lFirstAddr = (char*)(pInFlightMap.begin()->first);
            pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, ((ulong)lFirstAddr)-1);
            lFetchAddress = lFirstAddr;
            lStartIter = pInFlightMap.begin();
        }
        
        // Both start and end of new range have atleast one in flight range prior to them
        
        // Check if start and end of new range fall within their just prior ranges or outside
        bool lStartInside = ((lFetchAddress >= (char*)(lStartIter->first)) && (lFetchAddress < ((char*)(lStartIter->first) + lStartIter->second.first)));
        bool lEndInside = ((lLastFetchAddress >= (char*)(lEndIter->first)) && (lLastFetchAddress < ((char*)(lEndIter->first) + lEndIter->second.first)));
        
        // If both start and end of new range have the same in flight range just prior to them
        if(lStartIter == lEndIter)
        {
            // If both start and end lie within the same in flight range, then the new range is already being fetched
            if(lStartInside && lEndInside)
            {
                pCommandVector.emplace_back(lStartIter->second.second.receiveCommand);
                return;
            }
            else if(lStartInside && !lEndInside)
            {
                // If start of new range is within an in flight range and that range is just prior to the end of new range
                pCommandVector.emplace_back(lStartIter->second.second.receiveCommand);
                
                pRegionsToBeFetched.emplace_back((ulong)((char*)(lStartIter->first) + lStartIter->second.first), (ulong)lLastFetchAddress);
            }
            else
            {
                // If both start and end of new range have the same in flight range just prior to them and they don't fall within that range
                pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, (ulong)lLastFetchAddress);
            }
        }
        else
        {
            // If start and end of new range have different in flight ranges prior to them
            
            // If start of new range does not fall within the in flight range
            if(!lStartInside)
            {
                ++lStartIter;
                pRegionsToBeFetched.emplace_back((ulong)lFetchAddress, ((ulong)(lStartIter->first))-1);
            }
            
            // If end of new range does not fall within the in flight range
            if(!lEndInside)
            {
                pRegionsToBeFetched.emplace_back((ulong)((char*)(lEndIter->first) + lEndIter->second.first), (ulong)lLastFetchAddress);
            }
            
            pCommandVector.emplace_back(lEndIter->second.second.receiveCommand);
            
            // Fetch all non in flight data between in flight ranges
            if(lStartIter != lEndIter)
            {
                for(pmInFlightRegions::iterator lTempIter = lStartIter; lTempIter != lEndIter; ++lTempIter)
                {
                    pCommandVector.emplace_back(lTempIter->second.second.receiveCommand);
                    
                    pmInFlightRegions::iterator lNextIter = lTempIter;
                    ++lNextIter;

                    // If there is any gap between the two ranges in flight
                    if((ulong)((char*)(lTempIter->first) + lTempIter->second.first) != ((ulong)(lNextIter->first)))
                        pRegionsToBeFetched.emplace_back((ulong)((char*)(lTempIter->first) + lTempIter->second.first), ((ulong)(lNextIter->first))-1);
                }
            }
        }
    }
}

pmScatteredTransferMapType pmMemoryDirectoryLinear::SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet)
{
	pmScatteredSubscriptionFilter lBlocksFilter(pScatteredSubscriptionInfo);
    pmScatteredSubscriptionFilterHelper lLocalFilter(lBlocksFilter, pAddressSpaceBaseAddr, *this, true);
    
    pmScatteredTransferMapType lMachineVersusTupleVectorMap;

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    const auto& lBlocks = lBlocksFilter.FilterBlocks([&] (size_t pRow)
    {
        std::vector<pmCommandPtr> lInnerCommandVector;

        FindRegionsNotInFlight(mInFlightLinearMap, pAddressSpaceBaseAddr, pScatteredSubscriptionInfo.offset + pRow * pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.size, lLocalFilter, lInnerCommandVector);

        // If the range is already in one or more scattered flights, then multiple general entries are put into inFlightMap
        // Having a set ensures that a command is inserted and subsequently waited upon only once
        std::move(lInnerCommandVector.begin(), lInnerCommandVector.end(), std::inserter(pCommandsAlreadyIssuedSet, pCommandsAlreadyIssuedSet.begin()));
    });

    for_each(lBlocks, [&] (const std::pair<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& pMapKeyValue)
    {
        for_each(pMapKeyValue.second, [&] (const std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>& pPair)
        {
            EXCEPTION_ASSERT(pPair.first.size && pPair.first.step && pPair.first.count);

            pmCommandPtr lCommand = pmCountDownCommand::CreateSharedPtr(pPair.first.count, pPriority, communicator::RECEIVE, 0);	// Dummy command just to allow threads to wait on it
            lCommand->MarkExecutionStart();

            for(size_t i = 0; i < pPair.first.count; ++i)
            {
                char* lAddr = (char*)pAddressSpaceBaseAddr + pPair.first.offset + i * pPair.first.step;
                mInFlightLinearMap.emplace(std::piecewise_construct, std::forward_as_tuple(lAddr), std::forward_as_tuple(pPair.first.size, regionFetchData(lCommand)));
            }
            
            lMachineVersusTupleVectorMap[pMapKeyValue.first].emplace_back(pPair.first, pPair.second, lCommand);
        });
    });
    
    return lMachineVersusTupleVectorMap;
}

pmRemoteRegionsInfoMapType pmMemoryDirectoryLinear::GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr)
{
	pmScatteredSubscriptionFilter lBlocksFilter(pScatteredSubscriptionInfo);
    pmScatteredSubscriptionFilterHelper lLocalFilter(lBlocksFilter, pAddressSpaceBaseAddr, *this, true);
    
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());
    
    const auto& lBlocks = lBlocksFilter.FilterBlocks([&] (size_t pRow)
    {
        ulong lStartAddr = (ulong)((char*)pAddressSpaceBaseAddr + pScatteredSubscriptionInfo.offset + pRow * pScatteredSubscriptionInfo.step);

        lLocalFilter.emplace_back(lStartAddr, lStartAddr + pScatteredSubscriptionInfo.size - 1);
    });
    
    return lBlocks;
}
    
pmLinearTransferVectorType pmMemoryDirectoryLinear::SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector)
{
    std::vector<std::pair<ulong, ulong>> lRegionsToBeFetched;	// Start address and last address of sub ranges to be fetched
    pmLinearTransferVectorType lTupleVector;

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    FindRegionsNotInFlight(mInFlightLinearMap, pAddressSpaceBaseAddr, pSubscriptionInfo.offset, pSubscriptionInfo.length, lRegionsToBeFetched, pCommandVector);

    for_each(lRegionsToBeFetched, [&] (const std::pair<ulong, ulong>& pPair)
    {
		ulong lOffset = pPair.first - (ulong)pAddressSpaceBaseAddr;
		ulong lLength = pPair.second - pPair.first + 1;
        
        if(lLength)
        {
            pmMemOwnership lOwnerships;
            GetOwnersUnprotected(lOffset, lLength, lOwnerships);

            for_each(lOwnerships, [&] (pmMemOwnership::value_type& pInnerPair)
            {
                vmRangeOwner& lRangeOwner = pInnerPair.second.second;

                if(lRangeOwner.host != PM_LOCAL_MACHINE)
                {
                    char* lAddr = (char*)pAddressSpaceBaseAddr + pInnerPair.first;

                    pmCommandPtr lCommand = pmCommand::CreateSharedPtr(pPriority, communicator::RECEIVE, 0);	// Dummy command just to allow threads to wait on it
                    lCommand->MarkExecutionStart();

                    mInFlightLinearMap.emplace(std::piecewise_construct, std::forward_as_tuple(lAddr), std::forward_as_tuple(pInnerPair.second.first, regionFetchData(lCommand)));
                    lTupleVector.emplace_back(pmSubscriptionInfo(pInnerPair.first, pInnerPair.second.first), lRangeOwner, lCommand);
                }
            });
        }
    });

    return lTupleVector;
}
    
void pmMemoryDirectoryLinear::CancelUnreferencedRequests()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    // During scattered transfers, a single command is replicated as general fetch commands and multiple entries are
    // made into the inFlightMap. This increases the use_count of the shared ptr. Here, it is important to find out
    // the use_count that is external to inFlightMap. To compute the external use_count, the following map is used.
    std::map<pmCommandPtr, std::vector<pmInFlightRegions::iterator>> lInFlightUseCountMap;
    
    auto lIter = mInFlightLinearMap.begin(), lEnd = mInFlightLinearMap.end();
    while(lIter != lEnd)
    {
        if(lIter->second.second.receiveCommand.unique())
        {
            mInFlightLinearMap.erase(lIter++);
        }
        else
        {
            decltype(lInFlightUseCountMap)::iterator lUseCountIter = lInFlightUseCountMap.find(lIter->second.second.receiveCommand);
            if(lUseCountIter == lInFlightUseCountMap.end())
                lUseCountIter = lInFlightUseCountMap.emplace(std::piecewise_construct, std::forward_as_tuple(lIter->second.second.receiveCommand), std::forward_as_tuple()).first;
            
            lUseCountIter->second.push_back(lIter);
            
            ++lIter;
        }
    }
    
    for_each(lInFlightUseCountMap, [&] (const decltype(lInFlightUseCountMap)::value_type& pPair)
    {
        if((ulong)pPair.first.use_count() == (ulong)pPair.second.size() + 1) // Plus 1 for lInFlightUseCountMap
        {
            for_each(pPair.second, [&] (const pmInFlightRegions::iterator& pIter)
            {
                mInFlightLinearMap.erase(pIter);
            });
        }
    });
}

void pmMemoryDirectoryLinear::CopyOrUpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    if(CopyOrUpdateReceivedMemoryInternal(pAddressSpace, pAddressSpaceBaseAddr, mInFlightLinearMap, pLockingTask, pOffset, pLength, pDataSource))
    {
    #ifdef ENABLE_TASK_PROFILING
        if(pLockingTask)
            pLockingTask->GetTaskProfiler()->RecordProfileEvent(pLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
    #endif
    }
}

void pmMemoryDirectoryLinear::UpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    bool lComplete = true;
    for(ulong i = 0; i < pCount; ++i)
        lComplete &= CopyOrUpdateReceivedMemoryInternal(pAddressSpace, pAddressSpaceBaseAddr, mInFlightLinearMap, pLockingTask, pOffset + i * pStep, pLength);

#ifdef ENABLE_TASK_PROFILING
    if(lComplete && pLockingTask)
        pLockingTask->GetTaskProfiler()->RecordProfileEvent(pLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
#endif
}
    
// Must be called with WriteLock on address space directory acquired
void pmMemoryDirectoryLinear::AcquireOwnershipImmediateInternal(ulong pOffset, ulong pLength)
{
    SetRangeOwnerInternal(vmRangeOwner(PM_LOCAL_MACHINE, pOffset, mMemoryIdentifierStruct), pOffset, pLength);
}

// Must be called with WriteLock on address space directory acquired
bool pmMemoryDirectoryLinear::CopyOrUpdateReceivedMemoryInternal(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmInFlightRegions& pInFlightMap, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource /* = NULL */)
{
    void* lDestMem = pAddressSpaceBaseAddr;
    char* lAddr = (char*)lDestMem + pOffset;
    
    bool lTransferCommandComplete = false;
    
    pmInFlightRegions::iterator lIter = pInFlightMap.find(lAddr);
    if((lIter != pInFlightMap.end()) && (lIter->second.first == pLength))
    {
        std::pair<size_t, regionFetchData>& lPair = lIter->second;
        
        if(pDataSource)
            (*pDataSource)(lAddr, pLength);

        regionFetchData& lData = lPair.second;
        AcquireOwnershipImmediateInternal(pOffset, lPair.first);

        pmCountDownCommand* lCountDownCommand = dynamic_cast<pmCountDownCommand*>(lData.receiveCommand.get());
        if(!lCountDownCommand || lCountDownCommand->GetOutstandingCount() == 1)
            lTransferCommandComplete = true;

        pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
        lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

        pInFlightMap.erase(lIter);
    }
    else
    {
        pmInFlightRegions::iterator lBaseIter;
        pmInFlightRegions::iterator* lBaseIterAddr = &lBaseIter;
        FIND_FLOOR_ELEM(pmInFlightRegions, pInFlightMap, lAddr, lBaseIterAddr);
        
        if(!lBaseIterAddr)
            PMTHROW(pmFatalErrorException());
        
        size_t lStartAddr = reinterpret_cast<size_t>(lBaseIter->first);
        std::pair<size_t, regionFetchData>& lPair = lBaseIter->second;
        
        size_t lRecvAddr = reinterpret_cast<size_t>(lAddr);
        if((lRecvAddr < lStartAddr) || (lRecvAddr + pLength > lStartAddr + lPair.first))
            PMTHROW(pmFatalErrorException());
        
        typedef std::map<size_t, size_t> partialReceiveRecordType;
        regionFetchData& lData = lPair.second;
        partialReceiveRecordType& lPartialReceiveRecordMap = lData.partialReceiveRecordMap;
                
        partialReceiveRecordType::iterator lPartialIter;
        partialReceiveRecordType::iterator* lPartialIterAddr = &lPartialIter;
        FIND_FLOOR_ELEM(partialReceiveRecordType, lPartialReceiveRecordMap, lRecvAddr, lPartialIterAddr);

        if(lPartialIterAddr && lPartialIter->first + lPartialIter->second - 1 >= lRecvAddr)
            PMTHROW(pmFatalErrorException());   // Multiple overlapping partial receives

        lData.accumulatedPartialReceivesLength += pLength;
        if(lData.accumulatedPartialReceivesLength > lPair.first)
            PMTHROW(pmFatalErrorException());

        bool lTransferComplete = (lData.accumulatedPartialReceivesLength == lPair.first);

        if(lTransferComplete)
        {
            pmCountDownCommand* lCountDownCommand = dynamic_cast<pmCountDownCommand*>(lData.receiveCommand.get());
            if(!lCountDownCommand || lCountDownCommand->GetOutstandingCount() == 1)
                lTransferCommandComplete = true;

            if(pDataSource)
                (*pDataSource)(lAddr, pLength);

            size_t lOffset = lStartAddr - reinterpret_cast<size_t>(lDestMem);
            AcquireOwnershipImmediateInternal(lOffset, lPair.first);

            pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
            lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

            pInFlightMap.erase(lBaseIter);
        }
        else
        {            
            // Make partial receive entry
            lPartialReceiveRecordMap[lRecvAddr] = pLength;
            
            if(pDataSource)
                (*pDataSource)(lAddr, pLength);

            size_t lOffset = lRecvAddr - reinterpret_cast<size_t>(lDestMem);
            AcquireOwnershipImmediateInternal(lOffset, pLength);
        }
    }
    
    return lTransferCommandComplete;
}


/* class pmMemoryDirectory2D */
pmMemoryDirectory2D::pmMemoryDirectory2D(ulong pAddressSpaceRows, ulong pAddressSpaceCols, const communicator::memoryIdentifierStruct& pMemoryIdentifierStruct)
    : pmMemoryDirectory(pMemoryIdentifierStruct)
    , mAddressSpaceRows(pAddressSpaceRows)
    , mAddressSpaceCols(pAddressSpaceCols)
{}

pmMemoryDirectory2D::boost_box_type pmMemoryDirectory2D::GetBox(ulong pOffset, ulong pLength, ulong pStep, ulong pCount) const
{
    boost_point_type lPoint1(pOffset % pStep, pOffset / pStep);
    boost_point_type lPoint2(lPoint1.get<0>() + pLength - 1, lPoint1.get<1>() + pCount - 1);
    
    return boost_box_type(lPoint1, lPoint2);
}

pmScatteredSubscriptionInfo pmMemoryDirectory2D::GetReverseBoxMapping(const boost_box_type& pBox) const
{
    const boost_point_type& lMinCorner = pBox.min_corner();
    const boost_point_type& lMaxCorner = pBox.max_corner();
    
    return pmScatteredSubscriptionInfo(lMinCorner.get<0>() + lMinCorner.get<1>() * mAddressSpaceCols, lMaxCorner.get<0>() - lMinCorner.get<0>() + 1, mAddressSpaceCols, lMaxCorner.get<1>() - lMinCorner.get<1>() + 1);
}
    
ulong pmMemoryDirectory2D::GetReverseBoxOffset(const boost_box_type& pBox) const
{
    const boost_point_type& lMinCorner = pBox.min_corner();
    return lMinCorner.get<0>() + lMinCorner.get<1>() * mAddressSpaceCols;
}
    
void pmMemoryDirectory2D::Reset(const pmMachine* pOwner)
{
    mOwnershipRTree.insert(std::make_pair(GetBox(0, mAddressSpaceCols, mAddressSpaceCols, mAddressSpaceRows), vmRangeOwner(pOwner, 0, mMemoryIdentifierStruct)));
}
    
void pmMemoryDirectory2D::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
{
    pmScatteredMemOwnership lScatteredOwnerships;
    
    ulong lRemainder = (pOffset % mAddressSpaceCols);
    ulong lFirstRowLength = lRemainder ? std::min(mAddressSpaceCols - lRemainder, pLength) : 0;
    
    ulong lRemainingLength = pLength - lFirstRowLength;
    ulong lIntermediateRowsOffset = pOffset + lFirstRowLength;
    
    ulong lScatteredCount = lRemainingLength / mAddressSpaceCols;
    ulong lLeftover = lRemainingLength % mAddressSpaceCols;

    if(lFirstRowLength)
        SetRangeOwner(pRangeOwner, pOffset, lFirstRowLength, mAddressSpaceCols, 1);
    
    if(lScatteredCount)
    {
        vmRangeOwner lRangeOwner(pRangeOwner);
        lRangeOwner.hostOffset += lFirstRowLength;

        SetRangeOwner(lRangeOwner, lIntermediateRowsOffset, mAddressSpaceCols, mAddressSpaceCols, lScatteredCount);
    }
    
    if(lLeftover)
    {
        vmRangeOwner lRangeOwner(pRangeOwner);
        lRangeOwner.hostOffset += (lFirstRowLength + lScatteredCount * mAddressSpaceCols);

        SetRangeOwner(lRangeOwner, lIntermediateRowsOffset + lScatteredCount * mAddressSpaceCols, lLeftover, mAddressSpaceCols, 1);
    }
}

void pmMemoryDirectory2D::SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    EXCEPTION_ASSERT(pLength <= mAddressSpaceCols && pStep == mAddressSpaceCols);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    SetRangeOwnerInternal(pRangeOwner, pOffset, pLength, pStep, pCount);
}

void pmMemoryDirectory2D::SetRangeOwnerInternal(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    boost_box_type lBox = GetBox(pOffset, pLength, pStep, pCount);
    
    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    std::vector<boost_box_type> lRemainingBoxes;
    lRemainingBoxes.reserve(4);
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        ulong lStoredBoxOffset = GetReverseBoxOffset(pPair.first);

        lRemainingBoxes.clear();
        GetDifferenceOfBoxes(pPair.first, lBox, lRemainingBoxes);

        mOwnershipRTree.remove(pPair);
        
        for_each(lRemainingBoxes, [&] (const boost_box_type& pBox)
        {
            vmRangeOwner lRangeOwner(pPair.second);
            lRangeOwner.hostOffset += GetReverseBoxOffset(pBox) - lStoredBoxOffset;

            CombineAndInsertBox(pBox, lRangeOwner);
        });
    });

#ifdef _DEBUG
    lOverlappingBoxes.clear();
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    EXCEPTION_ASSERT(lOverlappingBoxes.empty());

#endif

    CombineAndInsertBox(lBox, pRangeOwner);
    
#ifdef _DEBUG
#if 0
    PrintOwnerships();
    SanitizeOwnerships();
#endif
#endif
}
    
void pmMemoryDirectory2D::CombineAndInsertBox(boost_box_type pBox, vmRangeOwner pRangeOwner)
{
    const boost_point_type& lMinCorner = pBox.min_corner();
    const boost_point_type& lMaxCorner = pBox.max_corner();

    ulong lXMin = lMinCorner.get<0>();
    ulong lYMin = lMinCorner.get<1>();
    ulong lXMax = lMaxCorner.get<0>();
    ulong lYMax = lMaxCorner.get<1>();

    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    bool lCombined = false;

    // Combine pBox with the one at top if matching
    boost_box_type lTopEdgeBox(boost_point_type(lXMin, lYMin - 1), boost_point_type(lXMax, lYMin - 1));
    mOwnershipRTree.query(boost::geometry::index::intersects(lTopEdgeBox), std::back_inserter(lOverlappingBoxes));

    if(lOverlappingBoxes.size() == 1)
    {
        auto lPair = lOverlappingBoxes.begin();
        boost_point_type lOverlappingMinCorner = lPair->first.min_corner();
        boost_point_type lOverlappingMaxCorner = lPair->first.max_corner();
        
        if(lPair->second.host == pRangeOwner.host && lXMin == lOverlappingMinCorner.get<0>() && lXMax == lOverlappingMaxCorner.get<0>())
        {
            EXCEPTION_ASSERT(lYMin - 1 == lOverlappingMaxCorner.get<1>());

            pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(lPair->first);

            if(lPair->second.hostOffset + lScatteredSubscriptionInfo.count * lScatteredSubscriptionInfo.step == pRangeOwner.hostOffset)
            {
                pBox.min_corner() = lOverlappingMinCorner;
                pRangeOwner.hostOffset = lPair->second.hostOffset;
                
                lXMin = pBox.min_corner().get<0>();
                lYMin = pBox.min_corner().get<1>();
                
                mOwnershipRTree.remove(*lPair);
                lCombined = true;
            }
        }
    }

    // Combine pBox with the one at bottom if matching
    lOverlappingBoxes.clear();
    boost_box_type lBottomEdgeBox(boost_point_type(lXMin, lYMax + 1), boost_point_type(lXMax, lYMax + 1));
    mOwnershipRTree.query(boost::geometry::index::intersects(lBottomEdgeBox), std::back_inserter(lOverlappingBoxes));

    if(lOverlappingBoxes.size() == 1)
    {
        auto lPair = lOverlappingBoxes.begin();
        boost_point_type lOverlappingMinCorner = lPair->first.min_corner();
        boost_point_type lOverlappingMaxCorner = lPair->first.max_corner();
        
        if(lPair->second.host == pRangeOwner.host && lXMin == lOverlappingMinCorner.get<0>() && lXMax == lOverlappingMaxCorner.get<0>())
        {
            EXCEPTION_ASSERT(lYMax + 1 == lOverlappingMinCorner.get<1>());

            pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(pBox);

            if(lPair->second.hostOffset == pRangeOwner.hostOffset + lScatteredSubscriptionInfo.count * lScatteredSubscriptionInfo.step)
            {
                pBox.max_corner() = lOverlappingMaxCorner;

                lXMax = pBox.max_corner().get<0>();
                lYMax = pBox.max_corner().get<1>();

                mOwnershipRTree.remove(*lPair);
                lCombined = true;
            }
        }
    }

    // Combine pBox with the one at left if matching
    lOverlappingBoxes.clear();
    boost_box_type lLeftEdgeBox(boost_point_type(lXMin - 1, lYMin), boost_point_type(lXMin - 1, lYMax));
    mOwnershipRTree.query(boost::geometry::index::intersects(lLeftEdgeBox), std::back_inserter(lOverlappingBoxes));

    if(lOverlappingBoxes.size() == 1)
    {
        auto lPair = lOverlappingBoxes.begin();
        boost_point_type lOverlappingMinCorner = lPair->first.min_corner();
        boost_point_type lOverlappingMaxCorner = lPair->first.max_corner();
        
        if(lPair->second.host == pRangeOwner.host && lYMin == lOverlappingMinCorner.get<1>() && lYMax == lOverlappingMaxCorner.get<1>())
        {
            EXCEPTION_ASSERT(lXMin - 1 == lOverlappingMaxCorner.get<0>());

            pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(lPair->first);

            if(lPair->second.hostOffset + lScatteredSubscriptionInfo.size == pRangeOwner.hostOffset)
            {
                pBox.min_corner() = lOverlappingMinCorner;
                pRangeOwner.hostOffset = lPair->second.hostOffset;
                
                lXMin = pBox.min_corner().get<0>();
                lYMin = pBox.min_corner().get<1>();

                mOwnershipRTree.remove(*lPair);
                lCombined = true;
            }
        }
    }

    // Combine pBox with the one at right if matching
    lOverlappingBoxes.clear();
    boost_box_type lRightEdgeBox(boost_point_type(lXMax + 1, lYMin), boost_point_type(lXMax + 1, lYMax));
    mOwnershipRTree.query(boost::geometry::index::intersects(lRightEdgeBox), std::back_inserter(lOverlappingBoxes));

    if(lOverlappingBoxes.size() == 1)
    {
        auto lPair = lOverlappingBoxes.begin();
        boost_point_type lOverlappingMinCorner = lPair->first.min_corner();
        boost_point_type lOverlappingMaxCorner = lPair->first.max_corner();
        
        if(lPair->second.host == pRangeOwner.host && lYMin == lOverlappingMinCorner.get<1>() && lYMax == lOverlappingMaxCorner.get<1>())
        {
            EXCEPTION_ASSERT(lXMax + 1 == lOverlappingMinCorner.get<0>());

            pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(pBox);

            if(lPair->second.hostOffset == pRangeOwner.hostOffset + lScatteredSubscriptionInfo.size)
            {
                pBox.max_corner() = lOverlappingMaxCorner;

                lXMax = pBox.max_corner().get<0>();
                lYMax = pBox.max_corner().get<1>();
                
                mOwnershipRTree.remove(*lPair);
                lCombined = true;
            }
        }
    }

    if(lCombined)
        CombineAndInsertBox(pBox, pRangeOwner);
    else
        mOwnershipRTree.insert(std::make_pair(pBox, pRangeOwner));
}

void pmMemoryDirectory2D::GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    pmScatteredMemOwnership lScatteredOwnerships;
    
    ulong lRemainder = (pOffset % mAddressSpaceCols);
    ulong lFirstRowLength = lRemainder ? std::min(mAddressSpaceCols - lRemainder, pLength) : 0;
    
    ulong lRemainingLength = pLength - lFirstRowLength;
    ulong lIntermediateRowsOffset = pOffset + lFirstRowLength;
    
    ulong lScatteredCount = lRemainingLength / mAddressSpaceCols;
    ulong lLeftover = lRemainingLength % mAddressSpaceCols;

    if(lFirstRowLength)
        GetOwners(pOffset, lFirstRowLength, mAddressSpaceCols, 1, lScatteredOwnerships);
    
    if(lScatteredCount)
        GetOwners(lIntermediateRowsOffset, mAddressSpaceCols, mAddressSpaceCols, lScatteredCount, lScatteredOwnerships);
    
    if(lLeftover)
        GetOwners(lIntermediateRowsOffset + lScatteredCount * mAddressSpaceCols, lLeftover, mAddressSpaceCols, 1, lScatteredOwnerships);

    for_each(lScatteredOwnerships, [&] (pmScatteredMemOwnership::value_type& pPair)
    {
        vmRangeOwner lRangeOwner(pPair.second);
        for(ulong i = 0; i < pPair.first.count; ++i)
        {
            lRangeOwner.hostOffset = pPair.second.hostOffset + i * pPair.first.step;
            pOwnerships.emplace(std::piecewise_construct, std::forward_as_tuple(pPair.first.offset + i * pPair.first.step), std::forward_as_tuple(pPair.first.size, lRangeOwner));
        }
    });
}
    
void pmMemoryDirectory2D::GetOwners(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pOwnerships)
{
    EXCEPTION_ASSERT(mAddressSpaceCols == pStep);
    if(!((pOffset + pStep * (pCount - 1) + pLength) <= (mAddressSpaceRows * mAddressSpaceCols)))
    {
        std::cout << pOffset << " " << pLength << " " << pStep << " " << pCount << " " << mAddressSpaceRows << " " << mAddressSpaceCols << " " << (pOffset + pStep * (pCount - 1) + pLength) << std::endl;
        PMTHROW(pmFatalErrorException());
    }

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());
    
    GetOwnersInternal(pOffset, pLength, pStep, pCount, pOwnerships);
}

// Must be called with mOwnershipLock acquired
void pmMemoryDirectory2D::GetOwnersInternal(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships)
{
    EXCEPTION_ASSERT(pLength <= mAddressSpaceCols && pStep == mAddressSpaceCols);

    boost_box_type lBox = GetBox(pOffset, pLength, pStep, pCount);

    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        ulong lStoredBoxOffset = GetReverseBoxOffset(pPair.first);
        
        boost_box_type lOutBox(boost_point_type(0, 0), boost_point_type(0, 0));
        boost::geometry::intersection(lBox, pPair.first, lOutBox);

        pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(lOutBox);
        
        vmRangeOwner lRangeOwner(pPair.second);
        lRangeOwner.hostOffset += lScatteredSubscriptionInfo.offset - lStoredBoxOffset;

        pScatteredOwnerships.emplace_back(lScatteredSubscriptionInfo, lRangeOwner);
    });
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectory2D::GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships)
{
    pmScatteredMemOwnership lScatteredOwnerships;
    
    ulong lRemainder = (pOffset % mAddressSpaceCols);
    ulong lFirstRowLength = lRemainder ? std::min(mAddressSpaceCols - lRemainder, pLength) : 0;
    
    ulong lRemainingLength = pLength - lFirstRowLength;
    ulong lIntermediateRowsOffset = pOffset + lFirstRowLength;
    
    ulong lScatteredCount = lRemainingLength / mAddressSpaceCols;
    ulong lLeftover = lRemainingLength % mAddressSpaceCols;

    if(lFirstRowLength)
        GetOwnersInternal(pOffset, lFirstRowLength, mAddressSpaceCols, 1, lScatteredOwnerships);
    
    if(lScatteredCount)
        GetOwnersInternal(lIntermediateRowsOffset, mAddressSpaceCols, mAddressSpaceCols, lScatteredCount, lScatteredOwnerships);
    
    if(lLeftover)
        GetOwnersInternal(lIntermediateRowsOffset + lScatteredCount * mAddressSpaceCols, lLeftover, mAddressSpaceCols, 1, lScatteredOwnerships);

    for_each(lScatteredOwnerships, [&] (pmScatteredMemOwnership::value_type& pPair)
    {
        vmRangeOwner lRangeOwner(pPair.second);
        for(ulong i = 0; i < pPair.first.count; ++i)
        {
            lRangeOwner.hostOffset = pPair.second.hostOffset + i * pPair.first.step;
            pOwnerships.emplace(std::piecewise_construct, std::forward_as_tuple(pPair.first.offset + i * pPair.first.step), std::forward_as_tuple(pPair.first.size, lRangeOwner));
        }
    });
}

// This method is temporarily created for preprocessor task
void pmMemoryDirectory2D::GetOwnersUnprotected(ulong pOffset, ulong pLength, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships)
{
    GetOwnersInternal(pOffset, pLength, pStep, pCount, pScatteredOwnerships);
}

void pmMemoryDirectory2D::Clear()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    
    mOwnershipRTree.clear();
}

bool pmMemoryDirectory2D::IsEmpty()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, ReadLock(), Unlock());

    return mOwnershipRTree.empty();
}

void pmMemoryDirectory2D::CloneFrom(pmMemoryDirectory* pDirectory)
{
    EXCEPTION_ASSERT(dynamic_cast<pmMemoryDirectory2D*>(pDirectory) != NULL);

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());
    FINALIZE_RESOURCE_PTR(dSrcDirectoryOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &pDirectory->mOwnershipLock, ReadLock(), Unlock());

    mOwnershipRTree = static_cast<pmMemoryDirectory2D*>(pDirectory)->mOwnershipRTree;
}

// Subtracts pBox2 from pBox1
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
        pRemainingBoxes.emplace_back(boost_point_type(lXMin1, lYMax2 + 1), boost_point_type(lXMax1, lYMax1));

    // Create a rectangle at left if required
    if(lXMin1 < lXMin2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMin1, std::max(lYMin1, lYMin2)), boost_point_type(lXMin2 - 1, std::min(lYMax1, lYMax2)));

    // Create a rectangle at right if required
    if(lXMax1 > lXMax2)
        pRemainingBoxes.emplace_back(boost_point_type(lXMax2 + 1, std::max(lYMin1, lYMin2)), boost_point_type(lXMax1, std::min(lYMax1, lYMax2)));
}
    
void pmMemoryDirectory2D::CancelUnreferencedRequests()
{
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    boost_box_type lBounds = mInFlightRTree.bounds();
    
    std::vector<std::pair<boost_box_type, regionFetchData2D>> lOverlappingBoxes;
    mInFlightRTree.query(boost::geometry::index::intersects(lBounds), std::back_inserter(lOverlappingBoxes));
    
    std::vector<std::pair<boost_box_type, regionFetchData2D>> lRemovableRequests;
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, regionFetchData2D>& pPair)
    {
        if(pPair.second.receiveCommand.unique())
            lRemovableRequests.emplace_back(pPair);
    });

    for_each(lRemovableRequests, [&] (const std::pair<boost_box_type, regionFetchData2D>& pPair)
    {
        mInFlightRTree.remove(pPair);
    });
}

pmScatteredTransferMapType pmMemoryDirectory2D::SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet)
{
    pmScatteredMemOwnership lScatteredMemOwnerships;
    pmScatteredTransferMapType lMachineVersusTupleVectorMap;

    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    GetOwnersInternal(pScatteredSubscriptionInfo.offset, pScatteredSubscriptionInfo.size, pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.count, lScatteredMemOwnerships);
    
    for_each(lScatteredMemOwnerships, [&] (const pmScatteredMemOwnership::value_type& pPair)
    {
        if(pPair.second.host != PM_LOCAL_MACHINE)
        {
            std::vector<boost_box_type> lRemainingBoxes1, lRemainingBoxes2;
            boost_box_type lBox = GetBox(pPair.first.offset, pPair.first.size, pPair.first.step, pPair.first.count);
            
            lRemainingBoxes1.emplace_back(lBox);
            ushort lCurrentRemainingBoxVectorIndex = 0;
            
            std::vector<std::pair<boost_box_type, regionFetchData2D>> lOverlappingBoxes;
            mInFlightRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
            
            for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, regionFetchData2D>& pInnerPair)
            {
                pCommandsAlreadyIssuedSet.emplace(pInnerPair.second.receiveCommand);

                const std::vector<boost_box_type>& lSrcBoxes = lCurrentRemainingBoxVectorIndex ? lRemainingBoxes2 : lRemainingBoxes1;
                std::vector<boost_box_type>& lDestBoxes = lCurrentRemainingBoxVectorIndex ? lRemainingBoxes1 : lRemainingBoxes2;

                lDestBoxes.clear();

                lCurrentRemainingBoxVectorIndex = 1 - lCurrentRemainingBoxVectorIndex;

                for_each(lSrcBoxes, [&] (const boost_box_type& pSrcBox)
                {
                    GetDifferenceOfBoxes(pSrcBox, pInnerPair.first, lDestBoxes);
                });
            });

            const std::vector<boost_box_type>& lRemainingBoxes = lCurrentRemainingBoxVectorIndex ? lRemainingBoxes2 : lRemainingBoxes1;

            for_each(lRemainingBoxes, [&] (const boost_box_type& pBox)
            {
                pmCommandPtr lCommand = pmCommand::CreateSharedPtr(pPriority, communicator::RECEIVE, 0);	// Dummy command just to allow threads to wait on it
                lCommand->MarkExecutionStart();
                
                pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(pBox);

                vmRangeOwner lRangeOwner = pPair.second;
                lRangeOwner.hostOffset += lScatteredSubscriptionInfo.offset - pPair.first.offset;
                
                mInFlightRTree.insert(std::make_pair(pBox, regionFetchData2D(lCommand)));
                lMachineVersusTupleVectorMap[pPair.second.host].emplace_back(lScatteredSubscriptionInfo, lRangeOwner, lCommand);

            });
        }
    });
    
    return lMachineVersusTupleVectorMap;
}

pmLinearTransferVectorType pmMemoryDirectory2D::SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, void* pAddressSpaceBaseAddr, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector)
{
    pmLinearTransferVectorType lLinearTransferVector;
    pmScatteredTransferMapType lScatteredTransferMap1, lScatteredTransferMap2, lScatteredTransferMap3;
    std::set<pmCommandPtr> lCommandsAlreadyIssuedSet;

    ulong lRemainder = (pSubscriptionInfo.offset % mAddressSpaceCols);
    ulong lFirstRowLength = lRemainder ? std::min(mAddressSpaceCols - lRemainder, pSubscriptionInfo.length) : 0;
    
    ulong lRemainingLength = pSubscriptionInfo.length - lFirstRowLength;
    ulong lIntermediateRowsOffset = pSubscriptionInfo.offset + lFirstRowLength;
    
    ulong lScatteredCount = lRemainingLength / mAddressSpaceCols;
    ulong lLeftover = lRemainingLength % mAddressSpaceCols;

    if(lFirstRowLength)
        lScatteredTransferMap1 = SetupRemoteRegionsForFetching(pmScatteredSubscriptionInfo(pSubscriptionInfo.offset, lFirstRowLength, mAddressSpaceCols, 1), pAddressSpaceBaseAddr,  pPriority, lCommandsAlreadyIssuedSet);
    
    if(lScatteredCount)
        lScatteredTransferMap2 = SetupRemoteRegionsForFetching(pmScatteredSubscriptionInfo(lIntermediateRowsOffset, mAddressSpaceCols, mAddressSpaceCols, lScatteredCount), pAddressSpaceBaseAddr,  pPriority, lCommandsAlreadyIssuedSet);
    
    if(lLeftover)
        lScatteredTransferMap3 = SetupRemoteRegionsForFetching(pmScatteredSubscriptionInfo(lIntermediateRowsOffset + lScatteredCount * mAddressSpaceCols, lLeftover, mAddressSpaceCols, 1), pAddressSpaceBaseAddr,  pPriority, lCommandsAlreadyIssuedSet);
    
    pCommandVector.reserve(lCommandsAlreadyIssuedSet.size());
    std::copy(lCommandsAlreadyIssuedSet.begin(), lCommandsAlreadyIssuedSet.end(), std::back_inserter(pCommandVector));
    
    lLinearTransferVector.reserve(lScatteredTransferMap1.size() + lScatteredTransferMap2.size() + lScatteredTransferMap3.size());

    auto lLambda = [&lLinearTransferVector] (pmScatteredTransferMapType& pScatteredTransferMap)
    {
        for_each(pScatteredTransferMap, [&] (pmScatteredTransferMapType::value_type& pMapKeyValue)
        {
            for_each(pMapKeyValue.second, [&] (std::tuple<pmScatteredSubscriptionInfo, vmRangeOwner, pmCommandPtr>& pTuple)
            {
                const pmScatteredSubscriptionInfo& lInfo = std::get<0>(pTuple);
                vmRangeOwner lRangeOwner = std::get<1>(pTuple);
                pmCommandPtr& lCommandPtr = std::get<2>(pTuple);
                
                if(lInfo.size == lInfo.step)
                {
                    lLinearTransferVector.emplace_back(pmSubscriptionInfo(lInfo.offset, lInfo.size * lInfo.count), lRangeOwner, lCommandPtr);
                }
                else
                {
                    for(ulong i = 0; i < lInfo.count; ++i)
                    {
                        lRangeOwner.hostOffset += lInfo.step;
                        lLinearTransferVector.emplace_back(pmSubscriptionInfo(lInfo.offset + i * lInfo.step, lInfo.size), lRangeOwner, lCommandPtr);
                    }
                }
            });
        });
    };
    
    lLambda(lScatteredTransferMap1);
    lLambda(lScatteredTransferMap2);
    lLambda(lScatteredTransferMap3);

    return lLinearTransferVector;
}

pmRemoteRegionsInfoMapType pmMemoryDirectory2D::GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, void* pAddressSpaceBaseAddr)
{
    pmRemoteRegionsInfoMapType lRemoteRegionsInfoMap;
    pmScatteredMemOwnership lScatteredMemOwnerships;

    GetOwners(pScatteredSubscriptionInfo.offset, pScatteredSubscriptionInfo.size, pScatteredSubscriptionInfo.step, pScatteredSubscriptionInfo.count, lScatteredMemOwnerships);
    
    for_each(lScatteredMemOwnerships, [&] (const pmScatteredMemOwnership::value_type& pPair)
    {
        if(pPair.second.host != PM_LOCAL_MACHINE)
            lRemoteRegionsInfoMap[pPair.second.host].emplace_back(pPair);
    });
  
    return lRemoteRegionsInfoMap;
}

void pmMemoryDirectory2D::CopyOrUpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource)
{
    EXCEPTION_ASSERT(!pDataSource); // This should not happen (see pmHeavyOperationsThread::ServeGeneralMemoryRequest)
    
    if(UpdateReceivedMemoryInternal(pAddressSpace, pAddressSpaceBaseAddr, pOffset, pLength, mAddressSpaceCols, 1))
    {
    #ifdef ENABLE_TASK_PROFILING
        if(pLockingTask)
            pLockingTask->GetTaskProfiler()->RecordProfileEvent(pLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
    #endif
    }
}

void pmMemoryDirectory2D::UpdateReceivedMemory(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, pmTask* pLockingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    if(UpdateReceivedMemoryInternal(pAddressSpace, pAddressSpaceBaseAddr, pOffset, pLength, pStep, pCount))
    {
    #ifdef ENABLE_TASK_PROFILING
        if(pLockingTask)
            pLockingTask->GetTaskProfiler()->RecordProfileEvent(pLockingTask->IsReadOnly(pAddressSpace) ? taskProfiler::INPUT_MEMORY_TRANSFER : taskProfiler::OUTPUT_MEMORY_TRANSFER, false);
    #endif
    }
}
    
// Must be called with WriteLock on address space directory acquired
bool pmMemoryDirectory2D::UpdateReceivedMemoryInternal(pmAddressSpace* pAddressSpace, void* pAddressSpaceBaseAddr, ulong pOffset, ulong pLength, ulong pStep, ulong pCount)
{
    boost_box_type lIncomingBox = GetBox(pOffset, pLength, pStep, pCount);
    
    bool lTransferCommandComplete = false;
    
    FINALIZE_RESOURCE_PTR(dOwnershipLock, RW_RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mOwnershipLock, WriteLock(), Unlock());

    std::vector<std::pair<boost_box_type, regionFetchData2D>> lOverlappingBoxes;
    mInFlightRTree.query(boost::geometry::index::intersects(lIncomingBox), std::back_inserter(lOverlappingBoxes));
    
    EXCEPTION_ASSERT(lOverlappingBoxes.size() == 1);
    const boost_box_type& lOverlappingBox = lOverlappingBoxes.begin()->first;
    const regionFetchData2D& lData = lOverlappingBoxes.begin()->second;

    if(AreBoxesEqual(lIncomingBox, lOverlappingBox))
    {
        SetRangeOwnerInternal(vmRangeOwner(PM_LOCAL_MACHINE, pOffset, mMemoryIdentifierStruct), pOffset, pLength, pStep, pCount);

        pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
        lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

        lTransferCommandComplete = true;
        mInFlightRTree.remove(*lOverlappingBoxes.begin());
    }
    else
    {
    #ifdef _DEBUG
        std::vector<boost_box_type> lTestRemainingBoxes;

        GetDifferenceOfBoxes(lIncomingBox, lOverlappingBox, lTestRemainingBoxes);
        EXCEPTION_ASSERT(lTestRemainingBoxes.empty());
    #endif

        regionFetchData2D lNewData = lData;
        lNewData.accumulatedPartialReceivesLength += pLength * pCount;
        lNewData.partialReceiveRecordVector.emplace_back(lIncomingBox);
        
        pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(lOverlappingBox);

        size_t lExpectedLength = lScatteredSubscriptionInfo.size * lScatteredSubscriptionInfo.count;
        
        EXCEPTION_ASSERT(lNewData.accumulatedPartialReceivesLength <= lExpectedLength);
        
        if(lNewData.accumulatedPartialReceivesLength == lExpectedLength)
        {
            SetRangeOwnerInternal(vmRangeOwner(PM_LOCAL_MACHINE, lScatteredSubscriptionInfo.offset, mMemoryIdentifierStruct), lScatteredSubscriptionInfo.offset, lScatteredSubscriptionInfo.size, lScatteredSubscriptionInfo.step, lScatteredSubscriptionInfo.count);

            pmCommandPtr lCommandPtr = std::static_pointer_cast<pmCommand>(lData.receiveCommand);
            lData.receiveCommand->MarkExecutionEnd(pmSuccess, lCommandPtr);

            lTransferCommandComplete = true;
            mInFlightRTree.remove(*lOverlappingBoxes.begin());
        }
        else
        {
            mInFlightRTree.remove(*lOverlappingBoxes.begin());
            mInFlightRTree.insert(std::make_pair(lOverlappingBox, lNewData));
        }
    }

    return lTransferCommandComplete;
}

    
#ifdef _DEBUG
void pmMemoryDirectory2D::PrintOwnerships() const
{
    std::cout << "Host " << pmGetHostId() << " Ownership Dump (" << this << ")" << std::endl;
    
    boost_box_type lBox = GetBox(0, mAddressSpaceCols, mAddressSpaceCols, mAddressSpaceRows);
    
    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        pmScatteredSubscriptionInfo lScatteredSubscriptionInfo = GetReverseBoxMapping(pPair.first);

        std::cout << "Scattered Range (" << lScatteredSubscriptionInfo.offset << ", " << lScatteredSubscriptionInfo.size << ", " << lScatteredSubscriptionInfo.step << ", " << lScatteredSubscriptionInfo.count << ") is owned by host " << (uint)(*(pPair.second.host)) << " (offset " << pPair.second.hostOffset << ")" << std::endl;
    });

    std::cout << std::endl;
}
    
void pmMemoryDirectory2D::SanitizeOwnerships() const
{
    boost_box_type lBox = GetBox(0, mAddressSpaceCols, mAddressSpaceCols, mAddressSpaceRows);
    
    std::vector<std::pair<boost_box_type, vmRangeOwner>> lOverlappingBoxes;
    mOwnershipRTree.query(boost::geometry::index::intersects(lBox), std::back_inserter(lOverlappingBoxes));
    
    for_each(lOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pPair)
    {
        std::vector<std::pair<boost_box_type, vmRangeOwner>> lWronglyOverlappingBoxes;
        mOwnershipRTree.query(boost::geometry::index::intersects(pPair.first), std::back_inserter(lWronglyOverlappingBoxes));
        
        if(lWronglyOverlappingBoxes.size() != 1)
        {
            for_each(lWronglyOverlappingBoxes, [&] (const std::pair<boost_box_type, vmRangeOwner>& pInnerPair)
            {
                if(!AreBoxesEqual(pInnerPair.first, pPair.first))
                {
                    std::cout << "<<< ERROR >>> Host " << pmGetHostId() << " Boxes wrongly overlap. Box 1: ";
                    PrintBox(pPair.first);
                    std::cout << " Box 2: ";
                    PrintBox(pInnerPair.first);
                    std::cout << std::endl;
                }
            });
        }
    });
}
    
void pmMemoryDirectory2D::PrintBox(const boost_box_type& pBox) const
{
    const boost_point_type& lMinCorner = pBox.min_corner();
    const boost_point_type& lMaxCorner = pBox.max_corner();
    
    ulong lXMin = lMinCorner.get<0>();
    ulong lYMin = lMinCorner.get<1>();
    ulong lXMax = lMaxCorner.get<0>();
    ulong lYMax = lMaxCorner.get<1>();

    std::cout << "(" << lXMin << ", " << lYMin << ", " << lXMax << ", " << lYMax << ")";
}
#endif

bool pmMemoryDirectory2D::AreBoxesEqual(const boost_box_type& pBox1, const boost_box_type& pBox2) const
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
    
    return (lXMin1 == lXMin2 && lYMin1 == lYMin2 && lXMax1 == lXMax2 && lYMax1 == lYMax2);
}


/* class pmScatteredSubscriptionFilter */
pmScatteredSubscriptionFilter::pmScatteredSubscriptionFilter(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo)
    : mScatteredSubscriptionInfo(pScatteredSubscriptionInfo)
{}
    
const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& pmScatteredSubscriptionFilter::FilterBlocks(const std::function<void (size_t)>& pRowFunctor)
{
    for(size_t i = 0; i < mScatteredSubscriptionInfo.count; ++i)
        pRowFunctor(i);

    return GetLeftoverBlocks();
}

// AddNextSubRow must be called in increasing y first and then increasing x values (i.e. first one or more times with increasing x for row one, then similarly for row two and so on)
void pmScatteredSubscriptionFilter::AddNextSubRow(ulong pOffset, ulong pLength, vmRangeOwner& pRangeOwner)
{
    ulong lStartCol = (pOffset - (ulong)mScatteredSubscriptionInfo.offset) % (ulong)mScatteredSubscriptionInfo.step;
    
    bool lRangeCombined = false;

    auto lIter = mCurrentBlocks.begin(), lEndIter = mCurrentBlocks.end();
    while(lIter != lEndIter)
    {
        blockData& lData = (*lIter);

        ulong lEndCol1 = lData.startCol + lData.colCount - 1;
        ulong lEndCol2 = lStartCol + pLength - 1;
        
        bool lRemoveCurrentRange = false;
        
        if(!(lEndCol2 < lData.startCol || lStartCol > lEndCol1))    // If the ranges overlap
        {
            // Total overlap and to be fetched from same host and there is no gap between rows
            if(lData.startCol == lStartCol && lData.colCount == pLength && lData.rangeOwner.host == pRangeOwner.host
               && lData.rangeOwner.memIdentifier == pRangeOwner.memIdentifier
               && (lData.rangeOwner.hostOffset + lData.subscriptionInfo.count * lData.subscriptionInfo.step == pRangeOwner.hostOffset)
               && (lData.subscriptionInfo.offset + lData.subscriptionInfo.count * lData.subscriptionInfo.step == pOffset))
            {
                ++lData.subscriptionInfo.count; // Combine with previous range
                lRangeCombined = true;
            }
            else
            {
                lRemoveCurrentRange = true;
            }
        }
        
        if(lRemoveCurrentRange)
        {
            mBlocksToBeFetched[lData.rangeOwner.host].emplace_back(lData.subscriptionInfo, lData.rangeOwner);
            mCurrentBlocks.erase(lIter++);
        }
        else
        {
            ++lIter;
        }
    }
    
    if(!lRangeCombined)
        mCurrentBlocks.emplace_back(lStartCol, pLength, pmScatteredSubscriptionInfo(pOffset, pLength, mScatteredSubscriptionInfo.step, 1), pRangeOwner);
}

const std::map<const pmMachine*, std::vector<std::pair<pmScatteredSubscriptionInfo, vmRangeOwner>>>& pmScatteredSubscriptionFilter::GetLeftoverBlocks()
{
    PromoteCurrentBlocks();
    
    return mBlocksToBeFetched;
}

void pmScatteredSubscriptionFilter::PromoteCurrentBlocks()
{
    for_each(mCurrentBlocks, [&] (const blockData& pData)
    {
        mBlocksToBeFetched[pData.rangeOwner.host].emplace_back(pData.subscriptionInfo, pData.rangeOwner);
    });
    
    mCurrentBlocks.clear();
}


/* class pmScatteredSubscriptionFilterHelper */
pmScatteredSubscriptionFilterHelper::pmScatteredSubscriptionFilterHelper(pmScatteredSubscriptionFilter& pGlobalFilter, void* pBaseAddr, pmMemoryDirectoryLinear& pMemoryDirectoryLinear, bool pUnprotected, const pmMachine* pFilteredMachine /* = PM_LOCAL_MACHINE */)
: mGlobalFilter(pGlobalFilter)
, mMemoryDirectoryLinear(pMemoryDirectoryLinear)
, mMem(reinterpret_cast<ulong>(pBaseAddr))
, mUnprotected(pUnprotected)
, mFilteredMachine(pFilteredMachine)
{}

void pmScatteredSubscriptionFilterHelper::emplace_back(ulong pStartAddr, ulong pLastAddr)
{
    ulong lLength = pLastAddr - pStartAddr + 1;

    pmMemOwnership lOwnerships;
    
    if(mUnprotected)
        mMemoryDirectoryLinear.GetOwnersUnprotected(pStartAddr - mMem, lLength, lOwnerships);
    else
        mMemoryDirectoryLinear.GetOwners(pStartAddr - mMem, lLength, lOwnerships);

    for_each(lOwnerships, [&] (pmMemOwnership::value_type& pPair)
    {
        vmRangeOwner& lRangeOwner = pPair.second.second;

        if(lRangeOwner.host != mFilteredMachine)
            mGlobalFilter.AddNextSubRow(pPair.first, pPair.second.first, lRangeOwner);
    });
}

}

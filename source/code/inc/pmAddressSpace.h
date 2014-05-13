
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

#ifndef __PM_MEM_SECTION__
#define __PM_MEM_SECTION__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"

#include <vector>
#include <map>
#include <memory>
#include <list>

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

class pmMachine;
class pmAddressSpace;

typedef class pmUserMemHandle
{
public:
    pmUserMemHandle(pmAddressSpace* pAddressSpace);
    ~pmUserMemHandle();
    
    void Reset(pmAddressSpace* pAddressSpace);
    
    pmAddressSpace* GetAddressSpace();
    
private:
    pmAddressSpace* mAddressSpace;
} pmUserMemHandle;
    
typedef std::map<const pmMachine*, std::shared_ptr<std::vector<communicator::ownershipChangeStruct> > > pmOwnershipTransferMap;

/**
 * \brief Encapsulation of task memory
 */

class pmAddressSpace : public pmBase
{
    typedef std::map<std::pair<const pmMachine*, ulong>, pmAddressSpace*> addressSpaceMapType;
    typedef std::map<void*, pmAddressSpace*> augmentaryAddressSpaceMapType;
    
    friend void FetchCallback(const pmCommandPtr& pCommand);
    
	public:
		typedef struct vmRangeOwner
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
		} vmRangeOwner;
    
        typedef struct pmMemTransferData
        {
            vmRangeOwner rangeOwner;
            ulong offset;
            ulong length;
            
            pmMemTransferData(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength)
            : rangeOwner(pRangeOwner)
            , offset(pOffset)
            , length(pLength)
            {}
        } pmMemTransferData;

		typedef std::map<size_t, std::pair<size_t, vmRangeOwner> > pmMemOwnership;

        void Do1DBlockRowDistribution(uint pBlockDim, uint pMatrixWidth, uint pMatrixHeight, bool pRandomize);
        void Do1DBlockColDistribution(uint pBlockDim, uint pMatrixWidth, uint pMatrixHeight, bool pRandomize);
        void Do2DBlockDistribution(uint pBlockDim, uint pMatrixWidth, uint pMatrixHeight, bool pRandomize);

        static pmAddressSpace* CreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner = GetNextGenerationNumber());
        static pmAddressSpace* CheckAndCreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);

        ~pmAddressSpace();
    
		void* GetMem() const;
        size_t GetLength() const;
        size_t GetAllocatedLength() const;
    
        void DisposeMemory();
    
        void AcquireOwnershipImmediate(ulong pOffset, ulong pLength);
        void TransferOwnershipPostTaskCompletion(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
		void FlushOwnerships();

        bool IsRegionLocallyOwned(ulong pOffset, ulong pLength);
		void GetOwners(ulong pOffset, ulong pLength, pmAddressSpace::pmMemOwnership& pOwnerships);
        void GetOwnersInternal(pmMemOwnership& pMap, ulong pOffset, ulong pLength, pmAddressSpace::pmMemOwnership& pOwnerships);

        void Fetch(ushort pPriority);
        void FetchAsync(ushort pPriority, pmCommandPtr pCommand);
        void FetchRange(ushort pPriority, ulong pOffset, ulong pLength);
    
        void EnqueueForLock(pmTask* pTask, pmMemType pMemType, const pmMemDistributionInfo& pDistInfo, pmCommandPtr& pCountDownCommand);
        void Unlock(pmTask* pTask);
        pmTask* GetLockingTask();
    
        void ChangeOwnership(std::shared_ptr<std::vector<communicator::ownershipChangeStruct>>& pOwnershipData);
    
        void UserDelete();
        void SetUserMemHandle(pmUserMemHandle* pUserMemHandle);
        pmUserMemHandle* GetUserMemHandle();

        void Update(size_t pOffset, size_t pLength, void* pSrcAddr);

#ifdef SUPPORT_LAZY_MEMORY
        void* GetReadOnlyLazyMemoryMapping();
        uint GetLazyForwardPrefetchPageCount();
#endif
            
        void GetPageAlignedAddresses(size_t& pOffset, size_t& pLength);

#ifdef ENABLE_MEM_PROFILING
        void RecordMemReceive(size_t pReceiveSize);
        void RecordMemTransfer(size_t pTransferSize);
#endif
            
        static pmAddressSpace* FindAddressSpace(const pmMachine* pOwner, ulong pGenerationNumber);
        static pmAddressSpace* FindAddressSpaceContainingLazyAddress(void* pPtr);
        static void DeleteAllLocalAddressSpaces();

        ulong GetGenerationNumber() const;
        const pmMachine* GetMemOwnerHost() const;
    
        const char* GetName();

    private:    
		pmAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);
    
        static ulong GetNextGenerationNumber();
    
        void Init(const pmMachine* pOwner);
        void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
        void SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength, pmMemOwnership& pMap);
        void SendRemoteOwnershipChangeMessages(pmOwnershipTransferMap& pOwnershipTransferMap);
    
        void SetWaitingForOwnershipChange();
        bool IsWaitingForOwnershipChange();
        void TransferOwnershipImmediate(ulong pOffset, ulong pLength, const pmMachine* pNewOwnerHost);
    
        void Lock(pmTask* pTask, pmMemType pMemType, const pmMemDistributionInfo& pDistInfo);
        void FetchCompletionCallback(const pmCommandPtr& pCommand);
    
        void ScanLockQueue();
    
#ifdef _DEBUG
        void CheckMergability(pmMemOwnership::iterator& pRange1, pmMemOwnership::iterator& pRange2);
        void SanitizeOwnerships();
        void PrintOwnerships();
#endif

        std::vector<uint> GetMachinesForDistribution(bool pRandomize);

        const pmMachine* mOwner;
        ulong mGenerationNumberOnOwner;
        pmUserMemHandle* mUserMemHandle;
		size_t mRequestedLength;
		size_t mAllocatedLength;
		size_t mVMPageCount;
        bool mLazy;
        void* mMem;
        void* mReadOnlyLazyMapping;
        std::string mName;
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mWaitingTasksLock;
        std::list<std::pair<std::pair<pmTask*, std::pair<pmMemType, pmMemDistributionInfo>>, pmCommandPtr>> mTasksWaitingForLock;

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mWaitingFetchLock;
        std::map<pmCommandPtr, pmCommandPtr> mCommandsWaitingForFetch;

        static ulong& GetGenerationId();
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetGenerationLock();
    
        static addressSpaceMapType& GetAddressSpaceMap();
        static augmentaryAddressSpaceMapType& GetAugmentaryAddressSpaceMap(); // Maps actual allocated memory regions to pmAddressSpace objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
                
        pmMemOwnership mOwnershipMap;       // offset versus pair of length of region and vmRangeOwner
        pmMemOwnership mOriginalOwnershipMap;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;
        
        std::vector<pmMemTransferData> mOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
        bool mWaitingForOwnershipChange;  // The address space owner may have sent ownership change message that must be processed before allowing any lock on address space
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipTransferLock;

        pmTask* mLockingTask;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTaskLock;
    
        bool mUserDelete;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mDeleteLock;
    
    protected:
#ifdef ENABLE_MEM_PROFILING
        size_t mMemReceived;
        size_t mMemTransferred;
        ulong mMemReceiveEvents;
        ulong mMemTransferEvents;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mMemProfileLock;
#endif
};

} // end namespace pm

#endif

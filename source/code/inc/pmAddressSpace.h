
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
#include "pmMemoryDirectory.h"

#include <vector>
#include <map>
#include <memory>
#include <list>

namespace pm
{

class pmMachine;
class pmAddressSpace;
class pmMemoryDirectory;

class pmUserMemHandle
{
public:
    pmUserMemHandle(pmAddressSpace* pAddressSpace);
    ~pmUserMemHandle();
    
    void Reset(pmAddressSpace* pAddressSpace);
    
    pmAddressSpace* GetAddressSpace();
    
private:
    pmAddressSpace* mAddressSpace;
};

typedef std::map<const pmMachine*, std::shared_ptr<std::vector<communicator::ownershipChangeStruct>>> pmOwnershipTransferMap;
typedef std::map<const pmMachine*, std::shared_ptr<std::vector<communicator::scatteredOwnershipChangeStruct>>> pmScatteredOwnershipTransferMap;

/**
 * \brief Encapsulation of task memory
 */

class pmAddressSpace : public pmBase
{
    typedef std::map<std::pair<const pmMachine*, ulong>, pmAddressSpace*> addressSpaceMapType;
    typedef std::map<void*, pmAddressSpace*> augmentaryAddressSpaceMapType;
    
    friend void FetchCallback(const pmCommandPtr& pCommand);
    
	public:
        static pmAddressSpace* CreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner = GetNextGenerationNumber());
        static pmAddressSpace* CreateAddressSpace(size_t pRows, size_t pCols, const pmMachine* pOwner, ulong pGenerationNumberOnOwner = GetNextGenerationNumber());

        static pmAddressSpace* CheckAndCreateAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);
        static pmAddressSpace* CheckAndCreateAddressSpace(size_t pRows, size_t pCols, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);

        ~pmAddressSpace();
    
		void* GetMem() const;
        size_t GetLength() const;
        size_t GetAllocatedLength() const;
    
        void DisposeMemory();
    
        void TransferOwnershipPostTaskCompletion(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
        void TransferOwnershipPostTaskCompletion(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pSize, ulong pStep, ulong pCount);
        void FlushOwnerships();

        bool IsRegionLocallyOwned(ulong pOffset, ulong pLength);
        void GetOwners(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);
        void GetOwners(ulong pOffset, ulong pSize, ulong pStep, ulong pCount, pmScatteredMemOwnership& pScatteredOwnerships);
        void GetOwnersUnprotected(ulong pOffset, ulong pLength, pmMemOwnership& pOwnerships);

        void Fetch(ushort pPriority);
        void FetchAsync(ushort pPriority, pmCommandPtr pCommand);
        void FetchRange(ushort pPriority, ulong pOffset, ulong pLength);
    
        void EnqueueForLock(pmTask* pTask, pmMemType pMemType, pmCommandPtr& pCountDownCommand);
        void Unlock(pmTask* pTask);
        pmTask* GetLockingTask();
    
        ulong FindLocalDataSizeUnprotected(ulong pOffset, ulong pLength);
        std::set<const pmMachine*> FindRemoteDataSourcesUnprotected(ulong pOffset, ulong pLength);
    
    #ifdef CENTRALIZED_AFFINITY_COMPUTATION
        void FindLocalDataSizeOnMachinesUnprotected(ulong pOffset, ulong pLength, const std::vector<const pmMachine*>& pMachinesVector, ulong* pDataArray, size_t pStepSizeInBytes);
        void FindRemoteDataSourcesOnMachinesUnprotected(ulong pOffset, ulong pLength, const std::vector<const pmMachine*>& pMachinesVector, uint* pDataArray, size_t pStepSizeInBytes);
    #endif
    
        void ChangeOwnership(std::shared_ptr<std::vector<communicator::ownershipChangeStruct>>& pOwnershipData);
        void ChangeOwnership(std::shared_ptr<std::vector<communicator::scatteredOwnershipChangeStruct>>& pScatteredOwnershipData);
    
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
    
#ifdef DUMP_DATA_TRANSFER_FREQUENCY
        void RecordDataTransferFrequency(ulong pOffset, ulong pLength, ulong pStep, ulong pCount);
#endif

        pmScatteredTransferMapType SetupRemoteRegionsForFetching(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo, ulong pPriority, std::set<pmCommandPtr>& pCommandsAlreadyIssuedSet);
        pmLinearTransferVectorType SetupRemoteRegionsForFetching(const pmSubscriptionInfo& pSubscriptionInfo, ulong pPriority, std::vector<pmCommandPtr>& pCommandVector);
        virtual pmRemoteRegionsInfoMapType GetRemoteRegionsInfo(const pmScatteredSubscriptionInfo& pScatteredSubscriptionInfo);

        static pmAddressSpace* FindAddressSpace(const pmMachine* pOwner, ulong pGenerationNumber);
        static pmAddressSpace* FindAddressSpaceContainingLazyAddress(void* pPtr);
        static void DeleteAllLocalAddressSpaces();

        ulong GetGenerationNumber() const;
        const pmMachine* GetMemOwnerHost() const;
    
        const char* GetName();
    
        pmAddressSpaceType GetAddressSpaceType() const;
        size_t GetRows() const;
        size_t GetCols() const;
    
        void CopyOrUpdateReceivedMemory(pmTask* pRequestingTask, ulong pOffset, ulong pLength, std::function<void (char*, ulong)>* pDataSource);
        void UpdateReceivedMemory(pmTask* pRequestingTask, ulong pOffset, ulong pLength, ulong pStep, ulong pCount);

    private:
        pmAddressSpace(size_t pLength, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);
        pmAddressSpace(size_t pRows, size_t pCols, const pmMachine* pOwner, ulong pGenerationNumberOnOwner);
    
        static ulong GetNextGenerationNumber();
    
        void Init();
        void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
        void SetRangeOwner(const vmRangeOwner& pRangeOwner, ulong pOffset, ulong pSize, ulong pStep, ulong pCount);
        void SendRemoteOwnershipChangeMessages(pmOwnershipTransferMap& pOwnershipTransferMap);
        void SendRemoteOwnershipChangeMessages(pmScatteredOwnershipTransferMap& pScatteredOwnershipTransferMap);
    
        void SetWaitingForOwnershipChange();
        bool IsWaitingForOwnershipChange();
        void TransferOwnershipImmediate(ulong pOffset, ulong pLength, const pmMachine* pNewOwnerHost);
        void TransferOwnershipImmediate(ulong pOffset, ulong pSize, ulong pStep, ulong pCount, const pmMachine* pNewOwnerHost);
    
        void Lock(pmTask* pTask, pmMemType pMemType);
        void FetchCompletionCallback(const pmCommandPtr& pCommand);
    
        void ScanLockQueue();

        std::vector<uint> GetMachinesForDistribution(bool pRandomize);

        const pmMachine* mOwner;
        ulong mGenerationNumberOnOwner;
        pmUserMemHandle* mUserMemHandle;
		size_t mRequestedLength;
		size_t mAllocatedLength;
        size_t mRequestedRows;
        size_t mRequestedCols;
		size_t mVMPageCount;
        pmAddressSpaceType mAddressSpaceType;
        bool mLazy;
        void* mMem;
        void* mReadOnlyLazyMapping;
        std::string mName;
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mWaitingTasksLock;
        std::list<std::pair<std::pair<pmTask*, pmMemType>, pmCommandPtr>> mTasksWaitingForLock;

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mWaitingFetchLock;
        std::map<pmCommandPtr, pmCommandPtr> mCommandsWaitingForFetch;

        static ulong& GetGenerationId();
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetGenerationLock();
    
        static addressSpaceMapType& GetAddressSpaceMap();
        static augmentaryAddressSpaceMapType& GetAugmentaryAddressSpaceMap(); // Maps actual allocated memory regions to pmAddressSpace objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
    
        std::unique_ptr<pmMemoryDirectory> mDirectoryPtr, mOriginalDirectoryPtr;

        std::vector<pmMemTransferData> mOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
        std::vector<pmScatteredMemTransferData> mScatteredOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
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

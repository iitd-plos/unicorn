
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

#ifndef __PM_MEM_SECTION__
#define __PM_MEM_SECTION__

#include "pmBase.h"
#include "pmCommand.h"
#include "pmResourceLock.h"

#include <vector>
#include <map>
#include <tr1/memory>

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
class pmMemSection;

typedef class pmUserMemHandle
{
public:
    pmUserMemHandle(pmMemSection* pMemSection);
    ~pmUserMemHandle();
    
    void Reset(pmMemSection* pMemSection);
    
    pmMemSection* GetMemSection();
    
private:
    pmMemSection* mMemSection;
} pmUserMemHandle;
    
typedef std::map<pmMachine*, std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> > > pmOwnershipTransferMap;

/**
 * \brief Encapsulation of task memory
 */

class pmMemSection : public pmBase
{
    typedef std::map<std::pair<pmMachine*, ulong>, pmMemSection*> memSectionMapType;
    typedef std::map<void*, pmMemSection*> augmentaryMemSectionMapType;
    
	public:
		typedef struct vmRangeOwner
		{
			pmMachine* host;		// Host where memory page lives
            ulong hostOffset;       // Offset on host (in case of data redistribution offsets at source and destination hosts are different)
            pmCommunicatorCommand::memoryIdentifierStruct memIdentifier;    // a different memory might be holding the required data (e.g. redistribution)
		} vmRangeOwner;
    
        typedef struct pmMemTransferData
        {
            vmRangeOwner rangeOwner;
            ulong offset;
            ulong length;
        } pmMemTransferData;

		typedef std::map<size_t, std::pair<size_t, vmRangeOwner> > pmMemOwnership;

        static pmMemSection* CreateMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner = GetNextGenerationNumber());
        static pmMemSection* CheckAndCreateMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner);

        virtual ~pmMemSection();
    
		void* GetMem();
        size_t GetLength();
        size_t GetAllocatedLength();
    
        void DisposeMemory();
    
        pmStatus TransferOwnershipImmediate(ulong pOffset, ulong pLength, pmMachine* pNewOwnerHost);
        pmStatus AcquireOwnershipImmediate(ulong pOffset, ulong pLength);
        pmStatus TransferOwnershipPostTaskCompletion(vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
		pmStatus FlushOwnerships();

        bool IsRegionLocallyOwned(ulong pOffset, ulong pLength);
		pmStatus GetOwners(ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships);
        pmStatus GetOwnersInternal(pmMemOwnership& pMap, ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships);

        pmStatus Fetch(ushort pPriority);
        pmStatus FetchRange(ushort pPriority, ulong pOffset, ulong pLength);
    
        void Lock(pmTask* pTask, pmMemInfo pMemInfo);
        void Unlock(pmTask* pTask);
        pmTask* GetLockingTask();
    
        void UserDelete();
        void SetUserMemHandle(pmUserMemHandle* pUserMemHandle);
        pmUserMemHandle* GetUserMemHandle();

        pmStatus Update(size_t pOffset, size_t pLength, void* pSrcAddr);
    
        pmMemInfo GetMemInfo();
        bool IsInput() const;
        bool IsOutput() const;
        bool IsWriteOnly() const;
        bool IsReadWrite() const;
        bool IsLazy() const;
        bool IsLazyWriteOnly() const;
        bool IsLazyReadWrite() const;

#ifdef SUPPORT_LAZY_MEMORY
        void* GetReadOnlyLazyMemoryMapping();
        uint GetLazyForwardPrefetchPageCount();
#endif
            
        void GetPageAlignedAddresses(size_t& pOffset, size_t& pLength);

#ifdef ENABLE_MEM_PROFILING
        void RecordMemReceive(size_t pReceiveSize);
        void RecordMemTransfer(size_t pTransferSize);
#endif
            
        static pmMemSection* FindMemSection(pmMachine* pOwner, ulong pGenerationNumber);
        static pmMemSection* FindMemSectionContainingLazyAddress(void* pPtr);
        static void DeleteAllLocalMemSections();

        ulong GetGenerationNumber();
        pmMachine* GetMemOwnerHost();
    
        const char* GetName();

    private:    
		pmMemSection(size_t pLength, pmMachine* pOwner, ulong pGenerationNumberOnOwner);
    
        static ulong GetNextGenerationNumber();
    
        void Init(pmMachine* pOwner);
        pmStatus SetRangeOwner(vmRangeOwner& pRangeOwner, ulong pOffset, ulong pLength);
        pmStatus SetRangeOwnerInternal(vmRangeOwner pRangeOwner, ulong pOffset, ulong pLength, pmMemOwnership& pMap);
        void SendRemoteOwnershipChangeMessages(pmOwnershipTransferMap& pOwnershipTransferMap);
    
#ifdef _DEBUG
        void CheckMergability(pmMemOwnership::iterator& pRange1, pmMemOwnership::iterator& pRange2);
        void SanitizeOwnerships();
        void PrintOwnerships();
#endif
    
        pmMachine* mOwner;
        ulong mGenerationNumberOnOwner;
        pmUserMemHandle* mUserMemHandle;
		size_t mRequestedLength;
		size_t mAllocatedLength;
		size_t mVMPageCount;
        bool mLazy;
        void* mMem;
        void* mReadOnlyLazyMapping;
        std::string mName;

        static ulong& GetGenerationId();
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetGenerationLock();
    
        static memSectionMapType& GetMemSectionMap();
        static augmentaryMemSectionMapType& GetAugmentaryMemSectionMap(); // Maps actual allocated memory regions to pmMemSection objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS& GetResourceLock();
                
        pmMemOwnership mOwnershipMap;       // offset versus pair (of length of region and vmRangeOwner) - dynamic
        pmMemOwnership mOriginalOwnershipMap;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;
        
        std::vector<pmMemTransferData> mOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipTransferLock;

        pmTask* mLockingTask;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTaskLock;
    
        bool mUserDelete;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mDeleteLock;
    
        pmMemInfo mMemInfo;
    
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

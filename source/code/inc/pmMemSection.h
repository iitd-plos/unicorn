
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
#include "pmResourceLock.h"

#include <vector>
#include <map>

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
class pmInputMemSection;
class pmOutputMemSection;
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

/**
 * \brief Encapsulation of task memory
 */

class pmMemSection : public pmBase
{
	public:
		typedef struct vmRangeOwner
		{
			pmMachine* host;		// Host where memory page lives
			ulong hostBaseAddr;		// Actual base addr on host
            ulong hostOffset;       // Offset on host (in case of data redistribution offsets at source and destination hosts are different)
		} vmRangeOwner;
    
        typedef struct pmMemTransferData
        {
            vmRangeOwner rangeOwner;
            ulong offset;
            ulong length;
        } pmMemTransferData;

		typedef std::map<size_t, std::pair<size_t, vmRangeOwner> > pmMemOwnership;

		virtual ~pmMemSection();
		void* GetMem();
		size_t GetLength();
    
		static pmMemSection* FindMemSection(void* pMem);
        static pmMemSection* FindMemSectionContainingAddress(void* pPtr);
    
        void DeleteAssociations();
        void DeleteLocalAssociations();
        void DeleteRemoteAssociations();
    
        void CreateLocalAssociation(pmMemSection* pMemSection);
        void DisposeMemory();
    
        static void SwapMemoryAndOwnerships(pmMemSection* pMemSection1, pmMemSection* pMemSection2);
        static pmInputMemSection* ConvertOutputMemSectionToInputMemSection(pmOutputMemSection* pOutputMemSection);
	
        pmStatus AcquireOwnershipImmediate(ulong pOffset, ulong pLength);

#ifdef SUPPORT_LAZY_MEMORY
        pmStatus AcquireOwnershipLazy(ulong pOffset, ulong pLength);
        void AccessAllMemoryPages(ulong pOffset, ulong pLength);
#endif
    
        pmStatus TransferOwnershipPostTaskCompletion(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, ulong pOffset, ulong pLength);
    
		pmStatus FlushOwnerships();
		pmStatus GetOwners(ulong pOffset, ulong pLength, bool pIsLazyRegisteration, pmMemSection::pmMemOwnership& pOwnerships);

        bool IsLazy();
        pmStatus Fetch(ushort pPriority);
    
        void SetUserMemHandle(pmUserMemHandle* pUserMemHandle);
        pmUserMemHandle* GetUserMemHandle();
    
    protected:
		pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerBaseMemAddr, bool pIsLazy);
        pmMemSection(const pmMemSection& pMemSection);
    
    private:
        pmStatus SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, ulong pOffset, ulong pLength, bool pIsLazyAcquisition);

        pmMachine* mOwner;
        pmUserMemHandle* mUserMemHandle;
		size_t mRequestedLength;
		size_t mAllocatedLength;
		size_t mVMPageCount;
        bool mLazy;
        void* mMem;

        void ResetOwnerships(pmMachine* pOwner, ulong pBaseAddr);
        void ClearOwnerships();
    
        std::vector<pmMemSection*> mLocalAssociations;

        static std::map<void*, pmMemSection*> mMemSectionMap;	// Maps actual allocated memory regions to pmMemSection objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
        
        pmMemOwnership mLazyOwnershipMap;   // offset versus pair (of length of region and vmRangeOwner) - updated to mOwnershipMap at task end
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mLazyOwnershipLock;
        
        pmMemOwnership mOwnershipMap;       // offset versus pair (of length of region and vmRangeOwner) - dynamic
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;
        
        std::vector<pmMemTransferData> mOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipTransferLock;
};

class pmInputMemSection : public pmMemSection
{
	public:
		pmInputMemSection(size_t pLength, bool pIsLazy, pmMachine* pOwner = NULL, ulong pOwnerMemSectionAddr = 0);
        pmInputMemSection(const pmOutputMemSection& pOutputMemSection);
    
		~pmInputMemSection();

	private:
};

class pmOutputMemSection : public pmMemSection
{
	public:
		typedef enum accessType
		{
			READ_WRITE,
			WRITE_ONLY
		} accessType;

		pmOutputMemSection(size_t pLength, accessType pAccess, bool pIsLazy, pmMachine* pOwner = NULL, ulong pOwnerMemSectionAddr = 0);
		~pmOutputMemSection();

		pmStatus Update(size_t pOffset, size_t pLength, void* pSrcAddr);
		accessType GetAccessType();

	private:
		accessType mAccess;
};

} // end namespace pm

#endif

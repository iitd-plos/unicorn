
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
    
    protected:
		pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerBaseMemAddr, bool pIsLazy);
        pmMemSection(const pmMemSection& pMemSection);

        void ResetOwnerships(pmMachine* pOwner, ulong pBaseAddr);
        void ClearOwnerships();

        void* mMem;
    
        static std::map<void*, pmMemSection*> mMemSectionMap;	// Maps actual allocated memory regions to pmMemSection objects
        static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
    
	private:
        pmStatus SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOwnerOffset, ulong pOffset, ulong pLength, bool pIsLazyAcquisition);
    
		size_t mRequestedLength;
		size_t mAllocatedLength;
		size_t mVMPageCount;
        bool mLazy;
        
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

        void SetupPostRedistributionMemSection(bool pAllocateNewMemory);
        pmOutputMemSection* GetPostRedistributionMemSection();
    
	private:
		accessType mAccess;
        pmOutputMemSection* mPostRedistributionMemSection;
};

} // end namespace pm

#endif

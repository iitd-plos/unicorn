
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
	
        pmStatus AcquireOwnershipImmediate(ulong pOffset, ulong pLength);
        pmStatus TransferOwnershipPostTaskCompletion(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength);
    
		pmStatus FlushOwnerships();
		pmStatus GetOwners(ulong pOffset, ulong pLength, pmMemSection::pmMemOwnership& pOwnerships);
    
        pmStatus Fetch(ushort pPriority);

	protected:
		pmMemSection(size_t pLength, pmMachine* pOwner, ulong pOwnerBaseMemAddr);

	private:
        pmStatus SetRangeOwner(pmMachine* pOwner, ulong pOwnerBaseMemAddr, ulong pOffset, ulong pLength);
    
		void* mMem;
		size_t mRequestedLength;
		size_t mAllocatedLength;
		size_t mVMPageCount;

		pmMemOwnership mOwnershipMap;		// offset versus pair (of length of region and vmRangeOwner)
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipLock;

        std::vector<pmMemTransferData> mOwnershipTransferVector;	// memory subscriptions; updated to mOwnershipMap after task finishes
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mOwnershipTransferLock;

		static std::map<void*, pmMemSection*> mMemSectionMap;	// Maps actual allocated memory regions to pmMemSection objects
		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmInputMemSection : public pmMemSection
{
	public:
		pmInputMemSection(size_t pLength, pmMachine* pOwner = NULL, ulong pOwnerMemSectionAddr = 0);
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

		pmOutputMemSection(size_t pLength, accessType pAccess, pmMachine* pOwner = NULL, ulong pOwnerMemSectionAddr = 0);
		~pmOutputMemSection();

		pmStatus Update(size_t pOffset, size_t pLength, void* pSrcAddr);
		accessType GetAccessType();

	private:
		accessType mAccess;
};

} // end namespace pm

#endif

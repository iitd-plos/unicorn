
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

#ifndef __PM_REDISTRIBUTOR__
#define __PM_REDISTRIBUTOR__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmCommunicator.h"

#include <map>
#include <vector>
#include <limits>

namespace pm
{

class pmTask;
class pmMachine;
class pmAddressSpace;

class pmRedistributor : public pmBase
{
	public:
		pmRedistributor(pmTask* pTask, uint pAddressSpaceIndex);

        void RedistributeData(pmExecutionStub* pStub, ulong pSubtaskId, pmSplitInfo* pSplitInfo, ulong pOffset, ulong pLength, uint pOrder);
        void PerformRedistribution(const pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<communicator::redistributionOrderStruct>& pVector);
    
        void SendRedistributionInfo();
    
        void ProcessRedistributionBucket(size_t pBucketIndex);
        void ReceiveGlobalOffsets(const std::vector<ulong>& pGlobalOffsetsVector, ulong pGenerationNumber);
	
	private:
        typedef std::map<std::pair<uint, uint>, size_t> globalRedistributionMapType;
        typedef std::map<ulong, std::vector<size_t> > localRedistributionMapType;

        typedef struct localRedistributionBucket
        {
            localRedistributionMapType::iterator startIter;
            localRedistributionMapType::iterator endIter;
        } localRedistributionBucket;
    
        typedef struct globalRedistributionBucket
        {
            size_t bucketOffset;
            globalRedistributionMapType::iterator startIter;
            globalRedistributionMapType::iterator endIter;
        } globalRedistributionBucket;
    
        void ComputeRedistributionBuckets();
        void CreateRedistributedAddressSpace(ulong pGenerationNumber = std::numeric_limits<ulong>::max());

        void DoParallelRedistribution();
        void DoPostParallelRedistribution();
    
        void ComputeGlobalOffsets();
        void SendGlobalOffsets();

		pmTask* mTask;
        uint mAddressSpaceIndex;
        ulong mSubtasksAccounted;
        pmAddressSpace* mRedistributedAddressSpace;

        std::vector<localRedistributionBucket> mLocalRedistributionBucketsVector;
    
        globalRedistributionMapType mGlobalRedistributionMap;   // Pair of Order no. and Machine id vs. length
        std::map<uint, std::vector<ulong> > mGlobalOffsetsMap;  // Machine Id vs. vector of offsets for each order in the host's mLocalRedistributionMap
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mGlobalRedistributionLock;

        std::vector<communicator::redistributionOrderStruct> mLocalRedistributionVector;
        std::vector<size_t> mLocalRedistributionOffsets;
        localRedistributionMapType mLocalRedistributionMap;   // Order vs. vector of mLocalRedistributionVector indices
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mLocalRedistributionLock;
    
        size_t mPendingBucketsCount;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mPendingBucketsCountLock;
    
        std::vector<ulong> mGlobalOffsetsVector;
        size_t mOrdersPerBucket;
    };

} // end namespace pm

#endif

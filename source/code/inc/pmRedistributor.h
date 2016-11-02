
/*
 * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
 * Copyright (c) 2016 Indian Institute of Technology Delhi
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version. Any redistribution or
 * modification must retain this copyright notice and appropriately
 * highlight the credits.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * More information about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
 *
 * All bug reports and enhancement requests can be sent to the following
 * email addresses -
 * onlinetarun@gmail.com
 * sbansal@cse.iitd.ac.in
 * subodh@cse.iitd.ac.in
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

namespace redistribution
{

typedef std::map<std::pair<uint, uint>, size_t> globalRedistributionMapType;
typedef std::map<ulong, std::vector<std::pair<size_t, ulong>>> localRedistributionMapType;

struct localRedistributionData
{
    std::vector<communicator::redistributionOrderStruct> mLocalRedistributionVector;
    localRedistributionMapType mLocalRedistributionMap;   // Order vs. vector of offset/length pairs
};

}

class pmRedistributor : public pmBase
{
	public:
		pmRedistributor(pmTask* pTask, uint pAddressSpaceIndex);

        void PerformRedistribution(const pmMachine* pHost, ulong pSubtasksAccounted, const std::vector<communicator::redistributionOrderStruct>& pVector);
    
        uint GetAddressSpaceIndex() const;
        void SendRedistributionInfo();
    
        void ProcessRedistributionBucket(size_t pBucketIndex);
        void ReceiveGlobalOffsets(const std::vector<ulong>& pGlobalOffsetsVector, ulong pGenerationNumber);
    
        pmRedistributionMetadata* GetRedistributionMetadata(ulong* pCount);
	
	private:
        typedef struct localRedistributionBucket
        {
            redistribution::localRedistributionMapType::iterator startIter;
            redistribution::localRedistributionMapType::iterator endIter;
        } localRedistributionBucket;
    
        typedef struct globalRedistributionBucket
        {
            size_t bucketOffset;
            redistribution::globalRedistributionMapType::iterator startIter;
            redistribution::globalRedistributionMapType::iterator endIter;
        } globalRedistributionBucket;
    
        void BuildRedistributionData();

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
    
        redistribution::globalRedistributionMapType mGlobalRedistributionMap;   // Pair of Order no. and Machine id vs. length
        std::map<uint, std::vector<ulong> > mGlobalOffsetsMap;  // Machine Id vs. vector of offsets for each order in the host's mLocalRedistributionMap
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mGlobalRedistributionLock;

        redistribution::localRedistributionData mLocalRedistributionData;
    
        size_t mPendingBucketsCount;
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mPendingBucketsCountLock;
    
        std::vector<ulong> mGlobalOffsetsVector;
        size_t mOrdersPerBucket;
    
        std::vector<pmRedistributionMetadata> mRedistributionMetaData;
    };

} // end namespace pm

#endif

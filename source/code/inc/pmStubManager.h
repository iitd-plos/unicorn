
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

#ifndef __PM_STUB_MANAGER__
#define __PM_STUB_MANAGER__

#include "pmHardware.h"

#include <vector>

namespace pm
{

class pmExecutionStub;
class pmAddressSpace;

class pmStubManager : public pmBase
{
	public:
		static pmStubManager* GetStubManager();

		size_t GetProcessingElementsCPU();
		size_t GetProcessingElementsGPU();
		size_t GetStubCount();
    
    #ifdef DUMP_EVENT_TIMELINE
        void InitializeEventTimelines();
    #endif

		pmExecutionStub* GetStub(const pmProcessingElement* pDevice) const;
		pmExecutionStub* GetStub(uint pIndex) const;
    
        pmExecutionStub* GetCpuStub(uint pIndex) const;
        pmExecutionStub* GetGpuStub(uint pIndex) const;
    
    #ifdef SUPPORT_CUDA
        size_t GetMaxCpuDevicesPerHostForCpuPlusGpuTasks();
        void PurgeAddressSpaceEntriesFromGpuCaches(const pmAddressSpace* pAddressSpace);
    #endif

        ushort GetCpuNumaDomainsCount() const;
        const std::vector<std::vector<pmExecutionStub*>>& GetCpuNumaDomains() const;
        const std::vector<pmExecutionStub*>& GetCpuNumaDomain(ushort pDomainId) const;
        ushort GetNumaDomainIdForCpuDevice(uint pIndex) const;

        void WaitForAllStubsToFinish();

	private:
		pmStubManager();
		~pmStubManager();

        void GetCpuIdInfo(uint pRegA, uint pRegC, uint& pEAX, uint& pEBX, uint& pECX, uint& pEDX);
    
		void CreateExecutionStubs();
		void DestroyExecutionStubs();
    
        void CreateCpuNumaDomains();

		pmStatus CountAndProbeProcessingElements();

    #ifdef SUPPORT_CUDA
		void FreeGpuResources();
    #endif

		std::vector<pmExecutionStub*> mStubVector;    
        std::vector<std::vector<pmExecutionStub*>> mCpuNumaDomains;
        std::map<pmExecutionStub*, ushort> mCpuNumaDomainsMap;    // stub versus NUMA domain id
		
		size_t mProcessingElementsCPU;
		size_t mProcessingElementsGPU;
		size_t mStubCount;
};

} // end namespace pm

#endif

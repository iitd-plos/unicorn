
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

#ifndef __PM_STUB_MANAGER__
#define __PM_STUB_MANAGER__

#include "pmHardware.h"

#include <vector>

namespace pm
{

class pmExecutionStub;

/**
 * \brief The representation of a parallel task.
 */

class pmStubManager : public pmBase
{
    friend class pmController;
    
	public:
		static pmStubManager* GetStubManager();

		size_t GetProcessingElementsCPU();
		size_t GetProcessingElementsGPU();
		size_t GetStubCount();

		pmStatus FreeGpuResources();
    
    #ifdef DUMP_EVENT_TIMELINE
        void InitializeEventTimelines();
    #endif

		pmExecutionStub* GetStub(pmProcessingElement* pDevice);
		pmExecutionStub* GetStub(uint pIndex);

	private:
		pmStubManager();
		virtual ~pmStubManager();

		pmStatus CreateExecutionStubs();
		pmStatus DestroyExecutionStubs();

		pmStatus CountAndProbeProcessingElements();

		std::vector<pmExecutionStub*> mStubVector;
		
		size_t mProcessingElementsCPU;
		size_t mProcessingElementsGPU;
		size_t mStubCount;

		static pmStubManager* mStubManager;
};

} // end namespace pm

#endif

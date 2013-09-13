
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

#ifndef __PM_SAFE_PRIORITY_QUEUE__
#define __PM_SAFE_PRIORITY_QUEUE__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"

#include <vector>
#include <map>

namespace pm
{

/**
 * \brief An STL based thread safe priority queue implementation
 * Any number of priority levels can be set as long as template argument P allows.
 * Higher the priority number lesser is the actual priority, setting 0 to
 * be the highest priority. STL's priority queue can't be used as it does
 * not provide iterators and deletion/inspection of random elements.
 */

using namespace std;

template<typename T, typename P = ushort>
class pmSafePQ : public pmBase
{
	public:
		typedef bool (*matchFuncPtr)(T& pItem, void* pMatchCriterion);

		pmSafePQ();
		virtual ~pmSafePQ();

		pmStatus InsertItem(T& pItem, P pPriority);
		pmStatus GetTopItem(T& pItem);
    
        pmStatus MarkProcessingFinished();
        pmStatus UnblockSecondaryOperations();

        pmStatus WaitIfMatchingItemBeingProcessed(T& pItem, matchFuncPtr pMatchFunc, void* pMatchCriterion);
		pmStatus DeleteAndGetFirstMatchingItem(P pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem, bool pTemporarilyUnblockSecondaryOperations);
		pmStatus DeleteMatchingItems(P pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion);

		bool IsHighPriorityElementPresent(P pPriority);

		bool IsEmpty();
		uint GetSize();

	private:
		typedef map<P, typename std::vector<T> > priorityQueueType;
		priorityQueueType mQueue;
        bool mIsProcessing;
        bool mSecondaryOperationsBlocked;
    
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mCommandSignalWait;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mSecondaryOperationsWait;
};

} // end namespace pm

#include "../src/pmSafePriorityQueue.cpp"

#endif

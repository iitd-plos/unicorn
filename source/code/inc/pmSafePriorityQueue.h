
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

#ifndef __PM_SAFE_PRIORITY_QUEUE__
#define __PM_SAFE_PRIORITY_QUEUE__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"

#include <list>
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

template<typename T, typename P = ushort>
class pmSafePQ : public pmBase
{
	public:
		typedef bool (*matchFuncPtr)(const T& pItem, const void* pMatchCriterion);

		pmSafePQ();

        void InsertItem(const std::shared_ptr<T>& pItem, P pPriority);
        pmStatus GetTopItem(std::shared_ptr<T>& pItem);

        void MarkProcessingFinished();
    
        void UnblockSecondaryOperations();
        void BlockSecondaryOperations();

        void WaitForCurrentItem();
        void WaitIfMatchingItemBeingProcessed(matchFuncPtr pMatchFunc, void* pMatchCriterion);
        pmStatus DeleteAndGetFirstMatchingItem(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::shared_ptr<T>& pItem, bool pTemporarilyUnblockSecondaryOperations);
        void DeleteAndGetAllMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::vector<std::shared_ptr<T>>& pItems, bool pTemporarilyUnblockSecondaryOperations);
		void DeleteMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion);

		bool IsHighPriorityElementPresent(P pPriority);

		bool IsEmpty();
		uint GetSize();

	private:
        typedef std::map<P, typename std::list<std::shared_ptr<T>>> priorityQueueType;

        priorityQueueType mQueue;
        std::shared_ptr<T> mCurrentItem;
        std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> mCurrentSignalWait;
        bool mSecondaryOperationsBlocked;
    
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mSecondaryOperationsWait;
};

} // end namespace pm

#include "../src/pmSafePriorityQueue.cpp"

#endif

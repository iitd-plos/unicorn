
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

		pmSafePQ(void* pEventNotificationIdentifier);

        void InsertItem(const std::shared_ptr<T>& pItem, P pPriority);
        pmStatus GetTopItem(std::shared_ptr<T>& pItem);

        void MarkProcessingFinished();
    
        void UnblockSecondaryOperations();
        void BlockSecondaryOperations();
    
        void CallWhenSecondaryOperationsUnblocked(const std::function<void ()>& pFunc);

        void WaitForCurrentItem();
        void WaitIfMatchingItemBeingProcessed(matchFuncPtr pMatchFunc, void* pMatchCriterion);
    
        pmStatus DeleteAndGetFirstMatchingItem(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::shared_ptr<T>& pItem, bool pTemporarilyUnblockSecondaryOperations);
        void DeleteAndGetAllMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::vector<std::shared_ptr<T>>& pItems, bool pTemporarilyUnblockSecondaryOperations);
		void DeleteMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion);
        bool HasMatchingItem(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion);

		bool IsHighPriorityElementPresent(P pPriority);

		bool IsEmpty();
		uint GetSize();

	private:
        typedef std::map<P, typename std::list<std::shared_ptr<T>>> priorityQueueType;

        void* mEventNotificationIdentifier;
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

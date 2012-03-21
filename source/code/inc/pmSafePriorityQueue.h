
#ifndef __PM_SAFE_PRIORITY_QUEUE__
#define __PM_SAFE_PRIORITY_QUEUE__

#include "pmBase.h"
#include "pmResourceLock.h"

#include <vector>
#include <map>

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief An STL based thread safe priority queue implementation
 * Any number of priority levels can be set as long as ushort allows.
 * Higher the priority number lesser is the actual priority setting 0 to
 * be the highest priority. STL's priority queue can't be used as it does
 * not provide iterators and deleteion/inspection of random elements.
 */

using namespace std;

template<typename T>
class pmSafePQ : public pmBase
{
	public:
		typedef bool (*matchFuncPtr)(T& pItem, void* pMatchCriterion);

		pmSafePQ<T>();
		virtual ~pmSafePQ<T>();

		pmStatus InsertItem(T& pItem, ushort pPriority);
		pmStatus GetTopItem(T& pItem);
    
        pmStatus MarkProcessingFinished();

        pmStatus WaitIfMatchingItemBeingProcessed(T& pItem, matchFuncPtr pMatchFunc, void* pMatchCriterion);
		pmStatus DeleteAndGetFirstMatchingItem(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem);
		pmStatus DeleteMatchingItems(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion);

		bool IsHighPriorityElementPresent(ushort pPriority);

		bool IsEmpty();
		uint GetSize();

	private:
		typedef map<ushort, typename std::vector<T> > priorityQueueType;
		priorityQueueType mQueue;
        bool mIsProcessing;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mCommandSignalWait;
};

} // end namespace pm

#include "../src/pmSafePriorityQueue.cpp"

#endif

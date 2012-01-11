
#ifndef __PM_SAFE_PRIORITY_QUEUE__
#define __PM_SAFE_PRIORITY_QUEUE__

#include "pmInternalDefinitions.h"
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

		T& DeleteAndGetFirstMatchingItem(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion);
		pmStatus DeleteMatchingItems(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion);

		bool IsHighPriorityElementPresent(ushort pPriority);

		bool IsEmpty();
		uint GetSize();

	private:
		pmStatus LockQueue();
		pmStatus UnlockQueue();

		map<ushort, typename vector<T> > mQueue;

		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

} // end namespace pm

#endif


#ifndef __PM_SAFE_PRIORITY_QUEUE__
#define __PM_SAFE_PRIORITY_QUEUE__

#include "pmInternalDefinitions.h"

#include<vector>
#include<queue>

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief An STL based thread safe priority queue implementation
 * Any number of priority levels can be set as long as ushort allows.
 * Higher the priority number lesser is the actual priority setting 0 to
 * be the highest priority.
 */

using namespace std;

template<typename T>
bool operator< (const pair<ushort, T>& pData1, const pair<ushort, T>& pData2);

template<typename T>
class pmSafePQ
{
	public:
		typedef pair<ushort, T> PQDT;

		pmSafePQ<T>(ushort pPriorityLevels);
		virtual ~pmSafePQ<T>();

		pmStatus InsertItem(T& pItem, ushort pPriority);
		pmStatus GetTopItem(T& pItem);

		bool IsEmpty();
		uint GetSize();

	private:
		pmStatus LockQueue() = 0;
		pmStatus UnlockQueue() = 0;

		ushort mPriorityLevels; // 0 means highest priority
		priority_queue<PQDT, vector<PQDT>, less<typename vector<PQDT>::value_type> > mQueue;
};

template<typename T>
class pmPThreadPQ : public pmSafePQ<T>
{
	public:
		pmPThreadPQ<T>(ushort pPriorityLevels);
		virtual ~pmPThreadPQ<T>();

	private:
		pmStatus LockQueue();
		pmStatus UnlockQueue();

		pthread_mutex_t mMutex;
};

} // end namespace pm

#endif

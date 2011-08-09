
#include "pmSafePriorityQueue.h"

namespace pm
{

/* class pmSafePQ<T> */
template<typename T>
pmSafePQ<T>::pmSafePQ(ushort pPriorityLevels)
{
	mPriorityLevels = pPriorityLevels;
}

template<typename T>
pmSafePQ<T>::~pmSafePQ()
{
}

template<typename T>
pmStatus pmSafePQ<T>::InsertItem(T& pItem, ushort pPriority)
{
	LockQueue();
	mQueue.push(pItem);
	UnlockQueue();

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::GetTopItem(T& pItem)
{
	LockQueue();
	pItem = mQueue.top();
	mQueue.pop();
	UnlockQueue();

	return pmSuccess;
}

template<typename T>
bool pmSafePQ<T>::IsEmpty()
{
	LockQueue();
	bool lVal = mQueue.empty();
	UnlockQueue();

	return lVal;
}

template<typename T>
uint pmSafePQ<T>::GetSize()
{
	LockQueue();
	uint lSize = (uint)(mQueue.size());
	UnlockQueue();

	return lSize;
}

template<typename T>
bool operator< (const pair<ushort, T>& pData1, const pair<ushort, T>& pData2)
{
	ushort lPriority1 = pData1.first;
	ushort lPriority2 = pData2.first;

	// Lesser the number more the priority
	if(lPriority1 < lPriority2)
		return false;

	return true;
}

/* class pmPThreadPQ */
template<typename T>
pmPThreadPQ<T>::pmPThreadPQ(ushort pPriorityLevels) : pmSafePQ<T>(pPriorityLevels)
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_init(&mMutex, NULL), pmThreadFailure, pmThreadFailure::MUTEX_INIT_FAILURE );
}

template<typename T>
pmPThreadPQ<T>::~pmPThreadPQ()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_destroy(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_DESTROY_FAILURE );
}

template<typename T>
pmStatus pmPThreadPQ<T>::LockQueue()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_lock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_LOCK_FAILURE );

	return pmSuccess;
}

template<typename T>
pmStatus pmPThreadPQ<T>::UnlockQueue()
{
	THROW_ON_NON_ZERO_RET_VAL( pthread_mutex_unlock(&mMutex), pmThreadFailure, pmThreadFailure::MUTEX_UNLOCK_FAILURE );

	return pmSuccess;
}

}

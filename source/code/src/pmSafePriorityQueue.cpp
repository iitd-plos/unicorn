
#include "pmSafePriorityQueue.h"

namespace pm
{

/* class pmSafePQ<T> */
template<typename T>
pmSafePQ<T>::pmSafePQ()
{
}

template<typename T>
pmSafePQ<T>::~pmSafePQ()
{
}

template<typename T>
pmStatus pmSafePQ<T>::InsertItem(T& pItem, ushort pPriority)
{
	LockQueue();

	map<ushort, vector<T> >::iterator lIter = mQueue.find(pPriority);
	if(lIter != mQueue.end())
	{
		vector<T>& lVector = mQueue[pPriority];
		lVector.insert(lVector.begin(), pItem);
	}
	else
	{
		vector<T> lVector;
		lVector.push_back(pItem);
		mQueue[pPriority] = lVector;
	}

	UnlockQueue();

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::GetTopItem(T& pItem)
{
	LockQueue();

	map<ushort, typename vector<T> >::iterator lIter = mQueue.begin();
	if(lIter == mQueue.end())
	{
		UnlockQueue();
		throw pmFatalErrorException();
	}

	vector<T>& lVector = lIter->second;
	pItem = lVector.back();
	lVector.pop_back();

	if(lVector.empty())
		mQueue.erase(lIter);

	UnlockQueue();

	return pmSuccess;
}

template<typename T>
bool pmSafePQ<T>::IsHighPriorityElementPresent(ushort pPriority)
{
	LockQueue();
	if(mQueue.empty())
	{
		UnlockQueue();
		return false;
	}

	map<ushort, typename vector<T> >::iterator lIter = mQueue.begin();

	bool lTest = (lIter->first < pPriority);
	UnlockQueue();

	return lTest;
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
pmStatus pmSafePQ<T>::DeleteMatchingItems(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	LockQueue();

	map<ushort, typename vector<T> >::iterator lIter = mQueue.find(pPriority);
	if(lIter == mQueue.end())
	{
		UnlockQueue();
		return true;
	}

	vector<T>& lVector = lIter->second;
	size_t lSize = lVector.size();
	for(size_t i=lSize-1; i>=0; --i)
	{
		if(pMatchFunc(lVector[i], pMatchCriterion))
			lVector.erase(lVector.begin()+i);
	}

	UnlockQueue();

	return pmSuccess;
}

template<typename T>
const T& pmSafePQ<T>::DeleteAndGetFirstMatchingItem(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	LockQueue();

	map<ushort, typename vector<T> >::iterator lIter = mQueue.find(pPriority);
	if(lIter == mQueue.end())
	{
		UnlockQueue();
		return true;
	}

	vector<T>& lVector = lIter->second;
	size_t lSize = lVector.size();
	for(size_t i=lSize-1; i>=0; --i)
	{
		if(pMatchFunc(lVector[i], pMatchCriterion))
		{
			T lElem = lVector[i];
			lVector.erase(lVector.begin()+i);
			
			UnlockQueue();
			return lElem;
		}
	}

	UnlockQueue();

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::LockQueue()
{
	mResourceLock.Lock();

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::UnlockQueue()
{
	mResourceLock.Unlock();

	return pmSuccess;
}

}

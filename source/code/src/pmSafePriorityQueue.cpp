
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

namespace pm
{

/* class pmSafePQ<T> */
template<typename T>
pmSafePQ<T>::pmSafePQ()
{
    mIsProcessing = false;
}

template<typename T>
pmSafePQ<T>::~pmSafePQ()
{
}

template<typename T>
pmStatus pmSafePQ<T>::InsertItem(T& pItem, ushort pPriority)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
	if(lIter != mQueue.end())
	{
		typename std::vector<T>& lVector = mQueue[pPriority];
		lVector.insert(lVector.begin(), pItem);
	}
	else
	{
		typename std::vector<T> lVector;
		lVector.push_back(pItem);
		mQueue[pPriority] = lVector;
	}

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::GetTopItem(T& pItem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.begin();
	if(lIter == mQueue.end())
		return pmOk;

	typename std::vector<T>& lVector = lIter->second;
	pItem = lVector.back();
	lVector.pop_back();

    assert(mIsProcessing == false);
    mIsProcessing = true;
    
	if(lVector.empty())
		mQueue.erase(lIter);

	return pmSuccess;
}
    
template<typename T>
pmStatus pmSafePQ<T>::MarkProcessingFinished()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    assert(mIsProcessing == true);
    mIsProcessing = false;

    mCommandSignalWait.Signal();
    
    return pmSuccess;
}

template<typename T>
bool pmSafePQ<T>::IsHighPriorityElementPresent(ushort pPriority)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mQueue.empty())
		return false;

	typename priorityQueueType::iterator lIter = mQueue.begin();

	return (lIter->first < pPriority);
}

template<typename T>
bool pmSafePQ<T>::IsEmpty()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mQueue.empty();
}

template<typename T>
uint pmSafePQ<T>::GetSize()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return (uint)(mQueue.size());
}

template<typename T>
pmStatus pmSafePQ<T>::WaitIfMatchingItemBeingProcessed(T& pItem, matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            if(!mIsProcessing || !pMatchFunc(pItem, pMatchCriterion))
                return pmSuccess;
        }
    
        mCommandSignalWait.Wait();
    }
    
    return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::DeleteMatchingItems(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
	if(lIter == mQueue.end())
		return pmSuccess;

	typename std::vector<T>& lVector = lIter->second;
	size_t lSize = lVector.size();
	for(long i=lSize-1; i>=0; --i)
	{
		if(pMatchFunc(lVector[i], pMatchCriterion))
			lVector.erase(lVector.begin()+i);
	}
    
    if(lVector.empty())
		mQueue.erase(lIter);

	return pmSuccess;
}

template<typename T>
pmStatus pmSafePQ<T>::DeleteAndGetFirstMatchingItem(ushort pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
	if(lIter == mQueue.end())
		return pmOk;

	typename std::vector<T>& lVector = lIter->second;
	size_t lSize = lVector.size();
	for(long i=lSize-1; i>=0; --i)
	{
		if(pMatchFunc(lVector[i], pMatchCriterion))
		{
			pItem = lVector[i];
			lVector.erase(lVector.begin()+i);
			
            if(lVector.empty())
                mQueue.erase(lIter);
            
			return pmSuccess;
		}
	}

	return pmOk;
}

}


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
template<typename T, typename P>
pmSafePQ<T, P>::pmSafePQ()
    : mIsProcessing(false)
    , mSecondaryOperationsBlocked(false)
    , mResourceLock __LOCK_NAME__("pmSafePQ::mResourceLock")
{
}

template<typename T, typename P>
pmSafePQ<T, P>::~pmSafePQ()
{
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::InsertItem(T& pItem, P pPriority)
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

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::GetTopItem(T& pItem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.begin();
	if(lIter == mQueue.end())
		return pmOk;

#ifdef _DEBUG
    if(mSecondaryOperationsBlocked)
        PMTHROW(pmFatalErrorException());
#endif
    
	typename std::vector<T>& lVector = lIter->second;
	pItem = lVector.back();
	lVector.pop_back();

    assert(mIsProcessing == false);
    mIsProcessing = true;
    
    mSecondaryOperationsBlocked = pItem.BlocksSecondaryOperations();
    
	if(lVector.empty())
		mQueue.erase(lIter);

	return pmSuccess;
}
    
template<typename T, typename P>
pmStatus pmSafePQ<T, P>::MarkProcessingFinished()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    assert(mIsProcessing == true);
    mIsProcessing = false;
    
    if(mSecondaryOperationsBlocked)
    {
        mSecondaryOperationsWait.Signal();
        mSecondaryOperationsBlocked = false;
    }

    mCommandSignalWait.Signal();
    
    return pmSuccess;
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::UnblockSecondaryOperations()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    if(!mSecondaryOperationsBlocked)
        PMTHROW(pmFatalErrorException());
    
    mSecondaryOperationsBlocked = false;
    mSecondaryOperationsWait.Signal();
    
    return pmSuccess;
}    

template<typename T, typename P>
bool pmSafePQ<T, P>::IsHighPriorityElementPresent(P pPriority)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	if(mQueue.empty())
		return false;

	typename priorityQueueType::iterator lIter = mQueue.begin();

	return (lIter->first < pPriority);
}

template<typename T, typename P>
bool pmSafePQ<T, P>::IsEmpty()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mQueue.empty();
}

template<typename T, typename P>
uint pmSafePQ<T, P>::GetSize()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return (uint)(mQueue.size());
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::WaitIfMatchingItemBeingProcessed(T& pItem, matchFuncPtr pMatchFunc, void* pMatchCriterion)
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

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::DeleteMatchingItems(P pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

            if(!mSecondaryOperationsBlocked)
            {
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
        }

        mSecondaryOperationsWait.Wait();
    }

	return pmSuccess;
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::DeleteAndGetFirstMatchingItem(P pPriority, matchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem, bool pTemporarilyUnblockSecondaryOperations)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        
            if(pTemporarilyUnblockSecondaryOperations && !mSecondaryOperationsBlocked)
                PMTHROW(pmFatalErrorException());

            if(!mSecondaryOperationsBlocked || pTemporarilyUnblockSecondaryOperations)
            {
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

        mSecondaryOperationsWait.Wait();
    }

	return pmOk;
}

}

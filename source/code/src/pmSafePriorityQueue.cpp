
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

namespace pm
{

/* class pmSafePQ<T> */
template<typename T, typename P>
pmSafePQ<T, P>::pmSafePQ()
    : mSecondaryOperationsBlocked(false)
    , mResourceLock __LOCK_NAME__("pmSafePQ::mResourceLock")
    , mSecondaryOperationsWait(false)
{
}

template<typename T, typename P>
void pmSafePQ<T, P>::InsertItem(const std::shared_ptr<T>& pItem, P pPriority)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
    if(lIter == mQueue.end())
        lIter = mQueue.emplace(pPriority, std::list<std::shared_ptr<T>>()).first;

    lIter->second.push_front(pItem);
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::GetTopItem(std::shared_ptr<T>& pItem)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	typename priorityQueueType::iterator lIter = mQueue.begin();
	if(lIter == mQueue.end())
		return pmOk;

    DEBUG_EXCEPTION_ASSERT(!mSecondaryOperationsBlocked);
    
	typename std::list<std::shared_ptr<T>>& lInternalList = lIter->second;
	pItem = lInternalList.back();
	lInternalList.pop_back();

    DEBUG_EXCEPTION_ASSERT(!mCurrentItem.get());
    mCurrentItem = pItem;
    
    mSecondaryOperationsBlocked = pItem->BlocksSecondaryOperations();
    
	if(lInternalList.empty())
		mQueue.erase(lIter);

	return pmSuccess;
}
    
template<typename T, typename P>
void pmSafePQ<T, P>::MarkProcessingFinished()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    DEBUG_EXCEPTION_ASSERT(mCurrentItem.get());
    mCurrentItem.reset();
    
    if(mSecondaryOperationsBlocked)
    {
        mSecondaryOperationsWait.Signal();
        mSecondaryOperationsBlocked = false;
    }

    if(mCurrentSignalWait.get())
    {
        mCurrentSignalWait->Signal();
        mCurrentSignalWait.reset();
    }
}

template<typename T, typename P>
void pmSafePQ<T, P>::UnblockSecondaryOperations()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    EXCEPTION_ASSERT(mSecondaryOperationsBlocked);
    
    mSecondaryOperationsBlocked = false;
    mSecondaryOperationsWait.Signal();
}

template<typename T, typename P>
void pmSafePQ<T, P>::BlockSecondaryOperations()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
    
    EXCEPTION_ASSERT(!mSecondaryOperationsBlocked);
    
    mSecondaryOperationsBlocked = true;
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
void pmSafePQ<T, P>::WaitIfMatchingItemBeingProcessed(matchFuncPtr pMatchFunc, void* pMatchCriterion)
{
    while(1)
    {
        std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitPtr;

        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
            
            if(mCurrentItem.get() && pMatchFunc(*mCurrentItem, pMatchCriterion))
            {
                if(!mCurrentSignalWait.get())
                    mCurrentSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
                
                lSignalWaitPtr = mCurrentSignalWait;
            }
            else
            {
                return;
            }
        }
    
        if(lSignalWaitPtr.get())
            lSignalWaitPtr->Wait();
    }
}

template<typename T, typename P>
void pmSafePQ<T, P>::WaitForCurrentItem()
{
    std::shared_ptr<SIGNAL_WAIT_IMPLEMENTATION_CLASS> lSignalWaitPtr;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        if(mCurrentItem.get())
        {
            if(!mCurrentSignalWait.get())
                mCurrentSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
            
            lSignalWaitPtr = mCurrentSignalWait;
        }
        else
        {
            return;
        }
    }

    if(lSignalWaitPtr.get())
        lSignalWaitPtr->Wait();
}

template<typename T, typename P>
void pmSafePQ<T, P>::DeleteMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion)
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
                    return;

                typename std::list<std::shared_ptr<T>>& lInternalList = lIter->second;

                typename std::list<std::shared_ptr<T>>::iterator lListIter = lInternalList.begin();
                while(lListIter != lInternalList.end())
                {
                    if(pMatchFunc(*lListIter->get(), pMatchCriterion))
                        lListIter = lInternalList.erase(lListIter);
                    else
                        ++lListIter;
                }
                
                if(lInternalList.empty())
                    mQueue.erase(lIter);

                return;
            }
        }

        mSecondaryOperationsWait.Wait();
    }
}

template<typename T, typename P>
pmStatus pmSafePQ<T, P>::DeleteAndGetFirstMatchingItem(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::shared_ptr<T>& pItem, bool pTemporarilyUnblockSecondaryOperations)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        
            EXCEPTION_ASSERT(!pTemporarilyUnblockSecondaryOperations || mSecondaryOperationsBlocked);

            if(!mSecondaryOperationsBlocked || pTemporarilyUnblockSecondaryOperations)
            {
                typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
                if(lIter == mQueue.end())
                    return pmOk;

                typename std::list<std::shared_ptr<T>>& lInternalList = lIter->second;
                
                typename std::list<std::shared_ptr<T>>::iterator lListIter = lInternalList.begin();
                while(lListIter != lInternalList.end())
                {
                    if(pMatchFunc(*lListIter->get(), pMatchCriterion))
                    {
                        pItem = *lListIter;
                        lInternalList.erase(lListIter);

                        if(lInternalList.empty())
                            mQueue.erase(lIter);
                        
                        return pmSuccess;
                    }
                    
                    ++lListIter;
                }
            
                return pmOk;
            }
        }

        mSecondaryOperationsWait.Wait();
    }

	return pmOk;
}

template<typename T, typename P>
void pmSafePQ<T, P>::DeleteAndGetAllMatchingItems(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion, std::vector<std::shared_ptr<T>>& pItems, bool pTemporarilyUnblockSecondaryOperations)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        
            EXCEPTION_ASSERT(!pTemporarilyUnblockSecondaryOperations || mSecondaryOperationsBlocked);

            if(!mSecondaryOperationsBlocked || pTemporarilyUnblockSecondaryOperations)
            {
                typename priorityQueueType::iterator lIter = mQueue.find(pPriority);
                if(lIter == mQueue.end())
                    return;

                typename std::list<std::shared_ptr<T>>& lInternalList = lIter->second;
                
                typename std::list<std::shared_ptr<T>>::iterator lListIter = lInternalList.begin();
                while(lListIter != lInternalList.end())
                {
                    if(pMatchFunc(*lListIter->get(), pMatchCriterion))
                    {
                        pItems.emplace_back(std::move(*lListIter));
                        lInternalList.erase(lListIter);

                        if(lInternalList.empty())
                            mQueue.erase(lIter);
                    }
                    
                    ++lListIter;
                }
            }
        }

        mSecondaryOperationsWait.Wait();
    }
}
    
}

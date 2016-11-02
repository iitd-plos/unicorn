
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

namespace pm
{

/* class pmSafePQ<T> */
template<typename T, typename P>
pmSafePQ<T, P>::pmSafePQ(void* pEventNotificationIdentifier)
    : mEventNotificationIdentifier(pEventNotificationIdentifier)
    , mSecondaryOperationsBlocked(false)
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

    pItem->EventNotification(mEventNotificationIdentifier, true);
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

    pItem->EventNotification(mEventNotificationIdentifier, false);
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
void pmSafePQ<T, P>::CallWhenSecondaryOperationsUnblocked(const std::function<void ()>& pFunc)
{
    while(1)
    {
        // Auto lock/unlock scope
        {
            FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

            if(!mSecondaryOperationsBlocked)
            {
                pFunc();
                return;
            }
        }
    
        mSecondaryOperationsWait.Wait();
    }
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
                    {
                        (*lListIter)->EventNotification(mEventNotificationIdentifier, false);
                        lListIter = lInternalList.erase(lListIter);
                    }
                    else
                    {
                        ++lListIter;
                    }
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
bool pmSafePQ<T, P>::HasMatchingItem(P pPriority, matchFuncPtr pMatchFunc, const void* pMatchCriterion)
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
                    return false;

                typename std::list<std::shared_ptr<T>>& lInternalList = lIter->second;

                typename std::list<std::shared_ptr<T>>::iterator lListIter = lInternalList.begin();
                while(lListIter != lInternalList.end())
                {
                    if(pMatchFunc(*lListIter->get(), pMatchCriterion))
                        return true;
                    else
                        ++lListIter;
                }
                
                return false;
            }
        }

        mSecondaryOperationsWait.Wait();
    }

    return false;
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
                        pItem->EventNotification(mEventNotificationIdentifier, false);
                        
                        lInternalList.erase(lListIter++);

                        if(lInternalList.empty())
                            mQueue.erase(lIter);
                        
                        return pmSuccess;
                    }
                    else
                    {
                        ++lListIter;
                    }
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
                        (*lListIter)->EventNotification(mEventNotificationIdentifier, false);
                        pItems.emplace_back(std::move(*lListIter));
                        lInternalList.erase(lListIter++);

                        if(lInternalList.empty())
                            mQueue.erase(lIter);
                    }
                    else
                    {
                        ++lListIter;
                    }
                }
                
                return;
            }
        }

        mSecondaryOperationsWait.Wait();
    }
}
    
}

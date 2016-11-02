
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

#include "pmCommand.h"
#include "pmSignalWait.h"

namespace pm
{

using namespace communicator;
    
/* class pmCommand */
pmCommandPtr pmCommand::CreateSharedPtr(ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback, const void* pUserIdentifier /* = NULL */)
{
    return pmCommandPtr(new pmCommand(pPriority, pType, pCallback, pUserIdentifier));
}

pmStatus pmCommand::GetStatus()
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	return mStatus;
}

void pmCommand::SetStatus(pmStatus pStatus)
{
	FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

	mStatus = pStatus;
}

pmStatus pmCommand::WaitForFinish()
{
    pmSignalWait* lSignalWait = NULL;
    pmStatus lStatus = pmStatusUnavailable;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if((lStatus = mStatus) == pmStatusUnavailable)
        {
            if(!mSignalWait.get_ptr())
                mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
            
            lSignalWait = mSignalWait.get_ptr();
        }
    }

    if(lSignalWait)
    {
        lSignalWait->Wait();
        
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());
        return mStatus;
    }
    
	return lStatus;
}

bool pmCommand::WaitWithTimeOut(ulong pTriggerTime)
{
    pmSignalWait* lSignalWait = NULL;

    // Auto lock/unlock scope
    {
        FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

        if(mStatus == pmStatusUnavailable)
        {
            if(!mSignalWait.get_ptr())
                mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS(true));
            
            lSignalWait = mSignalWait.get_ptr();
        }
    }

    if(lSignalWait)
        return lSignalWait->WaitWithTimeOut(pTriggerTime);
    
	return false;
}
    
bool pmCommand::AddDependentIfPending(pmCommandPtr& pSharedPtr)
{
    DEBUG_EXCEPTION_ASSERT(dynamic_cast<pmAccumulatorCommand*>(pSharedPtr.get()) != NULL);

    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

    if(mStatus == pmStatusUnavailable)
        return mDependentCommands.insert(pSharedPtr).second;
    
    return false;
}

void pmCommand::MarkExecutionStart()
{
	mTimer.Start();
}
    
void pmCommand::SignalDependentCommands()
{
    for_each(mDependentCommands, [] (const pmCommandPtr& pCommandPtr)
    {
        static_cast<pmAccumulatorCommand*>(pCommandPtr.get())->FinishCommand(pCommandPtr);
    });

    mDependentCommands.clear();
}

void pmCommand::MarkExecutionEnd(pmStatus pStatus, const pmCommandPtr& pSharedPtr)
{
	DEBUG_EXCEPTION_ASSERT(pSharedPtr.get() == this);

    mTimer.Stop();

	// Auto Lock/Unlock Scope
	{
		FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mResourceLock, Lock(), Unlock());

		mStatus = pStatus;

		if(mSignalWait.get_ptr())
			mSignalWait->Signal();
	}

	if(mCallback)
		mCallback(pSharedPtr);
    
    SignalDependentCommands();
}

double pmCommand::GetExecutionTimeInSecs() const
{
	return mTimer.GetElapsedTimeInSecs();
}


/* class pmCountDownCommand */
pmCommandPtr pmCountDownCommand::CreateSharedPtr(size_t pCount, ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback, const void* pUserIdentifier /* = NULL */)
{
    return pmCommandPtr(new pmCountDownCommand(pCount, pPriority, pType, pCallback, pUserIdentifier));
}

void pmCountDownCommand::MarkExecutionEnd(pmStatus pStatus, const pmCommandPtr& pSharedPtr)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCountLock, Lock(), Unlock());
    
    DEBUG_EXCEPTION_ASSERT(mCount);
    --mCount;

    if(!mCount)
        pmCommand::MarkExecutionEnd(pStatus, pSharedPtr);
}
    
size_t pmCountDownCommand::GetOutstandingCount()
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCountLock, Lock(), Unlock());
    
    return mCount;
}



/* class pmAccumulatorCommand */
pmCommandPtr pmAccumulatorCommand::CreateSharedPtr(const std::vector<pmCommandPtr>& pVector, pmCommandCompletionCallbackType pCallback /* = NULL */, const void* pUserIdentifier /* = NULL */)
{
    pmCommandPtr lSharedPtr(new pmAccumulatorCommand(pCallback, pUserIdentifier));
    pmAccumulatorCommand* lCommand = (pmAccumulatorCommand*)lSharedPtr.get();

    FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lCommand->mAccumulatorResourceLock, Lock(), Unlock());

    for_each(pVector, [&lCommand, &lSharedPtr] (const pmCommandPtr& pCommandPtr)
    {
        if(pCommandPtr->AddDependentIfPending(lSharedPtr))
            ++lCommand->mCommandCount;
    });

    lCommand->MarkExecutionStart();
    lCommand->CheckFinish(lSharedPtr);

    return lSharedPtr;
}

void pmAccumulatorCommand::FinishCommand(const pmCommandPtr& pSharedPtr)
{
	FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAccumulatorResourceLock, Lock(), Unlock());

    --mCommandCount;
    CheckFinish(pSharedPtr);
}

void pmAccumulatorCommand::ForceComplete(const pmCommandPtr& pSharedPtr)
{
	FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAccumulatorResourceLock, Lock(), Unlock());

    if(mCommandCount)
    {
        this->MarkExecutionEnd(pmSuccess, pSharedPtr);
        mForceCompleted = true;
    }
}

/* This method must be called with mAccumulatorResourceLock acquired */
void pmAccumulatorCommand::CheckFinish(const pmCommandPtr& pSharedPtr)
{
    if(!mCommandCount && !mForceCompleted)
        this->MarkExecutionEnd(pmSuccess, pSharedPtr);
}
    
} // end namespace pm


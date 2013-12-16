
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

#include "pmCommand.h"
#include "pmSignalWait.h"

namespace pm
{

using namespace communicator;
    
/* class pmCommand */
pmCommandPtr pmCommand::CreateSharedPtr(ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback)
{
    return pmCommandPtr(new pmCommand(pPriority, pType, pCallback));
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
                mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());
            
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
                mSignalWait.reset(new SIGNAL_WAIT_IMPLEMENTATION_CLASS());
            
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
    {
        mDependentCommands.push_back(pSharedPtr);
        return true;
    }
    
    return false;
}

void pmCommand::MarkExecutionStart()
{
	mTimer.Start();
}
    
void pmCommand::SignalDependentCommands()
{
    std::vector<pmCommandPtr>::iterator lIter = mDependentCommands.begin(), lEndIter = mDependentCommands.end();
    for(; lIter != lEndIter; ++lIter)
        static_cast<pmAccumulatorCommand*>((*lIter).get())->FinishCommand((*lIter));
    
    mDependentCommands.clear();
}

void pmCommand::MarkExecutionEnd(pmStatus pStatus, pmCommandPtr& pSharedPtr)
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
pmCommandPtr pmCountDownCommand::CreateSharedPtr(size_t pCount, ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback)
{
    return pmCommandPtr(new pmCountDownCommand(pCount, pPriority, pType, pCallback));
}

void pmCountDownCommand::MarkExecutionEnd(pmStatus pStatus, pmCommandPtr& pSharedPtr)
{
    FINALIZE_RESOURCE_PTR(dResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mCountLock, Lock(), Unlock());
    
    DEBUG_EXCEPTION_ASSERT(mCount);
    --mCount;
    
    if(!mCount)
        pmCommand::MarkExecutionEnd(pStatus, pSharedPtr);
}


/* class pmAccumulatorCommand */
pmCommandPtr pmAccumulatorCommand::CreateSharedPtr(const std::vector<pmCommandPtr>& pVector)
{
    pmCommandPtr lSharedPtr(new pmAccumulatorCommand());
    pmAccumulatorCommand* lCommand = (pmAccumulatorCommand*)lSharedPtr.get();

    FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &lCommand->mAccumulatorResourceLock, Lock(), Unlock());

    std::vector<pmCommandPtr>::const_iterator lIter = pVector.begin(), lEndIter = pVector.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if(((*lIter).get())->AddDependentIfPending(lSharedPtr))
            ++lCommand->mCommandCount;
    }

    lCommand->MarkExecutionStart();
    lCommand->CheckFinish(lSharedPtr);

    return lSharedPtr;
}

void pmAccumulatorCommand::FinishCommand(pmCommandPtr& pSharedPtr)
{
	FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAccumulatorResourceLock, Lock(), Unlock());

    --mCommandCount;
    CheckFinish(pSharedPtr);
}

void pmAccumulatorCommand::ForceComplete(pmCommandPtr& pSharedPtr)
{
	FINALIZE_RESOURCE_PTR(dAccumulatorResourceLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mAccumulatorResourceLock, Lock(), Unlock());

    if(mCommandCount)
    {
        this->MarkExecutionEnd(pmSuccess, pSharedPtr);
        mForceCompleted = true;
    }
}

/* This method must be called with mAccumulatorResourceLock acquired */
void pmAccumulatorCommand::CheckFinish(pmCommandPtr& pSharedPtr)
{
    if(!mCommandCount && !mForceCompleted)
        this->MarkExecutionEnd(pmSuccess, pSharedPtr);
}
    
} // end namespace pm


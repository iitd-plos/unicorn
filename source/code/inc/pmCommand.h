
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

#ifndef __PM_COMMAND__
#define __PM_COMMAND__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmTimer.h"
#include "pmSignalWait.h"

#include <set>
#include <memory>

namespace pm
{

class pmTask;
class pmHardware;

class pmCommand;
class pmCommunicatorCommandBase;

template<typename T, typename D>
class pmCommunicatorCommand;
    
class pmAccumulatorCommand;

typedef std::shared_ptr<pmCommand> pmCommandPtr;
typedef void (*pmCommandCompletionCallbackType)(const pmCommandPtr& pCommand);

typedef std::shared_ptr<pmCommunicatorCommandBase> pmCommunicatorCommandPtr;


/**
 * \brief The command class of PMLIB. Serves as an interface between various PMLIB components like pmControllers.
 * This class defines commands that pmController's, pmThread's, etc. on same/differnt machines/clusters use to communicate.
 * This is the only communication mechanism between pmControllers. The pmCommands are opaque objects
 * and the data interpretation is only known to and handled by command listeners. A pmCommand belongs
 * to a particular category of commands e.g. controller command, thread command, etc.
 * Most command objects are passed among threads. So they should be allocated on heap rather
 * than on local thread stacks. Be cautious to keep alive the memory associated with command objects
 * and the encapsulated data until the execution of a command object finishes.
 * Callers can wait for command to finish by calling WaitForFinish() method.
 * The command executors must set the exit status of command via MarkExecutionEnd() method. This also wakes
 * up any awaiting threads.
*/

class pmCommand : public pmBase
{
	public:
        static pmCommandPtr CreateSharedPtr(ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback);
    
        ushort GetPriority() const {return mPriority;}
        ushort GetType() const {return mType;}
        const pmCommandCompletionCallbackType GetCommandCompletionCallback() const {return mCallback;}

        pmStatus GetStatus();

		void SetStatus(pmStatus pStatus);

		/** The following functions must be called by clients for command
         execution time measurement, status reporting and callback calling. */
		void MarkExecutionStart();
        virtual void MarkExecutionEnd(pmStatus pStatus, const pmCommandPtr& pSharedPtr);

		double GetExecutionTimeInSecs() const;

		/** Block the execution of the calling thread until the status
		 * of the command object becomes available. */
		pmStatus WaitForFinish();
        bool WaitWithTimeOut(ulong pTriggerTime);

        bool AddDependentIfPending(pmCommandPtr& pSharedPtr);

    protected:
		pmCommand(ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback)
        : mPriority(pPriority)
        , mType(pType)
        , mCallback(pCallback)
        , mStatus(pmStatusUnavailable)
        , mResourceLock __LOCK_NAME__("pmCommand::mResourceLock")
        {}
    
        pmCommand(const pmCommand& pCommand) = delete;
        pmCommand& operator=(const pmCommand& pCommand) = delete;

        pmCommand(pmCommand&& pCommand) = delete;
        pmCommand& operator=(pmCommand&& pCommand) = delete;

    private:
        void SignalDependentCommands();

        const ushort mPriority;
		const ushort mType;
        const pmCommandCompletionCallbackType mCallback;

		pmStatus mStatus;
		finalize_ptr<pmSignalWait> mSignalWait;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
    
        std::set<pmCommandPtr> mDependentCommands;
		TIMER_IMPLEMENTATION_CLASS mTimer;
};
    
class pmCountDownCommand : public pmCommand
{
	public:
        static pmCommandPtr CreateSharedPtr(size_t pCount, ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback);

        virtual void MarkExecutionEnd(pmStatus pStatus, const pmCommandPtr& pSharedPtr);

    protected:
        pmCountDownCommand(size_t pCount, ushort pPriority, ushort pType, pmCommandCompletionCallbackType pCallback)
        : pmCommand(pPriority, pType, pCallback)
        , mCount(pCount)
        , mCountLock __LOCK_NAME__("pmCountDownCommand::mCountLock")
        {
            EXCEPTION_ASSERT(mCount);
        }

    private:
        size_t mCount;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mCountLock;
};
    
class pmCommunicatorCommandBase : public pmCommand
{
public:
    communicator::communicatorCommandTags GetTag() const {return mTag;}
    const pmHardware* GetDestination() const {return mDestination;}
    communicator::communicatorDataTypes GetDataType() const {return mDataType;}
    
    virtual void* GetData() const = 0;
    virtual ulong GetDataUnits() const = 0;
    virtual ulong GetDataLength() const = 0;
    
    void SetPersistent()
    {
        mPersistent = true;
    }
    
    bool IsPersistent() const
    {
        return mPersistent;
    }
    
protected:
    pmCommunicatorCommandBase(ushort pPriority, ushort pType, communicator::communicatorCommandTags pTag, communicator::communicatorDataTypes pDataType, const pmHardware* pDestination, pmCommandCompletionCallbackType pCallback)
    : pmCommand(pPriority, pType, pCallback)
    , mTag(pTag)
    , mDataType(pDataType)
    , mDestination(pDestination)
    , mPersistent(false)
    {}

private:
    communicator::communicatorCommandTags mTag;
    communicator::communicatorDataTypes mDataType;
    const pmHardware* mDestination;
    bool mPersistent;
};

template<typename T, typename D = deleteDeallocator<T> >
class pmCommunicatorCommand : public pmCommunicatorCommandBase
{
	public:
        static pmCommunicatorCommandPtr CreateSharedPtr(ushort pPriority, communicator::communicatorCommandTypes pType, communicator::communicatorCommandTags pTag, const pmHardware* pDestination, communicator::communicatorDataTypes pDataType, finalize_ptr<T, D>& pData, ulong pDataUnits, pmCommandCompletionCallbackType pCallback = NULL)
        {
            return pmCommunicatorCommandPtr(new pmCommunicatorCommand<T, D>(pPriority, pType, pTag, pDestination, pDataType, pData, pDataUnits, pCallback));
        }

		void* GetData() const
        {
            return mData.get_ptr();
        }
    
		ulong GetDataUnits() const
        {
            return mDataUnits;
        }
    
        ulong GetDataLength() const
        {
            return GetDataUnits() * sizeof(T);
        }

    protected:
		pmCommunicatorCommand(ushort pPriority, communicator::communicatorCommandTypes pType, communicator::communicatorCommandTags pTag, const pmHardware* pDestination, communicator::communicatorDataTypes pDataType, finalize_ptr<T, D>& pData, ulong pDataUnits, pmCommandCompletionCallbackType pCallback)
        : pmCommunicatorCommandBase(pPriority, pType, pTag, pDataType, pDestination, pCallback)
        , mData(std::move(pData))
        , mDataUnits(pDataUnits)
        {}

	private:
		finalize_ptr<T, D> mData;
		ulong mDataUnits;
};

class pmAccumulatorCommand : public pmCommand
{
    public:
		static pmCommandPtr CreateSharedPtr(const std::vector<pmCommandPtr>& pVector);

        void FinishCommand(const pmCommandPtr& pSharedPtr);
        void ForceComplete(const pmCommandPtr& pSharedPtr);

	protected:
		pmAccumulatorCommand()
        : pmCommand(MAX_CONTROL_PRIORITY, 0, NULL)
        , mCommandCount(0)
        , mForceCompleted(false)
        , mAccumulatorResourceLock __LOCK_NAME__("pmAccumulatorCommand::mAccumulatorResourceLock")
        {}

	private:
        void CheckFinish(const pmCommandPtr& pSharedPtr);
    
        uint mCommandCount;
        bool mForceCompleted;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mAccumulatorResourceLock;
};

} // end namespace pm

#endif

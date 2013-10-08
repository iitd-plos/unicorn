
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

#ifndef __PM_NETWORK__
#define __PM_NETWORK__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmSignalWait.h"
#include "pmThread.h"
#include "pmCommand.h"
#include "pmPoolAllocator.h"
#include "pmAllocatorCollection.h"

#include "mpi.h"
#include <map>
#include <vector>

namespace pm
{

class pmCluster;

namespace network
{

typedef struct networkEvent : public pmBasicThreadEvent
{
} networkEvent;

}

/**
 * \brief The base network class of PMLIB.
 * This class implements non-blocking send, receive and broadcast operations
 * and also calls MarkExecutionStart and MarkExecutionEnd for timing and signal
 * operations on command objects. Callers call WaitForFinish on command objects
 * to simulate blocking send or receive operations.
 * This class serves as a factory class to various network implementations.
 * This class has interface to pmCommunicator. The communicator talks in pmCommand
 * objects only while the pmNetwork deals with the actual bytes transferred.
 * Only one instance of pmNetwork class is created on each machine.
*/

class pmNetwork : public pmBase
{
    friend class pmHeavyOperationsThread;

	public:
		virtual void SendNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void ReceiveNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;

        virtual void BroadcastNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void All2AllNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;

        virtual void FreezeReceptionAndFinishCommands() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;

        virtual void RegisterTransferDataType(communicator::communicatorDataTypes pDataType) = 0;
		virtual void UnregisterTransferDataType(communicator::communicatorDataTypes pDataType) = 0;
	
		virtual pmCommunicatorCommandPtr PackData(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual pmCommunicatorCommandPtr UnpackData(void* pPackedData, int pDataLength) = 0;

		virtual void InitializePersistentCommand(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void TerminatePersistentCommand(pmCommunicatorCommandPtr& pCommand) = 0;

		virtual void GlobalBarrier() = 0;

		pmNetwork();
		virtual ~pmNetwork();

    protected:
		void SendComplete(pmCommunicatorCommandPtr& pCommand, pmStatus pStatus);
		void ReceiveComplete(pmCommunicatorCommandPtr& pCommand, pmStatus pStatus);
};
    
struct pmMPIRequestAllocatorTraits
{
    typedef pmPoolAllocator allocator;

    static const bool alignedAllocations = false;
    static const size_t maxAllocationsPerPool = 1024;
    
    struct creator
    {
        std::shared_ptr<pmPoolAllocator> operator()(size_t pSize)
        {
            return std::shared_ptr<pmPoolAllocator>(new pmPoolAllocator(sizeof(MPI_Request), maxAllocationsPerPool, false));
        }
    };
    
    struct destructor
    {
        void operator()(const std::shared_ptr<pmPoolAllocator>& pPtr)
        {
        }
    };
};

class pmMPI : public pmNetwork, public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
{
    public:
		enum pmRequestTag
		{
			PM_MPI_DUMMY_TAG = communicator::MAX_COMMUNICATOR_COMMAND_TAGS,
			PM_MPI_REVERSE_DUMMY_TAG = communicator::MAX_COMMUNICATOR_COMMAND_TAGS
		};

		typedef class pmUnknownLengthReceiveThread : public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
		{
			public:
				pmUnknownLengthReceiveThread(pmMPI* mMPI);
				virtual ~pmUnknownLengthReceiveThread();

				void StopThreadExecution();

                virtual void ThreadSwitchCallback(std::shared_ptr<network::networkEvent>& pCommand);

			private:
				pmStatus SendDummyProbeCancellationMessage(MPI_Request& pRequest);
				pmStatus ReceiveDummyProbeCancellationMessage();

				pmMPI* mMPI;
				bool mThreadTerminationFlag;
                finalize_ptr<pmSignalWait> mSignalWait;
				std::vector<pmCommunicatorCommandPtr> mReceiveCommands;
				RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
		} pmUnknownLengthReceiveThread;

		static pmNetwork* GetNetwork();

		virtual void SendNonBlocking(pmCommunicatorCommandPtr& pCommand);
		virtual void ReceiveNonBlocking(pmCommunicatorCommandPtr& pCommand);

        /* MPI 2 currently does not support non-blocking collective messages */
        virtual void BroadcastNonBlocking(pmCommunicatorCommandPtr& pCommand);
		virtual void All2AllNonBlocking(pmCommunicatorCommandPtr& pCommand);
    
        virtual void FreezeReceptionAndFinishCommands();

		virtual pmCommunicatorCommandPtr PackData(pmCommunicatorCommandPtr& pCommand);
		virtual pmCommunicatorCommandPtr UnpackData(void* pPackedData, int pDataLength);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

		MPI_Datatype GetDataTypeMPI(communicator::communicatorDataTypes pDataType);
		virtual void RegisterTransferDataType(communicator::communicatorDataTypes pDataType);
		virtual void UnregisterTransferDataType(communicator::communicatorDataTypes pDataType);

		virtual void ThreadSwitchCallback(std::shared_ptr<network::networkEvent>& pCommand);

		virtual void InitializePersistentCommand(pmCommunicatorCommandPtr& pCommand);
		virtual void TerminatePersistentCommand(pmCommunicatorCommandPtr& pCommand);

		virtual void GlobalBarrier();

	private:
		pmMPI();
		virtual ~pmMPI();

		void StopThreadExecution();

        bool IsUnknownLengthTag(communicator::communicatorCommandTags pTag);
        void SendNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength);
		void ReceiveNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength);

		virtual MPI_Request* GetPersistentSendRequest(pmCommunicatorCommandPtr& pCommand);
		virtual MPI_Request* GetPersistentRecvRequest(pmCommunicatorCommandPtr& pCommand);

		void SetupDummyRequest();
		void CancelDummyRequest();

		uint mTotalHosts;
		uint mHostId;

		std::map<MPI_Request*, pmCommunicatorCommandPtr> mNonBlockingRequestMap;	// Map of MPI_Request objects and corresponding pmCommunicatorCommand objects
		std::map<pmCommunicatorCommandPtr, size_t> mRequestCountMap;	// Maps MpiCommunicatorCommand object to the number of MPI_Requests issued
		MPI_Request* mDummyReceiveRequest;

        std::map<pmCommunicatorCommandPtr, MPI_Request*> mPersistentSendRequest;
		std::map<pmCommunicatorCommandPtr, MPI_Request*> mPersistentRecvRequest;

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	   // Resource lock on mDummyReceiveRequest, mNonBlockingRequestMap, mResourceCountMap, mPersistentSendRequest, mPersistentRecvRequest

		std::map<communicator::communicatorDataTypes, MPI_Datatype*> mRegisteredDataTypes;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mDataTypesResourceLock;	   // Resource lock on mRegisteredDataTypes

        finalize_ptr<pmSignalWait> mSignalWait;
    
		bool mThreadTerminationFlag;
        bool mPersistentReceptionFreezed;
    
		pmUnknownLengthReceiveThread* mReceiveThread;
    
        SIGNAL_WAIT_IMPLEMENTATION_CLASS mCommandCompletionSignalWait;
    
        pmAllocatorCollection<pmMPIRequestAllocatorTraits> mMPIRequestAllocator;
};

} // end namespace pm

#endif

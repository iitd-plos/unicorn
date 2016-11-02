
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
class pmMachine;

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
	public:
		virtual void SendNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void ReceiveNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;

        virtual void SendMemory(pmCommunicatorCommandPtr& pCommand) = 0;
        virtual void ReceiveMemory(pmCommunicatorCommandPtr& pCommand) = 0;
    
        virtual void BroadcastNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void All2AllNonBlocking(pmCommunicatorCommandPtr& pCommand) = 0;
    
        virtual void SendReduce(pmCommunicatorCommandPtr& pCommand) = 0;
        virtual void ReceiveReduce(pmCommunicatorCommandPtr& pCommand) = 0;

        virtual void FreezeReceptionAndFinishCommands() = 0;

		virtual uint GetTotalHostCount() = 0;
		virtual uint GetHostId() = 0;

        virtual void RegisterTransferDataType(communicator::communicatorDataTypes pDataType) = 0;
		virtual void UnregisterTransferDataType(communicator::communicatorDataTypes pDataType) = 0;
	
		virtual pmCommunicatorCommandPtr PackData(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual pmCommunicatorCommandPtr UnpackData(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pDataLength) = 0;

		virtual void InitializePersistentCommand(pmCommunicatorCommandPtr& pCommand) = 0;
		virtual void TerminatePersistentCommand(pmCommunicatorCommandPtr& pCommand) = 0;

		virtual void GlobalBarrier() = 0;
        virtual void StartReceiving() = 0;
    
        virtual bool IsImplicitlyReducible(pmTask* pTask) const = 0;

		pmNetwork();
		virtual ~pmNetwork();
};
    
class pmMPI : public pmNetwork, public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
{
    public:
		enum pmRequestTag
		{
			PM_MPI_DUMMY_TAG = communicator::MAX_COMMUNICATOR_COMMAND_TAGS,
			PM_MPI_REVERSE_DUMMY_TAG = communicator::MAX_COMMUNICATOR_COMMAND_TAGS
		};
    
        class pmDynamicMpiTagProducer
        {
            public:
                pmDynamicMpiTagProducer();
                int GetNextTag(const pmMachine* pMachine);
            
            private:
                RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
                std::map<const pmMachine*, int> mMachineVersusNextTagMap;
        };

		class pmUnknownLengthReceiveThread : public THREADING_IMPLEMENTATION_CLASS<network::networkEvent>
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
		};
    
        class pmMpiSubArrayManager
        {
            public:
                pmMpiSubArrayManager(int pMatrixRows, int pMatrixCols, int pSubArrayRows, int pSubArrayCols);
                ~pmMpiSubArrayManager();
            
                MPI_Datatype GetMpiType() const;
            
            private:
                MPI_Datatype mSubArrayType;
        };
    
        class pmMpiCommWrapper
        {
            public:
                pmMpiCommWrapper(uint pFirstMachineGlobalRank, uint pSecondMachineGlobalRank);
                ~pmMpiCommWrapper();
            
                MPI_Comm GetCommunicator() const;
            
            private:
                MPI_Group mGroup;
                MPI_Comm mCommunicator;
        };

		static pmNetwork* GetNetwork();

		virtual void SendNonBlocking(pmCommunicatorCommandPtr& pCommand);
		virtual void ReceiveNonBlocking(pmCommunicatorCommandPtr& pCommand);

        virtual void SendMemory(pmCommunicatorCommandPtr& pCommand);
        virtual void ReceiveMemory(pmCommunicatorCommandPtr& pCommand);

        /* MPI 2 currently does not support non-blocking collective messages */
        virtual void BroadcastNonBlocking(pmCommunicatorCommandPtr& pCommand);
		virtual void All2AllNonBlocking(pmCommunicatorCommandPtr& pCommand);
    
        virtual void SendReduce(pmCommunicatorCommandPtr& pCommand);
        virtual void ReceiveReduce(pmCommunicatorCommandPtr& pCommand);

        virtual void FreezeReceptionAndFinishCommands();

		virtual pmCommunicatorCommandPtr PackData(pmCommunicatorCommandPtr& pCommand);
		virtual pmCommunicatorCommandPtr UnpackData(finalize_ptr<char, deleteArrayDeallocator<char>>&& pPackedData, int pDataLength);

		virtual uint GetTotalHostCount() {return mTotalHosts;}
		virtual uint GetHostId() {return mHostId;}

		MPI_Datatype GetDataTypeMPI(communicator::communicatorDataTypes pDataType);
		virtual void RegisterTransferDataType(communicator::communicatorDataTypes pDataType);
		virtual void UnregisterTransferDataType(communicator::communicatorDataTypes pDataType);

		virtual void ThreadSwitchCallback(std::shared_ptr<network::networkEvent>& pCommand);

		virtual void InitializePersistentCommand(pmCommunicatorCommandPtr& pCommand);
		virtual void TerminatePersistentCommand(pmCommunicatorCommandPtr& pCommand);
    
        virtual bool IsImplicitlyReducible(pmTask* pTask) const;

		virtual void GlobalBarrier();
        virtual void StartReceiving();

	private:
		pmMPI();
		virtual ~pmMPI();

		void StopThreadExecution();

        bool IsUnknownLengthTag(communicator::communicatorCommandTags pTag);

        void SendNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength, MPI_Datatype pDynamicDataType = MPI_DATATYPE_NULL);
        void ReceiveNonBlockingInternal(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength, MPI_Datatype pDynamicDataType = MPI_DATATYPE_NULL);
        void ReceiveNonBlockingInternalForMemoryReceive(pmCommunicatorCommandPtr& pCommand, void* pData, int pLength, MPI_Datatype pDynamicDataType);

		MPI_Request GetPersistentSendRequest(pmCommunicatorCommandPtr& pCommand);
		MPI_Request GetPersistentRecvRequest(pmCommunicatorCommandPtr& pCommand);

        MPI_Request GetPersistentSendRequestInternal(pmCommunicatorCommandPtr& pCommand);
        MPI_Request GetPersistentRecvRequestInternal(pmCommunicatorCommandPtr& pCommand);
    
        MPI_Datatype GetReducibleDataTypeAndSize(pmTask* pTask, size_t& pSize) const;
        MPI_Op GetReductionMpiOperation(pmTask* pTask) const;

        void SetupDummyRequest();
		void CancelDummyRequest();

        void CommandComplete(pmCommunicatorCommandPtr& pCommand, pmStatus pStatus);
    
        void CleanupFinishedSendRequests(bool pTerminating);

		uint mTotalHosts;
		uint mHostId;

        std::list<std::pair<MPI_Request, pmCommunicatorCommandPtr>> mOngoingSendRequests;  // This list exists only to free memory associated with buffers in commandPtr after the send operations complete
		std::map<MPI_Request, pmCommunicatorCommandPtr> mNonBlockingRequestMap;	// Map of MPI_Request objects and corresponding pmCommunicatorCommand objects
        std::map<pmCommunicatorCommandPtr, size_t> mRequestCountMap;	// Maps MpiCommunicatorCommand object to the number of MPI_Requests issued

        bool mDummyRequestInitiated;
        MPI_Request mPersistentDummyRecvRequest;

        std::map<pmCommunicatorCommandPtr, MPI_Request> mPersistentSendRequest;
        std::map<pmCommunicatorCommandPtr, MPI_Request> mPersistentRecvRequest;

        RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;	   // Resource lock on mOngoingSendRequests, mDummyRequestInitiated, mPersistentDummyRecvRequest, mNonBlockingRequestMap, mResourceCountMap, mPersistentSendRequest, mPersistentRecvRequest

		std::map<communicator::communicatorDataTypes, MPI_Datatype> mRegisteredDataTypes;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mDataTypesResourceLock;	   // Resource lock on mRegisteredDataTypes

        finalize_ptr<pmSignalWait> mSignalWait;
    
		bool mThreadTerminationFlag;
    
		pmUnknownLengthReceiveThread* mReceiveThread;
        pmDynamicMpiTagProducer mDynamicMpiTagProducer;

    #ifdef PRE_CREATE_SUB_COMMUNICATORS
        std::map<std::pair<uint, uint>, std::shared_ptr<pmMpiCommWrapper>> mSubCommunicators;
    #endif
};

} // end namespace pm

#endif

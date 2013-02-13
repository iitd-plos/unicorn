
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

#include <tr1/memory>	// For std::tr1

namespace pm
{

class pmTask;
class pmSignalWait;
class pmLocalTask;
class pmHardware;
class pmMachine;
class pmExecutionStub;
class pmMemSection;

class pmCommand;
typedef std::tr1::shared_ptr<pmCommand> pmCommandPtr;

class pmAccumulatorCommand;
typedef std::tr1::shared_ptr<pmAccumulatorCommand> pmAccumulatorCommandPtr;

typedef pmStatus (*pmCommandCompletionCallback)(pmCommandPtr pCommand);

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
		virtual bool IsValid() = 0;

		virtual ushort GetType();
		virtual void* GetData();
		virtual ulong GetDataLength();
		virtual pmStatus GetStatus();
		virtual ushort GetPriority();
		virtual pmCommandCompletionCallback GetCommandCompletionCallback();

		virtual pmStatus SetData(void* pCommandData, ulong pDataLength);
		virtual pmStatus SetStatus(pmStatus pStatus);
		virtual pmStatus SetCommandCompletionCallback(pmCommandCompletionCallback pCallback);

		/**
		 * The following functions must be called by clients for
		 * command execution time measurement and status reporting
		 * and callback calling
		*/
		virtual pmStatus MarkExecutionStart();
		virtual pmStatus MarkExecutionEnd(pmStatus pStatus, pmCommandPtr pSharedPtr);

		double GetExecutionTimeInSecs();

		/**
		 * Block the execution of the calling thread until the status
		 * of the command object becomes available.
		*/
		pmStatus WaitForFinish();
        bool WaitWithTimeOut(ulong pTriggerTime);

        bool AddDependentIfPending(pmAccumulatorCommandPtr pSharedPtr);

	protected:
		pmCommand(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0, pmCommandCompletionCallback pCallback = NULL);
        virtual ~pmCommand();
    
		ushort mCommandType;
		void* mCommandData;
		size_t mDataLength;
		pmCommandCompletionCallback mCallback;
		pmStatus mStatus;
		finalize_ptr<pmSignalWait> mSignalWait;
		ushort mPriority;
    
    private:
        void SignalDependentCommands();

        std::vector<pmAccumulatorCommandPtr> mDependentCommands;
		TIMER_IMPLEMENTATION_CLASS mTimer;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmCommunicatorCommand;
typedef std::tr1::shared_ptr<pmCommunicatorCommand> pmCommunicatorCommandPtr;

class pmCommunicatorCommand : public pmCommand
{
	public:
		typedef struct machinePool
		{
			uint cpuCores;
			uint gpuCards;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 2
			} fieldCount;

		} machinePool;

		typedef struct devicePool
		{
			char name[MAX_NAME_STR_LEN];
			char description[MAX_DESC_STR_LEN];

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 2
			} fieldCount;

		} devicePool;

		typedef struct remoteTaskAssignStruct
		{
			uint taskConfLength;
			ulong taskId;
			ulong inputMemLength;
			ulong outputMemLength;
			ushort inputMemInfo;    // enum pmMemInfo
			ushort outputMemInfo;   // enum pmMemInfo
			ulong subtaskCount;
			char callbackKey[MAX_CB_KEY_LEN];
			uint assignedDeviceCount;
			uint originatingHost;
            ulong sequenceNumber;   // Sequence number of task on originating host
			ushort priority;
			ushort schedModel;
			ulong inputMemGenerationNumber;
			ulong outputMemGenerationNumber;
            ushort flags;           // LSB - multiAssignEnabled

			remoteTaskAssignStruct();
			remoteTaskAssignStruct(pmLocalTask* pLocalTask);

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 16
			} fieldCount;

		} remoteTaskAssignStruct;

		typedef struct dataPtr
		{
			void* ptr;
			uint length;
		} dataPtr;

		typedef struct remoteTaskAssignPacked
		{
			remoteTaskAssignPacked(pmLocalTask* pLocalTask = NULL);
			~remoteTaskAssignPacked();

			remoteTaskAssignStruct taskStruct;
			dataPtr taskConf;
			dataPtr devices;
		} remoteTaskAssignPacked;
    
        typedef enum subtaskAssignmentType
        {
            SUBTASK_ASSIGNMENT_REGULAR,
            RANGE_NEGOTIATION,
            SUBTASK_ASSIGNMENT_RANGE_NEGOTIATED
        } subtaskAssignmentType;

		typedef struct remoteSubtaskAssignStruct
		{
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			ulong startSubtask;
			ulong endSubtask;
			uint originatingHost;
			uint targetDeviceGlobalIndex;
            uint originalAllotteeGlobalIndex;
            ushort assignmentType;  // enum subtaskAssignmentType

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 7
			} fieldCount;

		} remoteSubtaskAssignStruct;
    
        typedef struct ownershipDataStruct
        {
            ulong offset;
            ulong length;
        
            typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 2
			} fieldCount;            

        } ownershipDataStruct;

		typedef struct sendAcknowledgementStruct
		{
			uint sourceDeviceGlobalIndex;
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			ulong startSubtask;
			ulong endSubtask;
			uint execStatus;
            uint originalAllotteeGlobalIndex;
            uint ownershipDataElements;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 8
			} fieldCount;

		} sendAcknowledgementStruct;
    
        typedef struct sendAcknowledgementPacked
        {
            sendAcknowledgementPacked();
            sendAcknowledgementPacked(pmProcessingElement* pSourceDevice, pmSubtaskRange& pRange, ownershipDataStruct* pOwnershipData, uint pCount, pmStatus pExecStatus);
            ~sendAcknowledgementPacked();
        
            sendAcknowledgementStruct ackStruct;
            ownershipDataStruct* ownershipData;
        } sendAcknowledgementPacked;

		typedef enum taskEvents
		{
			TASK_FINISH_EVENT,
            TASK_COMPLETE_EVENT,
			TASK_CANCEL_EVENT
		} taskEvents;

		typedef struct taskEventStruct
		{
			uint taskEvent;			// Map to enum taskEvents
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 3
			} fieldCount;

		} taskEventStruct;

		typedef struct stealRequestStruct
		{
			uint stealingDeviceGlobalIndex;
			uint targetDeviceGlobalIndex;
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			double stealingDeviceExecutionRate;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 5
			} fieldCount;

		} stealRequestStruct;

		typedef enum stealResponseType
		{
			STEAL_SUCCESS_RESPONSE,
			STEAL_FAILURE_RESPONSE
		} stealResponseType;

		typedef struct stealResponseStruct
		{
			uint stealingDeviceGlobalIndex;
			uint targetDeviceGlobalIndex;
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			ushort success;			// enum stealResponseType
			ulong startSubtask;
			ulong endSubtask;
            uint originalAllotteeGlobalIndex;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 8
			} fieldCount;

		} stealResponseStruct;

        typedef struct ownershipChangeStruct
        {
            ulong offset;
            ulong length;
            uint newOwnerHost;
        
            typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 3
			} fieldCount;            

        } ownershipChangeStruct;

        typedef struct memoryIdentifierStruct
        {
            uint memOwnerHost;
            ulong generationNumber;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 2
			} fieldCount;
        
        } memoryIdentifierStruct;

        typedef struct ownershipTransferPacked
        {
            ownershipTransferPacked();
            ownershipTransferPacked(pmMemSection* pMemSection, std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> >& pChangeData);
            ~ownershipTransferPacked();
        
            memoryIdentifierStruct memIdentifier;
            std::tr1::shared_ptr<std::vector<pmCommunicatorCommand::ownershipChangeStruct> > transferData;
        } ownershipTransferPacked;

        typedef struct memoryTransferRequest
		{
            memoryIdentifierStruct sourceMemIdentifier;
            memoryIdentifierStruct destMemIdentifier;
            ulong receiverOffset;
			ulong offset;
			ulong length;
			uint destHost;			// Host that will receive the memory (generally same as the requesting host)
            ushort isForwarded;     // Signifies a forwarded memory request. Transfer is made directly from owner host to requesting host.

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 7
			} fieldCount;

		} memoryTransferRequest;
    
		typedef struct subtaskReduceStruct
		{
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			ulong subtaskId;
			ulong subtaskMemLength;
			ulong subscriptionOffset;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 5
			} fieldCount;

		} subtaskReduceStruct;

		typedef struct subtaskReducePacked
		{
			subtaskReducePacked();
			subtaskReducePacked(pmExecutionStub* pReducingStub, pmTask* pTask, ulong pSubtaskId);
			~subtaskReducePacked();

			subtaskReduceStruct reduceStruct;
			dataPtr subtaskMem;
		} subtaskReducePacked;

		typedef struct memoryReceiveStruct
		{
            uint memOwnerHost;
			ulong generationNumber;
			ulong offset;
			ulong length;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 4
			} fieldCount;

		} memoryReceiveStruct;

		typedef struct memoryReceivePacked
		{
			memoryReceivePacked();
			memoryReceivePacked(uint pMemOwnerHost, ulong pGenerationNumber, ulong pOffset, ulong pLength, void* pMemPtr);
			~memoryReceivePacked();

			memoryReceiveStruct receiveStruct;
			dataPtr mem;
		} memoryReceivePacked;

		typedef struct hostFinalizationStruct
		{
			ushort terminate;   // firstly all machines send to master with terminate false; then master sends to all machines with terminate true

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 1
			} fieldCount;
		    
		} hostFinalizationStruct;
    
        typedef struct redistributionOrderStruct
        {
            uint order;
            ulong offset;
            ulong length;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 3
			} fieldCount;            
            
        } redistributionOrderStruct;

        typedef struct dataRedistributionStruct
        {
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
            uint remoteHost;
			ulong subtasksAccounted;
			uint orderDataCount;
            
			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 5
			} fieldCount;            
            
        } dataRedistributionStruct;

        typedef struct dataRedistributionPacked
        {
            dataRedistributionPacked();
            dataRedistributionPacked(pmTask* pTask, redistributionOrderStruct* pRedistributionData, uint pCount);
            ~dataRedistributionPacked();
            
            dataRedistributionStruct redistributionStruct;
            redistributionOrderStruct* redistributionData;
        } dataRedistributionPacked;
    
        typedef struct subtaskRangeCancelStruct
        {
			uint targetDeviceGlobalIndex;
			uint originatingHost;
			ulong sequenceNumber;	// sequence number of local task object (on originating host)
			ulong startSubtask;
			ulong endSubtask;
            uint originalAllotteeGlobalIndex;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 6
			} fieldCount;

        } subtaskRangeCancelStruct;

        typedef enum communicatorCommandTypes
		{
			SEND,
			RECEIVE,
			BROADCAST,
			ALL2ALL,
			MAX_COMMUNICATOR_COMMAND_TYPES
		} communicatorCommandTypes;

		typedef enum communicatorCommandTags
		{
			MACHINE_POOL_TRANSFER_TAG,
			DEVICE_POOL_TRANSFER_TAG,
			REMOTE_TASK_ASSIGNMENT_TAG,
			REMOTE_SUBTASK_ASSIGNMENT_TAG,
			SEND_ACKNOWLEDGEMENT_TAG,
			TASK_EVENT_TAG,
			STEAL_REQUEST_TAG,
			STEAL_RESPONSE_TAG,
            OWNERSHIP_TRANSFER_TAG,
			MEMORY_TRANSFER_REQUEST_TAG,
			MEMORY_RECEIVE_TAG,
			SUBTASK_REDUCE_TAG,
			UNKNOWN_LENGTH_TAG,
			HOST_FINALIZATION_TAG,
            DATA_REDISTRIBUTION_TAG,
            SUBTASK_RANGE_CANCEL_TAG,
			MAX_COMMUNICATOR_COMMAND_TAGS
		} communicatorCommandTags;

		typedef enum communicatorDataTypes
		{
			BYTE,
			INT,
			UINT,
			MACHINE_POOL_STRUCT,
			DEVICE_POOL_STRUCT,
			REMOTE_TASK_ASSIGN_STRUCT,
			REMOTE_TASK_ASSIGN_PACKED,
			REMOTE_SUBTASK_ASSIGN_STRUCT,
            OWNERSHIP_DATA_STRUCT,
			SEND_ACKNOWLEDGEMENT_STRUCT,
            SEND_ACKNOWLEDGEMENT_PACKED,
			TASK_EVENT_STRUCT,
			STEAL_REQUEST_STRUCT,
			STEAL_RESPONSE_STRUCT,
            MEMORY_IDENTIFIER_STRUCT,
            OWNERSHIP_CHANGE_STRUCT,
            OWNERSHIP_TRANSFER_PACKED,
			MEMORY_TRANSFER_REQUEST_STRUCT,
			SUBTASK_REDUCE_STRUCT,
			SUBTASK_REDUCE_PACKED,
			MEMORY_RECEIVE_STRUCT,
			MEMORY_RECEIVE_PACKED,
			HOST_FINALIZATION_STRUCT,
            REDISTRIBUTION_ORDER_STRUCT,
            DATA_REDISTRIBUTION_STRUCT,
            DATA_REDISTRIBUTION_PACKED,
            SUBTASK_RANGE_CANCEL_STRUCT,
			MAX_COMMUNICATOR_DATA_TYPES
		} communicatorDataTypes;

		static pmCommunicatorCommandPtr CreateSharedPtr(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
			void* pCommandData, ulong pDataUnits, void* pSecondaryData = NULL, ulong pSecondaryDataUnits = 0, pmCommandCompletionCallback pCallback = NULL);

		virtual ~pmCommunicatorCommand() {}

		virtual pmStatus SetTag(communicatorCommandTags pTag);
		virtual pmStatus SetSecondaryData(void* pSecondaryData, ulong pSecondaryLength);
		virtual communicatorCommandTags GetTag();
		virtual pmHardware* GetDestination();
		virtual communicatorDataTypes GetDataType();
		virtual void* GetSecondaryData();
		virtual ulong GetSecondaryDataLength();
		virtual bool IsValid();

	protected:
		pmCommunicatorCommand(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
			void* pCommandData, ulong pDataUnits, void* pSecondaryData = NULL, ulong pSecondaryDataUnits = 0, pmCommandCompletionCallback pCallback = NULL);

	private:
		communicatorCommandTags mCommandTag;
		communicatorDataTypes mDataType;
		pmHardware* mDestination;
		void* mSecondaryData;
		ulong mSecondaryDataLength;

		static RESOURCE_LOCK_IMPLEMENTATION_CLASS mResourceLock;
};

class pmPersistentCommunicatorCommand;
typedef std::tr1::shared_ptr<pmPersistentCommunicatorCommand> pmPersistentCommunicatorCommandPtr;

class pmPersistentCommunicatorCommand : public pmCommunicatorCommand
{
	public:
		static pmPersistentCommunicatorCommandPtr CreateSharedPtr(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
			void* pCommandData, ulong pDataUnits, void* pSecondaryData = NULL, ulong pSecondaryDataUnits = 0, pmCommandCompletionCallback pCallback = NULL);

		virtual ~pmPersistentCommunicatorCommand();

	protected:
		pmPersistentCommunicatorCommand(ushort pPriority, communicatorCommandTypes pCommandType, communicatorCommandTags pCommandTag, pmHardware* pDestination, communicatorDataTypes pDataType, 
			void* pCommandData, ulong pDataUnits, void* pSecondaryData = NULL, ulong pSecondaryDataUnits = 0, pmCommandCompletionCallback pCallback = NULL);

	private:
};

class pmTaskCommand;
typedef std::tr1::shared_ptr<pmTaskCommand> pmTaskCommandPtr;

class pmTaskCommand : public pmCommand
{
	public:
		typedef enum taskCommandTypes
		{
			BASIC_TASK,
			MAX_TASK_COMMAND_TYPES
		} taskCommandTypes;
		
		static pmTaskCommandPtr CreateSharedPtr(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0);
		virtual ~pmTaskCommand() {}

		virtual bool IsValid();

	protected:
		pmTaskCommand(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0) : pmCommand(pPriority, pCommandType, pCommandData, pDataLength) {}

	private:
};

class pmSubtaskRangeCommand;
typedef std::tr1::shared_ptr<pmSubtaskRangeCommand> pmSubtaskRangeCommandPtr;

class pmSubtaskRangeCommand : public pmCommand
{
	public:
		typedef enum subtaskRangeCommandTypes
		{
			BASIC_SUBTASK_RANGE,
			MAX_TASK_COMMAND_TYPES
		} subtaskRangeCommandTypes;
		
		static pmSubtaskRangeCommandPtr CreateSharedPtr(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0);
		virtual ~pmSubtaskRangeCommand() {}

		virtual bool IsValid();

	protected:
		pmSubtaskRangeCommand(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0) : pmCommand(pPriority, pCommandType, pCommandData, pDataLength) {}

	private:
};

class pmAccumulatorCommand : public pmCommand
{
	public:
		static pmAccumulatorCommandPtr CreateSharedPtr(const std::vector<pmCommunicatorCommandPtr>& pVector);
		virtual ~pmAccumulatorCommand() {}

		virtual bool IsValid();
    
        void FinishCommand(pmAccumulatorCommandPtr pSharedPtr);
    
        void ForceComplete(pmAccumulatorCommandPtr pSharedPtr);

	protected:
		pmAccumulatorCommand();

	private:
        void CheckFinish(pmAccumulatorCommandPtr pSharedPtr);
    
        uint mCommandCount;
        bool mForceCompleted;
		RESOURCE_LOCK_IMPLEMENTATION_CLASS mAccumulatorResourceLock;
};
    
bool operator==(pmCommunicatorCommand::memoryIdentifierStruct& pIdentifier1, pmCommunicatorCommand::memoryIdentifierStruct& pIdentifier2);

} // end namespace pm

#endif

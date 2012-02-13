
#ifndef __PM_COMMAND__
#define __PM_COMMAND__

#include "pmBase.h"
#include "pmResourceLock.h"
#include "pmTimer.h"
#include TIMER_IMPLEMENTATION_HEADER

#include <tr1/memory>	// For std::tr1

namespace pm
{

class pmTask;
class pmSignalWait;
class pmLocalTask;
class pmHardware;
class pmMachine;

class pmCommand;
typedef std::tr1::shared_ptr<pmCommand> pmCommandPtr;

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
		pmCommand(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0, pmCommandCompletionCallback pCallback = NULL);
		virtual ~pmCommand();

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
		virtual pmStatus WaitForFinish();

	protected:
		ushort mCommandType;
		void* mCommandData;
		size_t mDataLength;
		pmCommandCompletionCallback mCallback;
		pmStatus mStatus;
		pmSignalWait* mSignalWait;
		ushort mPriority;
	
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
			ushort isOutputMemReadWrite;
			ulong subtaskCount;
			char callbackKey[MAX_CB_KEY_LEN];
			uint assignedDeviceCount;
			uint originatingHost;
			ulong internalTaskId;	// memory address of local task object (on originating host)
			ushort priority;
			ushort schedModel;
			ulong inputMemAddr;		// Actual base addr of input memory
			ulong outputMemAddr;	// Actual base addr of output memory

			remoteTaskAssignStruct();
			remoteTaskAssignStruct(pmLocalTask* pLocalTask);

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 14
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

		typedef struct remoteSubtaskAssignStruct
		{
			ulong internalTaskId;	// memory address of local task object (on originating host)
			ulong startSubtask;
			ulong endSubtask;
			uint originatingHost;
			uint targetDeviceGlobalIndex;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 5
			} fieldCount;
	
		} remoteSubtaskAssignStruct;

		typedef struct sendAcknowledgementStruct
		{
			uint sourceDeviceGlobalIndex;
			uint originatingHost;
			ulong internalTaskId;	// memory address of local task object (on originating host)
			ulong startSubtask;
			ulong endSubtask;
			uint execStatus;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 6
			} fieldCount;
	
		} sendAcknowledgementStruct;

		typedef enum taskEvents
		{
			TASK_FINISH_EVENT,
			TASK_CANCEL_EVENT
		} taskEvents;

		typedef struct taskEventStruct
		{
			uint taskEvent;			// Map to enum taskEvents
			uint originatingHost;
			ulong internalTaskId;	// memory address of local task object (on originating host)

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
			ulong internalTaskId;	// memory address of local task object (on originating host)
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
			ulong internalTaskId;	// memory address of local task object (on originating host)
			ushort success;			// enum stealResponseType
			ulong startSubtask;
			ulong endSubtask;			

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 7
			} fieldCount;
	
		} stealResponseStruct;

		typedef struct memorySubscriptionRequest
		{
			ulong ownerBaseAddr;	// Actual base memory address on serving host
			ulong receiverBaseAddr;	// Actual base memory address on receiving host (on destHost)
			ulong offset;
			ulong length;
			uint destHost;			// Host that will receive the memory (generally same as the requesting host)

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 5
			} fieldCount;
	
		} memorySubscriptionRequest;

		typedef struct subtaskReduceStruct
		{
			uint originatingHost;
			ulong internalTaskId;	// memory address of local task object (on originating host)
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
			subtaskReducePacked(pmTask* pTask, ulong pSubtaskId);
			~subtaskReducePacked();

			subtaskReduceStruct reduceStruct;
			dataPtr subtaskMem;
		} subtaskReducePacked;

		typedef struct memoryReceiveStruct
		{
			ulong receivingMemBaseAddr;
			ulong offset;
			ulong length;

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 3
			} fieldCount;
	
		} memoryReceiveStruct;

		typedef struct memoryReceivePacked
		{
			memoryReceivePacked();
			memoryReceivePacked(ulong pReceivingMemBaseAddr, ulong pOffset, ulong pLength, void* pMemPtr);
			~memoryReceivePacked();

			memoryReceiveStruct receiveStruct;
			dataPtr mem;
		} memoryReceivePacked;
    
        typdef struct hostFinalizationStruct
        {
            ushort terminate;   // firstly all machines send to master with terminate false; then master sends to all machines with terminate true

			typedef enum fieldCount
			{
				FIELD_COUNT_VALUE = 1
			} fieldCount;
            
        } hostFinalizationStruct;

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
			MACHINE_POOL_TRANSFER,
			DEVICE_POOL_TRANSFER,
			REMOTE_TASK_ASSIGNMENT,
			REMOTE_SUBTASK_ASSIGNMENT,
			SEND_ACKNOWLEDGEMENT_TAG,
			TASK_EVENT_TAG,
			STEAL_REQUEST_TAG,
			STEAL_RESPONSE_TAG,
			MEMORY_SUBSCRIPTION_TAG,
			MEMORY_RECEIVE_TAG,
			SUBTASK_REDUCE_TAG,
            HOST_FINALIZATION_TAG,
			UNKNOWN_LENGTH_TAG,
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
			SEND_ACKNOWLEDGEMENT_STRUCT,
			TASK_EVENT_STRUCT,
			STEAL_REQUEST_STRUCT,
			STEAL_RESPONSE_STRUCT,
			MEMORY_SUBSCRIPTION_STRUCT,
			SUBTASK_REDUCE_STRUCT,
			SUBTASK_REDUCE_PACKED,
			MEMORY_RECEIVE_STRUCT,
			MEMORY_RECEIVE_PACKED,
            HOST_FINALIZATION_STRUCT,
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

class pmThreadCommand;
typedef std::tr1::shared_ptr<pmThreadCommand> pmThreadCommandPtr;

class pmThreadCommand : public pmCommand
{
	public:
		typedef enum threadCommandTypes
		{
			COMMAND_WRAPPER,
			MAX_THREAD_COMMAND_TYPES
		} threadCommandTypes;

		static pmThreadCommandPtr CreateSharedPtr(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0);
		virtual ~pmThreadCommand() {}

		virtual bool IsValid();

	protected:
		pmThreadCommand(ushort pPriority, ushort pCommandType, void* pCommandData = NULL, ulong pDataLength = 0) : pmCommand(pPriority, pCommandType, pCommandData, pDataLength) {}

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

} // end namespace pm

#endif

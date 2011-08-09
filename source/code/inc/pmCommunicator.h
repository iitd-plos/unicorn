
#ifndef __PM_COMMUNICATOR__
#define __PM_COMMUNICATOR__

#include "pmInternalDefinitions.h"
#include "pmThread.h"

namespace pm
{

class pmCommunicatorCommand;
class pmThreadCommand;

/**
 * \brief The communicator class of PMLIB. Controllers on different machines talk through communicator.
 * This class is implemented over MPI and is the only class in PMLIB that provides communication between
 * different machines. All PMLIB components (like scheduler) talk to pmController which sends pmCommands
 * to other pmControllers using pmCommunicator's API. pmCommunicator only allows pmCommand objects to be
 * sent or received by various pmControllers. This is a per machine singleton class i.e. only one instance
 * of pmCommunicator exists on each machine. All pmCommunicators run in a separate thread.
*/

class pmCommunicator: public THREADING_IMPLEMENTATION_CLASS
{
	public:
		typedef enum internalIdentifier
		{
			SEND,
			BROADCAST,
			RECEIVE,
			MAX_IDENTIFIER_COMMANDS
		} internalIdentifier;

		typedef struct commandWrapper
		{
			internalIdentifier id;
			pmHardware hardware;
			bool blocking;
			pmCommunicatorCommand* communicatorCommand;
		} commandWrapper;

		static pmCommunicator* GetCommunicator();
		pmStatus DestroyCommunicator();

		pmStatus Send(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus Broadcast(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);

		pmStatus BuildThreadCommand(const pmCommunicatorCommand* pCommunicatorCommand, const pmThreadCommand* pThreadCommand, internalIdentifier pId, pmHardware pHardware, bool pBlocking);

		virtual pmStatus ThreadSwitchCallback(const pmThreadCommand* pThreadCommand);

	private:
		pmCommunicator();

		pmStatus SendInternal(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus BroadcastInternal(const pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus ReceiveInternal(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);

		static pmCommunicator* mCommunicator;
};

} // end namespace pm

#endif
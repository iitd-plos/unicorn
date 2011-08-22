
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
 * of pmCommunicator exists on each machine.
*/

class pmCommunicator
{
	public:
		static pmCommunicator* GetCommunicator();
		pmStatus DestroyCommunicator();

		pmStatus Send(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus Broadcast(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);
		pmStatus Receive(pmCommunicatorCommand* pCommand, pmHardware pHardware, bool pBlocking = false);

	private:
		pmCommunicator();

		static pmCommunicator* mCommunicator;
};

} // end namespace pm

#endif

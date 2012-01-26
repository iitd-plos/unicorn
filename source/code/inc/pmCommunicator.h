
#ifndef __PM_COMMUNICATOR__
#define __PM_COMMUNICATOR__

#include "pmBase.h"
#include "pmCommand.h"

namespace pm
{

/**
 * \brief The communicator class of PMLIB. Controllers on different machines talk through communicator.
 * This class is implemented over MPI and is the only class in PMLIB that provides communication between
 * different machines. All PMLIB components (like scheduler) talk to pmController which sends pmCommands
 * to other pmControllers using pmCommunicator's API. pmCommunicator only allows pmCommand objects to be
 * sent or received by various pmControllers. This is a per machine singleton class i.e. only one instance
 * of pmCommunicator exists on each machine.
*/

class pmCommunicator : public pmBase
{
	public:
		static pmCommunicator* GetCommunicator();
		pmStatus DestroyCommunicator();

		pmStatus Send(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		pmStatus Receive(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);	// If no source is provided, any machine is assumed (MPI_ANY) 
		//pmStatus ReceivePacked(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);	// If no source is provided, any machine is assumed (MPI_ANY) 
		pmStatus SendPacked(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		pmStatus ReceivePacked(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);	// If no source is provided, any machine is assumed (MPI_ANY) 
		pmStatus Broadcast(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		pmStatus All2All(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		
	private:
		pmCommunicator();

		static pmCommunicator* mCommunicator;
};

} // end namespace pm

#endif

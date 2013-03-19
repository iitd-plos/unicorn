
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
    friend class pmHeavyOperationsThread;

    public:
		static pmCommunicator* GetCommunicator();

		pmStatus Send(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		pmStatus Receive(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);	// If no source is provided, any machine is assumed (MPI_ANY) 
		pmStatus Broadcast(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		pmStatus All2All(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
		
	private:
		pmCommunicator();

		pmStatus SendPacked(pmCommunicatorCommandPtr pCommand, bool pBlocking = false);
};

} // end namespace pm

#endif


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

#ifndef __PM_CLUSTER__
#define __PM_CLUSTER__

#include "pmBase.h"
#include "pmHardware.h"
#include "mpi.h"

#include <set>

namespace pm
{

/**
 * \brief The base cluster class of PMLIB.
 * This class serves as a factory class to various cluster implementations.
 * This class creates a virtual group of machines and a task submission can
 * be confined to a cluster only. A cluster may consist of further sub-clusters
 * and is typically designed to capture various network topologies. Communication
 * within a cluster is assumed to be cheaper than globally to distant machines in
 * other clusters.
*/

class pmCluster : public pmHardware
{
	public:
		bool ContainsMachine(const pmMachine* pMachine) const;

	protected:
		pmCluster()	/* Only used for creating Global Cluster; Machine list is not stored for this cluster */
        {}
    
		pmCluster(std::set<const pmMachine*>& pMachines);

	private:
		std::set<const pmMachine*> mMachines;
};

extern pmCluster* PM_GLOBAL_CLUSTER;	/* The cluster of all machines */

class pmClusterMPI : public pmCluster
{
	friend class pmMPI;

	private:
		pmClusterMPI();	/* Creates Global Cluster; Access it using PM_GLOBAL_CLUSTER */

	public:		
		pmClusterMPI(std::set<const pmMachine*>& pMachines);

		MPI_Comm GetCommunicator() const;
		uint GetRankInCommunicator(const pmMachine* pMachine) const;

		bool operator==(const pmClusterMPI& pClusterMPI) const;

	private:
		MPI_Comm mCommunicator;
};

} // end namespace pm

#endif

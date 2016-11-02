
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

#include "pmCluster.h"
#include <algorithm>

namespace pm
{

/* class pmCluster */
pmCluster::pmCluster(std::set<const pmMachine*>& pMachines)
{
	mMachines = pMachines;
}

bool pmCluster::ContainsMachine(const pmMachine* pMachine) const
{
	if(this == PM_GLOBAL_CLUSTER || mMachines.find(pMachine) != mMachines.end())
		return true;

	return false;
}


/* class pmClusterMPI */
pmClusterMPI::pmClusterMPI()
    : pmCluster()
{
	mCommunicator = MPI_COMM_WORLD;
}

pmClusterMPI::pmClusterMPI(std::set<const pmMachine*>& pMachines)
    : pmCluster(pMachines)
{
}

MPI_Comm pmClusterMPI::GetCommunicator() const
{
	return mCommunicator;
}

uint pmClusterMPI::GetRankInCommunicator(const pmMachine* pMachine) const
{
	if(mCommunicator == MPI_COMM_WORLD)
		return *pMachine;

	PMTHROW(pmFatalErrorException());   // Unimplemented !!!
}

bool pmClusterMPI::operator==(const pmClusterMPI& pClusterMPI) const
{
	return false;
}

};


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

#include "pmCluster.h"
#include <algorithm>

namespace pm
{

/* class pmCluster */
pmCluster::pmCluster(std::set<pmMachine*>& pMachines)
{
	mMachines = pMachines;
}

pmCluster::pmCluster()
{
}

pmCluster::~pmCluster()
{
}

bool pmCluster::ContainsMachine(pmMachine* pMachine)
{
	if(this == PM_GLOBAL_CLUSTER)
		return true;

	if(mMachines.find(pMachine) != mMachines.end())
		return true;

	return false;
}


/* class pmClusterMPI */
pmClusterMPI::pmClusterMPI() : pmCluster()
{
	mCommunicator = MPI_COMM_WORLD;
}

pmClusterMPI::pmClusterMPI(std::set<pmMachine*>& pMachines) : pmCluster(pMachines)
{
}

pmClusterMPI::~pmClusterMPI()
{
}

MPI_Comm pmClusterMPI::GetCommunicator()
{
	return mCommunicator;
}

uint pmClusterMPI::GetRankInCommunicator(pmMachine* pMachine)
{
	if(mCommunicator == MPI_COMM_WORLD)
		return *pMachine;

	return 0;
}

bool pmClusterMPI::operator==(pmClusterMPI& pClusterMPI)
{
	return false;
}

};

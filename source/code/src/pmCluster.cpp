
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

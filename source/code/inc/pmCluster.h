
#ifndef __PM_CLUSTER__
#define __PM_CLUSTER__

#include "pmBase.h"
#include "pmHardware.h"
#include "pmNetwork.h"
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
		bool ContainsMachine(pmMachine* pMachine);

	protected:
		pmCluster();	/* Only used for creating Global Cluster; Machine list is not stored for this cluster */
		pmCluster(std::set<pmMachine*>& pMachines);
		virtual ~pmCluster();

	private:
		std::set<pmMachine*> mMachines;
};

extern pmCluster* PM_GLOBAL_CLUSTER;	/* The cluster of all machines */

class pmClusterMPI : public pmCluster
{
	friend class pmMPI;

	private:
		pmClusterMPI();	/* Creates Global Cluster; Access it using PM_GLOBAL_CLUSTER */

	public:		
		pmClusterMPI(std::set<pmMachine*>& pMachines);
		virtual ~pmClusterMPI();

		virtual MPI_Comm GetCommunicator();
		virtual uint GetRankInCommunicator(pmMachine* pMachine);

		virtual bool operator==(pmClusterMPI& pClusterMPI);

	private:
		MPI_Comm mCommunicator;
};

} // end namespace pm

#endif

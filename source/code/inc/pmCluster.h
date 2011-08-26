
#ifndef __PM_CLUSTER__
#define __PM_CLUSTER__

#include "pmInternalDefinitions.h"

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

class pmCluster : public pmBase
{
	public:
		

	private:
};

extern pmCluster PM_GLOBAL_CLUSTER;	/* The cluster of all machines */

class pmMPICluster : public pmCluster
{
	public:


	private:
};

} // end namespace pm

#endif

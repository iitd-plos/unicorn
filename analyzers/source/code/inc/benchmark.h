
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

#ifndef __ANALYZER_BENCHMARK__
#define __ANALYZER_BENCHMARK__

#include <map>
#include <set>
#include <vector>
#include <string>
#include <limits>

#include "graph.h"

struct StandardCurve
{
    std::string name;
    std::vector<std::pair<double, double>> points;     // For rect graphs, the first member of pair is the group number
    
    StandardCurve(const std::string& pName)
    : name(pName)
    {}
};

struct StandardGantt
{
    std::string name;
    std::vector<std::pair<size_t, std::pair<double, double>>> data;
    
    StandardGantt(const std::string& pName)
    : name(pName)
    {}
};

struct StandardChart
{
    std::unique_ptr<Axis> xAxis;
    std::unique_ptr<Axis> yAxis;
    
    std::vector<std::string> groups;    // Only used for rect/gantt graphs
    std::vector<StandardCurve> curves;
    std::vector<StandardGantt> gantts;
    
    StandardChart(std::unique_ptr<Axis> pXAxis, std::unique_ptr<Axis> pYAxis)
    : xAxis(std::move(pXAxis))
    , yAxis(std::move(pYAxis))
    {}
    
    void SetCurves(size_t pCurveCount, const char** pCurveNames)
    {
        curves.clear();
        
        for(size_t i = 0; i < pCurveCount; ++i)
            curves.push_back(StandardCurve(pCurveNames[i]));
    }
    
    std::unique_ptr<Graph> graph;
};

/* Define names of each type in array EventTypeNames in file benchmark.cpp */
enum EventTypes
{
    SUBTASK_EXECUTION,
    WAIT_ON_NETWORK,
    COPY_TO_PINNED_MEMORY,
    COPY_FROM_PINNED_MEMORY,
    MAX_EVENT_TYPES
};

struct DeviceStats
{
    size_t subtasksExecuted;
    double subtaskExecutionRate;
    size_t stealAttempts;   // Only valid in PULL policy
    size_t stealSuccesses;  // Only valid in PULL policy
    size_t stealFailures;   // Only valid in PULL policy

    std::map<size_t, std::pair<double, double>> subtaskTimeline; // Subtask Id versus pair of subtask start and end time
    std::vector<std::pair<EventTypes, std::pair<double, double>>> eventTimeline;
    std::pair<double, double> lastEventTimings;
    
    DeviceStats()
    : subtasksExecuted(0)
    , subtaskExecutionRate(0)
    , stealAttempts(0)
    , stealSuccesses(0)
    , stealFailures(0)
    , lastEventTimings(std::make_pair(0, 0))
    {}
};

struct MachineStats
{
    size_t subtasksExecuted;
    size_t cpuSubtasks;
    
    MachineStats()
    : subtasksExecuted(0)
    , cpuSubtasks(0)
    {}
};

enum SchedulingPolicy
{
    PUSH,
    PULL,
    STATIC_EQUAL,
    STATIC_BEST,
    PULL_WITH_AFFINITY,
    MAX_SCHEDULING_POLICY
};

enum clusterType
{
    CPU,
    GPU,
    CPU_PLUS_GPU,
    MAX_CLUSTER_TYPE
};

struct Level1Key
{
    size_t varying1;
    size_t varying2;

    Level1Key()
    : varying1(0)
    , varying2(0)
    {}

    friend bool operator< (const Level1Key& pFirst, const Level1Key& pSecond)
    {
        if(pFirst.varying1 == pSecond.varying1)
            return (pFirst.varying2 < pSecond.varying2);
        
        return(pFirst.varying1 < pSecond.varying1);
    }
};

struct Level1Value
{
    double sequentialTime;
    double singleGpuTime;
    
    Level1Value()
    : sequentialTime(std::numeric_limits<double>::infinity())
    , singleGpuTime(std::numeric_limits<double>::infinity())
    {}
};

struct Level2Key
{
    size_t hosts;
    enum SchedulingPolicy policy;
    enum clusterType cluster;
    bool multiAssign;
    bool lazyMem;
    bool overlapComputeCommunication;
    int affinityCriterion;
    
    Level2Key(size_t pHosts, enum SchedulingPolicy pPolicy, enum clusterType pCluster, bool pMultiAssign, bool pLazyMem, bool pOverlapComputeCommunication, int pAffinityCriterion = -1)
    : hosts(pHosts)
    , policy(pPolicy)
    , cluster(pCluster)
    , multiAssign(pMultiAssign)
    , lazyMem(pLazyMem)
    , overlapComputeCommunication(pOverlapComputeCommunication)
    , affinityCriterion(pAffinityCriterion)
    {}

    friend bool operator< (const Level2Key& pFirst, const Level2Key& pSecond)
    {
        if(pFirst.hosts == pSecond.hosts)
        {
            if(pFirst.policy == pSecond.policy)
            {
                if(pFirst.cluster == pSecond.cluster)
                {
                    if(pFirst.multiAssign == pSecond.multiAssign)
                    {
                        if(pFirst.lazyMem == pSecond.lazyMem)
                            return (pFirst.overlapComputeCommunication < pSecond.overlapComputeCommunication);
                            
                        return (pFirst.lazyMem < pSecond.lazyMem);
                    }
                    
                    return (pFirst.multiAssign < pSecond.multiAssign);
                }
                
                return (pFirst.cluster < pSecond.cluster);
            }
            
            return (pFirst.policy < pSecond.policy);
        }
        
        return (pFirst.hosts < pSecond.hosts);
    }
};

struct Level2InnerTaskValue
{
    size_t subtaskCount;
    double totalExecutionRate;
   
    std::map<std::string, std::pair<double, double> > workTimeStats;
    std::map<size_t, DeviceStats> deviceStats;
    std::map<size_t, MachineStats> machineStats;
    
    Level2InnerTaskValue()
    : subtaskCount(0)
    , totalExecutionRate(0)
    {}
};

struct Level2InnerTaskKey
{
    size_t originatingHost;
    size_t taskSequenceId;
    
    Level2InnerTaskKey()
    : originatingHost(std::numeric_limits<size_t>::infinity())
    , taskSequenceId(std::numeric_limits<size_t>::infinity())
    {}

    friend bool operator< (const Level2InnerTaskKey& pFirst, const Level2InnerTaskKey& pSecond)
    {
        if(pFirst.originatingHost == pSecond.originatingHost)
            return (pFirst.taskSequenceId < pSecond.taskSequenceId);
        
        return(pFirst.originatingHost < pSecond.originatingHost);
    }
};

struct Level2Value
{
    double execTime;
    std::map<Level2InnerTaskKey, Level2InnerTaskValue> innerTaskMap;

    Level2Value()
    : execTime(std::numeric_limits<double>::infinity())
    {}
};

struct BenchmarkResults
{
    typedef std::map<Level1Key, std::pair<Level1Value, std::map<Level2Key, Level2Value> > > mapType;
    mapType results;

    std::map<size_t, size_t> hostsMap;  // Maps no. of hosts used in the benchmark to a sequential order e.g. 1, 2, 4, 8 to 1, 2, 3, 4
};

class Benchmark
{
public:
    typedef std::map<std::string, std::vector<std::string> > keyValuePairs;
    typedef std::vector<std::pair<std::string, std::vector<std::string> > > panelConfigurationType;

    void SetExecPath(const std::string& pExecPath);
    
    void CollectResults();
    void ProcessResults();

    const std::string& GetName();
    const std::string& GetExecPath();

    static void RegisterBasePath(const std::string& pBasePath);
    static void LoadGlobalConfiguration();
    static void GetAllBenchmarks(std::vector<Benchmark>& pBenchmarks);
    static void WriteTopLevelHtmlPage(const std::vector<Benchmark>& pBenchmarks);
    static void CopyResourceFiles();
    
    void ExecuteInstance(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pUnderscoreSeparatedVaryingsStr, size_t pSampleIndex);
    void ExecuteSample(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pOutputFolder);
    
protected:
    Benchmark(const std::string& pName);

private:
    void LoadConfiguration();
    static void LoadKeyValuePairs(const std::string& pFilePath, keyValuePairs& pPairs);
    static void CreateDir(const std::string& pPath);
    
    void ParseResultsFile(const Level1Key& pLevel1Key, const std::string& pResultsFile, size_t pSampleIndex);
    
    void BeginHtmlSection(std::ofstream &pHtmlStream, const std::string& pSectionName);

    void GenerateAnalysis();
    void GenerateTable(std::ofstream& pHtmlStream, std::vector<size_t>& pRadioSetCount);
    void GeneratePlots(std::ofstream& pHtmlStream, std::vector<size_t>& pRadioSetCount);
    Graph& GenerateStandardChart(size_t pPlotWidth, size_t pPlotHeight, StandardChart& pChart);

    void GenerateTableInternal(std::ofstream& pHtmlStream, bool pSequential, bool pAbsoluteValues, bool pOverlap);

    void GeneratePreControlCode(std::ofstream& pHtmlStream);
    void GeneratePostControlCode(std::ofstream& pHtmlStream, const std::vector<size_t>& pRadioSetCount);
    
    size_t GeneratePerformanceGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateSchedulingModelsGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateLoadBalancingGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateMultiAssignComparisonGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateOverlapComparisonGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateTimelineGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    
    void GeneratePerformanceGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, SchedulingPolicy pPolicy, size_t pVarying2Val = 0);
    void GenerateSchedulingModelsGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, size_t pVarying1Val, size_t pVarying2Val);
    void GenerateLoadBalancingGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, size_t pHosts, size_t pVarying1Val, size_t pVarying2Val, SchedulingPolicy pPolicy, const Level2InnerTaskKey& pInnerTask);
    void GenerateMultiAssignComparisonGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, size_t pVarying1Val, size_t pVarying2Val, bool pOverlap);
    void GenerateOverlapComparisonGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, size_t pVarying1Val, size_t pVarying2Val, bool pMA);
    void GenerateTimelineGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, size_t pHosts, size_t pVarying1Val, size_t pVarying2Val, SchedulingPolicy pPolicy, const Level2InnerTaskKey& pInnerTask);
    
    void GenerateSelectionGroup(size_t pPanelIndex, const panelConfigurationType& pPanelConf, std::ofstream& pHtmlStream);

    void EmbedResultsInTable(std::ofstream& pHtmlStream, BenchmarkResults::mapType::iterator pLevel1Iter, size_t pHosts, bool pMultiAssign, bool pGenerateStaticBest, bool pSequential, bool pAbsoluteValues, bool pOverlap);
    void EmbedPlot(std::ofstream& pHtmlStream, Graph& pGraph, const std::string& pGraphTitle);
    
    void SelectSample(bool pMedianSample);
    void BuildInnerTaskVector();
    
    void ExecuteShellCommand(const std::string& pCmd, const std::string& pDisplayName, const std::string& pOutputFile);
    void RecordFailure(const std::string& pCmd);
    int RunCommand(const std::string& pCmd, const std::string& pDisplayName);
    bool CheckIfException(const std::string& pFilePath);
    const std::string& GetTempOutputFileName();
    
    static void CopyFile(const std::string& pSrcFile, const std::string& pDestFile);
    static keyValuePairs& GetGlobalConfiguration();
    static std::string& GetBasePath();
    
    std::string mName;
    std::string mExecPath;
    keyValuePairs mConfiguration;

    std::vector<BenchmarkResults> mSamples;
    std::vector<std::set<size_t> > mHostsSetVector;
    
    BenchmarkResults mResults;
    std::vector<Level2InnerTaskKey> mInnerTasks;
};

#endif

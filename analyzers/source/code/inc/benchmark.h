
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
    std::vector<std::pair<double, double> > points;     // For rect graphs, the first member of pair is the group number
    
    StandardCurve(const std::string& pName)
    : name(pName)
    {}
};

struct StandardChart
{
    std::auto_ptr<Axis> xAxis;
    std::auto_ptr<Axis> yAxis;
    
    std::vector<std::string> groups;    // Only used for rect graphs
    std::vector<StandardCurve> curves;
    
    StandardChart(std::auto_ptr<Axis> pXAxis, std::auto_ptr<Axis> pYAxis)
    : xAxis(pXAxis)
    , yAxis(pYAxis)
    {}
    
    void SetCurves(size_t pCurveCount, const char** pCurveNames)
    {
        curves.clear();
        
        for(size_t i = 0; i < pCurveCount; ++i)
            curves.push_back(StandardCurve(pCurveNames[i]));
    }
    
    std::auto_ptr<Graph> graph;
};

struct DeviceStats
{
    size_t subtasksExecuted;
    double subtaskExecutionRate;
    size_t stealAttempts;   // Only valid in PULL policy
    size_t stealSuccesses;  // Only valid in PULL policy
    size_t stealFailures;   // Only valid in PULL policy

    std::map<size_t, std::pair<double, double> > eventTimeline; // Subtask Id versus pair of subtask start and end time
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
    
    Level2Key(size_t pHosts, enum SchedulingPolicy pPolicy, enum clusterType pCluster, bool pMultiAssign, bool pLazyMem)
    : hosts(pHosts)
    , policy(pPolicy)
    , cluster(pCluster)
    , multiAssign(pMultiAssign)
    , lazyMem(pLazyMem)
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
                        return (pFirst.lazyMem < pSecond.lazyMem);
                    
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
    Benchmark(const std::string& pName, const std::string& pExecPath);

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

    void GenerateTableInternal(std::ofstream& pHtmlStream, bool pSequential, bool pAbsoluteValues);

    void GeneratePreControlCode(std::ofstream& pHtmlStream);
    void GeneratePostControlCode(std::ofstream& pHtmlStream, const std::vector<size_t>& pRadioSetCount);
    
    size_t GeneratePerformanceGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateSchedulingModelsGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateLoadBalancingGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    size_t GenerateMultiAssignComparisonGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream);
    
    void GeneratePerformanceGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, SchedulingPolicy pPolicy, size_t pVarying2Val = 0);
    void GenerateSchedulingModelsGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, size_t pVarying1Val, size_t pVarying2Val);
    void GenerateLoadBalancingGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, size_t pHosts, size_t pVarying1Val, size_t pVarying2Val, SchedulingPolicy pPolicy, const Level2InnerTaskKey& pInnerTask);
    void GenerateMultiAssignComparisonGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, size_t pVarying1Val, size_t pVarying2Val);
    
    void GenerateSelectionGroup(size_t pPanelIndex, const panelConfigurationType& pPanelConf, std::ofstream& pHtmlStream);

    void EmbedResultsInTable(std::ofstream& pHtmlStream, BenchmarkResults::mapType::iterator pLevel1Iter, size_t pHosts, bool pMultiAssign, bool pGenerateStaticBest, bool pSequential, bool pAbsoluteValues);
    void EmbedPlot(std::ofstream& pHtmlStream, Graph& pGraph, const std::string& pGraphTitle);
    
    void SelectSample(bool pMedianSample);
    void BuildInnerTaskVector();
    
    void ExecuteShellCommand(const std::string& pCmd, const std::string& pDisplayName, const std::string& pOutputFile);
    int RunCommand(const std::string& pCmd, const std::string& pDisplayName);
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

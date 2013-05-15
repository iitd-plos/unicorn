
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

struct DeviceStats
{
    size_t subtasksExecuted;
    double subtaskExecutionRate;
    size_t stealAttempts;   // Only valid in PULL policy
    size_t stealSuccesses;  // Only valid in PULL policy
    size_t stealFailures;   // Only valid in PULL policy
    
    std::map<size_t, std::pair<double, double> > eventTimeline; // Subtask Id versus pair of subtask start and end time
    
    DeviceStats()
    : subtasksExecuted(0)
    , subtaskExecutionRate(0)
    , stealAttempts(0)
    , stealSuccesses(0)
    , stealFailures(0)
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

struct ExecutionInstanceStats
{
    double execTimeCpu;
    double execTimeGpu;
    double execTimeCpuPlusGpu;

    size_t subtaskCount;
    bool serialComparisonResult;
   
    std::map<std::string, std::pair<double, double> > workTimeStats;
    std::map<size_t, DeviceStats> deviceStats;
    std::map<size_t, MachineStats> machineStats;
    
    ExecutionInstanceStats()
    : execTimeCpu(0)
    , execTimeGpu(0)
    , execTimeCpuPlusGpu(0)
    , subtaskCount(0)
    , serialComparisonResult(false)
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

struct ExecutionInstanceKey
{
    size_t hosts;
    enum SchedulingPolicy policy;
    size_t varying1;
    size_t varying2;
    
    ExecutionInstanceKey(size_t pHosts, enum SchedulingPolicy pPolicy, size_t pVarying1, size_t pVarying2 = 0)
    : hosts(pHosts)
    , policy(pPolicy)
    , varying1(pVarying1)
    , varying2(pVarying2)
    {}
    
    friend bool operator< (const ExecutionInstanceKey& pKey1, const ExecutionInstanceKey& pKey2);
};

struct BenchmarkResults
{
    double serialExecTime;
    std::map<ExecutionInstanceKey, ExecutionInstanceStats> parallelStats;

    BenchmarkResults()
    : serialExecTime(0)
    {}
};

class Benchmark
{
public:
    typedef std::map<std::string, std::vector<std::string> > keyValuePairs;

    void CollectResults();
    void ProcessResults();

    const std::string& GetName();
    const std::string& GetExecPath();

    static void RegisterBasePath(const std::string& pBasePath);
    static void LoadGlobalConfiguration();
    static void GetAllBenchmarks(std::vector<Benchmark>& pBenchmarks);
    
    void ExecuteInstance(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pUnderscoreSeparatedVaryingsStr);
    
protected:
    Benchmark(const std::string& pName, const std::string& pExecPath);

private:
    void LoadConfiguration();
    static void LoadKeyValuePairs(const std::string& pFilePath, keyValuePairs& pPairs);
    static void CreateDir(const std::string& pPath);
    
    void ParseResultsFile(const std::string& pResultsFile);
    void GenerateAnalysis();
    
    static keyValuePairs& GetGlobalConfiguration();
    static std::string& GetBasePath();
    
    std::string mName;
    std::string mExecPath;
    keyValuePairs mConfiguration;
    
    BenchmarkResults mResults;
};

#endif

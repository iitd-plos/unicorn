
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

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <unistd.h>
#include <sys/stat.h>

#include "benchmark.h"

#ifdef _WIN32
    const char PATH_SEPARATOR = '\\';
    const char* gIntermediatePath = "build\\windows\\release";
#else
    const char PATH_SEPARATOR = '/';
    const char* gIntermediatePath = "build/linux/release";
#endif

Benchmark::keyValuePairs mGlobalConfiguration;

template<typename calleeType>
struct ConfIterator
{
    ConfIterator(const std::vector<std::string>& pCollection, calleeType& pCallee)
    : mCollection(pCollection)
    , mCallee(pCallee)
    {}
    
    void operator() (std::string pSpaceSeparatedStr, std::string pUnderscoreSeparatedStr)
    {
        std::vector<std::string>::const_iterator lIter = mCollection.begin();
        const std::vector<std::string>::const_iterator lEndIter = mCollection.end();
        
        for(; lIter != lEndIter; ++lIter)
        {
            std::string lStr1(pSpaceSeparatedStr);
            std::string lStr2(pUnderscoreSeparatedStr);
            
            if(!lStr1.empty())
            {
                lStr1 += std::string(" ");
                lStr2 += std::string("_");
            }
            
            lStr1 += *lIter;
            lStr2 += *lIter;
            
            mCallee(lStr1, lStr2);
        }
    }
    
private:
    const std::vector<std::string>& mCollection;
    calleeType& mCallee;
};

struct ExecuteConf
{
    ExecuteConf(Benchmark* pBenchmark, const std::string& pHosts)
    : mBenchmark(pBenchmark), mHosts(pHosts)
    {}
    
    void operator() (std::string pSpaceSeparatedStr, std::string pUnderscoreSeparatedStr)
    {
        mBenchmark->ExecuteInstance(mHosts, pSpaceSeparatedStr, pUnderscoreSeparatedStr);
    }
    
private:
    Benchmark* mBenchmark;
    const std::string& mHosts;
};

Benchmark::Benchmark(const std::string& pName, const std::string& pExecPath)
: mName(pName)
, mExecPath(pExecPath)
{
}

void Benchmark::CollectResults()
{
    std::string lVarying1Str("Varying_1");
    std::string lVarying2Str("Varying_2");
    std::string lHostsStr("Hosts");
    
    std::vector<std::string>& lHostsVector = GetGlobalConfiguration()[lHostsStr];
    std::vector<std::string>::iterator lHostsIter = lHostsVector.begin(), lHostsEndIter = lHostsVector.end();
    
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        ExecuteConf lExecuteConf(this, *lHostsIter);
        
        if(mConfiguration[lVarying2Str].empty())
        {
            ConfIterator<ExecuteConf> lVarying1Iter(mConfiguration[lVarying1Str], lExecuteConf);
            
            lVarying1Iter(std::string(""), std::string(""));
        }
        else
        {
            ConfIterator<ExecuteConf> lVarying2Iter(mConfiguration[lVarying2Str], lExecuteConf);
            ConfIterator<ConfIterator<ExecuteConf> > lVarying1Iter(mConfiguration[lVarying1Str], lVarying2Iter);

            lVarying2Iter(std::string(""), std::string(""));
        }
    }
}

void Benchmark::ProcessResults()
{
    std::cout << "Processing results for benchmark " << mName << " ..." << std::endl;
    
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lDirPath(GetBasePath());
    lDirPath.append(lSeparator);
    lDirPath.append("analyzers");
    lDirPath.append(lSeparator);
    lDirPath.append("results");
    lDirPath.append(lSeparator);
    lDirPath.append("intermediate");
    lDirPath.append(lSeparator);
    lDirPath.append(mName);

    DIR* lDir = opendir(lDirPath.c_str());
    if(lDir)
    {
        struct dirent* lEntry;
        while((lEntry = readdir(lDir)) != NULL)
        {
            if(lEntry->d_type != DT_DIR && lEntry->d_namlen)
            {
                std::string lFilePath(lDirPath);
                lFilePath.append(lSeparator);
                lFilePath.append(std::string(lEntry->d_name, lEntry->d_namlen));

                ParseResultsFile(lFilePath);
            }
        }
        
        closedir(lDir);
    }
    
    GenerateAnalysis();
}

void Benchmark::GenerateAnalysis()
{
    std::cout << "Analyzing results for benchmark " << mName << " ..." << std::endl;
    
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lDirPath(GetBasePath());
    lDirPath.append(lSeparator);
    lDirPath.append("analyzers");
    lDirPath.append(lSeparator);
    lDirPath.append("results");
    lDirPath.append(lSeparator);
    lDirPath.append("htmls");
    
    CreateDir(lDirPath);
    
    std::string lHtmlPath(lDirPath);
    lHtmlPath += lSeparator + mName + std::string(".html");
    
    std::ofstream lHtmlStream;

    lHtmlStream.open(lHtmlPath.c_str());
    if(lHtmlStream.fail())
        throw std::exception();
    
    std::set<size_t> lHostsSet;

    std::map<ExecutionInstanceKey, ExecutionInstanceStats>::iterator lIter = mResults.parallelStats.begin(), lEndIter = mResults.parallelStats.end();
    for(; lIter != lEndIter; ++lIter)
        lHostsSet.insert(lIter->first.hosts);
    
    lHtmlStream << "<html>" << std::endl << "<head><center><b><u>" << mName << "</u></b></center></head>" << std::endl << "<br><br>" << std::endl << "<body>" << std::endl;
    
    lHtmlStream << "<table align=center border=1>" << std::endl;
    lHtmlStream << "<tr>" << std::endl;

    std::string lVarying2Str("Varying_2");
    if(mConfiguration[lVarying2Str].empty())
    {
        lHtmlStream << "<th rowSpan=3>Varying</th>" << std::endl;
    }
    else
    {
        lHtmlStream << "<th rowSpan=3>Varying&nbsp;1</th>" << std::endl;
        lHtmlStream << "<th rowSpan=3>Varying&nbsp;2</th>" << std::endl;
    }
    
    lHtmlStream << "<th rowSpan=3>Serial Time<br>(in s)</th>" << std::endl;

    lHtmlStream << "<th colSpan=" << 3 * lHostsSet.size() << ">Parallel Time (in s)</th>" << std::endl;
    lHtmlStream << "</tr>" << std::endl << "<tr>" << std::endl;
    
    std::set<size_t>::iterator lHostsIter = lHostsSet.begin(), lHostsEndIter = lHostsSet.end();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
        lHtmlStream << "<th colSpan=3>" << (*lHostsIter) << "&nbsp;" << (((*lHostsIter) == 1) ? "Host" : "Hosts") << "</th>" << std::endl;

    lHtmlStream << "</tr>" << std::endl << "<tr>" << std::endl;
    
    lHostsIter = lHostsSet.begin();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        lHtmlStream << "<th>CPUs</th>" << std::endl;
        lHtmlStream << "<th>GPUs</th>" << std::endl;
        lHtmlStream << "<th>CPUs+GPUs</th>" << std::endl;
    }
    
    lHtmlStream << "</tr>" << std::endl;
    
    lIter = mResults.parallelStats.begin();
    for(; lIter != lEndIter; ++lIter)
    {
        
    }
    
    lHtmlStream << "</table>" << std::endl;
    
    lHtmlStream << "</body>" << std::endl << "</html>" << std::endl;
    
    lHtmlStream.close();
}

void Benchmark::ParseResultsFile(const std::string& pResultsFile)
{
    std::ifstream lFileStream(pResultsFile.c_str());
    if(lFileStream.fail())
        throw std::exception();
    
    std::cmatch lResults;
    size_t lHosts = 0, lVarying1 = 0, lVarying2 = 0, lCurrentDevice = 0;

    std::regex lVarying1Exp("([0-9]+)_([0-9]+)$");
    std::regex lVarying2Exp("([0-9]+)_([0-9]+)_([0-9]+)$");
    if(std::regex_search(pResultsFile.c_str(), lResults, lVarying2Exp))
    {
        lVarying2 = atoi(std::string(lResults[3]).c_str());
    }
    else if(!std::regex_search(pResultsFile.c_str(), lResults, lVarying1Exp))
        throw std::exception();
    
    lVarying1 = atoi(std::string(lResults[2]).c_str());
    lHosts = atoi(std::string(lResults[1]).c_str());

    ExecutionInstanceKey lKey(lHosts, MAX_SCHEDULING_POLICY, lVarying1, lVarying2);
    
    std::map<std::pair<size_t, size_t>, enum SchedulingPolicy> lTaskIdentifierMap;  // Pair of task originating host and sequence id

    std::string lLine;
    std::regex lExp1("Serial Task Execution Time = ([0-9.]+)");
    std::regex lExp2("Subtask distribution for task \[([0-9]+), ([0-9]+)\] under scheduling policy ([0-9]+) ... ");
    std::regex lExp3("Device ([0-9]+) Subtasks ([0-9]+)");
    std::regex lExp4("Machine ([0-9]+) Subtasks ([0-9]+) CPU-Subtasks ([0-9]+)");
    std::regex lExp5("Total Acknowledgements Received ([0-9]+)");
    std::regex lExp6("([^ ]+) => Accumulated Time: ([0-9.]+)s; Actual Time = ([0-9.]+)s; Overlapped Time = ([0-9.]+)s");
    std::regex lExp7("Device ([0-9]+) - Subtask execution rate = ([0-9.]+); Steal attemps = ([0-9]+); Successful steals = ([0-9]+); Failed steals = ([0-9]+)");
    std::regex lExp8("Parallel Task ([0-9]+) Execution Time = ([0-9.]+) \[Scheduling Policy: ([0-9]+)\] \[Serial Comparison Test ([A-Za-z]+)\]");
    std::regex lExp9("PMLIB \[Host ([0-9]+)\] Event Timeline Device ([0-9]+)");
    std::regex lExp10("Task \[([0-9]+), ([0-9]+)\] Subtask ([0-9]+) ([0-9.]+) ([0-9.]+)");
    
    while(std::getline(lFileStream, lLine))
    {
        if(std::regex_search(lLine.c_str(), lResults, lExp1))
        {
            mResults.serialExecTime = atof(std::string(lResults[1]).c_str());
            break;
        }
    }
    
    while(std::getline(lFileStream, lLine))
    {
        if(std::regex_search(lLine.c_str(), lResults, lExp10))
        {
            std::pair<size_t, size_t> lPair(atoi(std::string(lResults[1]).c_str()), atoi(std::string(lResults[2]).c_str()));
            size_t lSubtaskId = atoi(std::string(lResults[3]).c_str());
            double lStartTime = atof(std::string(lResults[4]).c_str());
            double lEndTime = atof(std::string(lResults[5]).c_str());
            
            lKey.policy = lTaskIdentifierMap[lPair];
            
            mResults.parallelStats[lKey].deviceStats[lCurrentDevice].eventTimeline[lSubtaskId] = std::make_pair(lStartTime, lEndTime);
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp2))
        {
            lKey.policy = (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str()));
            lTaskIdentifierMap[std::make_pair(atoi(std::string(lResults[1]).c_str()), atoi(std::string(lResults[2]).c_str()))] = lKey.policy;
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp3))
        {
            mResults.parallelStats[lKey].deviceStats[atoi(std::string(lResults[1]).c_str())].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp4))
        {
            size_t lMachine = atoi(std::string(lResults[1]).c_str());
            
            mResults.parallelStats[lKey].machineStats[lMachine].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
            mResults.parallelStats[lKey].machineStats[lMachine].cpuSubtasks = atoi(std::string(lResults[3]).c_str());
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp5))
        {
            mResults.parallelStats[lKey].subtaskCount = atoi(std::string(lResults[1]).c_str());
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp6))
        {
            std::string lCriterion(lResults[1]);

            mResults.parallelStats[lKey].workTimeStats[lCriterion].first = atof(std::string(lResults[2]).c_str());
            mResults.parallelStats[lKey].workTimeStats[lCriterion].second = atof(std::string(lResults[3]).c_str());
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp7))
        {
            size_t lDevice = atoi(std::string(lResults[1]).c_str());
            
            mResults.parallelStats[lKey].deviceStats[lDevice].subtaskExecutionRate = atoi(std::string(lResults[2]).c_str());
            mResults.parallelStats[lKey].deviceStats[lDevice].stealAttempts = atoi(std::string(lResults[3]).c_str());
            mResults.parallelStats[lKey].deviceStats[lDevice].stealSuccesses = atoi(std::string(lResults[4]).c_str());
            mResults.parallelStats[lKey].deviceStats[lDevice].stealFailures = atoi(std::string(lResults[5]).c_str());
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp8))
        {
            lKey.policy = (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str()));
            mResults.parallelStats[lKey].serialComparisonResult = (bool)(!strcmp(std::string(lResults[4]).c_str(), "Passed"));
            size_t lParallelTask = (atoi(std::string(lResults[1]).c_str()));
            
            switch(lParallelTask)
            {
                case 4:
                    mResults.parallelStats[lKey].execTimeCpu = atof(std::string(lResults[2]).c_str());
                    break;
                    
                case 5:
                    mResults.parallelStats[lKey].execTimeGpu = atof(std::string(lResults[2]).c_str());
                    break;

                case 6:
                    mResults.parallelStats[lKey].execTimeCpuPlusGpu = atof(std::string(lResults[2]).c_str());
                    break;
            }
        }
        else if(std::regex_search(lLine.c_str(), lResults, lExp9))
        {
            lCurrentDevice = atoi(std::string(lResults[2]).c_str());
        }
    }

    lFileStream.close();
}

void Benchmark::ExecuteInstance(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pUnderscoreSeparatedVaryingsStr)
{
    std::cout << "Running benchmark " << mName << " on " << pHosts.c_str() << " hosts with varyings " << pSpaceSeparatedVaryingsStr << " ..." << std::endl;

    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lOutputFile(GetBasePath());
    lOutputFile.append(lSeparator);
    lOutputFile.append("analyzers");
    lOutputFile.append(lSeparator);
    lOutputFile.append("results");
    
    CreateDir(lOutputFile);
    
    lOutputFile.append(lSeparator);
    lOutputFile.append("intermediate");
    
    CreateDir(lOutputFile);

    lOutputFile.append(lSeparator);
    lOutputFile.append(mName);

    CreateDir(lOutputFile);

    lOutputFile.append(lSeparator);
    lOutputFile.append(pHosts + std::string("_") + pUnderscoreSeparatedVaryingsStr);

    std::ifstream lFileStream(lOutputFile.c_str());
    
    if(lFileStream.fail())
    {
        std::stringstream lStream;
        lStream << "source ~/.pmlibrc; ";
        lStream << "mpirun -n " << pHosts << " " << mExecPath << " 1 7 4 " << pSpaceSeparatedVaryingsStr;
        lStream << " 2>&1 > " << lOutputFile.c_str();
        
        system(lStream.str().c_str());
    }
    else
    {
        lFileStream.close();
    }
}

void Benchmark::CreateDir(const std::string& pPath)
{
    struct stat lFileStat;

    if(stat(pPath.c_str(), &lFileStat) != 0)
        mkdir(pPath.c_str(), 0777);
}

const std::string& Benchmark::GetName()
{
    return mName;
}

const std::string& Benchmark::GetExecPath()
{
    return mExecPath;
}

void Benchmark::GetAllBenchmarks(std::vector<Benchmark>& pBenchmarks)
{
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lTestSuitePath(GetBasePath());
    lTestSuitePath.append(lSeparator);
    lTestSuitePath.append("testsuite");

    DIR* lDir = opendir(lTestSuitePath.c_str());
    if(lDir)
    {
        std::string lBenchmarksStr("Benchmarks");
        const std::vector<std::string>& lSelectiveBenchmarks = GetGlobalConfiguration()[lBenchmarksStr];
        
        struct dirent* lEntry;
        while((lEntry = readdir(lDir)) != NULL)
        {
            if(lEntry->d_type == DT_DIR && lEntry->d_namlen)
            {
                std::string lName(lEntry->d_name, lEntry->d_namlen);
                
                if(lSelectiveBenchmarks.empty() || std::find(lSelectiveBenchmarks.begin(), lSelectiveBenchmarks.end(), lName) != lSelectiveBenchmarks.end())
                {
                    std::string lExecPath(lTestSuitePath);
                    lExecPath.append(lSeparator);
                    lExecPath += lName;
                    lExecPath.append(lSeparator);
                    lExecPath.append(gIntermediatePath);
                    lExecPath.append(lSeparator);
                    lExecPath.append(lName);
                    lExecPath.append(".exe");
                    
                    FILE* lExecFile = fopen(lExecPath.c_str(), "rb");
                    if(lExecFile)
                    {
                        fclose(lExecFile);
                        
                        Benchmark b(lName, lExecPath);
                        
                        try
                        {
                            b.LoadConfiguration();
                        }
                        catch(...)
                        {
                            continue;
                        }

                        pBenchmarks.push_back(b);
                    }
                }
            }
        }

        closedir(lDir);
    }
}

void Benchmark::LoadGlobalConfiguration()
{
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lConfPath(GetBasePath());
    lConfPath.append(lSeparator);
    lConfPath.append("analyzers");
    lConfPath.append(lSeparator);
    lConfPath.append("conf");
    lConfPath.append(lSeparator);
    lConfPath.append("global.txt");

    LoadKeyValuePairs(lConfPath, GetGlobalConfiguration());
    
    if(GetGlobalConfiguration()[std::string("Hosts")].empty())
        throw std::string("Hosts not defined in global conf file\n");
}

void Benchmark::LoadConfiguration()
{
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lConfPath(GetBasePath());
    lConfPath.append(lSeparator);
    lConfPath.append("analyzers");
    lConfPath.append(lSeparator);
    lConfPath.append("conf");
    lConfPath.append(lSeparator);
    lConfPath.append(mName);
    lConfPath.append(".txt");

    LoadKeyValuePairs(lConfPath, mConfiguration);

    if(mConfiguration[std::string("Varying_1")].empty())
        throw std::string("Varyings not defined in conf file ") + lConfPath.c_str();
}

void Benchmark::LoadKeyValuePairs(const std::string& pFilePath, keyValuePairs& pPairs)
{
    std::ifstream lFileStream(pFilePath.c_str());
    if(lFileStream.fail())
    {
        std::cout << "ERROR: Failed to load conf file " << pFilePath.c_str() << std::endl;
        throw std::exception();
    }

    std::string lLine;
    std::regex lExp1("^[ \t]*#");
    std::regex lExp2("^[ \t]*$");
    std::regex lExp3("^[ \t]*(.*)=[ \t]*(.*)$");
    std::regex lExp4("[ \t]+$");
    std::regex lExp5("[ \t]+");
    std::cmatch lResults;

    while(std::getline(lFileStream, lLine))
    {
        if(!std::regex_search(lLine.begin(), lLine.end(), lExp1) && !std::regex_search(lLine.begin(), lLine.end(), lExp2))
        {
            if(std::regex_search(lLine.c_str(), lResults, lExp3))
            {
                std::string lKey(lResults[1]);
                std::string lValue(lResults[2]);
                
                lKey = std::regex_replace(lKey, lExp4, std::string(""));
                lValue = std::regex_replace(lValue, lExp4, std::string(""));
                
                std::string lValueStr = std::regex_replace(lValue, lExp5, std::string(" "));
                
                std::stringstream lStream(lValueStr);
                std::string lTempBuf;
                
                while(lStream >> lTempBuf)
                    pPairs[lKey].push_back(lTempBuf);
            }
        }
    }

    lFileStream.close();
}

void Benchmark::RegisterBasePath(const std::string& pBasePath)
{
    GetBasePath() = pBasePath;
}

Benchmark::keyValuePairs& Benchmark::GetGlobalConfiguration()
{
    static Benchmark::keyValuePairs sKeyValuePairs;
    return sKeyValuePairs;
}
    
std::string& Benchmark::GetBasePath()
{
    static std::string sBasePath;
    return sBasePath;
}

bool operator< (const ExecutionInstanceKey& pKey1, const ExecutionInstanceKey& pKey2)
{
    if(pKey1.hosts == pKey2.hosts)
    {
        if(pKey1.policy == pKey2.policy)
        {
            if(pKey1.varying1 == pKey2.varying1)
                return (pKey1.varying2 < pKey2.varying2);
            
            return (pKey1.varying1 < pKey2.varying1);
        }
        
        return (pKey1.policy < pKey2.policy);
    }
    
    return (pKey1.hosts < pKey2.hosts);
}



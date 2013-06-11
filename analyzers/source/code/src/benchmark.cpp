
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
#include <unistd.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <sstream>
#include <set>
#include <algorithm>
#include <limits>

#include <boost/regex.hpp>

#include "benchmark.h"

#ifdef _WIN32
    const char PATH_SEPARATOR = '\\';
    const char* gIntermediatePath = "build\\windows\\release";
#else
    const char PATH_SEPARATOR = '/';
    const char* gIntermediatePath = "build/linux/release";
#endif

#define SEQUENTIAL_FILE_NAME "sequential"
#define SINGLE_GPU_FILE_NAME "singleGpu"

#define SAMPLE_COUNT 3

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
        for(size_t i = 0; i < SAMPLE_COUNT; ++i)
            mBenchmark->ExecuteInstance(mHosts, pSpaceSeparatedStr, pUnderscoreSeparatedStr, i);
    }
    
private:
    Benchmark* mBenchmark;
    const std::string& mHosts;
};

class MedianFinder
{
public:
    MedianFinder()
    {}
    
    void AddData(double pData)
    {
        mData.insert(std::make_pair(pData, mData.size()));
    }
    
    size_t GetMedianIndex()
    {
        size_t lMedian = mData.size() / 2;
        
        std::set<std::pair<double, size_t> >::iterator lIter = mData.begin();
        std::advance(lIter, lMedian);
        
        return (*lIter).second;
    }
    
private:
    std::set<std::pair<double, size_t> > mData;
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
    
    mSamples.resize(SAMPLE_COUNT);
    
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lFolderPath(GetBasePath());
    lFolderPath.append(lSeparator);
    lFolderPath.append("analyzers");
    lFolderPath.append(lSeparator);
    lFolderPath.append("results");
    lFolderPath.append(lSeparator);
    lFolderPath.append("intermediate");
    lFolderPath.append(lSeparator);
    lFolderPath.append(mName);

    boost::cmatch lResults;
    boost::regex lVarying1Exp("([0-9]+)$");
    boost::regex lVarying2Exp("([0-9]+)_([0-9]+)$");
    boost::regex lSampleExp("sample_([0-9]+)$");
    
    DIR* lDir = opendir(lFolderPath.c_str());
    if(lDir)
    {
        struct dirent* lEntry;
        while((lEntry = readdir(lDir)) != NULL)
        {
            if(lEntry->d_type == DT_DIR && strlen(lEntry->d_name) && strcmp(lEntry->d_name, ".") && strcmp(lEntry->d_name, ".."))
            {
                std::string lDirPath(lFolderPath);
                lDirPath.append(lSeparator);
                lDirPath.append(std::string(lEntry->d_name));

                Level1Key lLevel1Key;
                if(boost::regex_search(lDirPath.c_str(), lResults, lVarying2Exp))
                    lLevel1Key.varying2 = atoi(std::string(lResults[2]).c_str());
                else if(!boost::regex_search(lDirPath.c_str(), lResults, lVarying1Exp))
                    throw std::exception();
                
                lLevel1Key.varying1 = atoi(std::string(lResults[1]).c_str());

                DIR* lSamplesDir = opendir(lDirPath.c_str());
                if(lSamplesDir)
                {
                    struct dirent* lSamplesEntry;
                    while((lSamplesEntry = readdir(lSamplesDir)) != NULL)
                    {
                        if(lSamplesEntry->d_type == DT_DIR && strlen(lSamplesEntry->d_name) && strcmp(lSamplesEntry->d_name, ".") && strcmp(lSamplesEntry->d_name, ".."))
                        {
                            std::string lSamplePath(lDirPath);
                            lSamplePath.append(lSeparator);
                            lSamplePath.append(std::string(lSamplesEntry->d_name));
                            
                            if(!boost::regex_search(lSamplePath.c_str(), lResults, lSampleExp))
                                throw std::exception();
                            
                            size_t lSampleIndex = atoi(std::string(lResults[1]).c_str());

                            DIR* lResultsDir = opendir(lSamplePath.c_str());
                            if(lResultsDir)
                            {
                                struct dirent* lDirEntry;
                                while((lDirEntry = readdir(lResultsDir)) != NULL)
                                {
                                    if(lDirEntry->d_type != DT_DIR && strlen(lDirEntry->d_name))
                                    {
                                        std::string lFilePath(lSamplePath);
                                        lFilePath.append(lSeparator);
                                        lFilePath.append(std::string(lDirEntry->d_name));

                                        ParseResultsFile(lLevel1Key, lFilePath, lSampleIndex);
                                    }
                                }
                                
                                closedir(lResultsDir);
                            }
                        }
                    }
                    
                    closedir(lSamplesDir);
                }
            }
        }
        
        closedir(lDir);
    }
    
    SelectMedianSample();

    GenerateAnalysis();
}

void Benchmark::SelectMedianSample()
{
    if(mSamples.size() != SAMPLE_COUNT)
        throw std::exception();

    for(size_t i = 1; i < SAMPLE_COUNT; ++i)
    {
        if(mSamples[0].results.size() != mSamples[i].results.size())
            throw std::exception();
        
        if(mSamples[0].hostsMap != mSamples[i].hostsMap)
            throw std::exception();
    }
    
#if 0
    for(size_t i = 0; i < SAMPLE_COUNT; ++i)
    {
        std::cout << "Sample " << i << std::endl;

        BenchmarkResults::mapType::const_iterator lIter = mSamples[i].results.begin(), lEndIter = mSamples[i].results.end();
        for(; lIter != lEndIter; ++lIter)
        {
            const Level1Key& lLevel1Key = lIter->first;
            const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
            
            std::cout << lLevel1Key.varying1 << " " << lLevel1Key.varying2 << std::endl;

            std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
            for(; lInnerIter != lInnerEndIter; ++lInnerIter)
            {
                const Level2Key& lLevel2Key = lInnerIter->first;

                std::cout << lLevel2Key.hosts << " " << lLevel2Key.policy << " " << lLevel2Key.cluster << " " << lLevel2Key.multiAssign << " " << lLevel2Key.lazyMem << std::endl;
            }
        }
    }
#endif

    BenchmarkResults::mapType::const_iterator lIter = mSamples[0].results.begin(), lEndIter = mSamples[0].results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const Level1Key& lLevel1Key = lIter->first;
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;

        MedianFinder lSequentialMedianFinder, lSingleGpuMedianFinder;
        
        for(size_t i = 0; i < SAMPLE_COUNT; ++i)
        {
            if(mSamples[i].results.find(lLevel1Key) == mSamples[i].results.end())
                throw std::exception();

            lSequentialMedianFinder.AddData(mSamples[i].results[lLevel1Key].first.sequentialTime);
            lSingleGpuMedianFinder.AddData(mSamples[i].results[lLevel1Key].first.singleGpuTime);
        }
        
        mResults.results[lLevel1Key].first.sequentialTime = mSamples[lSequentialMedianFinder.GetMedianIndex()].results[lLevel1Key].first.sequentialTime;
        mResults.results[lLevel1Key].first.singleGpuTime = mSamples[lSingleGpuMedianFinder.GetMedianIndex()].results[lLevel1Key].first.singleGpuTime;
        
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            const Level2Key& lLevel2Key = lInnerIter->first;
            
            MedianFinder lParallelMedianFinder;
            for(size_t i = 0; i < SAMPLE_COUNT; ++i)
            {
                if(mSamples[i].results[lLevel1Key].second.find(lLevel2Key) == mSamples[i].results[lLevel1Key].second.end())
                    throw std::exception();

                lParallelMedianFinder.AddData(mSamples[i].results[lLevel1Key].second[lLevel2Key].execTime);
            }
            
            mResults.results[lLevel1Key].second[lLevel2Key] = mSamples[lParallelMedianFinder.GetMedianIndex()].results[lLevel1Key].second[lLevel2Key];
        }
    }
    
    mResults.hostsMap = mSamples[0].hostsMap;
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

    lHtmlStream << std::fixed << std::setprecision(2);

    lHtmlStream << "<html>" << std::endl;
    lHtmlStream << "<head><center><b><u>" << mConfiguration["Benchmark_Name"][0] << "</u></b></center></head>" << std::endl;
    lHtmlStream << "<body>" << std::endl;
    
    GenerateTable(lHtmlStream);
    GeneratePlots(lHtmlStream);
    
    lHtmlStream << "</body>" << std::endl;
    lHtmlStream << "</html>" << std::endl;
    
    lHtmlStream.close();
}

void Benchmark::BeginHtmlSection(std::ofstream &pHtmlStream, const std::string& pSectionName)
{
    pHtmlStream << "<br><br><hr><div style='text-align:center'><b>" << pSectionName << "</b></div><hr>" << std::endl;
}

void Benchmark::GenerateTable(std::ofstream& pHtmlStream)
{
    BeginHtmlSection(pHtmlStream, "Experimental Results");

    pHtmlStream << "<table align=center border=1>" << std::endl;
    pHtmlStream << "<tr>" << std::endl;

    std::string lVarying2Str("Varying_2");
    bool lVarying2Defined = !(mConfiguration[lVarying2Str].empty());
    
    if(lVarying2Defined)
    {
        boost::regex lSpaceExp("[ \t]");
        std::string lVarying1Name = boost::regex_replace(mConfiguration["Varying1_Name"][0], lSpaceExp, std::string("<br>"));
        std::string lVarying2Name = boost::regex_replace(mConfiguration["Varying2_Name"][0], lSpaceExp, std::string("<br>"));
        
        pHtmlStream << "<th rowSpan=3>" << lVarying1Name << "</th>" << std::endl;
        pHtmlStream << "<th rowSpan=3>" << lVarying2Name << "</th>" << std::endl;
    }
    else
    {
        boost::regex lSpaceExp("[ \t]");
        std::string lVarying1Name = boost::regex_replace(mConfiguration["Varying1_Name"][0], lSpaceExp, std::string("<br>"));
        
        pHtmlStream << "<th rowSpan=3>" << lVarying1Name << "</th>" << std::endl;
    }
    
    size_t lVarying2Count = (lVarying2Defined ? mConfiguration[lVarying2Str].size() : 0);
    
    pHtmlStream << "<th rowSpan=3>Sequential<br>Time<br>(in s)</th>" << std::endl;

    pHtmlStream << "<th colSpan=" << 3 * mResults.hostsMap.size() << ">Parallel Time (in s)</th>" << std::endl;
    pHtmlStream << "</tr>" << std::endl << "<tr>" << std::endl;
    
    std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
        pHtmlStream << "<th colSpan=3>" << lHostsIter->first << "&nbsp;" << ((lHostsIter->first == 1) ? "Host" : "Hosts") << "</th>" << std::endl;

    pHtmlStream << "</tr>" << std::endl << "<tr>" << std::endl;
    
    lHostsIter = mResults.hostsMap.begin();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        pHtmlStream << "<th>CPUs</th>" << std::endl;
        pHtmlStream << "<th>GPUs</th>" << std::endl;
        pHtmlStream << "<th>CPUs+GPUs</th>" << std::endl;
    }
    
    pHtmlStream << "</tr>" << std::endl;

    BenchmarkResults::mapType::iterator lLevel1Iter = mResults.results.begin(), lLevel1EndIter = mResults.results.end();
    for(; lLevel1Iter != lLevel1EndIter; ++lLevel1Iter)
    {
        pHtmlStream << "<tr>" << std::endl;
        pHtmlStream << "<td align=center rowspan = " << ((lVarying2Count == 0) ? 1 : lVarying2Count) << ">" << lLevel1Iter->first.varying1 << "</td>" << std::endl;
        
        if(lVarying2Defined)
            pHtmlStream << "<td align=center>" << lLevel1Iter->first.varying2 << "</td>" << std::endl;

        pHtmlStream << "<td align=center>" << lLevel1Iter->second.first.sequentialTime << "</td>" << std::endl;

        std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
        for(; lHostsIter != lHostsEndIter; ++lHostsIter)
        {
            Level2Key lKey1(lHostsIter->first, PUSH, CPU, false, false);
            Level2Key lKey2(lHostsIter->first, PULL, CPU, false, false);
            Level2Key lKey3(lHostsIter->first, STATIC_EQUAL, CPU, false, false);
            Level2Key lKey4(lHostsIter->first, STATIC_BEST, CPU, false, false);
            
            Level2Key lKey5(lHostsIter->first, PUSH, GPU, false, false);
            Level2Key lKey6(lHostsIter->first, PULL, GPU, false, false);
            Level2Key lKey7(lHostsIter->first, STATIC_EQUAL, GPU, false, false);
            Level2Key lKey8(lHostsIter->first, STATIC_BEST, GPU, false, false);

            Level2Key lKey9(lHostsIter->first, PUSH, CPU_PLUS_GPU, false, false);
            Level2Key lKey10(lHostsIter->first, PULL, CPU_PLUS_GPU, false, false);
            Level2Key lKey11(lHostsIter->first, STATIC_EQUAL, CPU_PLUS_GPU, false, false);
            Level2Key lKey12(lHostsIter->first, STATIC_BEST, CPU_PLUS_GPU, false, false);

            pHtmlStream << "<td><table align = center>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey1].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey2].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey3].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey4].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

            pHtmlStream << "<td><table align = center>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey5].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey6].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey7].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey8].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

            pHtmlStream << "<td><table align = center>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey9].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey10].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey11].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "<tr><td>" << lLevel1Iter->second.second[lKey12].execTime << "</td></tr>" << std::endl;
            pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;
        }

        pHtmlStream << "</tr>" << std::endl;
    }
    
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::GeneratePlots(std::ofstream& pHtmlStream)
{
    size_t lPlotWidth = 400;
    size_t lPlotHeight = 400;
    
    pHtmlStream << "<style type='text/css'> div.plotTitle { border-width:1px; border-style:solid; border-color:gray; background:lightgray; text-align:center; } </style>" << std::endl;
    pHtmlStream << "<style type='text/css'> tr.horizSpacing > td { padding-left: 2em; padding-right: 2em; } </style>" << std::endl;
    
    BeginHtmlSection(pHtmlStream, "Performance Graphs - PUSH Model");
    GeneratePerformanceGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PUSH);

    BeginHtmlSection(pHtmlStream, "Performance Graphs - PULL Model");
    GeneratePerformanceGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PULL);

    BeginHtmlSection(pHtmlStream, "Scheduling Models Comparison");
    GenerateSchedulingModelsGraphs(lPlotWidth, lPlotHeight, pHtmlStream);

    BeginHtmlSection(pHtmlStream, "Load Balancing Graphs");
    GenerateLoadBalancingGraphs(lPlotWidth, lPlotHeight, pHtmlStream);

    BeginHtmlSection(pHtmlStream, "Overhead Graphs - PUSH Model");
    GenerateOverheadGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PUSH);

    BeginHtmlSection(pHtmlStream, "Overhead Graphs - PULL Model");
    GenerateOverheadGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PULL);

    BeginHtmlSection(pHtmlStream, "Work Time Graphs - PUSH Model");
    GenerateWorkTimeGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PUSH);

    BeginHtmlSection(pHtmlStream, "Work Time Graphs - PULL Model");
    GenerateWorkTimeGraphs(lPlotWidth, lPlotHeight, pHtmlStream, PULL);

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;
}

Graph& Benchmark::GenerateStandardChart(size_t pPlotWidth, size_t pPlotHeight, StandardChart& pChart)
{
    if(pChart.groups.empty())
    {
        pChart.graph.reset(new LineGraph(pPlotWidth, pPlotHeight, pChart.xAxis, pChart.yAxis, pChart.curves.size()));
        LineGraph* lGraph = static_cast<LineGraph*>(pChart.graph.get());

        std::vector<StandardCurve>::iterator lIter = pChart.curves.begin(), lEndIter = pChart.curves.end();
        for(size_t i = 0; lIter != lEndIter; ++lIter, ++i)
        {
            lGraph->SetLineName(i, (*lIter).name);
            
            std::vector<std::pair<double, double> >::iterator lInnerIter = (*lIter).points.begin(), lInnerEndIter = (*lIter).points.end();
            for(; lInnerIter != lInnerEndIter; ++lInnerIter)
                lGraph->AddLineDataPoint(i, *lInnerIter);
        }
        
        return *lGraph;
    }
    
    pChart.graph.reset(new RectGraph(pPlotWidth, pPlotHeight, pChart.xAxis, pChart.yAxis, pChart.groups.size(), pChart.curves.size()));
    RectGraph* lGraph = static_cast<RectGraph*>(pChart.graph.get());
    
    std::vector<std::string>::iterator lGroupIter = pChart.groups.begin(), lGroupEndIter = pChart.groups.end();
    for(size_t i = 0; lGroupIter != lGroupEndIter; ++lGroupIter, ++i)
        lGraph->SetGroupName(i, *lGroupIter);
        
    std::vector<StandardCurve>::iterator lIter = pChart.curves.begin(), lEndIter = pChart.curves.end();
    for(size_t i = 0; lIter != lEndIter; ++lIter, ++i)
    {
        lGraph->SetRectName(i, (*lIter).name);
        
        std::vector<std::pair<double, double> >::iterator lInnerIter = (*lIter).points.begin(), lInnerEndIter = (*lIter).points.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
            lGraph->AddRect((size_t)((*lInnerIter).first), Rect(i, 0, i+1, (*lInnerIter).second));
    }

    return *lGraph;
}

void Benchmark::GeneratePerformanceGraphs(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, SchedulingPolicy pPolicy)
{
    std::string lVarying1Name = mConfiguration["Varying1_Name"][0];
    lVarying1Name.append(" --->");

    const char* lOneHostSvpCurveNames[] = {"Sequential", "CPUs", "GPUs", "CPUs+GPUs"};
    StandardChart lOneHostSvpGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->"))); // Svp means Sequential versus Parallel
    lOneHostSvpGraph.SetCurves(4, lOneHostSvpCurveNames);
    
    const char* lOneHostParallelCurveNames[] = {"CPUs", "GPUs", "CPUs+GPUs"};
    StandardChart lOneHostParallelGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    lOneHostParallelGraph.SetCurves(3, lOneHostParallelCurveNames);

    StandardChart lCpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    StandardChart lGpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    StandardChart lCpgAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));   // Cpg means CPUs+GPUs

    std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(size_t lHostIndex = 0; lHostsIter != lHostsEndIter; ++lHostsIter, ++lHostIndex)
    {
        std::stringstream lHostStrStream;
        lHostStrStream << lHostsIter->first << ((lHostsIter->first == 1) ? " Host" : " Hosts") << std::endl;
        
        lCpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lGpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lCpgAllHostsGraph.curves.push_back(lHostStrStream.str());
    }

    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        lOneHostSvpGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lIter->second.first.sequentialTime));
        
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.policy == pPolicy && !lInnerIter->first.multiAssign && !lInnerIter->first.lazyMem)
            {
                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostParallelGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        }

                        lCpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        break;
                        
                    case GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostParallelGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        }
                        
                        lGpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));

                        break;
                        
                    case CPU_PLUS_GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[3].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostParallelGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        }
                        
                        lCpgAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        
                        break;
                        
                    default:
                        throw std::exception();
                }
            }
        }
    }
    
    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostSvpGraph), "Sequential vs. PMLIB - 1 Host");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostParallelGraph), "PMLIB Tasks - 1 Host");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;

    pHtmlStream << "<br>" << std::endl;

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpusAllHostsGraph), "PMLIB CPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lGpusAllHostsGraph), "PMLIB GPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpgAllHostsGraph), "PMLIB CPUs+GPUs - All Hosts");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::GenerateSchedulingModelsGraphs(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    
    bool lFirst = true;
    for(; lIter != lEndIter; ++lIter)
    {
        if(lFirst)
            lFirst = false;
        else
            pHtmlStream << "<br>" << std::endl;

        std::stringstream lVaryingStr;
        lVaryingStr << mConfiguration["Varying1_Name"][0] << " = " << lIter->first.varying1;
        
        StandardChart lCpusGraph(std::auto_ptr<Axis>(new Axis(lVaryingStr.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lGpusGraph(std::auto_ptr<Axis>(new Axis(lVaryingStr.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lCpusPlusGpusGraph(std::auto_ptr<Axis>(new Axis(lVaryingStr.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));

        const char* lCurveNames[] = {"Push", "Pull", "Static Equal", "Static Best"};
        lCpusGraph.SetCurves(4, lCurveNames);
        lGpusGraph.SetCurves(4, lCurveNames);
        lCpusPlusGpusGraph.SetCurves(4, lCurveNames);

        std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
        for(size_t lHostIndex = 0; lHostsIter != lHostsEndIter; ++lHostsIter, ++lHostIndex)
        {
            std::stringstream lHostStrStream;
            lHostStrStream << lHostsIter->first << ((lHostsIter->first == 1) ? " Host" : " Hosts") << std::endl;
            
            lCpusGraph.groups.push_back(lHostStrStream.str());
            lGpusGraph.groups.push_back(lHostStrStream.str());
            lCpusPlusGpusGraph.groups.push_back(lHostStrStream.str());
        }

        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(!lInnerIter->first.multiAssign && !lInnerIter->first.lazyMem)
            {
                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        lCpusGraph.curves[(size_t)(lInnerIter->first.policy)].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                    
                        break;
                        
                    case GPU:
                        lGpusGraph.curves[(size_t)(lInnerIter->first.policy)].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                    
                        break;

                    case CPU_PLUS_GPU:
                        lCpusPlusGpusGraph.curves[(size_t)(lInnerIter->first.policy)].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                    
                        break;
                        
                    default:
                        throw std::exception();
                }
            }
        }
                
        pHtmlStream << "<table align=center>" << std::endl;
        pHtmlStream << "<tr class=horizSpacing>" << std::endl;
        EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpusGraph), "CPUs");
        EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lGpusGraph), "GPUs");
        EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpusPlusGpusGraph), "CPUs+GPUs");
        pHtmlStream << "</tr>" << std::endl;
        pHtmlStream << "</table>" << std::endl;
    }
}

void Benchmark::GenerateLoadBalancingGraphs(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    size_t lCount = 0;

    const char* lChartTitle[] = {"CPUs", "GPUs", "CPUs+GPUs"};

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;

    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end(), lPenultimateIter = lEndIter;
    --lPenultimateIter;
    
    for(lIter = lPenultimateIter; lIter != lEndIter; ++lIter)
//    for(; lIter != lEndIter; ++lIter)
    {
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.multiAssign || lInnerIter->first.lazyMem)
                continue;
            
            if(lInnerIter->first.policy != PUSH && lInnerIter->first.policy != PULL)
                continue;
    
            if(lCount && (lCount % 3 == 0))
            {
                pHtmlStream << "</tr>" << std::endl;
                pHtmlStream << "</table>" << std::endl;
                pHtmlStream << "<br>" << std::endl;
                pHtmlStream << "<table align=center>" << std::endl;
                pHtmlStream << "<tr class=horizSpacing>" << std::endl;
            }
            
            std::stringstream lStr;
            
            if(lInnerIter->first.policy == PUSH)
                lStr << mConfiguration["Varying1_Name"][0] << " = " << lIter->first.varying1 << "; Hosts = " << lInnerIter->first.hosts << " [Push]";
            else
                lStr << mConfiguration["Varying1_Name"][0] << " = " << lIter->first.varying1 << "; Hosts = " << lInnerIter->first.hosts << " [Pull]";                
            
            StandardChart lGraph(std::auto_ptr<Axis>(new Axis(lStr.str(), false)), std::auto_ptr<Axis>(new Axis("Time (in s) --->")));

            std::map<size_t, DeviceStats>::const_iterator lDeviceIter = lInnerIter->second.deviceStats.begin(), lDeviceEndIter = lInnerIter->second.deviceStats.end();
            for(; lDeviceIter != lDeviceEndIter; ++lDeviceIter)
            {
                std::stringstream lDeviceName;
                lDeviceName << "Device " << lDeviceIter->first;
                
                lGraph.curves.push_back(lDeviceName.str());
                (*(lGraph.curves.rbegin())).points.push_back(std::make_pair(0, lDeviceIter->second.lastEventTimings.second));
            }

            lGraph.groups.push_back("");

            EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lGraph), lChartTitle[(size_t)(lInnerIter->first.cluster)]);
            
            ++lCount;
        }
    }

    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::GenerateOverheadGraphs(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, SchedulingPolicy pPolicy)
{
    std::string lVarying1Name = mConfiguration["Varying1_Name"][0];
    lVarying1Name.append(" --->");

    const char* lOneHostSvpCurveNames[] = {"Sequential", "CPUs", "GPUs", "CPUs+GPUs"};  // Svp means sequential versus parallel
    StandardChart lOneHostSvpGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Relative Execution Rate (subtasks/s) --->")));
    lOneHostSvpGraph.SetCurves(4, lOneHostSvpCurveNames);
    
    StandardChart lCpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Rate (in subtasks/s) --->")));
    StandardChart lGpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Rate (in subtasks/s) --->")));
    StandardChart lCpgAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Execution Rate (in subtasks/s) --->")));   // Cpg means CPUs+GPUs

    std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(size_t lHostIndex = 0; lHostsIter != lHostsEndIter; ++lHostsIter, ++lHostIndex)
    {
        std::stringstream lHostStrStream;
        lHostStrStream << lHostsIter->first << ((lHostsIter->first == 1) ? " Host" : " Hosts") << std::endl;
        
        lCpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lGpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lCpgAllHostsGraph.curves.push_back(lHostStrStream.str());
    }
    
    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        lOneHostSvpGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, 1.0));

        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.policy == pPolicy && !lInnerIter->first.multiAssign && !lInnerIter->first.lazyMem)
            {
                double lVal = lInnerIter->second.totalExecutionRate / (lInnerIter->second.subtaskCount / lIter->second.first.sequentialTime);

                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        if(lInnerIter->first.hosts == 1)
                            lOneHostSvpGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        
                        lCpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        break;
                        
                    case GPU:
                        if(lInnerIter->first.hosts == 1)
                            lOneHostSvpGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        
                        lGpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));

                        break;
                        
                    case CPU_PLUS_GPU:
                        if(lInnerIter->first.hosts == 1)
                            lOneHostSvpGraph.curves[3].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        
                        lCpgAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        
                        break;
                        
                    default:
                        throw std::exception();
                }
            }
        }
    }
    
    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostSvpGraph), "Sequential vs. PMLIB - 1 Host");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;

    pHtmlStream << "<br>" << std::endl;

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpusAllHostsGraph), "PMLIB CPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lGpusAllHostsGraph), "PMLIB GPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpgAllHostsGraph), "PMLIB CPUs+GPUs - All Hosts");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::GenerateWorkTimeGraphs(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, SchedulingPolicy pPolicy)
{
    std::string lStat("SUBTASK_EXECUTION");

    std::string lVarying1Name = mConfiguration["Varying1_Name"][0];
    lVarying1Name.append(" --->");

    const char* lOneHostSvpCurveNames[] = {"Sequential", "CPUs", "GPUs", "CPUs+GPUs"};
    StandardChart lOneHostSvpGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Work Time (in s) --->"))); // Svp means Sequential versus Parallel
    lOneHostSvpGraph.SetCurves(4, lOneHostSvpCurveNames);
    
    const char* lOneHostParallelCurveNames[] = {"CPUs", "GPUs", "CPUs+GPUs"};
    StandardChart lOneHostParallelGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Work Time (in s) --->")));
    lOneHostParallelGraph.SetCurves(3, lOneHostParallelCurveNames);

    StandardChart lCpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Work Time (in s) --->")));
    StandardChart lGpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Work Time (in s) --->")));
    StandardChart lCpgAllHostsGraph(std::auto_ptr<Axis>(new Axis(lVarying1Name)), std::auto_ptr<Axis>(new Axis("Work Time (in s) --->")));   // Cpg means CPUs+GPUs

    std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(size_t lHostIndex = 0; lHostsIter != lHostsEndIter; ++lHostsIter, ++lHostIndex)
    {
        std::stringstream lHostStrStream;
        lHostStrStream << lHostsIter->first << ((lHostsIter->first == 1) ? " Host" : " Hosts") << std::endl;
        
        lCpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lGpusAllHostsGraph.curves.push_back(lHostStrStream.str());
        lCpgAllHostsGraph.curves.push_back(lHostStrStream.str());
    }

    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        lOneHostSvpGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lIter->second.first.sequentialTime));
        
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.policy == pPolicy && !lInnerIter->first.multiAssign && !lInnerIter->first.lazyMem)
            {
                double lVal = lInnerIter->second.workTimeStats.find(lStat)->second.first;

                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                            lOneHostParallelGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        }
                        
                        lCpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        break;
                        
                    case GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                            lOneHostParallelGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        }
                        
                        lGpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));

                        break;
                        
                    case CPU_PLUS_GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[3].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                            lOneHostParallelGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        }
                        
                        lCpgAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lVal));
                        
                        break;
                        
                    default:
                        throw std::exception();
                }
            }
        }
    }
    
    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostSvpGraph), "Sequential vs. PMLIB - 1 Host");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostParallelGraph), "PMLIB Tasks - 1 Host");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;

    pHtmlStream << "<br>" << std::endl;

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpusAllHostsGraph), "PMLIB CPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lGpusAllHostsGraph), "PMLIB GPUs - All Hosts");
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lCpgAllHostsGraph), "PMLIB CPUs+GPUs - All Hosts");
    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::EmbedPlot(std::ofstream& pHtmlStream, Graph& pGraph, const std::string& pGraphTitle)
{
    pHtmlStream << "<td>" << std::endl;

    pHtmlStream << "<div class='plotTitle' style='width: " << pGraph.GetWidth() - 2 << "px;'>" << pGraphTitle << "</div>" << std::endl;
    pHtmlStream << "<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='" << pGraph.GetWidth() << "px' height='" << pGraph.GetHeight() << "px'>" << std::endl;
    pHtmlStream << "<rect x='0' y='0' width='100%' height='100%' fill='none' stroke='gray' />" << std::endl;
    pHtmlStream << pGraph.GetSvg();
    pHtmlStream << "</svg>" << std::endl;

    pHtmlStream << "</td>" << std::endl;
}

void Benchmark::ParseResultsFile(const Level1Key& pLevel1Key, const std::string& pResultsFile, size_t pSampleIndex)
{
    std::ifstream lFileStream(pResultsFile.c_str());
    if(lFileStream.fail())
        throw std::exception();

    std::string lLine;
    boost::cmatch lResults;

    if(!strcmp(pResultsFile.c_str() + strlen(pResultsFile.c_str()) - strlen(SEQUENTIAL_FILE_NAME), SEQUENTIAL_FILE_NAME))
    {
        boost::regex lExp("^Serial Task Execution Time = ([0-9.]+)");
        while(std::getline(lFileStream, lLine))
        {
            if(boost::regex_search(lLine.c_str(), lResults, lExp))
            {
                mSamples[pSampleIndex].results[pLevel1Key].first.sequentialTime = atof(std::string(lResults[1]).c_str());
                break;
            }
        }
        
        return;
    }
    
//    if(!strcmp(pResultsFile.c_str() + strlen(pResultsFile.c_str()) - strlen(SINGLE_GPU_FILE_NAME), SINGLE_GPU_FILE_NAME))
//    {
//        std::regex lExp("^Single GPU Task Execution Time = ([0-9.]+)");
//        while(std::getline(lFileStream, lLine))
//        {
//            if(std::regex_search(lLine.c_str(), lResults, lExp))
//            {
//                mSamples[pSampleIndex].results[pLevel1Key].first.singleGpuTime = atof(std::string(lResults[1]).c_str());
//                break;
//            }
//        }
//        
//        return;
//    }

    boost::regex lKeyExp("([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)$");
    if(!boost::regex_search(pResultsFile.c_str(), lResults, lKeyExp))
        throw std::exception();

    Level2Key lLevel2Key((size_t)(atoi(std::string(lResults[1]).c_str())), (enum SchedulingPolicy)(atoi(std::string(lResults[2]).c_str())), (enum clusterType)(atoi(std::string(lResults[3]).c_str())), (bool)(atoi(std::string(lResults[4]).c_str())), (bool)(atoi(std::string(lResults[5]).c_str())));

    if(mSamples[pSampleIndex].hostsMap.find(lLevel2Key.hosts) == mSamples[pSampleIndex].hostsMap.end())
    {
        size_t lSize = mSamples[pSampleIndex].hostsMap.size();
        mSamples[pSampleIndex].hostsMap[lLevel2Key.hosts] = lSize;
    }
    
    size_t lCurrentDevice = 0;
    std::pair<size_t, size_t> lCurrentTask(std::numeric_limits<size_t>::infinity(), std::numeric_limits<size_t>::infinity());  // task originating host and sequence id

    boost::regex lExp2("^Subtask distribution for task \\[([0-9]+), ([0-9]+)\\] under scheduling policy ([0-9]+) ... ");
    boost::regex lExp3("^Device ([0-9]+) Subtasks ([0-9]+)");
    boost::regex lExp4("^Machine ([0-9]+) Subtasks ([0-9]+) CPU-Subtasks ([0-9]+)");
    boost::regex lExp5("^Total Acknowledgements Received ([0-9]+)");
    boost::regex lExp6("^([^ ]+) => Accumulated Time: ([0-9.]+)s; Actual Time = ([0-9.]+)s; Overlapped Time = ([0-9.]+)s");
    boost::regex lExp7("^Device ([0-9]+) - Subtask execution rate = ([0-9.]+); Steal attemps = ([0-9]+); Successful steals = ([0-9]+); Failed steals = ([0-9]+)");
    boost::regex lExp8("^Parallel Task ([0-9]+) Execution Time = ([0-9.]+) \\[Scheduling Policy: ([0-9]+)\\]");
    boost::regex lExp9("^PMLIB \\[Host ([0-9]+)\\] Event Timeline Device ([0-9]+)");
    boost::regex lExp10("^Task \\[([0-9]+), ([0-9]+)\\] Subtask ([0-9]+) ([0-9.]+) ([0-9.]+)");
    
    while(std::getline(lFileStream, lLine))
    {
        if(boost::regex_search(lLine.c_str(), lResults, lExp10))
        {
            size_t lSubtaskId = atoi(std::string(lResults[3]).c_str());
            double lStartTime = atof(std::string(lResults[4]).c_str());
            double lEndTime = atof(std::string(lResults[5]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lCurrentDevice].eventTimeline[lSubtaskId] = std::make_pair(lStartTime, lEndTime);
            if(lEndTime > mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lCurrentDevice].lastEventTimings.second)
                mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lCurrentDevice].lastEventTimings = std::make_pair(lStartTime, lEndTime);
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp2))
        {
            if(lLevel2Key.policy != (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str())))
                throw std::exception();
            
            lCurrentTask.first = atoi(std::string(lResults[1]).c_str());
            lCurrentTask.second = atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp3))
        {
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[atoi(std::string(lResults[1]).c_str())].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp4))
        {
            size_t lMachine = atoi(std::string(lResults[1]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].machineStats[lMachine].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].machineStats[lMachine].cpuSubtasks = atoi(std::string(lResults[3]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp5))
        {
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].subtaskCount = atoi(std::string(lResults[1]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp6))
        {
            std::string lCriterion(lResults[1]);

            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].workTimeStats[lCriterion].first = atof(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].workTimeStats[lCriterion].second = atof(std::string(lResults[3]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp7))
        {
            size_t lDevice = atoi(std::string(lResults[1]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lDevice].subtaskExecutionRate = atoi(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lDevice].stealAttempts = atoi(std::string(lResults[3]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lDevice].stealSuccesses = atoi(std::string(lResults[4]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].deviceStats[lDevice].stealFailures = atoi(std::string(lResults[5]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].totalExecutionRate += atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp8))
        {
            if(lLevel2Key.policy != (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str())))
                throw std::exception();

            if(4 + (size_t)lLevel2Key.cluster != (size_t)(atoi(std::string(lResults[1]).c_str())))
                throw std::exception();

            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].serialComparisonResult = true;  //(bool)(!strcmp(std::string(lResults[4]).c_str(), "Passed"));
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].execTime = atof(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp9))
        {
            lCurrentDevice = atoi(std::string(lResults[2]).c_str());
        }
    }

    lFileStream.close();
}

void Benchmark::ExecuteInstance(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pUnderscoreSeparatedVaryingsStr, size_t pSampleIndex)
{
    std::cout << "Running benchmark " << mName << " on " << pHosts.c_str() << " hosts with varyings " << pSpaceSeparatedVaryingsStr << " [Sample " << pSampleIndex << "] ..." << std::endl;

    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);
    
    std::stringstream lSampleIndex;
    lSampleIndex << "sample_" << pSampleIndex;

    std::string lOutputFolder(GetBasePath());
    lOutputFolder.append(lSeparator);
    lOutputFolder.append("analyzers");
    lOutputFolder.append(lSeparator);
    lOutputFolder.append("results");
    
    CreateDir(lOutputFolder);
    
    lOutputFolder.append(lSeparator);
    lOutputFolder.append("intermediate");
    
    CreateDir(lOutputFolder);

    lOutputFolder.append(lSeparator);
    lOutputFolder.append(mName);

    CreateDir(lOutputFolder);

    lOutputFolder.append(lSeparator);
    lOutputFolder.append(pUnderscoreSeparatedVaryingsStr);
    
    CreateDir(lOutputFolder);
    
    lOutputFolder.append(lSeparator);
    lOutputFolder.append(lSampleIndex.str());
    
    CreateDir(lOutputFolder);

    ExecuteSample(pHosts, pSpaceSeparatedVaryingsStr, lOutputFolder);
}

void Benchmark::ExecuteSample(const std::string& pHosts, const std::string& pSpaceSeparatedVaryingsStr, const std::string& pOutputFolder)
{
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    /* Generate sequential task output */
    std::string lSequentialFile(pOutputFolder);
    lSequentialFile.append(lSeparator);
    lSequentialFile.append(SEQUENTIAL_FILE_NAME);
    
    std::ifstream lSequentialFileStream(lSequentialFile.c_str());
    
    if(lSequentialFileStream.fail())
    {
        std::stringstream lStream;
        lStream << "source ~/.pmlibrc; ";
        lStream << "mpirun -n " << 1 << " " << mExecPath << " 2 0 0 " << pSpaceSeparatedVaryingsStr;
        lStream << " 2>&1 > " << lSequentialFile.c_str();

        system(lStream.str().c_str());
    }
    else
    {
        lSequentialFileStream.close();
    }
    
    /* Generate single GPU task output */
//    std::string lSingleGpuFile(pOutputFolder);
//    lSingleGpuFile.append(lSeparator);
//    lSingleGpuFile.append(SINGLE_GPU_FILE_NAME);
//    
//    std::ifstream lSingleGpuFileStream(lSingleGpuFile.c_str());
//    
//    if(lSingleGpuFileStream.fail())
//    {
//        std::stringstream lStream;
//        lStream << "source ~/.pmlibrc; ";
//        lStream << "mpirun -n " << 1 << " " << mExecPath << " 3 0 0 " << pSpaceSeparatedVaryingsStr;
//        lStream << " 2>&1 > " << lSequentialFile.c_str();
//        
//        system(lStream.str().c_str());
//    }
//    else
//    {
//        lSingleGpuFileStream.close();
//    }
    
    const std::string lMpiOptionsStr("Mpi_Options");
    std::vector<std::string>& lMpiOptionsVector = GetGlobalConfiguration()[lMpiOptionsStr];
    
    const std::string& lMpiOptions = lMpiOptionsVector.empty() ? std::string("") : lMpiOptionsVector[0];

    /* Generate PMLIB tasks output */
    for(size_t i = 0; i < (size_t)MAX_SCHEDULING_POLICY; ++i)
    {
        for(size_t j = 0; j < (size_t)MAX_CLUSTER_TYPE; ++j)
        {
            for(size_t k = 0; k <= 1; ++k) // Multi Assign
            {
                for(size_t l = 0; l <= 1; ++l)  // Lazy Mem
                {
                    if(k == 1 || l == 1)
                        continue;
                    
                    std::stringstream lOutputFile;
                    
                    lOutputFile << pOutputFolder << lSeparator << pHosts << "_" << i << "_" << j << "_" << k << "_" << l;

                    std::ifstream lFileStream(lOutputFile.str().c_str());

                    if(lFileStream.fail())
                    {
                        std::stringstream lStream;
                        lStream << "source ~/.pmlibrc; ";
                        lStream << "mpirun -n " << pHosts << " " << lMpiOptions << " " << mExecPath << " 0 " << 4+j << " " << i << " " << pSpaceSeparatedVaryingsStr;
                        lStream << " 2>&1 > " << lOutputFile.str().c_str();

                        system(lStream.str().c_str());
                    }
                    else
                    {
                        lFileStream.close();
                    }
                }
            }
        }
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
    lTestSuitePath.append("testSuite");

    DIR* lDir = opendir(lTestSuitePath.c_str());
    if(lDir)
    {
        std::string lBenchmarksStr("Benchmarks");
        const std::vector<std::string>& lSelectiveBenchmarks = GetGlobalConfiguration()[lBenchmarksStr];
        
        struct dirent* lEntry;
        while((lEntry = readdir(lDir)) != NULL)
        {
            if(lEntry->d_type == DT_DIR && strlen(lEntry->d_name))
            {
                std::string lName(lEntry->d_name);
                
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

    if(mConfiguration[std::string("Varying_1")].empty() || mConfiguration[std::string("Benchmark_Name")].empty() || mConfiguration[std::string("Varying1_Name")].empty()
       || (!mConfiguration[std::string("Varying_2")].empty() && mConfiguration[std::string("Varying2_Name")].empty()))
        throw std::string("Mandatory configurations not defined in conf file ") + lConfPath.c_str();
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
    boost::regex lExp1("^[ \t]*#");
    boost::regex lExp2("^[ \t]*$");
    boost::regex lExp3("^[ \t]*(.*)=[ \t]*(.*)$");
    boost::regex lExp4("[ \t]+$");
    boost::regex lExp5("[ \t]+");
    boost::cmatch lResults;

    while(std::getline(lFileStream, lLine))
    {
        if(!boost::regex_search(lLine.begin(), lLine.end(), lExp1) && !boost::regex_search(lLine.begin(), lLine.end(), lExp2))
        {
            if(boost::regex_search(lLine.c_str(), lResults, lExp3))
            {
                std::string lKey(lResults[1]);
                std::string lValue(lResults[2]);
                
                lKey = boost::regex_replace(lKey, lExp4, std::string(""));
                lValue = boost::regex_replace(lValue, lExp4, std::string(""));
                
                std::string lValueStr = boost::regex_replace(lValue, lExp5, std::string(" "));
                
                if(lKey == std::string("Mpi_Options") || lKey == std::string("Benchmark_Name")
                || lKey == std::string("Varying1_Name") || lKey == std::string("Varying2_Name"))
                {
                    pPairs[lKey].push_back(lValueStr);
                }
                else
                {
                    std::stringstream lStream(lValueStr);
                    std::string lTempBuf;
                    
                    while(lStream >> lTempBuf)
                        pPairs[lKey].push_back(lTempBuf);
                }
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



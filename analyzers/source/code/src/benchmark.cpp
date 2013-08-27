
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
#include <pthread.h>
#include <signal.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <sys/wait.h>

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

#define HANDLE_BENCHMARK_HANGS

#define SEQUENTIAL_FILE_NAME "sequential"
#define SINGLE_GPU_FILE_NAME "singleGpu"

#define SAMPLE_COUNT 3
#define TIMEOUT_IN_SECS 600

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

class SampleFinder
{
public:
    SampleFinder(bool pMedianSample)
    : mMedianSample(pMedianSample)
    {}
    
    void AddData(double pData)
    {
        mData.insert(std::make_pair(pData, mData.size()));
    }
    
    size_t GetSampleIndex()
    {
        if(mMedianSample)
        {
            size_t lMedian = mData.size() / 2;
            
            std::set<std::pair<double, size_t> >::iterator lIter = mData.begin();
            std::advance(lIter, lMedian);
            
            return (*lIter).second;
        }
        
        return mData.begin()->second;   // best sample
    }
    
private:
    bool mMedianSample;
    std::set<std::pair<double, size_t> > mData;
};

Benchmark::Benchmark(const std::string& pName)
: mName(pName)
{
    mHostsSetVector.resize(SAMPLE_COUNT);
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

            lVarying1Iter(std::string(""), std::string(""));
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
    
    for(size_t i = 0; i < SAMPLE_COUNT; ++i)
    {
        std::set<size_t>::iterator lIter = mHostsSetVector[i].begin(), lEndIter = mHostsSetVector[i].end();
        for(; lIter != lEndIter; ++lIter)
        {
            size_t lSize = mSamples[i].hostsMap.size();
            mSamples[i].hostsMap[(*lIter)] = lSize;
        }
    }
    
    SelectSample(false);
    BuildInnerTaskVector();

    GenerateAnalysis();
}

/* If pMedianSample is false, then it selects best sample */
void Benchmark::SelectSample(bool pMedianSample)
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

                std::cout << lLevel2Key.hosts << " " << lLevel2Key.policy << " " << lLevel2Key.cluster << " " << lLevel2Key.multiAssign << " " << lLevel2Key.lazyMem << " " << lLevel2Key.overlapComputeCommunication << std::endl;
                
                const std::map<Level2InnerTaskKey, Level2InnerTaskValue>& lLevel2Value = lInnerIter->second.innerTaskMap;
                std::map<Level2InnerTaskKey, Level2InnerTaskValue>::const_iterator lInnerTaskIter = lLevel2Value.begin(), lInnerTaskEndIter = lLevel2Value.end();
                for(; lInnerTaskIter != lInnerTaskEndIter; ++lInnerTaskIter)
                {
                    const Level2InnerTaskKey& lInnerTask = lInnerTaskIter->first;

                    std::cout << lInnerTask.originatingHost << " " << lInnerTask.taskSequenceId << std::endl;
                }
            }
        }
    }
#endif

    BenchmarkResults::mapType::const_iterator lIter = mSamples[0].results.begin(), lEndIter = mSamples[0].results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const Level1Key& lLevel1Key = lIter->first;
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;

        SampleFinder lSequentialSampleFinder(pMedianSample), lSingleGpuSampleFinder(pMedianSample);
        
        for(size_t i = 0; i < SAMPLE_COUNT; ++i)
        {
            if(mSamples[i].results.find(lLevel1Key) == mSamples[i].results.end())
                throw std::exception();

            lSequentialSampleFinder.AddData(mSamples[i].results[lLevel1Key].first.sequentialTime);
            lSingleGpuSampleFinder.AddData(mSamples[i].results[lLevel1Key].first.singleGpuTime);
        }
        
        mResults.results[lLevel1Key].first.sequentialTime = mSamples[lSequentialSampleFinder.GetSampleIndex()].results[lLevel1Key].first.sequentialTime;
        mResults.results[lLevel1Key].first.singleGpuTime = mSamples[lSingleGpuSampleFinder.GetSampleIndex()].results[lLevel1Key].first.singleGpuTime;
        
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            const Level2Key& lLevel2Key = lInnerIter->first;
            
            SampleFinder lParallelSampleFinder(pMedianSample);
            for(size_t i = 0; i < SAMPLE_COUNT; ++i)
            {
                if(mSamples[i].results[lLevel1Key].second.find(lLevel2Key) == mSamples[i].results[lLevel1Key].second.end())
                    throw std::exception();

                lParallelSampleFinder.AddData(mSamples[i].results[lLevel1Key].second[lLevel2Key].execTime);
            }
            
            mResults.results[lLevel1Key].second[lLevel2Key] = mSamples[lParallelSampleFinder.GetSampleIndex()].results[lLevel1Key].second[lLevel2Key];
        }
    }
    
    mResults.hostsMap = mSamples[0].hostsMap;
}

void Benchmark::BuildInnerTaskVector()
{
    BenchmarkResults::mapType::const_iterator lLevel1Iter = mResults.results.begin();
    std::map<Level2Key, Level2Value>::const_iterator lLevel2Iter = lLevel1Iter->second.second.begin();

    std::map<Level2InnerTaskKey, Level2InnerTaskValue>::const_iterator lIter = lLevel2Iter->second.innerTaskMap.begin(), lEndIter = lLevel2Iter->second.innerTaskMap.end();
    for(; lIter != lEndIter; ++lIter)
        mInnerTasks.push_back(lIter->first);
    
    if((mInnerTasks.size() > 1) && (mConfiguration["Inner_Task_Names"].empty() || (mInnerTasks.size() != mConfiguration["Inner_Task_Names"].size())))
        throw std::exception();
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
    lHtmlStream << "<head><center>" << std::endl;
    lHtmlStream << "<b><u>" << mConfiguration["Benchmark_Name"][0] << "</u></b><br><br><a href=\"pmlibResults.html\">Home</a>" << std::endl;
    lHtmlStream << "</center></head>" << std::endl;

    lHtmlStream << "<body>" << std::endl;
    lHtmlStream << "<style type='text/css'> .boxed { border:1px solid #000; background:lightgray; display:inline-block; } </style>" << std::endl;
    lHtmlStream << "<br><center><div class='boxed'><b>&nbsp;&nbsp;One Subtask&nbsp;&nbsp;</b>" << std::endl;
    lHtmlStream << "<br>&nbsp;&nbsp;" << mConfiguration["Subtask_Definition"][0] << "&nbsp;&nbsp;" << std::endl;
    
    if(!mConfiguration["Other_Information"].empty())
        lHtmlStream << "<br>&nbsp;&nbsp;" << mConfiguration["Other_Information"][0] << "&nbsp;&nbsp;" << std::endl;
    
    lHtmlStream << "</div></center>" << std::endl;
    
    std::vector<size_t> lRadioSetCount;
    
    GeneratePreControlCode(lHtmlStream);
    GenerateTable(lHtmlStream, lRadioSetCount);
    GeneratePlots(lHtmlStream, lRadioSetCount);
    GeneratePostControlCode(lHtmlStream, lRadioSetCount);
    
    lHtmlStream << "</body>" << std::endl;
    lHtmlStream << "</html>" << std::endl;

    lHtmlStream.close();
}

void Benchmark::WriteTopLevelHtmlPage(const std::vector<Benchmark>& pBenchmarks)
{
    std::cout << "Writing top level html page ..." << std::endl;
    
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
    lHtmlPath += lSeparator + std::string("pmlibResults.html");
    
    std::ofstream lHtmlStream;

    lHtmlStream.open(lHtmlPath.c_str());
    if(lHtmlStream.fail())
        throw std::exception();

    lHtmlStream << "<html>" << std::endl;
    lHtmlStream << "<head><center><b><u> PMLIB Results </u></b></center></head>" << std::endl;
    lHtmlStream << "<body><br><br>" << std::endl;

    lHtmlStream << "<table align=center border=1>" << std::endl;
    std::vector<Benchmark>::const_iterator lIter = pBenchmarks.begin(), lEndIter = pBenchmarks.end();
    for(; lIter != lEndIter; ++lIter)
    {
        lHtmlStream << "<tr>" << std::endl;
        
        lHtmlStream << "<td align=center>" << std::endl;
        lHtmlStream << "<a href=\"" << (*lIter).mName << ".html\">" << const_cast<Benchmark&>(*lIter).mConfiguration["Benchmark_Name"][0] << "</a>" << std::endl;
        lHtmlStream << "</td>" << std::endl;
        
        lHtmlStream << "</tr>" << std::endl;
    }
    
    lHtmlStream << "</table>" << std::endl;
    
    lHtmlStream << "</body>" << std::endl;
    lHtmlStream << "</html>" << std::endl;

    lHtmlStream.close();    
}

void Benchmark::CopyResourceFiles()
{
    std::cout << "Copying resource files ..." << std::endl;

    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lDirPath(GetBasePath());
    lDirPath.append(lSeparator);
    lDirPath.append("analyzers");
    lDirPath.append(lSeparator);
    lDirPath.append("thirdparty");
    lDirPath.append(lSeparator);
    lDirPath.append("jquery");

    std::string lDestDirPath(GetBasePath());
    lDestDirPath.append(lSeparator);
    lDestDirPath.append("analyzers");
    lDestDirPath.append(lSeparator);
    lDestDirPath.append("results");
    lDestDirPath.append(lSeparator);
    lDestDirPath.append("htmls");

    DIR* lSrcDir = opendir(lDirPath.c_str());
    if(!lSrcDir)
        throw std::exception();
    
    struct dirent* lDirEntry;
    while((lDirEntry = readdir(lSrcDir)) != NULL)
    {
        if(lDirEntry->d_type != DT_DIR && strlen(lDirEntry->d_name))
        {
            std::string lSrcFilePath(lDirPath);
            lSrcFilePath.append(lSeparator);
            lSrcFilePath.append(std::string(lDirEntry->d_name));

            std::string lDestFilePath(lDestDirPath);
            lDestFilePath.append(lSeparator);
            lDestFilePath.append(std::string(lDirEntry->d_name));
            
            CopyFile(lSrcFilePath, lDestFilePath);
        }
    }
    
    closedir(lSrcDir);
}

void Benchmark::CopyFile(const std::string& pSrcFile, const std::string& pDestFile)
{
    std::ifstream lSrcStream(pSrcFile.c_str());
    std::ofstream lDestStream(pDestFile.c_str());
    
    if(lSrcStream.fail())
        throw std::exception();

    lDestStream << lSrcStream.rdbuf();
}

void Benchmark::BeginHtmlSection(std::ofstream &pHtmlStream, const std::string& pSectionName)
{
    pHtmlStream << std::endl << "<br><br><hr><div style='text-align:center'><b>" << pSectionName << "</b></div><hr>" << std::endl;
    pHtmlStream << std::endl;
}

void Benchmark::GeneratePreControlCode(std::ofstream& pHtmlStream)
{
    pHtmlStream << std::endl;
    pHtmlStream << "<link rel='stylesheet' href='jquery-ui-1.10.1.css' />" << std::endl;
    pHtmlStream << "<script src='jquery-1.10.1.js'></script>" << std::endl;
    pHtmlStream << "<script src='jquery-ui-1.10.1.js'></script>" << std::endl;

    pHtmlStream << std::endl;
    pHtmlStream << "<script>" << std::endl;
    pHtmlStream << "\
        function checkPanelState(panelName) \n\
        { \n\
            var className = '.' + panelName + '_toggler'; \n\
            $(className).hide(); \n\
            \n\
            var panelIdName = '#' + panelName; \n\
            var radioSetCount = $(panelIdName).attr('value'); \n\
            \n\
            var tableIdName = '#' + panelName + '_table'; \n\
            for(var i = 0; i < radioSetCount; ++i) \n\
            { \n\
                var radioSetIdName = '#' + panelName + '_rs' + (i+1) + ' input:radio:checked'; \n\
                var selectedId = $(radioSetIdName).attr('id'); \n\
                var selectedIndex = selectedId.substr(selectedId.lastIndexOf('_') + 1); \n\
            \n\
                tableIdName += '_' + selectedIndex; \n\
            } \n\
            \n\
            $(tableIdName).show(); \n\
        } \n\
        \n\
        function checkState(radioButton) \n\
        { \n\
            var buttonName = radioButton.id; \n\
            var panelName = buttonName.substring(0, buttonName.indexOf('_')); \n\
            \n\
            checkPanelState(panelName); \n\
        }\n" << std::endl;

    pHtmlStream << "</script>" << std::endl;
    pHtmlStream << std::endl;

    pHtmlStream << "<style type='text/css'> .selectionGroup { border:2px solid #000; display:inline-block; } </style>" << std::endl;
    pHtmlStream << "<style type='text/css'> .selectionName { background:#564; color:#FFF; display:inline-block; margin-left:6px; margin-right:4px; } </style>" << std::endl;
    pHtmlStream << "<style type='text/css'> .selectionBox { margin-top: 4px; margin-bottom: 4px; } </style>" << std::endl;
    pHtmlStream << "<style type='text/css'> div.plotTitle { border-width:1px; border-style:solid; border-color:gray; background:lightgray; text-align:center; } </style>" << std::endl;
    pHtmlStream << "<style type='text/css'> tr.horizSpacing > td { padding-left: 2em; padding-right: 2em; } </style>" << std::endl;
}

void Benchmark::GeneratePostControlCode(std::ofstream& pHtmlStream, const std::vector<size_t>& pRadioSetCount)
{
    pHtmlStream << "<script>" << std::endl;
    pHtmlStream << "$(function() {" << std::endl;
    
    std::vector<size_t>::const_iterator lIter = pRadioSetCount.begin(), lEndIter = pRadioSetCount.end();
    for(size_t lPanelIndex = 1; lIter != lEndIter; ++lIter, ++lPanelIndex)
    {
        for(size_t i = 1; i <= (*lIter); ++i)
            pHtmlStream << "    $(\"#p" << lPanelIndex << "_rs" << i << "\").buttonset();" << std::endl;

        if((*lIter))
            pHtmlStream << "    checkPanelState('p" << lPanelIndex << "');" << std::endl;
    }

    pHtmlStream << "});" << std::endl;
    pHtmlStream << "</script>" << std::endl;
}

void Benchmark::GenerateTable(std::ofstream& pHtmlStream, std::vector<size_t>& pRadioSetCount)
{
    BeginHtmlSection(pHtmlStream, "Experimental Results");

    panelConfigurationType lPanelConf;

    const char* lBaselines[] = {"Sequential", "Single GPU"};
    lPanelConf.push_back(std::make_pair("Baseline", std::vector<std::string>(lBaselines, lBaselines + sizeof(lBaselines)/sizeof(lBaselines[0]))));
    
    const char* lDisplay[] = {"Absolute Values", "Speedup"};
    lPanelConf.push_back(std::make_pair("Display", std::vector<std::string>(lDisplay, lDisplay + (sizeof(lDisplay)/sizeof(lDisplay[0])))));
    
    const char* lOverlap[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Overlap&nbsp;Comp/Comm", std::vector<std::string>(lOverlap, lOverlap + (sizeof(lOverlap)/sizeof(lOverlap[0])))));

    size_t lPanelIndex = pRadioSetCount.size() + 1;

    pHtmlStream << "<div id='p" << lPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(lPanelIndex, lPanelConf, pHtmlStream);

    for(int baseline = 1; baseline <= 2; ++baseline)
    {
        for(int display = 1; display <= 2; ++display)
        {
            for(int overlap = 1; overlap <= 2; ++overlap)
            {
                pHtmlStream << "<div class='p" << lPanelIndex << "_toggler' id='p" << lPanelIndex << "_table_" << baseline << "_" << display << "_" << overlap << "' style='display:none'>" << std::endl;
                GenerateTableInternal(pHtmlStream, (baseline == 1), (display == 1), (overlap == 1));
                pHtmlStream << "</div>" << std::endl;
            }
        }
    }

    pHtmlStream << "</div>" << std::endl;
    
    pRadioSetCount.push_back(lPanelConf.size());
}

void Benchmark::GenerateTableInternal(std::ofstream& pHtmlStream, bool pSequential, bool pAbsoluteValues, bool pOverlap)
{
    pHtmlStream << "<table align=center border=1>" << std::endl;
    pHtmlStream << "<tr>" << std::endl;

    std::string lVarying2Str("Varying_2");
    bool lVarying2Defined = !(mConfiguration[lVarying2Str].empty());
    
    if(lVarying2Defined)
    {
        boost::regex lSpaceExp("[ \t]");
        std::string lVarying1Name = boost::regex_replace(mConfiguration["Varying1_Name"][0], lSpaceExp, std::string("<br>"));
        std::string lVarying2Name = boost::regex_replace(mConfiguration["Varying2_Name"][0], lSpaceExp, std::string("<br>"));
        
        pHtmlStream << "<th rowSpan=4>" << lVarying1Name << "</th>" << std::endl;
        pHtmlStream << "<th rowSpan=4>" << lVarying2Name << "</th>" << std::endl;
    }
    else
    {
        boost::regex lSpaceExp("[ \t]");
        std::string lVarying1Name = boost::regex_replace(mConfiguration["Varying1_Name"][0], lSpaceExp, std::string("<br>"));
        
        pHtmlStream << "<th rowSpan=4>" << lVarying1Name << "</th>" << std::endl;
    }
    
    size_t lVarying2Count = (lVarying2Defined ? mConfiguration[lVarying2Str].size() : 0);
    
    if(pSequential)
        pHtmlStream << "<th rowSpan=4>Sequential<br>Time<br>(in s)</th>" << std::endl;
    else
        pHtmlStream << "<th rowSpan=4>Single&nbsp;GPU<br>Time<br>(in s)</th>" << std::endl;
    
    pHtmlStream << "<th rowSpan=4>Parallel<br>Scheduling<br>Policy</th>" << std::endl;

    pHtmlStream << "<th colSpan=" << 6 * mResults.hostsMap.size() << ">Parallel Time (in s)</th>" << std::endl;
    pHtmlStream << "</tr>" << std::endl << "<tr>" << std::endl;
    
    std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
        pHtmlStream << "<th colSpan=6>" << lHostsIter->first << "&nbsp;" << ((lHostsIter->first == 1) ? "Host" : "Hosts") << "</th>" << std::endl;

    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "<tr>" << std::endl;

    for(lHostsIter = mResults.hostsMap.begin(); lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        pHtmlStream << "<th colspan=3>No Multi Assign</th>" << std::endl;
        pHtmlStream << "<th colspan=3>Multi Assign</th>" << std::endl;
    }

    pHtmlStream << "</tr>" << std::endl;
    pHtmlStream << "<tr>" << std::endl;
    
    for(lHostsIter = mResults.hostsMap.begin(); lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        for(size_t i = 0; i < 2; ++i)
        {
            pHtmlStream << "<th>CPUs</th>" << std::endl;
            pHtmlStream << "<th>GPUs</th>" << std::endl;
            pHtmlStream << "<th>Both</th>" << std::endl;
        }
    }

    pHtmlStream << "</tr>" << std::endl;

    const std::string lStaticBestStr("Generate_Static_Best");
    std::vector<std::string>& lStaticBestVector = GetGlobalConfiguration()[lStaticBestStr];
    bool lGenerateStaticBest = (!lStaticBestVector.empty() && !lStaticBestVector[0].compare(std::string("false"))) ? false : true;

    size_t lLastVarying1Value = 0;
    
    BenchmarkResults::mapType::iterator lLevel1Iter = mResults.results.begin(), lLevel1EndIter = mResults.results.end();
    for(; lLevel1Iter != lLevel1EndIter; ++lLevel1Iter)
    {
        pHtmlStream << "<tr>" << std::endl;
        
        if(!lVarying2Defined || lLevel1Iter->first.varying1 != lLastVarying1Value)
            pHtmlStream << "<td align=center rowspan = " << ((lVarying2Count == 0) ? 1 : lVarying2Count) << ">" << lLevel1Iter->first.varying1 << "</td>" << std::endl;
        
        lLastVarying1Value = lLevel1Iter->first.varying1;
        
        if(lVarying2Defined)
            pHtmlStream << "<td align=center>" << lLevel1Iter->first.varying2 << "</td>" << std::endl;

        if(pSequential)
            pHtmlStream << "<td align=center>" << lLevel1Iter->second.first.sequentialTime << "</td>" << std::endl;
        else
            pHtmlStream << "<td align=center>" << lLevel1Iter->second.first.singleGpuTime << "</td>" << std::endl;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr bgcolor=lightgray><td align=center>Push</td></tr>" << std::endl;
        pHtmlStream << "<tr bgcolor=lightgray><td align=center>Pull</td></tr>" << std::endl;
        pHtmlStream << "<tr bgcolor=lightgray><td align=center>Static&nbsp;Equal</td></tr>" << std::endl;
        if(lGenerateStaticBest)
            pHtmlStream << "<th><td>Static&nbsp;Best</td></th>" << std::endl;

        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

        std::map<size_t, size_t>::iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
        for(; lHostsIter != lHostsEndIter; ++lHostsIter)
        {
            EmbedResultsInTable(pHtmlStream, lLevel1Iter, lHostsIter->first, false, lGenerateStaticBest, pSequential, pAbsoluteValues, pOverlap);
            EmbedResultsInTable(pHtmlStream, lLevel1Iter, lHostsIter->first, true, lGenerateStaticBest, pSequential, pAbsoluteValues, pOverlap);
        }

        pHtmlStream << "</tr>" << std::endl;
    }
    
    pHtmlStream << "</table>" << std::endl;
}

void Benchmark::EmbedResultsInTable(std::ofstream& pHtmlStream, BenchmarkResults::mapType::iterator pLevel1Iter, size_t pHosts, bool pMultiAssign, bool pGenerateStaticBest, bool pSequential, bool pAbsoluteValues, bool pOverlap)
{
    Level2Key lKey1(pHosts, PUSH, CPU, pMultiAssign, false, pOverlap);
    Level2Key lKey2(pHosts, PULL, CPU, pMultiAssign, false, pOverlap);
    Level2Key lKey3(pHosts, STATIC_EQUAL, CPU, false, false, false);
    Level2Key lKey4(pHosts, STATIC_BEST, CPU, false, false, false);
    
    Level2Key lKey5(pHosts, PUSH, GPU, pMultiAssign, false, pOverlap);
    Level2Key lKey6(pHosts, PULL, GPU, pMultiAssign, false, pOverlap);
    Level2Key lKey7(pHosts, STATIC_EQUAL, GPU, false, false, false);
    Level2Key lKey8(pHosts, STATIC_BEST, GPU, false, false, false);

    Level2Key lKey9(pHosts, PUSH, CPU_PLUS_GPU, pMultiAssign, false, pOverlap);
    Level2Key lKey10(pHosts, PULL, CPU_PLUS_GPU, pMultiAssign, false, pOverlap);
    Level2Key lKey11(pHosts, STATIC_EQUAL, CPU_PLUS_GPU, false, false, false);
    Level2Key lKey12(pHosts, STATIC_BEST, CPU_PLUS_GPU, false, false, false);
    
    if(pAbsoluteValues)
    {
        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey1].execTime << "</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey2].execTime << "</td></tr>" << std::endl;
        
        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey3].execTime << "</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey4].execTime << "</td></tr>" << std::endl;
        }

        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey5].execTime << "</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey6].execTime << "</td></tr>" << std::endl;

        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey7].execTime << "</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey8].execTime << "</td></tr>" << std::endl;
        }

        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey9].execTime << "</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey10].execTime << "</td></tr>" << std::endl;

        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey11].execTime << "</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << pLevel1Iter->second.second[lKey12].execTime << "</td></tr>" << std::endl;
        }
        
        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;
    }
    else
    {
        double lFactor = 1.0;

        if(pSequential)
            lFactor = pLevel1Iter->second.first.sequentialTime;
        else
            lFactor = pLevel1Iter->second.first.singleGpuTime;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey1].execTime << "x</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey2].execTime << "x</td></tr>" << std::endl;
        
        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey3].execTime << "x</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey4].execTime << "x</td></tr>" << std::endl;
        }

        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey5].execTime << "x</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey6].execTime << "x</td></tr>" << std::endl;

        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey7].execTime << "x</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey8].execTime << "x</td></tr>" << std::endl;
        }

        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;

        pHtmlStream << "<td><table align=center>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey9].execTime << "x</td></tr>" << std::endl;
        pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey10].execTime << "x</td></tr>" << std::endl;

        if(pMultiAssign || pOverlap)
        {
            pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>N.A.</td></tr>" << std::endl;
        }
        else
        {
            pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey11].execTime << "x</td></tr>" << std::endl;
            if(pGenerateStaticBest)
                pHtmlStream << "<tr><td>" << lFactor / pLevel1Iter->second.second[lKey12].execTime << "x</td></tr>" << std::endl;
        }
        
        pHtmlStream << "</table>" << std::endl << "</td>" << std::endl;        
    }
}

void Benchmark::GeneratePlots(std::ofstream& pHtmlStream, std::vector<size_t>& pRadioSetCount)
{
    size_t lPlotWidth = 400;
    size_t lPlotHeight = 400;

    size_t lPanelIndex = pRadioSetCount.size() + 1;
    
    BeginHtmlSection(pHtmlStream, "Performance Graphs");
    pRadioSetCount.push_back( GeneratePerformanceGraphs(lPanelIndex, lPlotWidth, lPlotHeight, pHtmlStream) );

    BeginHtmlSection(pHtmlStream, "Scheduling Models Comparison");
    pRadioSetCount.push_back( GenerateSchedulingModelsGraphs(lPanelIndex + 1, lPlotWidth, lPlotHeight, pHtmlStream) );

    BeginHtmlSection(pHtmlStream, "Load Balancing Graphs");
    pRadioSetCount.push_back( GenerateLoadBalancingGraphs(lPanelIndex + 2, lPlotWidth, lPlotHeight, pHtmlStream) );

    BeginHtmlSection(pHtmlStream, "Multi Assign Comparison Graphs");
    pRadioSetCount.push_back( GenerateMultiAssignComparisonGraphs(lPanelIndex + 3, lPlotWidth, lPlotHeight, pHtmlStream) );

    BeginHtmlSection(pHtmlStream, "Compute Communication Overlap Comparison Graphs");
    pRadioSetCount.push_back( GenerateOverlapComparisonGraphs(lPanelIndex + 4, lPlotWidth, lPlotHeight, pHtmlStream) );
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

void Benchmark::GenerateSelectionGroup(size_t pPanelIndex, const panelConfigurationType& pPanelConf, std::ofstream& pHtmlStream)
{
    pHtmlStream << "<center>" << std::endl;
    pHtmlStream << "<div align=center class=selectionGroup>" << std::endl;
    pHtmlStream << std::endl;

    panelConfigurationType::const_iterator lIter = pPanelConf.begin(), lEndIter = pPanelConf.end();
    for(size_t lRadioSetIndex = 1; lIter != lEndIter; ++lIter, ++lRadioSetIndex)
    {
        pHtmlStream << "<div align=left id='p" << pPanelIndex << "_rs" << lRadioSetIndex << "' class=selectionBox>" << std::endl;
        pHtmlStream << "<div class=selectionName>" << std::endl;
        pHtmlStream << "<div style='display:inline-block; width:130px;'>&nbsp;&nbsp;" << (*lIter).first << "</div>" << std::endl;
        pHtmlStream << "<small style='margin-left: 40px;'>" << std::endl;
        
        std::vector<std::string>::const_iterator lInnerIter = (*lIter).second.begin(), lInnerEndIter = (*lIter).second.end(), lPenultimateInnerIter = lInnerEndIter;
        --lPenultimateInnerIter;
        for(size_t lRadioIndex = 1; lInnerIter != lInnerEndIter; ++lInnerIter, ++lRadioIndex)
        {
            pHtmlStream << "<input type='radio' id='p" << pPanelIndex << "_rs" << lRadioSetIndex << "_" << lRadioIndex << "' name='p" << pPanelIndex << "_rs" << lRadioSetIndex << "_button'" << ((lInnerIter == lPenultimateInnerIter) ? " checked='checked'" : "") << " onClick='checkState(this)' />" << std::endl;
            pHtmlStream << "<label for='p" << pPanelIndex << "_rs" << lRadioSetIndex << "_" << lRadioIndex << "'>" << (*lInnerIter) << "</label>" << std::endl;
        }

        pHtmlStream << "</small>" << std::endl;
        pHtmlStream << "</div>" << std::endl;
        pHtmlStream << "</div>" << std::endl;
        pHtmlStream << std::endl;
    }
    
    pHtmlStream << "</div>" << std::endl;
    pHtmlStream << "</center>" << std::endl;
    pHtmlStream << "<br>" << std::endl;
    pHtmlStream << std::endl;
}

size_t Benchmark::GeneratePerformanceGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    panelConfigurationType lPanelConf;
    
    const char* lPolicies[] = {"PUSH", "PULL"};
    lPanelConf.push_back(std::make_pair("Scheduling&nbsp;Policy", std::vector<std::string>(lPolicies, lPolicies + sizeof(lPolicies)/sizeof(lPolicies[0]))));
    
    std::string lVarying2Str("Varying_2");
    if(!mConfiguration[lVarying2Str].empty())
        lPanelConf.push_back(std::make_pair(mConfiguration["Varying2_Name"][0], mConfiguration[lVarying2Str]));
    
    const char* lMaOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Multi&nbsp;Assign", std::vector<std::string>(lMaOptions, lMaOptions + (sizeof(lMaOptions)/sizeof(lMaOptions[0])))));
    
    const char* lOverlapOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Overlap&nbsp;Comp/Comm", std::vector<std::string>(lOverlapOptions, lOverlapOptions + (sizeof(lOverlapOptions)/sizeof(lOverlapOptions[0])))));

    pHtmlStream << "<div id='p" << pPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(pPanelIndex, lPanelConf, pHtmlStream);

    if(!mConfiguration[lVarying2Str].empty())
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying2Str].begin(), lEndIter = mConfiguration[lVarying2Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            size_t lVarying2Val = (size_t)atoi((*lIter).c_str());
         
            for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
            {
                std::string lMaStr((maVal == 0) ? "2" : "1");
                
                for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                {
                    std::string lOverlapStr((overlap == 0) ? "2" : "1");

                    pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << 1 << "_" << lIndex << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;

                    GeneratePerformanceGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, PUSH, lVarying2Val);
                    pHtmlStream << "</div>" << std::endl;

                    pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << 2 << "_" << lIndex << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                    GeneratePerformanceGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, PULL, lVarying2Val);
                    pHtmlStream << "</div>" << std::endl;
                }
            }
        }
    }
    else
    {
        for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
        {
            std::string lMaStr((maVal == 0) ? "2" : "1");
            
            for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
            {
                std::string lOverlapStr((overlap == 0) ? "2" : "1");

                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << 1 << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                GeneratePerformanceGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, PUSH);
                pHtmlStream << "</div>" << std::endl;

                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << 2 << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                GeneratePerformanceGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, PULL);
                pHtmlStream << "</div>" << std::endl;
            }
        }
    }
    
    pHtmlStream << "</div>" << std::endl;
    
    return lPanelConf.size();
}

size_t Benchmark::GenerateSchedulingModelsGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    panelConfigurationType lPanelConf;

    std::string lVarying1Str("Varying_1");
    std::string lVarying2Str("Varying_2");
    
    lPanelConf.push_back(std::make_pair(mConfiguration["Varying1_Name"][0], mConfiguration[lVarying1Str]));
    
    if(!mConfiguration[lVarying2Str].empty())
        lPanelConf.push_back(std::make_pair(mConfiguration["Varying2_Name"][0], mConfiguration[lVarying2Str]));
    
    const char* lMaOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Multi&nbsp;Assign", std::vector<std::string>(lMaOptions, lMaOptions + (sizeof(lMaOptions)/sizeof(lMaOptions[0])))));

    const char* lOverlapOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Overlap&nbsp;Comp/Comm", std::vector<std::string>(lOverlapOptions, lOverlapOptions + (sizeof(lOverlapOptions)/sizeof(lOverlapOptions[0])))));

    pHtmlStream << "<div id='p" << pPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(pPanelIndex, lPanelConf, pHtmlStream);
    
    if(!mConfiguration[lVarying2Str].empty())
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            std::vector<std::string>::const_iterator lInnerIter = mConfiguration[lVarying2Str].begin(), lInnerEndIter = mConfiguration[lVarying2Str].end();
            for(size_t lInnerIndex = 1; lInnerIter != lInnerEndIter; ++lInnerIter, ++lInnerIndex)
            {
                for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
                {
                    std::string lMaStr((maVal == 0) ? "2" : "1");
                
                    for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                    {
                        std::string lOverlapStr((overlap == 0) ? "2" : "1");

                        pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lInnerIndex << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                        
                        GenerateSchedulingModelsGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, (size_t)atoi(((*lIter).c_str())), (size_t)atoi(((*lInnerIter).c_str())));
                        pHtmlStream << "</div>" << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
            {
                std::string lMaStr((maVal == 0) ? "2" : "1");
                
                for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                {
                    std::string lOverlapStr((overlap == 0) ? "2" : "1");

                    pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lMaStr << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                    GenerateSchedulingModelsGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, (size_t)atoi((*lIter).c_str()), 0);
                    pHtmlStream << "</div>" << std::endl;
                }
            }
        }
    }
    
    pHtmlStream << "</div>" << std::endl;
    
    return lPanelConf.size();
}

size_t Benchmark::GenerateLoadBalancingGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    panelConfigurationType lPanelConf;
    
    bool lHasInnerTasks = (mInnerTasks.size() > 1);
    
    std::vector<std::string> lHostsVector;
    std::map<size_t, size_t>::const_iterator lHostsIter = mResults.hostsMap.begin(), lHostsEndIter = mResults.hostsMap.end();
    for(; lHostsIter != lHostsEndIter; ++lHostsIter)
    {
        std::stringstream lHostsStream;
        lHostsStream << lHostsIter->first;
        
        lHostsVector.push_back(lHostsStream.str());
    }

    std::string lVarying1Str("Varying_1");
    std::string lVarying2Str("Varying_2");
    
    lPanelConf.push_back(std::make_pair("Hosts", lHostsVector));
    lPanelConf.push_back(std::make_pair(mConfiguration["Varying1_Name"][0], mConfiguration[lVarying1Str]));
    
    if(!mConfiguration[lVarying2Str].empty())
        lPanelConf.push_back(std::make_pair(mConfiguration["Varying2_Name"][0], mConfiguration[lVarying2Str]));

    const char* lPolicies[] = {"PUSH", "PULL"};
    lPanelConf.push_back(std::make_pair("Scheduling&nbsp;Policy", std::vector<std::string>(lPolicies, lPolicies + sizeof(lPolicies)/sizeof(lPolicies[0]))));

    const char* lMaOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Multi&nbsp;Assign", std::vector<std::string>(lMaOptions, lMaOptions + (sizeof(lMaOptions)/sizeof(lMaOptions[0])))));

    const char* lOverlapOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Overlap&nbsp;Comp/Comm", std::vector<std::string>(lOverlapOptions, lOverlapOptions + (sizeof(lOverlapOptions)/sizeof(lOverlapOptions[0])))));

    if(lHasInnerTasks)
        lPanelConf.push_back(std::make_pair("Inner&nbsp;Task", mConfiguration["Inner_Task_Names"]));

    pHtmlStream << "<div id='p" << pPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(pPanelIndex, lPanelConf, pHtmlStream);

    lHostsIter = mResults.hostsMap.begin();
    for(size_t lHostIndex = 1; lHostsIter != lHostsEndIter; ++lHostsIter, ++lHostIndex)
    {
        if(!mConfiguration[lVarying2Str].empty())
        {
            std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
            for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
            {
                std::vector<std::string>::const_iterator lInnerIter = mConfiguration[lVarying2Str].begin(), lInnerEndIter = mConfiguration[lVarying2Str].end();
                for(size_t lInnerIndex = 1; lInnerIter != lInnerEndIter; ++lInnerIter, ++lInnerIndex)
                {
                    for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
                    {
                        std::string lMaStr((maVal == 0) ? "2" : "1");
                        
                        for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                        {
                            std::string lOverlapStr((overlap == 0) ? "2" : "1");

                            std::vector<Level2InnerTaskKey>::const_iterator lInnerTaskIter = mInnerTasks.begin(), lInnerTaskEndIter = mInnerTasks.end();
                            for(size_t lInnerTaskIndex = 1; lInnerTaskIter != lInnerTaskEndIter; ++lInnerTaskIter, ++lInnerTaskIndex)
                            {
                                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lHostIndex << "_" << lIndex << "_" << lInnerIndex << "_" << 1 << "_" << lMaStr << "_" << lOverlapStr;
                                
                                if(lHasInnerTasks)
                                    pHtmlStream << "_" << lInnerTaskIndex;

                                pHtmlStream << "' style='display:none'>" << std::endl;
                                
                                GenerateLoadBalancingGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, lHostsIter->first, (size_t)atoi(((*lIter).c_str())), (size_t)atoi(((*lInnerIter).c_str())), PUSH, *lInnerTaskIter);

                                pHtmlStream << "</div>" << std::endl;

                                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lHostIndex << "_" << lIndex << "_" << lInnerIndex << "_" << 2 << "_" << lMaStr << "_" << lOverlapStr;
                                
                                if(lHasInnerTasks)
                                    pHtmlStream << "_" << lInnerTaskIndex;
                                
                                pHtmlStream << "' style='display:none'>" << std::endl;
                                
                                GenerateLoadBalancingGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, lHostsIter->first, (size_t)atoi(((*lIter).c_str())), (size_t)atoi(((*lInnerIter).c_str())), PULL, *lInnerTaskIter);
                                
                                pHtmlStream << "</div>" << std::endl;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
            for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
            {
                for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
                {
                    std::string lMaStr((maVal == 0) ? "2" : "1");
                
                    for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                    {
                        std::string lOverlapStr((overlap == 0) ? "2" : "1");

                        std::vector<Level2InnerTaskKey>::const_iterator lInnerTaskIter = mInnerTasks.begin(), lInnerTaskEndIter = mInnerTasks.end();
                        for(size_t lInnerTaskIndex = 1; lInnerTaskIter != lInnerTaskEndIter; ++lInnerTaskIter, ++lInnerTaskIndex)
                        {
                            pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lHostIndex << "_" << lIndex << "_" << 1 << "_" << lMaStr << "_" << lOverlapStr;
                         
                            if(lHasInnerTasks)
                                pHtmlStream << "_" << lInnerTaskIndex;
                            
                            pHtmlStream << "' style='display:none'>" << std::endl;
                            
                            GenerateLoadBalancingGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, lHostsIter->first, (size_t)atoi((*lIter).c_str()), 0, PUSH, *lInnerTaskIter);
                            
                            pHtmlStream << "</div>" << std::endl;

                            pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lHostIndex << "_" << lIndex << "_" << 2 << "_" << lMaStr << "_" << lOverlapStr;
                            
                            if(lHasInnerTasks)
                                pHtmlStream << "_" << lInnerTaskIndex;
                            
                            pHtmlStream << "' style='display:none'>" << std::endl;
                            
                            GenerateLoadBalancingGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (bool)maVal, (bool)overlap, lHostsIter->first, (size_t)atoi((*lIter).c_str()), 0, PULL, *lInnerTaskIter);
                            
                            pHtmlStream << "</div>" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    pHtmlStream << "</div>" << std::endl;
    
    return lPanelConf.size();
}

size_t Benchmark::GenerateMultiAssignComparisonGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    panelConfigurationType lPanelConf;

    std::string lVarying1Str("Varying_1");
    std::string lVarying2Str("Varying_2");
    
    lPanelConf.push_back(std::make_pair(mConfiguration["Varying1_Name"][0], mConfiguration[lVarying1Str]));
    
    if(!mConfiguration[lVarying2Str].empty())
        lPanelConf.push_back(std::make_pair(mConfiguration["Varying2_Name"][0], mConfiguration[lVarying2Str]));
    
    const char* lOverlapOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Overlap&nbsp;Comp/Comm", std::vector<std::string>(lOverlapOptions, lOverlapOptions + (sizeof(lOverlapOptions)/sizeof(lOverlapOptions[0])))));

    pHtmlStream << "<div id='p" << pPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(pPanelIndex, lPanelConf, pHtmlStream);
    
    if(!mConfiguration[lVarying2Str].empty())
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            std::vector<std::string>::const_iterator lInnerIter = mConfiguration[lVarying2Str].begin(), lInnerEndIter = mConfiguration[lVarying2Str].end();
            for(size_t lInnerIndex = 1; lInnerIter != lInnerEndIter; ++lInnerIter, ++lInnerIndex)
            {
                for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
                {
                    std::string lOverlapStr((overlap == 0) ? "2" : "1");

                    pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lInnerIndex << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                    
                    GenerateMultiAssignComparisonGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (size_t)atoi(((*lIter).c_str())), (size_t)atoi(((*lInnerIter).c_str())), (bool)overlap);
                    pHtmlStream << "</div>" << std::endl;
                }
            }
        }
    }
    else
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            for(int overlap = 1; overlap >= 0; --overlap)   // Compute Communication Overlap
            {
                std::string lOverlapStr((overlap == 0) ? "2" : "1");

                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lOverlapStr << "' style='display:none'>" << std::endl;
                GenerateMultiAssignComparisonGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (size_t)atoi((*lIter).c_str()), 0, (bool)overlap);
                pHtmlStream << "</div>" << std::endl;
            }
        }
    }
    
    pHtmlStream << "</div>" << std::endl;
    
    return lPanelConf.size();    
}

size_t Benchmark::GenerateOverlapComparisonGraphs(size_t pPanelIndex, size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream)
{
    panelConfigurationType lPanelConf;

    std::string lVarying1Str("Varying_1");
    std::string lVarying2Str("Varying_2");
    
    lPanelConf.push_back(std::make_pair(mConfiguration["Varying1_Name"][0], mConfiguration[lVarying1Str]));
    
    if(!mConfiguration[lVarying2Str].empty())
        lPanelConf.push_back(std::make_pair(mConfiguration["Varying2_Name"][0], mConfiguration[lVarying2Str]));
    
    const char* lMaOptions[] = {"Yes", "No"};
    lPanelConf.push_back(std::make_pair("Multi&nbsp;Assign", std::vector<std::string>(lMaOptions, lMaOptions + (sizeof(lMaOptions)/sizeof(lMaOptions[0])))));

    pHtmlStream << "<div id='p" << pPanelIndex << "' value='" << lPanelConf.size() << "'>" << std::endl;
    
    GenerateSelectionGroup(pPanelIndex, lPanelConf, pHtmlStream);
    
    if(!mConfiguration[lVarying2Str].empty())
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            std::vector<std::string>::const_iterator lInnerIter = mConfiguration[lVarying2Str].begin(), lInnerEndIter = mConfiguration[lVarying2Str].end();
            for(size_t lInnerIndex = 1; lInnerIter != lInnerEndIter; ++lInnerIter, ++lInnerIndex)
            {
                for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
                {
                    std::string lMaStr((maVal == 0) ? "2" : "1");
                    
                    pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lInnerIndex << "_" << lMaStr << "' style='display:none'>" << std::endl;
                    
                    GenerateOverlapComparisonGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (size_t)atoi(((*lIter).c_str())), (size_t)atoi(((*lInnerIter).c_str())), (bool)maVal);
                    pHtmlStream << "</div>" << std::endl;
                }
            }
        }
    }
    else
    {
        std::vector<std::string>::const_iterator lIter = mConfiguration[lVarying1Str].begin(), lEndIter = mConfiguration[lVarying1Str].end();
        for(size_t lIndex = 1; lIter != lEndIter; ++lIter, ++lIndex)
        {
            for(int maVal = 1; maVal >= 0; --maVal)   // Multi Assign
            {
                std::string lMaStr((maVal == 0) ? "2" : "1");
                
                pHtmlStream << "<div class='p" << pPanelIndex << "_toggler' id='p" << pPanelIndex << "_table_" << lIndex << "_" << lMaStr << "' style='display:none'>" << std::endl;
                GenerateOverlapComparisonGraphsInternal(pPlotWidth, pPlotHeight, pHtmlStream, (size_t)atoi((*lIter).c_str()), 0, (bool)maVal);
                pHtmlStream << "</div>" << std::endl;
            }
        }
    }
    
    pHtmlStream << "</div>" << std::endl;
    
    return lPanelConf.size();    
}

void Benchmark::GeneratePerformanceGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, SchedulingPolicy pPolicy, size_t pVarying2Val /* = 0 */)
{
    std::string lVarying1DisplayName = mConfiguration["Varying1_Name"][0];

    std::stringstream lVarying2Stream;
    if(!mConfiguration["Varying_2"].empty())
        lVarying2Stream << mConfiguration["Varying2_Name"][0] << "=" << pVarying2Val << ", ";
    
    std::stringstream lGraphDisplayNameStream;
    lGraphDisplayNameStream << lVarying1DisplayName << "&nbsp;&nbsp;[" << lVarying2Stream.str() << ((pPolicy == PUSH) ? "Push" : "Pull") << (pMA ? ", MA" : "") << (pOverlap ? ", Overlap" : "") << "] --->";
    
    const std::string& lGraphDisplayName = lGraphDisplayNameStream.str();

    const char* lOneHostSvpCurveNames[] = {"Sequential", "CPUs", "GPUs", "CPUs+GPUs"}; // Svp means Sequential versus Parallel
    StandardChart lOneHostSvpGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    lOneHostSvpGraph.SetCurves(4, lOneHostSvpCurveNames);
    
    const char* lOneHostSgvpCurveNames[] = {"Single GPU", "CPUs", "GPUs", "CPUs+GPUs"}; // Sgvp means Single GPU versus Parallel
    StandardChart lOneHostSgvpGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    lOneHostSgvpGraph.SetCurves(4, lOneHostSgvpCurveNames);

    const char* lOneHostParallelCurveNames[] = {"CPUs", "GPUs", "CPUs+GPUs"};
    StandardChart lOneHostParallelGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    lOneHostParallelGraph.SetCurves(3, lOneHostParallelCurveNames);

    StandardChart lCpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    StandardChart lGpusAllHostsGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
    StandardChart lCpgAllHostsGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayName)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));   // Cpg means CPUs+GPUs

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
        if(lIter->first.varying2 != pVarying2Val)
            continue;

        lOneHostSvpGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lIter->second.first.sequentialTime));
        lOneHostSgvpGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lIter->second.first.singleGpuTime));
        
        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.multiAssign != pMA || lInnerIter->first.overlapComputeCommunication != pOverlap)
                continue;

            if(lInnerIter->first.policy == pPolicy && !lInnerIter->first.lazyMem)
            {
                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostSgvpGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostParallelGraph.curves[0].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        }

                        lCpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        break;
                        
                    case GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostSgvpGraph.curves[2].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostParallelGraph.curves[1].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                        }
                        
                        lGpusAllHostsGraph.curves[mResults.hostsMap[lInnerIter->first.hosts]].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));

                        break;
                        
                    case CPU_PLUS_GPU:
                        if(lInnerIter->first.hosts == 1)
                        {
                            lOneHostSvpGraph.curves[3].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
                            lOneHostSgvpGraph.curves[3].points.push_back(std::make_pair(lIter->first.varying1, lInnerIter->second.execTime));
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
    EmbedPlot(pHtmlStream, GenerateStandardChart(pPlotWidth, pPlotHeight, lOneHostSgvpGraph), "Single GPU vs. PMLIB - 1 Host");
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

void Benchmark::GenerateSchedulingModelsGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, size_t pVarying1Val, size_t pVarying2Val)
{
    const std::string lStaticBestStr("Generate_Static_Best");
    std::vector<std::string>& lStaticBestVector = GetGlobalConfiguration()[lStaticBestStr];
    bool lGenerateStaticBest = (!lStaticBestVector.empty() && !lStaticBestVector[0].compare(std::string("false"))) ? false : true;

    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    
    std::stringstream lGraphDisplayNameStream;
    lGraphDisplayNameStream << mConfiguration["Varying1_Name"][0] << "=" << pVarying1Val;

    if(!mConfiguration["Varying_2"].empty())
        lGraphDisplayNameStream << ", " << mConfiguration["Varying2_Name"][0] << "=" << pVarying2Val;

    if(pMA)
        lGraphDisplayNameStream << ", MA";

    if(pOverlap)
        lGraphDisplayNameStream << ", Overlap";
    
    bool lFirst = true;
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.varying1 != pVarying1Val || lIter->first.varying2 != pVarying2Val)
            continue;

        if(lFirst)
            lFirst = false;
        else
            pHtmlStream << "<br>" << std::endl;

        StandardChart lCpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lCpusPlusGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));

        const char* lCurveNames[] = {"Push", "Pull", "Static Equal", "Static Best"};
        const char* lCurveNamesMA[] = {"Push_MA", "Pull_MA", "Static Equal", "Static Best"};
        lCpusGraph.SetCurves(lGenerateStaticBest ? 4 : 3, pMA ? lCurveNamesMA : lCurveNames);
        lGpusGraph.SetCurves(lGenerateStaticBest ? 4 : 3, pMA ? lCurveNamesMA : lCurveNames);
        lCpusPlusGpusGraph.SetCurves(lGenerateStaticBest ? 4 : 3, pMA ? lCurveNamesMA : lCurveNames);

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
            if(!lGenerateStaticBest && lInnerIter->first.policy == STATIC_BEST)
                continue;
            
            if(lInnerIter->first.policy == PUSH || lInnerIter->first.policy == PULL)
            {
                if(lInnerIter->first.multiAssign != pMA || lInnerIter->first.overlapComputeCommunication != pOverlap)
                    continue;
            }

            if(!lInnerIter->first.lazyMem)
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

void Benchmark::GenerateLoadBalancingGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, bool pMA, bool pOverlap, size_t pHosts, size_t pVarying1Val, size_t pVarying2Val, SchedulingPolicy pPolicy, const Level2InnerTaskKey& pInnerTask)
{
    size_t lCount = 0;

    const char* lChartTitle[] = {"CPUs", "GPUs", "CPUs+GPUs"};

    pHtmlStream << "<table align=center>" << std::endl;
    pHtmlStream << "<tr class=horizSpacing>" << std::endl;

    std::stringstream lGraphDisplayNameStream;
    lGraphDisplayNameStream << mConfiguration["Varying1_Name"][0] << "=" << pVarying1Val;

    if(!mConfiguration["Varying_2"].empty())
        lGraphDisplayNameStream << ", " << mConfiguration["Varying2_Name"][0] << "=" << pVarying2Val;

    if(pMA)
        lGraphDisplayNameStream << ", MA";
    
    if(pOverlap)
        lGraphDisplayNameStream << ", Overlap";

    lGraphDisplayNameStream << ", Hosts=" << pHosts;
    lGraphDisplayNameStream << ", " << ((pPolicy == PUSH) ? "Push" : "Pull") << std::endl;

    BenchmarkResults::mapType::const_iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.varying1 != pVarying1Val || lIter->first.varying2 != pVarying2Val)
            continue;

        const std::map<Level2Key, Level2Value>& lMap = lIter->second.second;
        std::map<Level2Key, Level2Value>::const_iterator lInnerIter = lMap.begin(), lInnerEndIter = lMap.end();
        for(; lInnerIter != lInnerEndIter; ++lInnerIter)
        {
            if(lInnerIter->first.lazyMem)
                continue;
            
            if(lInnerIter->first.hosts != pHosts || lInnerIter->first.policy != pPolicy || lInnerIter->first.multiAssign != pMA || lInnerIter->first.overlapComputeCommunication != pOverlap)
                continue;

            if(lCount && (lCount % 3 == 0))
            {
                pHtmlStream << "</tr>" << std::endl;
                pHtmlStream << "</table>" << std::endl;
                pHtmlStream << "<br>" << std::endl;
                pHtmlStream << "<table align=center>" << std::endl;
                pHtmlStream << "<tr class=horizSpacing>" << std::endl;
            }
            
            StandardChart lGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Finishing Time (in s) --->")));

            const std::map<Level2InnerTaskKey, Level2InnerTaskValue>& lInnerTaskMap = lInnerIter->second.innerTaskMap;
            const Level2InnerTaskValue& lInnerTaskVal = lInnerTaskMap.find(pInnerTask)->second;
            std::map<size_t, DeviceStats>::const_iterator lDeviceIter = lInnerTaskVal.deviceStats.begin(), lDeviceEndIter = lInnerTaskVal.deviceStats.end();
            
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

void Benchmark::GenerateMultiAssignComparisonGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, size_t pVarying1Val, size_t pVarying2Val, bool pOverlap)
{
    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    
    std::stringstream lGraphDisplayNameStream;
    lGraphDisplayNameStream << mConfiguration["Varying1_Name"][0] << "=" << pVarying1Val;

    if(!mConfiguration["Varying_2"].empty())
        lGraphDisplayNameStream << ", " << mConfiguration["Varying2_Name"][0] << "=" << pVarying2Val;

    bool lFirst = true;
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.varying1 != pVarying1Val || lIter->first.varying2 != pVarying2Val)
            continue;

        if(lFirst)
            lFirst = false;
        else
            pHtmlStream << "<br>" << std::endl;
        
        StandardChart lCpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lCpusPlusGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));

        const char* lCurveNames[] = {"Push", "Push_MA", "Pull", "Pull_MA"};
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
            if(lInnerIter->first.policy != PUSH && lInnerIter->first.policy != PULL)
                continue;
            
            if(lInnerIter->first.overlapComputeCommunication != pOverlap)
                continue;
            
            if(!lInnerIter->first.lazyMem)
            {
                size_t lCurveIndex = 0;
                switch(lInnerIter->first.policy)
                {
                    case PUSH:
                        lCurveIndex = ((lInnerIter->first.multiAssign) ? 1 : 0);
                        break;
                        
                    case PULL:
                        lCurveIndex = ((lInnerIter->first.multiAssign) ? 3 : 2);
                        break;
                        
                    default:
                        throw std::exception();
                }
                
                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        lCpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                        break;
                        
                    case GPU:
                        lGpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                        break;

                    case CPU_PLUS_GPU:
                        lCpusPlusGpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
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

void Benchmark::GenerateOverlapComparisonGraphsInternal(size_t pPlotWidth, size_t pPlotHeight, std::ofstream& pHtmlStream, size_t pVarying1Val, size_t pVarying2Val, bool pMA)
{
    BenchmarkResults::mapType::iterator lIter = mResults.results.begin(), lEndIter = mResults.results.end();
    
    std::stringstream lGraphDisplayNameStream;
    lGraphDisplayNameStream << mConfiguration["Varying1_Name"][0] << "=" << pVarying1Val;

    if(!mConfiguration["Varying_2"].empty())
        lGraphDisplayNameStream << ", " << mConfiguration["Varying2_Name"][0] << "=" << pVarying2Val;

    bool lFirst = true;
    for(; lIter != lEndIter; ++lIter)
    {
        if(lIter->first.varying1 != pVarying1Val || lIter->first.varying2 != pVarying2Val)
            continue;

        if(lFirst)
            lFirst = false;
        else
            pHtmlStream << "<br>" << std::endl;
        
        StandardChart lCpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));
        StandardChart lCpusPlusGpusGraph(std::auto_ptr<Axis>(new Axis(lGraphDisplayNameStream.str(), false)), std::auto_ptr<Axis>(new Axis("Execution Time (in s) --->")));

        const char* lCurveNames[] = {"Push", "Push_Overlap", "Pull", "Pull_Overlap"};
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
            if(lInnerIter->first.policy != PUSH && lInnerIter->first.policy != PULL)
                continue;
            
            if(lInnerIter->first.multiAssign != pMA)
                continue;
            
            if(!lInnerIter->first.lazyMem)
            {
                size_t lCurveIndex = 0;
                switch(lInnerIter->first.policy)
                {
                    case PUSH:
                        lCurveIndex = ((lInnerIter->first.overlapComputeCommunication) ? 1 : 0);
                        break;
                        
                    case PULL:
                        lCurveIndex = ((lInnerIter->first.overlapComputeCommunication) ? 3 : 2);
                        break;
                        
                    default:
                        throw std::exception();
                }
                
                switch(lInnerIter->first.cluster)
                {
                    case CPU:
                        lCpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                        break;
                        
                    case GPU:
                        lGpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
                        break;

                    case CPU_PLUS_GPU:
                        lCpusPlusGpusGraph.curves[lCurveIndex].points.push_back(std::make_pair(mResults.hostsMap[lInnerIter->first.hosts], lInnerIter->second.execTime));
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
    
    if(!strcmp(pResultsFile.c_str() + strlen(pResultsFile.c_str()) - strlen(SINGLE_GPU_FILE_NAME), SINGLE_GPU_FILE_NAME))
    {
        boost::regex lExp("^Single GPU Task Execution Time = ([0-9.]+)");
        while(std::getline(lFileStream, lLine))
        {
            if(boost::regex_search(lLine.c_str(), lResults, lExp))
            {
                mSamples[pSampleIndex].results[pLevel1Key].first.singleGpuTime = atof(std::string(lResults[1]).c_str());
                break;
            }
        }
        
        return;
    }

    boost::regex lKeyExp("([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)$");
    if(!boost::regex_search(pResultsFile.c_str(), lResults, lKeyExp))
        throw std::exception();

    Level2Key lLevel2Key((size_t)(atoi(std::string(lResults[1]).c_str())), (enum SchedulingPolicy)(atoi(std::string(lResults[2]).c_str())), (enum clusterType)(atoi(std::string(lResults[3]).c_str())), (bool)(atoi(std::string(lResults[4]).c_str())), (bool)(atoi(std::string(lResults[5]).c_str())), (bool)(atoi(std::string(lResults[6]).c_str())));

    if(mHostsSetVector[pSampleIndex].find(lLevel2Key.hosts) == mHostsSetVector[pSampleIndex].end())
        mHostsSetVector[pSampleIndex].insert(lLevel2Key.hosts);
    
    size_t lCurrentDevice = 0;
    Level2InnerTaskKey lCurrentTask;

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

            lCurrentTask.originatingHost = atoi(std::string(lResults[1]).c_str());
            lCurrentTask.taskSequenceId = atoi(std::string(lResults[2]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lCurrentDevice].eventTimeline[lSubtaskId] = std::make_pair(lStartTime, lEndTime);
            if(lEndTime > mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lCurrentDevice].lastEventTimings.second)
                mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lCurrentDevice].lastEventTimings = std::make_pair(lStartTime, lEndTime);
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp2))
        {
            if(lLevel2Key.policy != (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str())))
                throw std::exception();
            
            lCurrentTask.originatingHost = atoi(std::string(lResults[1]).c_str());
            lCurrentTask.taskSequenceId = atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp3))
        {
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[atoi(std::string(lResults[1]).c_str())].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp4))
        {
            size_t lMachine = atoi(std::string(lResults[1]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].machineStats[lMachine].subtasksExecuted = atoi(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].machineStats[lMachine].cpuSubtasks = atoi(std::string(lResults[3]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp5))
        {
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].subtaskCount = atoi(std::string(lResults[1]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp6))
        {
            std::string lCriterion(lResults[1]);

            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].workTimeStats[lCriterion].first = atof(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].workTimeStats[lCriterion].second = atof(std::string(lResults[3]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp7))
        {
            size_t lDevice = atoi(std::string(lResults[1]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lDevice].subtaskExecutionRate = atoi(std::string(lResults[2]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lDevice].stealAttempts = atoi(std::string(lResults[3]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lDevice].stealSuccesses = atoi(std::string(lResults[4]).c_str());
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].deviceStats[lDevice].stealFailures = atoi(std::string(lResults[5]).c_str());
            
            mSamples[pSampleIndex].results[pLevel1Key].second[lLevel2Key].innerTaskMap[lCurrentTask].totalExecutionRate += atoi(std::string(lResults[2]).c_str());
        }
        else if(boost::regex_search(lLine.c_str(), lResults, lExp8))
        {
            if(lLevel2Key.policy != (enum SchedulingPolicy)(atoi(std::string(lResults[3]).c_str())))
                throw std::exception();

            if(4 + (size_t)lLevel2Key.cluster != (size_t)(atoi(std::string(lResults[1]).c_str())))
                throw std::exception();

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
    
    const std::string& lFixedArgs = (mConfiguration["Fixed_Args"].empty() ? std::string("") : std::string(" ").append(mConfiguration["Fixed_Args"][0]));

    /* Generate sequential task output */
    std::string lSequentialFile(pOutputFolder);
    lSequentialFile.append(lSeparator);
    lSequentialFile.append(SEQUENTIAL_FILE_NAME);
    
    const std::string& lTempFile = GetTempOutputFileName();
    
    std::ifstream lSequentialFileStream(lSequentialFile.c_str());

    if(lSequentialFileStream.fail())
    {
        std::stringstream lStream;
        lStream << "source ~/.pmlibrc; ";
        lStream << "mpirun -n " << 1 << " " << mExecPath << " 2 0 0 " << pSpaceSeparatedVaryingsStr << lFixedArgs;
        lStream << " > " << lTempFile.c_str() << " 2>&1";

        ExecuteShellCommand(lStream.str(), "sequential", lSequentialFile);
    }
    else
    {
        lSequentialFileStream.close();
    }
    
    /* Generate single GPU task output */
    std::string lSingleGpuFile(pOutputFolder);
    lSingleGpuFile.append(lSeparator);
    lSingleGpuFile.append(SINGLE_GPU_FILE_NAME);
    
    std::ifstream lSingleGpuFileStream(lSingleGpuFile.c_str());
    
    if(lSingleGpuFileStream.fail())
    {
        std::stringstream lStream;
        lStream << "source ~/.pmlibrc; ";
        lStream << "mpirun -n " << 1 << " " << mExecPath << " 3 0 0 " << pSpaceSeparatedVaryingsStr << lFixedArgs;
        lStream << " > " << lTempFile.c_str() << " 2>&1";
        
        ExecuteShellCommand(lStream.str(), "single gpu", lSingleGpuFile);
    }
    else
    {
        lSingleGpuFileStream.close();
    }
    
    const std::string lStaticBestStr("Generate_Static_Best");
    std::vector<std::string>& lStaticBestVector = GetGlobalConfiguration()[lStaticBestStr];
    bool lGenerateStaticBest = (!lStaticBestVector.empty() && !lStaticBestVector[0].compare(std::string("false"))) ? false : true;
    
    const std::string lMpiOptionsStr("Mpi_Options");
    std::vector<std::string>& lMpiOptionsVector = GetGlobalConfiguration()[lMpiOptionsStr];
    
    const std::string& lMpiOptions = lMpiOptionsVector.empty() ? std::string("") : lMpiOptionsVector[0];
    
    const char* lSchedulingModelNames[] = {"Push", "Pull", "StaticEqual", "StaticBest"};
    const char* lClusterTypeNames[] = {"CPU", "GPU", "CPU+GPU"};

    /* Generate PMLIB tasks output */
    for(size_t i = 0; i < (size_t)MAX_SCHEDULING_POLICY; ++i)
    {
        if(!lGenerateStaticBest && ((enum SchedulingPolicy)i == STATIC_BEST))
            continue;
        
        for(size_t j = 0; j < (size_t)MAX_CLUSTER_TYPE; ++j)
        {
            for(size_t k = 0; k <= 1; ++k) // Multi Assign
            {
                for(size_t l = 0; l <= 1; ++l)  // Lazy Mem
                {
                    if(l == 1)
                        continue;
                    
                    for(size_t m = 0; m <= 1; ++m)  // Compute Communication Overlap
                    {
                        if((k == 1 || l == 1 || m == 1) && (((SchedulingPolicy)i == STATIC_BEST) || ((SchedulingPolicy)i == STATIC_EQUAL)))
                            continue;

                        setenv("PMLIB_DISABLE_MA", ((k == 0) ? "1" : "0"), 1);
                        setenv("PMLIB_ENABLE_LAZY_MEM", ((l == 0) ? "0" : "1"), 1);
                        setenv("PMLIB_DISABLE_COMPUTE_COMMUNICATION_OVERLAP", ((m == 0) ? "1" : "0"), 1);
                        
                        std::stringstream lOutputFile, lDisplayName;
                        
                        lOutputFile << pOutputFolder << lSeparator << pHosts << "_" << i << "_" << j << "_" << k << "_" << l << "_" << m;
                        lDisplayName << lSchedulingModelNames[i] << "_" << lClusterTypeNames[j] << "_" << ((k == 0) ? "NonMA" : "MA") << "_" << ((l == 0) ? "NonLazy" : "Lazy") << "_" << ((m == 0) ? "NoCompCommOverlap" : "CompCommOverlap");

                        std::ifstream lFileStream(lOutputFile.str().c_str());

                        if(lFileStream.fail())
                        {
                            std::stringstream lStream;
                            lStream << "source ~/.pmlibrc; ";
                            lStream << "mpirun -n " << pHosts << " " << lMpiOptions << " " << mExecPath << " 0 " << 4+j << " " << i << " " << pSpaceSeparatedVaryingsStr;
                            lStream << lFixedArgs;
                            lStream << " > " << lTempFile.c_str() << " 2>&1";

                            ExecuteShellCommand(lStream.str(), lDisplayName.str(), lOutputFile.str());
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
}

int Benchmark::RunCommand(const std::string& pCmd, const std::string& pDisplayName)
{
    std::cout << "      " << pDisplayName << "..." << std::endl;

    return system(pCmd.c_str());
}

void Benchmark::ExecuteShellCommand(const std::string& pCmd, const std::string& pDisplayName, const std::string& pOutputFile)
{
#ifdef HANDLE_BENCHMARK_HANGS
    bool lHangDetected = false;
    int lRetVal = -1;
    int lPipeDescriptors[2];

    if(pipe(lPipeDescriptors) != 0)
    {
        std::cout << "Pipe creation failed" << std::endl;
        exit(1);
    }
    
    pid_t lExecutorPid = fork();
    if(lExecutorPid == -1)
    {
        std::cout << "Fork Failed" << std::endl;
        exit(1);
    }
    
    if(lExecutorPid == 0) // child process
    {
        close(lPipeDescriptors[0]);

        lRetVal = RunCommand(pCmd, pDisplayName);

        write(lPipeDescriptors[1], (void*)(&lRetVal), sizeof(lRetVal));
        close(lPipeDescriptors[1]);

        exit(0);
    }

    close(lPipeDescriptors[1]);

    pid_t lTimerPid = fork();
    if(lTimerPid == -1)
    {
        std::cout << "Secondary fork failed" << std::endl;
        exit(1);
    }

    if(lTimerPid == 0) // child process
    {
        unsigned int lLeftTime = sleep(TIMEOUT_IN_SECS);
        while(lLeftTime != 0)
            lLeftTime = sleep(lLeftTime);

        exit(0);
    }

    int lPid = wait(NULL);
    if(lPid == lExecutorPid)
    {
        read(lPipeDescriptors[0], (void*)(&lRetVal), sizeof(lRetVal));
        close(lPipeDescriptors[0]);
        
        if(kill(lTimerPid, SIGKILL) != 0)
        {
            std::cout << "Failed to kill timer process" << std::endl;
            exit(1);
        }
    }
    else if(lPid == lTimerPid)
    {
        close(lPipeDescriptors[0]);

        lHangDetected = true;
        
        if(kill(lExecutorPid, SIGKILL) != 0)
        {
            std::cout << "Failed to kill executor process" << std::endl;
            exit(1);
        }
        
        std::stringstream lKillStream;
        lKillStream << "ps -ef | grep '" << mExecPath << "' | awk '{print $2}' | xargs -n1 kill -9 ";
        system(lKillStream.str().c_str());
    }

    while(1)
    {
        if(wait(NULL) == -1 && errno == ECHILD)
            break;
    }

    if(lHangDetected)
    {
        std::cerr << "[ERROR]: Command hanged - " << pCmd << std::endl;
        ExecuteShellCommand(pCmd, pDisplayName, pOutputFile);
    }
    else
#else
    int lRetVal = RunCommand(pCmd, pDisplayName);
#endif
    if(lRetVal != -1 && lRetVal != 127)
    {
        if(lRetVal != 0 || WIFSIGNALED(lRetVal) || !WIFEXITED(lRetVal))
        {
            std::cerr << "[ERROR]: Command abnormally exited - " << pCmd << std::endl;
            ExecuteShellCommand(pCmd, pDisplayName, pOutputFile);
        }
        else
        {
            const std::string& lTempFile = GetTempOutputFileName();
            if(std::rename(lTempFile.c_str(), pOutputFile.c_str()) != 0)
                std::cout << "Failed to move file " << lTempFile.c_str() << " to " << pOutputFile.c_str() << std::endl;
        }
    }

    unlink(GetTempOutputFileName().c_str());
}

const std::string& Benchmark::GetTempOutputFileName()
{
    static std::string lTempFileName;
    if(lTempFileName.empty())
    {
        std::stringstream lPid;
        srand((unsigned int)time(NULL));

        lPid << "../../../.pmlib_" << getpid() << "_" << rand();
        
        lTempFileName = std::string(lPid.str());
    }
    
    return lTempFileName;
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

void Benchmark::SetExecPath(const std::string& pExecPath)
{
    mExecPath = pExecPath;
}

void Benchmark::GetAllBenchmarks(std::vector<Benchmark>& pBenchmarks)
{
    char lPathSeparator[1];
    lPathSeparator[0] = PATH_SEPARATOR;
    std::string lSeparator(lPathSeparator, 1);

    std::string lTestSuitePath(GetBasePath());
    lTestSuitePath.append(lSeparator);
    lTestSuitePath.append("testSuite");

    std::string lBenchmarksStr("Benchmarks");
    const std::vector<std::string>& lSelectiveBenchmarks = GetGlobalConfiguration()[lBenchmarksStr];

    std::set<std::string> lChosenBenchmarks;
    std::vector<std::string> lReorderedBenchmarks;
    
    DIR* lDir = opendir(lTestSuitePath.c_str());
    if(lDir)
    {
        struct dirent* lEntry;
        while((lEntry = readdir(lDir)) != NULL)
        {
            if(lEntry->d_type == DT_DIR && strlen(lEntry->d_name))
            {
                std::string lName(lEntry->d_name);
                
                if(lSelectiveBenchmarks.empty() || std::find(lSelectiveBenchmarks.begin(), lSelectiveBenchmarks.end(), lName) != lSelectiveBenchmarks.end())
                    lChosenBenchmarks.insert(lName);
            }
        }

        closedir(lDir);
    }

    if(lSelectiveBenchmarks.empty())
    {
        lReorderedBenchmarks.insert(lReorderedBenchmarks.end(), lChosenBenchmarks.begin(), lChosenBenchmarks.end());
    }
    else
    {
        std::vector<std::string>::const_iterator lSelectionIter = lSelectiveBenchmarks.begin(), lSelectionEndIter = lSelectiveBenchmarks.end();
        for(; lSelectionIter != lSelectionEndIter; ++lSelectionIter)
        {
            if(lChosenBenchmarks.find((*lSelectionIter)) != lChosenBenchmarks.end())
                lReorderedBenchmarks.push_back((*lSelectionIter));
        }
    }
    
    std::vector<std::string>::iterator lIter = lReorderedBenchmarks.begin(), lEndIter = lReorderedBenchmarks.end();
    for(; lIter != lEndIter; ++lIter)
    {
        const std::string& lName = (*lIter);
        
        std::stringstream lConfigurationsKeyName;
        lConfigurationsKeyName << lName << "_Configurations";

        size_t lConfigurations = 1;
        if(!GetGlobalConfiguration()[lConfigurationsKeyName.str()].empty())
            lConfigurations = atoi((GetGlobalConfiguration()[lConfigurationsKeyName.str()][0]).c_str());

        std::vector<std::string> lConfigurationalNames;
        if(lConfigurations == 1)
        {
            lConfigurationalNames.push_back(lName);
        }
        else
        {
            for(size_t confIndex = 0; confIndex < lConfigurations; ++confIndex)
            {
                std::stringstream lConfName;
                lConfName << lName << "_" << (confIndex + 1);
                
                lConfigurationalNames.push_back(lConfName.str());
            }
        }
        
        std::vector<std::string>::iterator lConfIter = lConfigurationalNames.begin(), lConfEndIter = lConfigurationalNames.end();
        for(size_t lConfIndex = 0; lConfIter != lConfEndIter; ++lConfIter, ++lConfIndex)
        {
            Benchmark b(*lConfIter);
            
            try
            {
                b.LoadConfiguration();
                
                std::string lExecName(b.mConfiguration[std::string("Exec_Name")][0]);
                if(lExecName.empty())
                    lExecName = lName;
                
                std::string lExecPath(lTestSuitePath);
                lExecPath.append(lSeparator);
                lExecPath += lName;
                lExecPath.append(lSeparator);
                lExecPath.append(gIntermediatePath);
                lExecPath.append(lSeparator);
                lExecPath.append(lExecName);
                lExecPath.append(".exe");

                FILE* lExecFile = fopen(lExecPath.c_str(), "rb");
                if(!lExecFile)
                    throw std::string("Exec file not found");
                    
                fclose(lExecFile);
                b.SetExecPath(lExecPath);
            }
            catch(const std::string& pExecStr)
            {
                std::cout << "Caught Exception - " << pExecStr << std::endl;
                continue;
            }
            catch(...)
            {
                continue;
            }
            
            pBenchmarks.push_back(b);
        }
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

    if(mConfiguration[std::string("Benchmark_Name")].empty()  || mConfiguration[std::string("Subtask_Definition")].empty()
    || mConfiguration[std::string("Varying_1")].empty() || mConfiguration[std::string("Varying1_Name")].empty()
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
                || lKey == std::string("Varying1_Name") || lKey == std::string("Varying2_Name")
                || lKey == std::string("Fixed_Args") || lKey == std::string("Subtask_Definition")
                || lKey == std::string("Other_Information"))
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



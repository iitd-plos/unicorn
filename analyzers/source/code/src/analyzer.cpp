
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

#include <stdlib.h>
#include <iostream>
#include <vector>

#include "analyzer.h"
#include "benchmark.h"

#define PMLIB_INSTALL_PATH "/Users/tberi/Development/git-repositories/pmlib"

Analyzer::Analyzer(const std::string& pBasePath)
: mBasePath(pBasePath)
{
}

void Analyzer::Analyze()
{
    std::vector<Benchmark> lBenchmarks;
    Benchmark::RegisterBasePath(mBasePath);
    Benchmark::LoadGlobalConfiguration();
    Benchmark::GetAllBenchmarks(lBenchmarks);
    
    std::vector<Benchmark>::iterator lIter = lBenchmarks.begin(), lEndIter = lBenchmarks.end();
    for(; lIter != lEndIter; ++lIter)
    {
        lIter->CollectResults();
        lIter->ProcessResults();
    }
    
    Benchmark::WriteTopLevelHtmlPage(lBenchmarks);
    Benchmark::CopyResourceFiles();
}

int main(int argc, const char* argv[])
{
    const char* lBasePath = getenv("PMLIB_INSTALL_PATH");
    if(!lBasePath)
    {
    #ifdef PMLIB_INSTALL_PATH
        lBasePath = PMLIB_INSTALL_PATH;
    #endif
    }
        
    if(!lBasePath)
    {
        std::cout << "PMLIB_INSTALL_PATH not defined" << std::endl;
        return 1;
    }
    
    Analyzer lAnalyzer(lBasePath);
    lAnalyzer.Analyze();
    
    return 0;
}


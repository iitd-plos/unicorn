
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

#include <map>

namespace pageRankInvertedIndex
{

using namespace pm;
    
#define PAGE_RANK_DATA_TYPE float

#define PAGE_RANK_ITERATIONS 2
#define DAMPENING_FACTOR (float)0.85
#define INITIAL_PAGE_RANK (float)1.0

#define WEB_PAGES_PER_SUBTASK 10000000

#define MAX_BASE_PATH_LENGTH 256
//#define DEFAULT_BASE_PATH (char*)"../../web_dump"
#define DEFAULT_BASE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/pageRank/web_dump_100M"
    
struct invertedIndex
{
    unsigned int srcPage;
    unsigned int destPage;  // srcPage has an outlink to destPage
    unsigned int srcPageOutlinks;    // number of outlinks of srcPage
};
    
struct subtaskMetadata
{
    unsigned long inlinksStartOffset;
    unsigned int inlinksCount;
};
    
struct invertIndexTaskConf
{
	unsigned int totalWebPages;
    unsigned int maxOutlinksPerWebPage;
    unsigned int webPagesPerFile;
    unsigned int webPagesPerSubtask;    
    char basePath[MAX_BASE_PATH_LENGTH];
};

struct pageRankTaskConf
{
	unsigned int totalWebPages;
    unsigned int maxOutlinksPerWebPage;
    unsigned int webPagesPerFile;
    unsigned int webPagesPerSubtask;
    unsigned int iteration;
    PAGE_RANK_DATA_TYPE initialPageRank;
    subtaskMetadata* metadata;  // The number of metadata entries in this array is equal to the number of subtasks
};
    
#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus invertIndex_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
pmStatus pageRank_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuPageRank(pageRankTaskConf& pTaskConf, unsigned int* pWebDump, void* pOutputMem);
#endif

enum memIndex
{
    INVERTED_INDEX_MEM_INDEX = 0,
    INPUT_PAGES_MEM_INDEX = 1,
    MAX_MEM_INDICES
};

enum memIndex2
{
    INVERTED_LINKS_MEM_INDEX = 0,
    OUTPUT_PAGE_RANKS_MEM_INDEX = 1,
    INPUT_PAGE_RANKS_MEM_INDEX = 2,
    MAX_MEM_INDICES2
};
    
void** LoadMappedFiles(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);

void LoadSubtaskBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer, pageRankTaskConf* pTaskConf, unsigned long pSubtaskId);
void LoadReductionBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer);
void PlaceMapIntoSubtaskReductionBuffer(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap);
    
}

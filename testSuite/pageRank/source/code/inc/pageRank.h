
#include <map>

namespace pageRank
{

using namespace pm;
    
#define PAGE_RANK_DATA_TYPE float

#define PAGE_RANK_ITERATIONS 1
#define DAMPENING_FACTOR (float)0.85
#define INITIAL_PAGE_RANK (float)1.0

#define WEB_PAGES_PER_SUBTASK 100000

#define MAX_BASE_PATH_LENGTH 256
//#define DEFAULT_BASE_PATH (char*)"../../web_dump"
#define DEFAULT_BASE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/pageRank/web_dump_1M"

typedef struct pageRankTaskConf
{
	unsigned int totalWebPages;
    unsigned int maxOutlinksPerWebPage;
    unsigned int webPagesPerFile;
    unsigned int webPagesPerSubtask;
    unsigned int iteration;
    PAGE_RANK_DATA_TYPE initialPageRank;
    char basePath[MAX_BASE_PATH_LENGTH];
} pageRankTaskConf;
    
#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus pageRank_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuPageRank(pageRankTaskConf& pTaskConf, unsigned int* pWebDump, void* pOutputMem);
#endif

enum memIndex
{
    MEM_INDEX = 0,
    MAX_MEM_INDICES
};
    
void** LoadMappedFiles(pmTaskInfo pTaskInfo, pmSubtaskInfo pSubtaskInfo);

void LoadSubtaskBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer, pageRankTaskConf* pTaskConf, unsigned long pSubtaskId);
void LoadReductionBufferInMap(std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap, char* pBuffer);
void PlaceMapIntoSubtaskReductionBuffer(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, std::map<unsigned int, PAGE_RANK_DATA_TYPE>& pMap);
    
}

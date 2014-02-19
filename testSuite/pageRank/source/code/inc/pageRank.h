
namespace pageRank
{

using namespace pm;
    
#define PAGE_RANK_DATA_TYPE float

#define PAGE_RANK_ITERATIONS 1
#define DAMPENING_FACTOR (float)0.85
#define INITIAL_PAGE_RANK (float)1.0

#define ONE_SUBTASK_PER_DEVICE

#define WEB_PAGES_PER_SUBTASK 100000

#define MAX_BASE_PATH_LENGTH 256
//#define DEFAULT_BASE_PATH (char*)"../../web_dump"
#define DEFAULT_BASE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/pageRank/web_dump"

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
    OUTPUT_MEM_INDEX = 0,
    INPUT_MEM_INDEX,
    MAX_MEM_INDICES
};
    
void** LoadMappedFiles(pageRankTaskConf* pTaskConf, ulong pSubtaskId);
    
}

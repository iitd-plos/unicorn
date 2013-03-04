
namespace pageRank
{

using namespace pm;
    
#define PAGE_RANK_DATA_TYPE float

#define PAGE_RANK_ITERATIONS 1
#define DAMPENING_FACTOR (float)0.85
#define INITIAL_PAGE_RANK (float)1.0
    
#define WEB_PAGES_PER_SUBTASK 10000

#define MAX_BASE_PATH_LENGTH 256
#define DEFAULT_BASE_PATH (char*)"/Users/tarunberi/Development/git-repositories/pmlib/testSuite/pageRank/build/linux/../../web_dump"
//#define DEFAULT_BASE_PATH (char*)"../../web_dump"

#ifdef BUILD_CUDA
#include <cuda.h>
typedef void (*pageRank_cudaFuncPtr)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
extern pageRank_cudaFuncPtr pageRank_cudaFunc;
#endif
    
typedef struct pageRankTaskConf
{
	unsigned int totalWebPages;
    unsigned int maxOutlinksPerWebPage;
    unsigned int webPagesPerFile;
    unsigned int webPagesPerSubtask;
    unsigned int iteration;
    unsigned int initialPageRank;
    char basePath[MAX_BASE_PATH_LENGTH];
} pageRankTaskConf;

}

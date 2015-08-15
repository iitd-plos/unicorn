
namespace imageFiltering
{

//#define DEFAULT_IMAGE_PATH (char*)"../../images/default.bmp"
#define DEFAULT_IMAGE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/imageFiltering/images/default.bmp"
#define PIXEL_COUNT (size_t)3    // assumes 24-bit RGB
#define IMAGE_SIZE ((size_t)gImageWidth * gImageHeight * PIXEL_COUNT)
    
#define LOAD_IMAGE_INTO_ADDRESS_SPACE
//#define USE_ELLIPTICAL_FILTER
    
#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
#define GENERATE_RANDOM_IMAGE_IN_MEMORY
#endif
    
#ifdef GENERATE_RANDOM_IMAGE_IN_MEMORY
#define DEFAULT_IMAGE_WIDTH 32768
#define DEFAULT_IMAGE_HEIGHT 32768
#endif
    
#define TILE_DIM 2048
#define GPU_BLOCK_DIM 32

#define DEFAULT_FILTER_RADIUS 1
#define MIN_FILTER_RADIUS 1
#define MAX_FILTER_RADIUS 15
#define MAX_FILTER_DIM 32
    
#define DEFAULT_FILTER_RADIUS_STEP 1

#define DEFAULT_ITERATION_COUNT 10
#define MAX_ITERATIONS 25

#define DO_MULTIPLE_CONVOLUTIONS

using namespace pm;

#define MAX_IMAGE_PATH_LENGTH 256

#ifdef BUILD_CUDA
#include <cuda.h>
size_t computeSubtaskReservedMemRequirement(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, unsigned long pSubtaskId, int pSubscriptionStartCol, int pSubscriptionEndCol, int pSubscriptionStartRow, int pSubscriptionEndRow);
pmStatus imageFilter_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo, void* pCudaStream);
int singleGpuImageFilter(void* pInvertedImageData, size_t pImageWidth, size_t pImageHeight, char pFilter[MAX_FILTER_DIM][MAX_FILTER_DIM], size_t pFilterRadius, size_t pImageBytesPerLine, void* pOutputMem);
#endif

#ifdef LOAD_IMAGE_INTO_ADDRESS_SPACE
enum memIndex
{
    INPUT_MEM_INDEX = 0,
    OUTPUT_MEM_INDEX = 1,
    MAX_MEM_INDICES
};
#else
enum memIndex
{
    OUTPUT_MEM_INDEX = 0,
    MAX_MEM_INDICES
};
#endif

typedef struct imageFilterTaskConf
{
    size_t imageWidth;
    size_t imageHeight;
    size_t imageOffset;
    size_t imageBytesPerLine;
    size_t filterRadius;
    char imagePath[MAX_IMAGE_PATH_LENGTH];
    char filter[MAX_FILTER_DIM][MAX_FILTER_DIM];
} imageFilterTaskConf;
    
bool GetSubtaskSubscription(imageFilterTaskConf* pTaskConf, unsigned long pSubtaskId, pmSplitInfo& pSplitInfo, int* pStartCol, int* pEndCol, int* pStartRow, int* pEndRow);

#pragma pack(push)
#pragma pack(1)
typedef struct bitmapHeader
{
    char identifier[2];
    unsigned int filesize;
    short reserved[2];
    int headersize;
    int infoSize;
    int width;
    int height;
    short bitPlanes;
    short bitCount;
    int compression;
    unsigned int imageSize;
    int pixelsPerMeterX;
    int pixelsPerMeterY;
    int colorsUsed;
    int importantColors;
} bitmapHeader;
#pragma pack(pop)
    
}
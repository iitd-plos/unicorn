
namespace imageFiltering
{

//#define DEFAULT_IMAGE_PATH (char*)"../../images/default.bmp"
#define DEFAULT_IMAGE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/imageFiltering/images/default.bmp"
#define PIXEL_COUNT 3    // assumes 24-bit RGB
#define IMAGE_SIZE (gImageWidth * gImageHeight * PIXEL_COUNT)
    
#define TILE_DIM 512
#define GPU_BLOCK_DIM 32
    
#define SOBEL_FILTER
//#define AVERAGE_FILTER_3
//#define AVERAGE_FILTER_5
//#define RANDOM_FILTER

#if defined(SOBEL_FILTER) || defined(AVERAGE_FILTER_3)
#define FILTER_RADIUS 1
#elif defined(AVERAGE_FILTER_5) || defined(RANDOM_FILTER)
#define FILTER_RADIUS 2
#else
#error "No Filter Defined"
#endif
    
#define FILTER_DIM ((2 * FILTER_RADIUS) + 1)
    
#if FILTER_DIM >= GPU_BLOCK_DIM
#error "Too large filter"
#endif

using namespace pm;

#define MAX_IMAGE_PATH_LENGTH 256

#ifdef BUILD_CUDA
#include <cuda.h>
pmStatus imageFilter_cudaLaunchFunc(pmTaskInfo pTaskInfo, pmDeviceInfo pDeviceInfo, pmSubtaskInfo pSubtaskInfo);
#endif

typedef struct imageFilterTaskConf
{
    int imageWidth;
    int imageHeight;
    int imageOffset;
    int imageBytesPerLine;
    char imagePath[MAX_IMAGE_PATH_LENGTH];
    char filter[FILTER_DIM][FILTER_DIM];
} imageFilterTaskConf;

#pragma pack(push)
#pragma pack(1)
typedef struct bitmapHeader
{
    char identifier[2];
    int filesize;
    short reserved[2];
    int headersize;
    int infoSize;
    int width;
    int height;
    short bitPlanes;
    short bitCount;
    int compression;
    int imageSize;
    int pixelsPerMeterX;
    int pixelsPerMeterY;
    int colorsUsed;
    int importantColors;
} bitmapHeader;
#pragma pack(pop)
    
}
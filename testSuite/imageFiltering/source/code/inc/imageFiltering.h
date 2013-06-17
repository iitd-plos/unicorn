
namespace imageFiltering
{

#define DEFAULT_IMAGE_PATH (char*)"../../images/default.bmp"
//#define DEFAULT_IMAGE_PATH (char*)"/Users/tberi/Development/git-repositories/pmlib/testSuite/imageFiltering/images/default.bmp"
#define PIXEL_COUNT 3    // assumes 24-bit RGB
#define IMAGE_SIZE (gImageWidth * gImageHeight * PIXEL_COUNT)
    
#define TILE_DIM 2048
#define GPU_BLOCK_DIM 32

#define DEFAULT_FILTER_RADIUS 1
#define MIN_FILTER_RADIUS 1
#define MAX_FILTER_RADIUS 15
#define MAX_FILTER_DIM 32

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
    int filterRadius;
    char imagePath[MAX_IMAGE_PATH_LENGTH];
    char filter[MAX_FILTER_DIM][MAX_FILTER_DIM];
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
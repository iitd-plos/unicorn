
namespace imageFiltering
{

//#define DEFAULT_IMAGE_PATH (char*)"../../images/default.bmp"
#define DEFAULT_IMAGE_PATH (char*)"/Users/tarunberi/Development/git-repositories/pmlib/testSuite/imageFiltering/images/default.bmp"
#define PIXEL_COUNT 3    // assumes 24-bit RGB
#define IMAGE_SIZE (gImageWidth * gImageHeight * PIXEL_COUNT)
    
#define TILE_DIM 512

using namespace pm;

#define MAX_IMAGE_PATH_LENGTH 256

#ifdef BUILD_CUDA
#include <cuda.h>
typedef void (*imageFiltering_cudaFuncPtr)(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
extern imageFiltering_cudaFuncPtr imageFiltering_cudaFunc;
#endif

typedef struct imageFilterTaskConf
{
    unsigned int imageWidth;
    unsigned int imageHeight;
    unsigned int imageOffset;
    unsigned int imageBytesPerLine;
    char imagePath[MAX_IMAGE_PATH_LENGTH];
#ifdef BUILD_CUDA
	pmCudaLaunchConf cudaLaunchConf;
#endif
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

# Linux Makefile for fft test suite for pmlib

# Usage:
# make - Builds the test suite in release mode
# make DEBUG=1 - Builds the test suite in debug mode
# make clean - Cleans test suite's release build files
# make DEBUG=1 clean - Cleans test suite's debug build files

FFT_1D=0

ifeq ($(FFT_1D), 1)
NAME_SUFFIX=_1d
FLAGS=-DFFT_1D
else
NAME_SUFFIX=_2d
FLAGS=-DFFT_2D
endif

SAMPLE_NAME=fft

# 1 if CUDA code is included in the test suite; 0 otherwise
BUILD_CUDA=$(BUILD_FOR_CUDA)

# 1 if common code is included in the test suite; 0 otherwise
BUILD_COMMON=1

DEBUG=0

OBJECTS= $(SAMPLE_NAME)$(NAME_SUFFIX).o 
CUDA_OBJECTS = $(SAMPLE_NAME)CUDA$(NAME_SUFFIX).o
FLAGS += -DFFT_DATA_TYPE=complex

ifeq ($(FFT_1D), 0)
OBJECTS += matrixTranspose.o
CUDA_OBJECTS += matrixTransposeCUDA.o
FLAGS += -DMATRIX_DATA_TYPE=complex
endif

ifeq ($(BUILD_CUDA), 1)
CUDAFLAGS += -DFFT_DATA_TYPE=complex
LIBRARIES += -lcufft

ifeq ($(FFT_1D), 1)
CUDAFLAGS += -DFFT_1D
else
CUDAFLAGS += -DMATRIX_DATA_TYPE=complex
CUDAFLAGS += -DFFT_2D
endif

endif

LIBRARIES += -L$(PATH_PREFIX)../../../fft/thirdparty/fftw-3.3.3/installation/lib -lfftw3f
INCLUDES += -I$(PATH_PREFIX)../../../fft/thirdparty/fftw-3.3.3/installation/include

ifeq ($(FFT_1D), 0)
INCLUDES += -I$(PATH_PREFIX)../../../matrixTranspose/source/code/inc
endif

include ../../../common/build/linux/Makefile.common

ifeq ($(FFT_1D), 0)
$(OUTDIR)/%.o: $(PATH_PREFIX)../../../matrixTranspose/source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: $(PATH_PREFIX)../../../matrixTranspose/source/code/src/%.cu
	$(NVCC_COMPILER) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@
endif

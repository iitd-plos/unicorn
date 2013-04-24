
# Linux Makefile for matrixTranspose test suite for pmlib

# Usage:
# make - Builds the test suite in release mode
# make DEBUG=1 - Builds the test suite in debug mode
# make clean - Cleans test suite's release build files
# make DEBUG=1 clean - Cleans test suite's debug build files

SAMPLE_NAME=fft

# 1 if CUDA code is included in the test suite; 0 otherwise
BUILD_CUDA=$(BUILD_FOR_CUDA)

# 1 if common code is included in the test suite; 0 otherwise
BUILD_COMMON=1

DEBUG=0

OBJECTS= $(SAMPLE_NAME).o matrixTranspose.o

CUDA_OBJECTS = $(SAMPLE_NAME)CUDA.o matrixTransposeCUDA.o

FLAGS += -DFFT_DATA_TYPE=complex -DMATRIX_DATA_TYPE=complex

ifeq ($(BUILD_CUDA), 1)
CUDAFLAGS += -DFFT_DATA_TYPE=complex -DMATRIX_DATA_TYPE=complex
LIBRARIES += -lcufft
endif

INCLUDES += -I$(PATH_PREFIX)../../../matrixTranspose/source/code/inc

include ../../../common/build/linux/Makefile.common

%.o: $(PATH_PREFIX)../../../matrixTranspose/source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $(OUTDIR)/$@

%.o: $(PATH_PREFIX)../../../matrixTranspose/source/code/src/%.cu
	$(NVCC_COMPILER) $(CUDAFLAGS) $(INCLUDES) -c $< -o $(OUTDIR)/$@


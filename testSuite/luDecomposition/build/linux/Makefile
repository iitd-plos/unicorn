
# Linux Makefile for luDecomposition test suite for pmlib

# Usage:
# make - Builds the test suite in release mode
# make DEBUG=1 - Builds the test suite in debug mode
# make clean - Cleans test suite's release build files
# make DEBUG=1 clean - Cleans test suite's debug build files

SAMPLE_NAME=luDecomposition

# 1 if CUDA code is included in the test suite; 0 otherwise
BUILD_CUDA=$(BUILD_FOR_CUDA)

# 1 if common code is included in the test suite; 0 otherwise
BUILD_COMMON=1

DEBUG=0

OBJECTS= $(SAMPLE_NAME).o

CUDA_OBJECTS = $(SAMPLE_NAME)CUDA.o

ENABLE_BLAS=1

ifeq ($(BUILD_CUDA), 1)
#CUDAFLAGS += -gencode arch=compute_30,code=sm_30
endif

include ../../../common/build/linux/Makefile.common




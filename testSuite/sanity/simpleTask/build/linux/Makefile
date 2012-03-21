
# Linux Makefile for matrixMultiply test suite for pmlib

# Usage:
# make - Builds the test suite in release mode
# make DEBUG=1 - Builds the test suite in debug mode
# make clean - Cleans test suite's release build files
# make DEBUG=1 clean - Cleans test suite's debug build files

SAMPLE_NAME=simpleTask
BUILD_CUDA=0	# 1 if CUDA code is included in the test suite; 0 otherwise
BUILD_COMMON=0	# 1 if common code is included in the test suite; 0 otherwise

OBJECTS= $(SAMPLE_NAME).o

PATH_PREFIX= ../

include $(PATH_PREFIX)../../../common/build/linux/Makefile.common


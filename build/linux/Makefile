
# Linux Makefile for pmlib library

# Usage:
# make - Builds the library in release mode
# make DEBUG=1 - Builds the library in debug mode
# make clean - Cleans library's release build files
# make DEBUG=1 clean - Cleans library's debug build files
# make regressions - Builds the entire regression suite in release mode [Not Implemented Yet]
# make regressions clean - Cleans the entire regression suite's release build files [Not Implemented Yet]

SUPPORT_CUDA=$(BUILD_FOR_CUDA)
SUPPORT_OPENCL=$(BUILD_FOR_OPENCL)

DEBUG=0

ifndef NVCC_PATH
NVCC_PATH=nvcc
endif

PTHREAD_LIBS= -lpthread

BOOST_INCLUDE= -I../../thirdparty/boost

COMPILER=g++
NVCC_COMPILER=$(NVCC_PATH)
FLAGS=-Wall -std=c++11 -fPIC -DUNIX -D$(PLATFORM) -MMD -fopenmp
CUDAFLAGS= -DUNIX --compiler-options -fPIC --machine 64

ifeq ($(PLATFORM), LINUX)
STRIP=strip -s
else
STRIP=strip -x
endif

ifeq ($(DEBUG), 1)
MPI_INCLUDE= -I$(MPI_DEBUG_INSTALL_PATH)/include
MPI_LIB_PATH=$(MPI_DEBUG_INSTALL_PATH)/lib
MPI_LIBS= -L$(MPI_LIB_PATH) -lmpi -lmpi_cxx
ifeq ($(PLATFORM), LINUX)
SO_FLAGS=-Wall -DUNIX -D$(PLATFORM) -shared -Wl,-soname,libpmlib_g.so
else
SO_FLAGS=-Wall -DUNIX -D$(PLATFORM) -shared
endif
else
MPI_INCLUDE= -I$(MPI_RELEASE_INSTALL_PATH)/include
MPI_LIB_PATH=$(MPI_RELEASE_INSTALL_PATH)/lib
MPI_LIBS= -L$(MPI_LIB_PATH) -lmpi -lmpi_cxx
ifeq ($(PLATFORM), LINUX)
SO_FLAGS=-Wall -DUNIX -D$(PLATFORM) -shared -Wl,-soname,libpmlib.so
else
SO_FLAGS=-Wall -DUNIX -D$(PLATFORM) -shared
endif
endif

LIBRARIES= $(PTHREAD_LIBS) $(MPI_LIBS) -lstdc++ -lm -lgomp
ifeq ($(PLATFORM), LINUX)
LIBRARIES += -lrt
endif

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	CUDAFLAGS += -DDEBUG -D_DEBUG -G -g
	BUILD=debug
else
	FLAGS += -O3 -DNDEBUG -g
	BUILD=release
endif

OUTDIR= $(BUILD)

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/libpmlib_g.so
else
	PROGRAM=$(OUTDIR)/libpmlib.so
endif

OBJECTS= $(OUTDIR)/pmAddressSpace.o \
	$(OUTDIR)/pmAffinityTable.o \
	$(OUTDIR)/pmBase.o \
	$(OUTDIR)/pmCallback.o \
	$(OUTDIR)/pmCallbackUnit.o \
	$(OUTDIR)/pmCluster.o \
	$(OUTDIR)/pmCommand.o \
	$(OUTDIR)/pmCommunicator.o \
	$(OUTDIR)/pmController.o \
	$(OUTDIR)/pmDataTypes.o \
	$(OUTDIR)/pmDevicePool.o \
	$(OUTDIR)/pmDispatcherGPU.o \
	$(OUTDIR)/pmExecutionStub.o \
	$(OUTDIR)/pmEventTimeline.o \
	$(OUTDIR)/pmHardware.o \
	$(OUTDIR)/pmHeavyOperations.o \
	$(OUTDIR)/pmLogger.o \
	$(OUTDIR)/pmMemChunk.o \
	$(OUTDIR)/pmMemoryDirectory.o \
	$(OUTDIR)/pmMemoryManager.o \
	$(OUTDIR)/pmOpenCLManager.o \
	$(OUTDIR)/pmStealAgent.o \
	$(OUTDIR)/pmNetwork.o \
	$(OUTDIR)/pmReducer.o \
	$(OUTDIR)/pmRedistributor.o \
	$(OUTDIR)/pmPoolAllocator.o \
	$(OUTDIR)/pmPreprocessorTask.o \
	$(OUTDIR)/pmPublicDefinitions.o \
	$(OUTDIR)/pmResourceLock.o \
	$(OUTDIR)/pmScheduler.o \
	$(OUTDIR)/pmSignalWait.o \
	$(OUTDIR)/pmStubManager.o \
	$(OUTDIR)/pmSubscriptionManager.o \
	$(OUTDIR)/pmSubtaskManager.o \
	$(OUTDIR)/pmSubtaskSplitter.o \
	$(OUTDIR)/pmTask.o \
	$(OUTDIR)/pmTaskExecStats.o \
	$(OUTDIR)/pmTaskManager.o \
	$(OUTDIR)/pmTaskProfiler.o \
	$(OUTDIR)/pmTimer.o \
	$(OUTDIR)/pmTimedEventManager.o \
	$(OUTDIR)/pmTls.o \
	$(OUTDIR)/pmUtility.o 

CUDA_OBJECTS = $(OUTDIR)/pmCudaInterface.o

ifeq ($(SUPPORT_CUDA), 1)
FLAGS += -DSUPPORT_CUDA
CUDAFLAGS += -DSUPPORT_CUDA
INCLUDES += -I$(CUDA_INCLUDE_PATH)
OBJECTS += $(CUDA_OBJECTS)
LIBRARIES += -L$(CUDA_LIB_PATH) -L$(CUDA_DRIVER_LIB_PATH) -lcuda -lcudart
endif

ifeq ($(AUTOCONF), 1)
LIBRARIES += -Wl,-rpath,$(MPI_LIB_PATH),-rpath,$(CUDA_LIB_PATH)
endif

INCLUDES += -I../../source/code/inc $(MPI_INCLUDE) $(BOOST_INCLUDE)

all: precompile compile prelink link final

precompile:
	@echo "*** Compiling ***"

compile: $(OBJECTS)

prelink:
	@echo "\n*** Linking ***"

link: $(PROGRAM)

ifeq ($(DEBUG), 1)
$(PROGRAM): $(OBJECTS)
	$(COMPILER) $(SO_FLAGS) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)
else
$(PROGRAM): $(OBJECTS)
	$(COMPILER) $(SO_FLAGS) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)
	@echo "\n*** Stripping ***"
	$(STRIP) $(PROGRAM)
endif

final:
	@echo "\n*** Done ***"

# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
$(OUTDIR)/%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: ../../source/code/src/%.cu
	$(NVCC_COMPILER) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

clean: 
	rm -f $(OUTDIR)/*.o $(OUTDIR)/*.d core.* $(PROGRAM)

install:
	$(INSTALL) -d $(prefix)
	$(INSTALL) -d $(prefix)/lib
	$(INSTALL) -d $(prefix)/include
	$(INSTALL) $(OUTDIR)/libpmlib.so $(prefix)/lib
	$(INSTALL) ../../source/code/inc/pmPublicDefinitions.h ../../source/code/inc/pmPublicUtilities.h $(prefix)/include

-include $(OUTDIR)/*.d

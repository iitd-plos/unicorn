
# Linux Makefile for pmlib library

# Usage:
# make - Builds the library in release mode
# make DEBUG=1 - Builds the library in debug mode
# make clean - Cleans library's release build files
# make DEBUG=1 clean - Cleans library's debug build files
# make regressions - Builds the entire regression suite in release mode [Not Implemented Yet]
# make regressions clean - Cleans the entire regression suite's release build files [Not Implemented Yet]

SUPPORT_CUDA=1
DEBUG=0

MPI_INCLUDE= -I/usr/lib/openmpi/include
MPI_LIBS= -L/usr/lib/openmpi/lib -lmpi -lmpi_cxx

PTHREAD_LIBS= -lpthread

COMPILER=g++
NVCC_COMPILER=nvcc
STRIP=strip
FLAGS=-Wall -fPIC -DUNIX -DLINUX
CUDAFLAGS = -DUNIX --compiler-options -fPIC
SO_FLAGS=-Wall -DUNIX -DLINUX -shared -Wl,-soname,pmlib.so

LIBRARIES= -lstdc++ -lm $(PTHREAD_LIBS) $(MPI_LIBS)

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	CUDAFLAGS += -DDEBUG -D_DEBUG -g
	BUILD=debug
else
	FLAGS += -O3
	BUILD=release
endif

OUTDIR= $(BUILD)

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/libpmlib_g.so
else
	PROGRAM=$(OUTDIR)/libpmlib.so
endif

OBJECTS= $(OUTDIR)/pmBase.o \
	$(OUTDIR)/pmCallback.o \
	$(OUTDIR)/pmCallbackChain.o \
	$(OUTDIR)/pmCallbackUnit.o \
	$(OUTDIR)/pmCluster.o \
	$(OUTDIR)/pmCommand.o \
	$(OUTDIR)/pmCommunicator.o \
	$(OUTDIR)/pmController.o \
	$(OUTDIR)/pmDevicePool.o \
	$(OUTDIR)/pmDispatcherGPU.o \
	$(OUTDIR)/pmExecutionStub.o \
	$(OUTDIR)/pmMemoryManager.o \
	$(OUTDIR)/pmLogger.o \
	$(OUTDIR)/pmMemSection.o \
	$(OUTDIR)/pmNetwork.o \
	$(OUTDIR)/pmPublicDefinitions.o \
	$(OUTDIR)/pmResourceLock.o \
	$(OUTDIR)/pmSafePriorityQueue.o \
	$(OUTDIR)/pmScheduler.o \
	$(OUTDIR)/pmSignalWait.o \
	$(OUTDIR)/pmStubManager.o \
	$(OUTDIR)/pmTask.o \
	$(OUTDIR)/pmTaskManager.o \
	$(OUTDIR)/pmThread.o \
	$(OUTDIR)/pmTimer.o

CUDA_OBJECTS = $(OUTDIR)/pmDispatcherCUDA.o

ifeq ($(SUPPORT_CUDA), 1)
FLAGS += -DSUPPORT_CUDA
CUDAFLAGS += -DSUPPORT_CUDA
INCLUDES += -I/usr/local/cuda/include
OBJECTS += $(CUDA_OBJECTS)
#LIBRARIES += -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart
endif

INCLUDES= -I../../source/code/inc $(MPI_INCLUDE)

all: prebuild compile link postbuild final

prebuild:

compile: precompile $(OBJECTS)

precompile:
	@echo "*** Compiling ***"

link: 
	@echo "\n*** Linking ***"
	$(COMPILER) $(SO_FLAGS) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)

ifeq ($(DEBUG), 1)
postbuild:
else
postbuild: strip
endif

final:
	@echo "\n*** Done ***"

# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
$(OUTDIR)/%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: ../../source/code/src/%.cu
	$(NVCC_COMPILER) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

strip: $(PROGRAM)
	@echo "\n*** Stripping ***"
	$(STRIP) -s $(PROGRAM)

clean: 
	rm -f $(OUTDIR)/*.o core.* $(PROGRAM)



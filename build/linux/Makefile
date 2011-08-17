
# Linux Makefile for pmlib library

# Usage:
# make - Builds the library in release mode
# make DEBUG=1 - Builds the library in debug mode
# make clean - Cleans library's release build files
# make DEBUG=1 clean - Cleans library's debug build files
# make regressions - Builds the entire regression suite in release mode [Not Implemented Yet]
# make regressions clean - Cleans the entire regression suite's release build files [Not Implemented Yet]

MPI_INCLUDE= -I/usr/lib/openmpi/include
MPI_LIBS= -L/usr/lib/openmpi/lib -lmpi -lmpi_cxx

PTHREAD_LIBS= -lpthread

COMPILER=g++
STRIP=strip
FLAGS=-Wall -fPIC -DUNIX -DLINUX
SO_FLAGS=-Wall -DUNIX -DLINUX -shared -Wl,-soname,pmlib.so

#CUDAFLAGS = -DUNIX --compiler-options -fPIC

DEBUG=0

LIBRARIES= -lstdc++ -lm $(PTHREAD_LIBS) $(MPI_LIBS)

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	#CUDAFLAGS += -DDEBUG -D_DEBUG -g
	BUILD=debug
else
	FLAGS += -O3
	BUILD=release
endif

OUTDIR= $(BUILD)
#LIBRARIES += -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/libpmlib_g.so
else
	PROGRAM=$(OUTDIR)/libpmlib.so
endif

OBJECTS= $(OUTDIR)/pmCluster.o \
	 $(OUTDIR)/pmCommand.o \
	 $(OUTDIR)/pmCommunicator.o \
	 $(OUTDIR)/pmController.o \
	 $(OUTDIR)/pmDevicePool.o \
	 $(OUTDIR)/pmNetwork.o \
	 $(OUTDIR)/pmPublicDefinitions.o \
	 $(OUTDIR)/pmResourceLock.o \
	 $(OUTDIR)/pmSafePriorityQueue.o \
	 $(OUTDIR)/pmSignalWait.o \
	 $(OUTDIR)/pmThread.o \
	 $(OUTDIR)/pmTimer.o \
	 $(OUTDIR)/pmMemoryManager.o \
	 $(OUTDIR)/pmScheduler.o \
	 $(OUTDIR)/pmTaskManager.o

INCLUDES= -I../../source/code/inc $(MPI_INCLUDE)
#INCLUDES += -I/usr/local/cuda/include

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

strip: $(PROGRAM)
	@echo "\n*** Stripping ***"
	$(STRIP) -s $(PROGRAM)

clean: 
	rm -f $(OUTDIR)/*.o core.* $(PROGRAM)




# Linux Makefile for pmlib library

# Usage:
# make - Builds the library in release mode
# make DEBUG=1 - Builds the library in debug mode
# make clean - Cleans library's release build files
# make DEBUG=1 clean - Cleans library's debug build files
# make regressions - Builds the entire regression suite in release mode [Not Implemented Yet]
# make regressions clean - Cleans the entire regression suite's release build files [Not Implemented Yet]

COMPILER=g++
FLAGS=-Wall -fPIC -DUNIX -DLINUX
SO_FLAGS=-Wall -DUNIX -DLINUX -shared -Wl,-soname,pmlib.so

#CUDAFLAGS = -DUNIX --compiler-options -fPIC

DEBUG=0

LIBRARIES= -lstdc++ -lmpi -lmpi_cxx -lm -lpthread

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
	 $(OUTDIR)/pmSafePriorityQueue.o \
	 $(OUTDIR)/pmSignalWait.o \
	 $(OUTDIR)/pmThread.o

INCLUDES= -I..
#INCLUDES += -I/usr/local/cuda/include

all: $(OBJECTS)
	$(COMPILER) $(SO_FLAGS) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)


# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
$(OUTDIR)/%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $(input) -o $(output)


clean: 
	rm -f $(OUTDIR)/*.o core.* $(PROGRAM)



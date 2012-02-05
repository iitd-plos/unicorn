
# Linux Makefile for pmlib library

DEBUG=0

COMPILER=g++
NVCC_COMPILER=nvcc
STRIP=strip
FLAGS=-Wall -DUNIX -DLINUX
CUDAFLAGS= -DUNIX --compiler-options

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	CUDAFLAGS += -DDEBUG -D_DEBUG -g
	BUILD=debug
	LIBRARIES=-L../../../../../build/linux/debug -lpmlib_g
else
	FLAGS += -O3 -DNDEBUG
	BUILD=release
	LIBRARIES=-L../../../../../build/linux/release -lpmlib
endif

OUTDIR= $(BUILD)

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/basic_g.exe
else
	PROGRAM=$(OUTDIR)/basic.exe
endif

OBJECTS= $(OUTDIR)/basic.o

CUDA_OBJECTS = 

INCLUDES += -I../../../../../source/code/inc

all: prebuild compile link postbuild final

prebuild:

compile: precompile $(OBJECTS)

precompile:
	@echo "*** Compiling ***"

link: 
	@echo "\n*** Linking ***"
	$(COMPILER) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)

ifeq ($(DEBUG), 1)
postbuild:
else
postbuild: strip
endif

final:
	@echo "\n*** Done ***"

# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
$(OUTDIR)/%.o: ../../src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: ../../src/%.cu
	$(NVCC_COMPILER) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

strip: $(PROGRAM)
	@echo "\n*** Stripping ***"
	$(STRIP) -s $(PROGRAM)

clean: 
	rm -f $(OUTDIR)/*.o core.* $(PROGRAM)



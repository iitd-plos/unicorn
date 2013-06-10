# Linux Makefile for pmlib analyzer

# Usage:
# make - Builds the analyzer in release mode
# make DEBUG=1 - Builds the analyzer in debug mode
# make clean - Cleans analyzer's release build files
# make DEBUG=1 clean - Cleans analyzer's debug build files

COMPILER=g++
NVCC_COMPILER=nvcc

STD_TR1_INCLUDE= -I/usr/include/c++/4.4.3

FLAGS +=-Wall -DUNIX -DLINUX

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	BUILD=debug
else
	FLAGS += -O3 -DNDEBUG
	BUILD=release
endif

OUTDIR= $(BUILD)

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/analyzer_g.exe
else
	PROGRAM=$(OUTDIR)/analyzer.exe
endif

INCLUDES += -I../../source/code/inc $(STD_TR1_INCLUDE)

ifeq ($(PLATFORM), LINUX)
	STRIP=strip -s
else
	STRIP=strip -u -r
endif

OBJECTS= analyzer.o \
        benchmark.o \
        graph.o

all: prebuild compile link postbuild final

prebuild:

compile: precompile $(OBJECTS)

precompile:
	@echo "*** Compiling ***"

link: 
	@echo "\n*** Linking ***"
	$(COMPILER) -o $(PROGRAM) $(OBJECTS:%.o=$(OUTDIR)/%.o) $(LIBRARIES)

ifeq ($(DEBUG), 1)
postbuild:
else
postbuild: strip
endif

final:
	@echo "\n*** Done ***"

# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $(OUTDIR)/$@

strip: $(PROGRAM)
	@echo "\n*** Stripping ***"
	$(STRIP) $(PROGRAM)

clean: 
	rm -f $(OUTDIR)/*.o core.* $(PROGRAM)



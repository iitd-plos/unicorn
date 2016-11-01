# Linux Makefile for pmlib analyzer

# Usage:
# make - Builds the analyzer in release mode
# make DEBUG=1 - Builds the analyzer in debug mode
# make clean - Cleans analyzer's release build files
# make DEBUG=1 clean - Cleans analyzer's debug build files

COMPILER=g++

BOOST_INCLUDE= -I../../thirdparty/regex_boost

FLAGS +=-Wall -DUNIX -DLINUX -MMD -std=c++11

ifeq ($(DEBUG), 1)
	FLAGS += -DDEBUG -D_DEBUG -g
	BUILD=debug
else
	FLAGS += -O3 -DNDEBUG
	BUILD=release
endif

ifeq ($(AUTOCONF), 1)
FLAGS += -DBUILD_FOR_DISTRIBUTION -DDISTRIB_INSTALL_PATH=\"$(prefix)\"
endif

OUTDIR= $(BUILD)

ifeq ($(DEBUG), 1)
	PROGRAM=$(OUTDIR)/analyzer_g.exe
else
	PROGRAM=$(OUTDIR)/analyzer.exe
endif

INCLUDES += -I../../source/code/inc $(BOOST_INCLUDE)
LIBRARIES += -lpthread

ifeq ($(PLATFORM), LINUX)
	STRIP=strip -s
else
	STRIP=strip -u -r
endif

BOOST_REGEX_SRCS= $(wildcard ../../thirdparty/regex_boost/libs/regex/src/*.cpp)
ANALYZER_SRCS= $(wildcard ../../source/code/src/*.cpp)

OBJS= $(BOOST_REGEX_SRCS:../../thirdparty/regex_boost/libs/regex/src/%.cpp=%.o)
OBJS += $(ANALYZER_SRCS:../../source/code/src/%.cpp=%.o)

OBJECTS= $(OBJS:%.o=$(OUTDIR)/%.o)

all: precompile compile prelink link final

precompile:
	@echo "*** Compiling ***"

compile: $(OBJECTS)

prelink:
	@echo "\n*** Linking ***"

link: $(PROGRAM)

ifeq ($(DEBUG), 1)
$(PROGRAM): $(OBJECTS)
	$(COMPILER) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)
else
$(PROGRAM): $(OBJECTS)
	$(COMPILER) -o $(PROGRAM) $(OBJECTS) $(LIBRARIES)
	@echo "\n*** Stripping ***"
	$(STRIP) $(PROGRAM)
endif

final:
	@echo "\n*** Done ***"

# Generic rule for compiling any cpp file. Any specific rules must
# be added after this rule as the later one overrides
$(OUTDIR)/%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: ../../thirdparty/regex_boost/libs/regex/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

clean: 
	rm -f $(OUTDIR)/*.o $(OUTDIR)/*.d core.* $(PROGRAM)

install:
	$(INSTALL) -d $(prefix)
	$(INSTALL) -d $(prefix)/analyzers
	$(INSTALL) -d $(prefix)/analyzers/bin
	$(INSTALL) $(PROGRAM) $(prefix)/analyzers/bin
	rm -rf $(prefix)/analyzers/conf
	cp -R $(PMLIB_INSTALL_PATH)/analyzers/conf $(prefix)/analyzers/conf

-include $(OUTDIR)/*.d


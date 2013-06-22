# Linux Makefile for pmlib analyzer

# Usage:
# make - Builds the analyzer in release mode
# make DEBUG=1 - Builds the analyzer in debug mode
# make clean - Cleans analyzer's release build files
# make DEBUG=1 clean - Cleans analyzer's debug build files

COMPILER=g++

BOOST_INCLUDE= -I../../thirdparty/regex_boost

FLAGS +=-Wall -DUNIX -DLINUX -MMD

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
$(OUTDIR)/%.o: ../../source/code/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: ../../thirdparty/regex_boost/libs/regex/src/%.cpp
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

strip: $(PROGRAM)
	@echo "\n*** Stripping ***"
	$(STRIP) $(PROGRAM)

clean: 
	rm -f $(OUTDIR)/*.o $(OUTDIR)/*.d core.* $(PROGRAM)

-include $(OUTDIR)/*.d


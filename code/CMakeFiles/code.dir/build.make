# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/wdcdrive/cpp/poroelasticity/fixed-stress-split

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wdcdrive/cpp/poroelasticity/fixed-stress-split

# Include any dependencies generated for this target.
include code/CMakeFiles/code.dir/depend.make

# Include the progress variables for this target.
include code/CMakeFiles/code.dir/progress.make

# Include the compile flags for this target's objects.
include code/CMakeFiles/code.dir/flags.make

code/CMakeFiles/code.dir/source/fss-poroel.cc.o: code/CMakeFiles/code.dir/flags.make
code/CMakeFiles/code.dir/source/fss-poroel.cc.o: code/source/fss-poroel.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/wdcdrive/cpp/poroelasticity/fixed-stress-split/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object code/CMakeFiles/code.dir/source/fss-poroel.cc.o"
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/code.dir/source/fss-poroel.cc.o -c /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code/source/fss-poroel.cc

code/CMakeFiles/code.dir/source/fss-poroel.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code.dir/source/fss-poroel.cc.i"
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code/source/fss-poroel.cc > CMakeFiles/code.dir/source/fss-poroel.cc.i

code/CMakeFiles/code.dir/source/fss-poroel.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code.dir/source/fss-poroel.cc.s"
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code/source/fss-poroel.cc -o CMakeFiles/code.dir/source/fss-poroel.cc.s

code/CMakeFiles/code.dir/source/fss-poroel.cc.o.requires:

.PHONY : code/CMakeFiles/code.dir/source/fss-poroel.cc.o.requires

code/CMakeFiles/code.dir/source/fss-poroel.cc.o.provides: code/CMakeFiles/code.dir/source/fss-poroel.cc.o.requires
	$(MAKE) -f code/CMakeFiles/code.dir/build.make code/CMakeFiles/code.dir/source/fss-poroel.cc.o.provides.build
.PHONY : code/CMakeFiles/code.dir/source/fss-poroel.cc.o.provides

code/CMakeFiles/code.dir/source/fss-poroel.cc.o.provides.build: code/CMakeFiles/code.dir/source/fss-poroel.cc.o


# Object files for target code
code_OBJECTS = \
"CMakeFiles/code.dir/source/fss-poroel.cc.o"

# External object files for target code
code_EXTERNAL_OBJECTS =

code/code: code/CMakeFiles/code.dir/source/fss-poroel.cc.o
code/code: code/CMakeFiles/code.dir/build.make
code/code: lib/liblib.a
code/code: /usr/local/lib/libdeal_II.g.so.8.4.1
code/code: /usr/lib/liblapack.so
code/code: /usr/lib/libblas.so
code/code: /usr/lib/x86_64-linux-gnu/libz.so
code/code: code/CMakeFiles/code.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/wdcdrive/cpp/poroelasticity/fixed-stress-split/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable code"
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/code.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
code/CMakeFiles/code.dir/build: code/code

.PHONY : code/CMakeFiles/code.dir/build

code/CMakeFiles/code.dir/requires: code/CMakeFiles/code.dir/source/fss-poroel.cc.o.requires

.PHONY : code/CMakeFiles/code.dir/requires

code/CMakeFiles/code.dir/clean:
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code && $(CMAKE_COMMAND) -P CMakeFiles/code.dir/cmake_clean.cmake
.PHONY : code/CMakeFiles/code.dir/clean

code/CMakeFiles/code.dir/depend:
	cd /media/wdcdrive/cpp/poroelasticity/fixed-stress-split && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wdcdrive/cpp/poroelasticity/fixed-stress-split /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code /media/wdcdrive/cpp/poroelasticity/fixed-stress-split /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code /media/wdcdrive/cpp/poroelasticity/fixed-stress-split/code/CMakeFiles/code.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : code/CMakeFiles/code.dir/depend

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /home/karen/Documents/jetbrain/clion-2017.2.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/karen/Documents/jetbrain/clion-2017.2.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/karen/workspace/data_science/opencv/learnopencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/learningcpp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/learningcpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/learningcpp.dir/flags.make

CMakeFiles/learningcpp.dir/main.cpp.o: CMakeFiles/learningcpp.dir/flags.make
CMakeFiles/learningcpp.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/learningcpp.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/learningcpp.dir/main.cpp.o -c /home/karen/workspace/data_science/opencv/learnopencv/main.cpp

CMakeFiles/learningcpp.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learningcpp.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/karen/workspace/data_science/opencv/learnopencv/main.cpp > CMakeFiles/learningcpp.dir/main.cpp.i

CMakeFiles/learningcpp.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learningcpp.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/karen/workspace/data_science/opencv/learnopencv/main.cpp -o CMakeFiles/learningcpp.dir/main.cpp.s

CMakeFiles/learningcpp.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/learningcpp.dir/main.cpp.o.requires

CMakeFiles/learningcpp.dir/main.cpp.o.provides: CMakeFiles/learningcpp.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/learningcpp.dir/build.make CMakeFiles/learningcpp.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/learningcpp.dir/main.cpp.o.provides

CMakeFiles/learningcpp.dir/main.cpp.o.provides.build: CMakeFiles/learningcpp.dir/main.cpp.o


# Object files for target learningcpp
learningcpp_OBJECTS = \
"CMakeFiles/learningcpp.dir/main.cpp.o"

# External object files for target learningcpp
learningcpp_EXTERNAL_OBJECTS =

learningcpp: CMakeFiles/learningcpp.dir/main.cpp.o
learningcpp: CMakeFiles/learningcpp.dir/build.make
learningcpp: CMakeFiles/learningcpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable learningcpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/learningcpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/learningcpp.dir/build: learningcpp

.PHONY : CMakeFiles/learningcpp.dir/build

CMakeFiles/learningcpp.dir/requires: CMakeFiles/learningcpp.dir/main.cpp.o.requires

.PHONY : CMakeFiles/learningcpp.dir/requires

CMakeFiles/learningcpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/learningcpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/learningcpp.dir/clean

CMakeFiles/learningcpp.dir/depend:
	cd /home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/karen/workspace/data_science/opencv/learnopencv /home/karen/workspace/data_science/opencv/learnopencv /home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug /home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug /home/karen/workspace/data_science/opencv/learnopencv/cmake-build-debug/CMakeFiles/learningcpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/learningcpp.dir/depend


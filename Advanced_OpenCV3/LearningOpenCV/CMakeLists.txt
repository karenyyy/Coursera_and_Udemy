cmake_minimum_required(VERSION 3.8)
project(learningopencv)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp AlphaBlending)
add_executable(learningcpp ${SOURCE_FILES})



find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(OpenCVTest ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( OpenCVTest ${OpenCV_LIBS} )

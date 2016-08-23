cmake_minimum_required(VERSION 2.6)
project(optimization_toolkit)

# If the user doesn't specify a build type (e.g. with -DCMAKE_BUILD_TYPE=Release),
# choose Debug as the default value.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build Type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Add compiler flags here (e.g. -O3, -Wall, etc.)
# This is also needed to use C++11. "-std=c++11"
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3")

# Set the location of the final executable to be where cmake/make are called
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Tell the compiler where to find the headers
include_directories(include)
include_directories(include/Functions)
include_directories(include/Trainers)
include_directories(include/Utils)

# All the code is in the src subdirectory
file(GLOB_RECURSE SOURCES "src/*.cpp")

#Ignore the listed files
list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/Functions/NeuralNetwork.cpp")

# The first argument will be the name of the executable (e.g. "[NAME].exe")
add_executable(optimization_toolkit ${SOURCES})

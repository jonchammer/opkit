cmake_minimum_required(VERSION 2.6)
project(demo)

# If the user doesn't specify a build type (e.g. with -DCMAKE_BUILD_TYPE=Release),
# choose Release as the default value.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Compiling in ${CMAKE_BUILD_TYPE} mode.")

# Add compiler flags here (e.g. -O3, -Wall, etc.)
# This is also needed to use C++11. "-std=c++11"
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -g -gdwarf-2 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -Wextra -g -Wfatal-errors")

# Set the location of the final executable to be where cmake/make are called
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Since the library is header-only, we need to specify where they are
include_directories("/usr/local/include/opkit")

include_directories("/usr/include/openblas")

# Create lists of all of the header files and the source files
set(SOURCES src/neuralNetworkBatchTest.cpp)
add_executable(demo ${SOURCES})

# Find and link the BLAS Library
find_package(BLAS REQUIRED)
message("BLAS Library Found: ${BLAS_LIBRARIES}")
target_link_libraries(demo ${BLAS_LIBRARIES})

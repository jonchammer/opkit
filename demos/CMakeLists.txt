cmake_minimum_required(VERSION 2.6)
project(demo)

# If the user doesn't specify a build type (e.g. with -DCMAKE_BUILD_TYPE=Release),
# choose Release as the default value.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message("Compiling in ${CMAKE_BUILD_TYPE} mode.")

# If the user doesn't specify which type of acceleration to use (e.g. with
# -DCMAKE_ACCELERATION_TYPE=CPU or -DCMAKE_ACCELERATION_TYPE=GPU), choose CPU
# as the default value.
if (NOT CMAKE_ACCELERATION_TYPE)
    set (CMAKE_ACCELERATION_TYPE "CPU")
endif (NOT CMAKE_ACCELERATION_TYPE)

# Let the user know what type of acceleration they are using
if (CMAKE_ACCELERATION_TYPE STREQUAL "CPU")
    message("Applying CPU acceleration only.")
elseif (CMAKE_ACCELERATION_TYPE STREQUAL "GPU")
    message("Applying CPU + GPU acceleration.")
else(CMAKE_ACCELERATION_TYPE STREQUAL "CPU")
    message(WARNING "Acceleration type: ${CMAKE_ACCELERATION_TYPE} not recognized. Please select 'CPU' or 'GPU'")
endif(CMAKE_ACCELERATION_TYPE STREQUAL "CPU")

# Add compiler flags here (e.g. -O3, -Wall, etc.)
# This is also needed to use C++11. "-std=c++11"
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -g -gdwarf-2 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -Wextra -g -Wfatal-errors")

# Set the location of the final executable to be where cmake/make are called
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Specify the location of the headers that are used
include_directories("/usr/local/include/opkit")
include_directories("/usr/include/openblas")

# Compile using the GPU if the user wanted that
if (CMAKE_ACCELERATION_TYPE STREQUAL "GPU")
    add_definitions(-DOPKIT_NVBLAS)
endif(CMAKE_ACCELERATION_TYPE STREQUAL "GPU")

# Create lists of all of the header files and the source files
set(SOURCES src/NeuralNetworkTestSuite.cpp)
add_executable(demo ${SOURCES})

# Add other GPU libraries if the user wanted that
set(GPU_LIBRARIES "")
if (CMAKE_ACCELERATION_TYPE STREQUAL "GPU")
    set(GPU_LIBRARIES "nvblas")
endif(CMAKE_ACCELERATION_TYPE STREQUAL "GPU")

# Find and link the BLAS Library
find_package(BLAS REQUIRED)
message("BLAS Library Found: ${BLAS_LIBRARIES}")

# Link all necessary libraries
target_link_libraries(demo ${GPU_LIBRARIES} ${BLAS_LIBRARIES})

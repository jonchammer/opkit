#!/bin/bash

# Declare variables
COMPILER="g++"   #"clang++-3.8"
INCLUDE_DIR="../include"
SRC_FILE=""
DEST_NAME=""

# NOTE: Values should be "RELEASE" or "DEBUG"
MODE="RELEASE"

# NOTE: Values should be "CPU", "GPU", or "NONE"
ACCELERATION="CPU"

# Text for the help menu
MENU="Usage: ./build.sh [args]
-s: Source file name (.cpp)
-o: Output filename
-c: Switch compiler (e.g. g++ or clang++-3.8)
-d: Compile with debug flags (release is default)
-a: Change acceleration type. Values should be: CPU, GPU, or NONE.
-h: Display this menu."

while getopts s:o:c:d:a:h option
do
 case "${option}"
 in
 s) SRC_FILE=${OPTARG};;
 o) DEST_NAME=${OPTARG};;
 c) COMPILER=${OPTARG};;
 d) MODE="DEBUG";;
 a) ACCELERATION=${OPTARG};;
 h) echo "$MENU"
exit

 esac
done

# Ensure the user has provided a source file
if [[ $SRC_FILE == "" ]]; then
    echo "Source file required! Please run again with -s."
    echo "$MENU"
    exit
fi

# Create a default destination name based on the src file
if [[ $DEST_NAME == "" ]]; then
    mkdir -p bin
    DEST_NAME="./bin/"
    DEST_NAME+="${SRC_FILE%%.*}"
fi

DEBUG_FLAGS="-Wall -Wfatal-errors -Wno-unused-value -g -std=c++11 -rdynamic -O0"
RELEASE_FLAGS="-O3 -std=c++11 -g -rdynamic"

# Specify where to find OpenBLAS (if necessary)
OPENBLAS_INCLUDE_DIR="/usr/local/include/openblas/"
OPENBLAS_LIBRARY_DIR="/usr/local/lib/openblas/"

# NOTE: When compiling with IL, set EXTRA_INCLUDES to "-I /usr/include/IL" and EXTRA_LIBS to "-lIL"
EXTRA_INCLUDES=""
EXTRA_LIBS=""


if [ $MODE == "RELEASE" ]; then
    COMPILE_FLAGS=$RELEASE_FLAGS
    echo "Compiling in RELEASE mode"
else
    COMPILE_FLAGS=$DEBUG_FLAGS
    echo "Compiling in DEBUG mode"
fi

# Compile with CPU accleration
if [ $ACCELERATION == "CPU" ] ; then
    echo "Compiling with CPU Acceleration"
    $COMPILER $COMPILE_FLAGS -I $OPENBLAS_INCLUDE_DIR -I $INCLUDE_DIR $EXTRA_INCLUDES -DOPKIT_OPEN_BLAS $SRC_FILE -o $DEST_NAME -Wl,-rpath,$OPENBLAS_LIBRARY_DIR -L$OPENBLAS_LIBRARY_DIR -lopenblas $EXTRA_LIBS

# Compile with GPU acceleration
elif [ $ACCELERATION == "GPU" ] ; then
    echo "Compiling with GPU Acceleration"
    export LD_LIBRARY_PATH=$OPENBLAS_LIBRARY_DIR
    $COMPILER $COMPILE_FLAGS -I $OPENBLAS_INCLUDE_DIR -I $INCLUDE_DIR $EXTRA_INCLUDES -DOPKIT_NVBLAS $SRC_FILE -o $DEST_NAME -Wl,-rpath,$OPENBLAS_LIBRARY_DIR -L$OPENBLAS_LIBRARY_DIR -lnvblas -lopenblas $EXTRA_LIBS

# Compile with no acceleration
else
    echo "Compiling with NO acceleration"
    $COMPILER $COMPILE_FLAGS -I $INCLUDE_DIR $EXTRA_INCLUDES -DOPKIT_CPU_ONLY $SRC_FILE -o $DEST_NAME $EXTRA_LIBS
fi

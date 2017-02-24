# opkit - A fast, comprehensive C++11 optimization toolkit

## Important Features
* Templated
* Header-only
* Extensive support for training and using neural networks
* Wide support for various optimization techniques (e.g. gradient descent, evolutionary optimization, etc.)
* Supports CPU & GPU acceleration via BLAS

## Installation Instructions

#### Dependencies
* OpenBlas - opkit requires linking with a BLAS library in order to accelerate
certain computations (anything that uses "Acceleration.h"). OpenBLAS is
recommended for Linux-based platforms.

* NVBlas - If you need GPU acceleration, you will also need NVBlas
(provided by NVidia). This will offload some of the more expensive arithmetic
instructions onto a local GPU to (ideally) improve performance. See the GPU
Acceleration section below for more information.

* Cmake - CMake is not strictly required since the library is header-only. It
is used to automatically generate the master header, ```opkit.h```, and copy
all the library headers into the default installation directory. If you would
prefer to take over that process, CMake is not necessary.

On Linux-based platforms, OpenBlas and Cmake can be installed by using apt-get:

```bash
sudo apt-get install openblas cmake
```

Installing NVBlas is more complicated. See NVidia's documentation for installing
the Cuda Development Kit, which includes NVBlas.

#### Installation
The following commands can be used to download and install the latest version
of opkit:

```bash
git clone https://gitlab.com/jhammer/OptimizationToolkit.git
cd ./build
cmake ..
sudo make install
```

#### Default Install Directory
On Ubuntu 16, the headers will be copied to:
/usr/local/include/opkit

On Windows (assuming Administrator privileges have been granted):
C:/Program Files (x86)/optimization_toolkit/include/opkit

## Compilation
This is an example compilation command (using g++):

```bash
g++ -std=c++11 -O3 test.cpp -o test -lopenblas
```

**NOTE:** This assumes that the header files that comprise the library are in a
directory accessible by the path. If that is not the case, you will have to
compile with the -I flag.

**NOTE 2:** It may also be necessary to tell the compiler where to find the BLAS
headers (again using the -I) flag.

## GPU Acceleration
opkit can take advantage of a local GPU to accelerate certain operations by
using NVBlas. Doing so is not guaranteed to speed up training, however. Assuming
NVBlas is already installed, a GPU-accelerated application can be compiled as
follows:

```bash
g++ -O3 -std=c++11 -DOPKIT_NVBLAS test.cpp -o test -lnvblas -lopenblas
```

The variable ```OPKIT_NVBLAS``` is used to tell opkit to make use of NVBlas
acceleration. It needs to be set either in the compilation command or via an
explicit ```#define OPKIT_NVBLAS``` in the user's application.

**NOTE:** In order for NVBlas to operate correctly, a file named 'nvblas.conf' must
reside in the executable directory. An example file can be found in the /docs
folder.

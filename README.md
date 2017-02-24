opkit - A fast, comprehensive C++11 optimization toolkit

Important Features
------------------
* Templated
* Header-only
* Extensive support for training and using neural networks
* Wide support for various optimization techniques (e.g. gradient descent, evolutionary optimization, etc.)
* Supports CPU & GPU acceleration via BLAS

Installation Instructions
-------------------------
git clone https://gitlab.com/jhammer/OptimizationToolkit.git
cd ./build
cmake ..
sudo make install

When the library is installed, the headers will be copied into
/usr/local/include/opkit. On Windows, this command will attempt to install to
C:/Program Files (x86)/optimization_toolkit, but you have to have Administrator
privileges for the installation to succeed.

Dependencies
------------
opkit requires linking with a BLAS library in order to accelerate certain
computations (anything that uses "Acceleration.h"). OpenBLAS is recommended for
Linux-based platforms. Some BLAS libraries (e.g. OpenBLAS) require a threading
library for parallelism, so it's probably a good idea to link to that too.

Example compilation command*
----------------------------
g++ -std=c++11 -O3 test.cpp -o test -lopenblas -lpthread

* This assumes that the header files that comprise the library are in a directory
accessible by the path. If that is not the case, you will have to compile with
the -I flag.

GPU Acceleration
-----------------
opkit can take advantage of a local GPU to accelerate certain operations by
using NVBlas. Doing so is not guaranteed to speed up training, however. Assuming
NVBlas is already installed, an application can be compiled as follows:

g++ -O3 -std=c++11 -DOPKIT_NVBLAS test.cpp -o test -lnvblas -lopenblas

NOTE: In order for NVBlas to operate correctly, a file named 'nvblas.conf' must
reside in the executable directory. An example file can be found in the /docs
folder.

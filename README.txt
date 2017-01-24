opkit - A comprehensive C++11 header only optimization toolkit

Installation Instructions
-------------------------
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

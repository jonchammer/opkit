opkit - A comprehensive optimization toolkit

Build Instructions
-------------------------
cd ./build
cmake ..
make

The static library (libopkit.a) will be saved in the 'build' directory.

Installation Instructions
-------------------------
cd ./build
cmake ..
sudo make install

When the library is installed, the static library file (libopkit.a) will be copied into
/usr/local/lib, and the headers will be copied into /usr/local/include/opkit.
On Windows, this command will attempt to install to C:/Program Files (x86)/optimization_toolkit,
but you have to have Administrator priviledges for the installation to succeed.

Example compilation command
---------------------------
g++ -std=c++11 -O3 test.cpp -o test -lopkit


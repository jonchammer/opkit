Athena - A comprehensive optimization toolkit

Build Instructions
-------------------------
cd ./build
cmake ..
make

The static library (libathena.a) will be saved in the 'build' directory.

Installation Instructions
-------------------------
cd ./build
cmake ..
sudo make install

When the library is installed, the static library file (libathena.a) will be copied into
/usr/local/lib, and the headers will be copied into /usr/local/include/athena.

Example compilation command
---------------------------
g++ -std=c++11 -O3 test.cpp -o test -lathena


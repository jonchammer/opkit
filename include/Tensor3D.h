/* 
 * File:   Tensor3D.h
 * Author: Jon C. Hammer
 *
 * Created on August 20, 2016, 10:27 AM
 */

#ifndef TENSOR3D_H
#define TENSOR3D_H

#include <vector>
using namespace std;

// This class provides a 3D array wrapper around a normal C++ vector. It handles
// the math necessary to provide the user with a 3-index interface, plus range
// checking. This is used primarily with Convolutional Models.
class Tensor3D
{
public:
    // Create a new Tensor that wraps the given vector (starting with
    // 'parametersStart'). The virtual dimensions are 'width'x'height'x'depth'.
    Tensor3D(vector<double>& parameters, const size_t parametersStart, 
        const size_t width, const size_t height, const size_t depth);
    
    Tensor3D(double* parameters, const size_t parametersStart,
        const size_t width, const size_t height, const size_t depth);
    
    // Retrieve and modify a given cell in the virtual 3D Tensor
    double get(const int x, const int y, const int z) const;
    void set(const int x, const int y, const int z, const double val);
    void add(const int x, const int y, const int z, const double val);
    
    // Getters / Setters
    size_t getWidth()  const;
    size_t getHeight() const;
    size_t getDepth()  const;
    
private:
    // The vector that is wrapped (and where to start looking)
    //vector<double>& mParameters;
    double* mParameters;
    size_t mParametersStart;
    
    // The dimensions of this tensor (x, y, z)
    size_t mWidth, mHeight, mDepth;
};

#endif /* TENSOR3D_H */

